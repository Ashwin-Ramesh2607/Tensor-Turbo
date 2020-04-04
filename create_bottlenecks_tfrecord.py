import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def get_class_label(image_path):
    class_label = tf.strings.split(image_path, os.path.sep)
    boolean_encoded = class_label[-2] == CLASS_LABELS
    one_hot_encoded = tf.dtypes.cast(boolean_encoded, tf.uint8)
    return one_hot_encoded

def decode_image(image):
    image = tf.io.decode_image(image, channels = 3, dtype = tf.float32, expand_animations = False)
    image = tf.image.resize(image, [224, 224], method = tf.image.ResizeMethod.BICUBIC, antialias = True)
    image = tf.clip_by_value(image, clip_value_min = 0., clip_value_max = 1.)
    image = tf.ensure_shape(image, (224, 224, 3))
    return image

def forward_pass(image):
    image = tf.expand_dims(image, axis = 0)
    bottlenecks = feature_extractor(image)
    return bottlenecks[0]
	
def create_bottlenecks_vectors(image_path):
	class_label = get_class_label(image_path)
	image = tf.io.read_file(image_path)
	image = decode_image(image)
	bottlenecks = forward_pass(image)
	
	bottlenecks = tf.io.serialize_tensor(bottlenecks)
	class_label = tf.io.serialize_tensor(class_label)
	return bottlenecks, class_label

def create_train_feature(value):
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
	train_feature = tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
	return train_feature

def create_train_example(bottleneck_vector, class_label):
	feature = {
		'bottleneck_vector': create_train_feature(bottleneck_vector),
		'label': create_train_feature(class_label)
	}
	train_example = tf.train.Example(features = tf.train.Features(feature = feature))
	return train_example

def create_bottlenecks_tfrecord():
    image_path_DS = tf.data.Dataset.list_files('local/tf_files/images/*/*', shuffle = True)
    CLASS_LABELS = np.array(['chocolate_cake', 'dumplings', 'french_fries', 'hamburger', 'macarons', 'onion_rings', 'pho', 'pizza', 'samosa', 'sushi'])

    feature_extractor =  hub.load('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4')
    bottleneck_DS = image_path_DS.map(create_bottlenecks_vectors, num_parallel_calls =  tf.data.experimental.AUTOTUNE)

    record_file = 'images.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for bottleneck_vector, class_label in bottleneck_DS:
            train_example = create_train_example(bottleneck_vector, class_label)	
            writer.write(train_example.SerializeToString())

