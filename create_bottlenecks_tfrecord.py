import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def get_class_label(image_path, class_labels):
    class_label = tf.strings.split(image_path, os.path.sep)
    boolean_encoded = class_label[-2] == class_labels
    one_hot_encoded = tf.dtypes.cast(boolean_encoded, tf.uint8)
    return one_hot_encoded


def decode_image(image, input_image_size):
    image = tf.io.decode_image(image, channels=3, dtype=tf.float32, expand_animations=False)
    image = tf.image.resize(image, input_image_size[:2], method=tf.image.ResizeMethod.BICUBIC, antialias=True)
    image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=1.)
    image = tf.ensure_shape(image, tuple(input_image_size))
    return image


def forward_pass(image, feature_extractor):
    image = tf.expand_dims(image, axis=0)
    bottlenecks = feature_extractor(image)
    return bottlenecks[0]


def create_bottlenecks_vectors(image_path, class_labels, input_image_size, feature_extractor):
    class_label = get_class_label(image_path, class_labels)
    image = tf.io.read_file(image_path)
    image = decode_image(image, input_image_size)
    bottlenecks = forward_pass(image, feature_extractor)

    bottlenecks = tf.io.serialize_tensor(bottlenecks)
    class_label = tf.io.serialize_tensor(class_label)
    return bottlenecks, class_label


def create_train_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    train_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return train_feature


def create_train_example(bottleneck_vector, class_label):
    feature = {
        'bottleneck_vector': create_train_feature(bottleneck_vector),
        'label': create_train_feature(class_label)
    }
    train_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return train_example


def create_bottlenecks_tfrecord(image_dir, class_labels, input_image_size, feature_extractor, tfrecord_path):

    start = time.time()
    image_path_DS = tf.data.Dataset.list_files(image_dir + '/*/*', shuffle=True)
    bottleneck_DS = image_path_DS.map(lambda image_path: create_bottlenecks_vectors(image_path, class_labels, input_image_size, feature_extractor), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    end = time.time()
    print(f'Time to create bottlenecks: {end - start:.3f} seconds')

    start = time.time()
    total_images = tf.data.experimental.cardinality(bottleneck_DS).numpy()
    image_count = 1

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for bottleneck_vector, class_label in bottleneck_DS:
            train_example = create_train_example(bottleneck_vector, class_label)
            writer.write(train_example.SerializeToString())

            image_count += 1
            images_written = 'Images Written to TFRecord File: ' + str(image_count) + ' of ' + str(total_images)
            print('\r' + images_written, end='')

    end = time.time()
    print(f'\nTime to create TFRecord File: {end - start:.3f} seconds')
