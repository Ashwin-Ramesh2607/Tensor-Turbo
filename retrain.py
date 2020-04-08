import re
import os
import sys
import hashlib
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_hub as hub

import hub_models
import create_model
import train_classifier
import create_bottlenecks_tfrecord


def prepare_dir_tree():

    if not tf.io.gfile.exists(FLAGS.bottleneck_dir):
        tf.io.gfile.mkdir(FLAGS.bottleneck_dir)
    if not tf.io.gfile.exists(os.path.join(FLAGS.bottleneck_dir, os.path.basename(FLAGS.image_dir))):
        tf.io.gfile.mkdir(os.path.join(FLAGS.bottleneck_dir, os.path.basename(FLAGS.image_dir)))
    if not tf.io.gfile.exists(os.path.join(FLAGS.bottleneck_dir, os.path.basename(FLAGS.image_dir), FLAGS.architecture)):
        tf.io.gfile.mkdir(os.path.join(FLAGS.bottleneck_dir, os.path.basename(FLAGS.image_dir), FLAGS.architecture))

    if tf.io.gfile.exists(FLAGS.summaries_dir):
        tf.io.gfile.rmtree(FLAGS.summaries_dir)
    tf.io.gfile.mkdir(FLAGS.summaries_dir)


def image_metadata():
    if not tf.io.gfile.exists(FLAGS.image_dir):
        sys.exit(f'Image Dataset could not be found at: {FLAGS.image_dir}')

    labels_metadata = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    class_dirs = [
        os.path.join(FLAGS.image_dir, class_label)
        for class_label in tf.io.gfile.listdir(FLAGS.image_dir)]

    class_dirs = sorted(
        class_label for class_label in class_dirs
        if tf.io.gfile.isdir(class_label))

    for class_dir in class_dirs:
        image_count = len([
            image_name for image_name in tf.io.gfile.listdir(class_dir)
            if os.path.splitext(image_name)[1].lower() in image_extensions])

        labels_metadata[os.path.basename(class_dir)] = image_count

    for label, image_count in labels_metadata.items():
        print(f'{label} contains {image_count} images.')

    if 0 in labels_metadata.values():
        sys.exit('Ensure that no label has 0 images. \nEither remove the folder or add images.')

    class_labels = np.array([
        os.path.basename(class_label)
        for class_label in class_dirs])

    total_classes = len(class_labels)
    total_images = sum(labels_metadata.values())

    return class_labels, total_classes, total_images


def check_existing_tfrecord(total_classes, total_images):
    expected_tfrecord_name = f'{total_classes}-classes_{total_images}-images.tfrecord'
    expected_tfrecord_path = os.path.join(
        FLAGS.bottleneck_dir,
        os.path.basename(FLAGS.image_dir),
        FLAGS.architecture,
        expected_tfrecord_name
        )

    tfrecord_exists = tf.io.gfile.exists(expected_tfrecord_path)

    return tfrecord_exists


def main():
    prepare_dir_tree()

    class_labels, total_classes, total_images = image_metadata()

    input_image_size, bottleneck_shape = hub_models.get_model_shapes(FLAGS.architecture)
    feature_extractor = hub_models.get_hub_model(FLAGS.architecture)

    expected_tfrecord_name = f'{total_classes}-classes_{total_images}-images.tfrecord'
    expected_tfrecord_path = os.path.join(
        FLAGS.bottleneck_dir,
        os.path.basename(FLAGS.image_dir),
        FLAGS.architecture,
        expected_tfrecord_name
        )

    if not check_existing_tfrecord(total_classes, total_images):
        create_bottlenecks_tfrecord.create_bottlenecks_tfrecord(
            FLAGS.image_dir,
            class_labels,
            input_image_size,
            feature_extractor,
            expected_tfrecord_path)
    else:
        print('An existing and compatible TFRecord file has been found which will be used for training.')

    train_classifier.train(
        expected_tfrecord_path,
        bottleneck_shape,
        total_classes,
        total_images,
        FLAGS)

'''
    if FLAGS.custom_classifier:
        model = create_model.CustomModel(input_dim = (28, 28, 1), num_classes = 10, classifier_head = {
            '1_conv2d': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
            '2_maxpooling2d': {'pool_size': (2, 2)},
            '3_conv2d': {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
            '4_maxpooling2d': {'pool_size': (2, 2)},
            '5_conv2d': {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
            '6_flatten': {},
            '7_dense': {'units': 64, 'activation': 'relu'},
            '8_dense': {'units': 10, 'activation': 'softmax'},
        })
    else:
        model = create_model.CustomModel(input_dim = (1280, ), num_classes = 10, classifier_head = {
            '1_dense': {'units': 10, 'activation': 'softmax'}
        })

    print(model.get_summary().summary())'''


def parse_arguments():
    parser = argparse.ArgumentParser(description='Transfer Learning Parameters')

    parser.add_argument(
        '--architecture',
        type=str,
        default='',
        help='''
            Specify the architecture to use as a Feature Extractor.
            A classifier head will be added which uses the Feature Vector to classify images.
            The pre-trained feature extractor will be downloaded from TensorFlow Hub.'''
    )

    parser.add_argument(
        '--custom_classifier',
        action='store_true',
        help='''\
            Include this flag if you want to specify a custom classifier head.
            If this flag is not set, only a dense layer (with number of nodes = number of image classes)
            as a default classifier on top of the tf-hub feature extractor model.'''
    )

    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='''
            Path to the folders containing images.
            Train, Validation and Test splits will be automatically made using Hashing Function.'''
    )

    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='bottlenecks',
        help='''
            Directory to store csv files containing bottleneck values for each image.'''
    )

    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=15,
        help='''
            Percentage of images to reserve for validation.'''
    )

    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=15,
        help='''
            Percentage of images to reserve for testing.'''
    )

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=64,
        help='''\
            Number of training images to be used in one batch while training.'''
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='''
            Number of Epochs to train before stopping.'''
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='''
            Learning Rate to use during Training.'''
    )

    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='logs',
        help='''
            Directory to store summaries for TensorBoard visualizations.'''
    )

    parser.add_argument(
        '--saved_model',
        type=str,
        default='saved_models',
        help='''
            Path to store Saved Model in TensorFlow 2 Format.'''
    )

    return parser.parse_args()


if __name__ == '__main__':

    FLAGS = parse_arguments()
    main()
