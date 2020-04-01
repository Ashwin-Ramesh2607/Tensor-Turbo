import re
import os
import sys
import hashlib
import argparse
import tensorflow as tf
import tensorflow_hub as hub

import hub_urls
import create_model

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1

def create_image_splits():
    if not tf.io.gfile.exists(FLAGS.image_dir):
        print(f'Image Dataset could not be found at: {FLAGS.image_dir}')
        return None

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    dataset_splits = {}

    class_dirs = [
        os.path.join(FLAGS.image_dir, class_label)
        for class_label in tf.io.gfile.listdir(FLAGS.image_dir)]

    class_dirs = sorted(
        class_label for class_label in class_dirs
        if tf.io.gfile.isdir(class_label))

    for class_dir in class_dirs:
        image_list = [
            image_name for image_name in tf.io.gfile.listdir(class_dir)
            if os.path.splitext(image_name)[1].lower() in image_extensions]

        training_images = []
        validation_images = []
        testing_images = []

        if not image_list:
            print(f'No Images found under label: {class_dir}')

        for image_file in image_list:
            image_name = os.path.splitext(image_file)[0]
            hash_name = re.sub(r'_nohash_.*$', '', image_name)
            hash_name_hashed = hashlib.sha512(tf.compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                               (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))

            if percentage_hash < FLAGS.validation_percentage:
                validation_images.append(image_file)
            elif percentage_hash < (FLAGS.testing_percentage + FLAGS.validation_percentage):
                testing_images.append(image_file)
            else:
                training_images.append(image_file)

        dataset_splits[os.path.basename(class_dir)] = {
            'directory': class_dir,
            'train': training_images,
            'validation': validation_images,
            'test': testing_images
        }

    return dataset_splits

def prepare_dir_tree():
    
    if not tf.io.gfile.exists(FLAGS.bottleneck_dir):
        tf.io.gfile.mkdir(FLAGS.bottleneck_dir)
    elif not tf.io.gfile.exists(os.path.join(FLAGS.bottleneck_dir, FLAGS.architecture)):
        tf.io.gfile.mkdir(os.path.join(FLAGS.bottleneck_dir, FLAGS.architecture))

    if tf.io.gfile.exists(FLAGS.summaries_dir):
        tf.io.gfile.rmtree(FLAGS.summaries_dir)
    tf.io.gfile.mkdir(FLAGS.summaries_dir)

def main():
    prepare_dir_tree()
    create_image_splits()
'''
    print('URL', hub_urls.get_hub_url(FLAGS.architecture))
    
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

    print(model.get_summary().summary())
    '''


def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Transfer Learning Parameters')
    
    parser.add_argument(
        '--architecture',
        type = str,
        default = '',
        help = '''
            Specify the architecture to use as a Feature Extractor.
            A classifier head will be added which uses the Feature Vector to classify images.
            The pre-trained feature extractor will be downloaded from TensorFlow Hub.'''
    )

    parser.add_argument(
        '--image_dir',
        type = str,
        default = '',
        help = '''
            Path to the folders containing images.
            Train, Validation and Test splits will be automatically made using Hashing Function.'''
    )

    parser.add_argument(
        '--bottleneck_dir',
        type = str,
        default = 'bottlenecks',
        help = '''
            Directory to store csv files containing bottleneck values for each image.'''     
    )

    parser.add_argument(
        '--validation_percentage',
        type = int,
        default = 15,
        help = '''
            Percentage of images to reserve for validation.'''
    )

    parser.add_argument(
        '--testing_percentage',
        type = int,
        default = 15,
        help = '''
            Percentage of images to reserve for testing.'''
    )

    parser.add_argument(
        '--train_batch_size',
        type = int,
        default = 128,
        help = '''\
            Number of training images to be used in one batch while training.'''
    )

    parser.add_argument(
        '--epochs',
        type = int,
        default = 20,
        help = '''
            Number of Epochs to train before stopping.'''
    )

    parser.add_argument(
        '--learning_rate',
        type = float,
        default = 0.01,
        help = '''
            Learning Rate to use during Training.'''
    )

    parser.add_argument(
        '--summaries_dir',
        type = str,
        default = 'logs',
        help = '''
            Directory to store summaries for TensorBoard visualizations.'''
    )
    
    parser.add_argument(
        '--saved_model',
        type = str,
        default = 'saved_models',
        help = '''
            Path to store Saved Model in TensorFlow 2 Format.'''
    )

    return parser.parse_args()

if __name__ == '__main__':
    
    FLAGS = parse_arguments()
    main()