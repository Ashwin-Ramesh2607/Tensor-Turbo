import os
import sys
import argparse
import tensorflow as tf
import tensorflow_hub as hub

import hub_urls
import create_model

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