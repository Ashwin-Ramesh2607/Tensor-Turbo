import os
import time

import tensorflow as tf


def convert_to_tensors(train_example, image_feature_description,
                       bottleneck_shape, total_classes):
    bottleneck_label_pair = tf.io.parse_single_example(
        train_example, image_feature_description)
    bottleneck_vector = tf.ensure_shape(
        tf.io.parse_tensor(bottleneck_label_pair['bottleneck_vector'],
                           out_type=tf.float32), bottleneck_shape)
    class_label = tf.ensure_shape(
        tf.io.parse_tensor(bottleneck_label_pair['label'], out_type=tf.uint8),
        tf.zeros([total_classes], tf.uint8).shape)
    return bottleneck_vector, class_label


def calculate_loss(model, x, y, training):
    y_ = model(x, training=training)
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true=y,
                                                                     y_pred=y_)


def calculate_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = calculate_loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(tfrecord_path, bottleneck_shape, total_classes, total_images, FLAGS):

    bottleneck_tfrecord_DS = tf.data.TFRecordDataset(tfrecord_path)

    image_feature_description = {
        'bottleneck_vector': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }

    bottleneck_whole_DS = bottleneck_tfrecord_DS.map(
        lambda train_example: convert_to_tensors(
            train_example, image_feature_description, bottleneck_shape,
            total_classes),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_samples = int(
        ((100 - (FLAGS.validation_percentage + FLAGS.testing_percentage)) *
         total_images) / 100.)
    test_samples = int((FLAGS.testing_percentage * total_images) / 100.)

    bottleneck_train_DS = bottleneck_whole_DS.take(train_samples)
    bottleneck_test_DS = bottleneck_whole_DS.skip(train_samples)
    bottleneck_val_DS = bottleneck_test_DS.skip(test_samples)
    bottleneck_test_DS = bottleneck_test_DS.take(test_samples)

    bottleneck_train_DS = bottleneck_train_DS.batch(
        FLAGS.train_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    bottleneck_val_DS = bottleneck_val_DS.batch(
        FLAGS.train_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    bottleneck_test_DS = bottleneck_test_DS.batch(
        FLAGS.train_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.summaries_dir, 'train'))
    validation_summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.summaries_dir, 'validation'))

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=bottleneck_shape),
        tf.keras.layers.Dense(total_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate, momentum=0.9)

    for epoch in range(1, FLAGS.epochs + 1):
        
        train_loss_avg = tf.keras.metrics.Mean()
        validation_loss_avg = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        validation_accuracy = tf.keras.metrics.CategoricalAccuracy()
        epoch_start_time = time.time()

        for x, y in bottleneck_train_DS:
            loss_value, gradients = calculate_gradients(model, x, y)
            optimizer.apply_gradients(zip(gradients,
                                          model.trainable_variables))

            train_loss_avg.update_state(loss_value)
            train_accuracy.update_state(y, model(x, training=True))

        for x, y in bottleneck_val_DS:
            loss_value = calculate_loss(model, x, y, training=False)

            validation_loss_avg.update_state(loss_value)
            validation_accuracy.update_state(y, model(x, training=False))

        epoch_end_time = time.time()

        with train_summary_writer.as_default():
            tf.summary.scalar('Loss', train_loss_avg.result(), step=epoch)
            tf.summary.scalar('Accuracy', train_accuracy.result(), step=epoch)

        with validation_summary_writer.as_default():
            tf.summary.scalar('Loss', validation_loss_avg.result(), step=epoch)
            tf.summary.scalar('Accuracy',
                              validation_accuracy.result(),
                              step=epoch)

        print(f'Epoch {epoch:03d}: {epoch_end_time - epoch_start_time:.3f} seconds')
        print(f'Train Loss: {train_loss_avg.result():.3f}, Train Accuracy: {train_accuracy.result():.3%}')
        print(f'Validation Loss: {validation_loss_avg.result():.3f}, Validation Accuracy: {validation_accuracy.result():.3%}')
        print('---------------------------------------------------------------------')

        train_loss_avg.reset_states()
        train_accuracy.reset_states()
        validation_loss_avg.reset_states()
        validation_accuracy.reset_states()

    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for x, y in bottleneck_test_DS:
        test_accuracy.update_state(y, model(x, training=False))

    print(f'Final Test Accuracy: {test_accuracy.result():.3%}')