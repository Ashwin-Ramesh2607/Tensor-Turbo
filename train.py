import time
import tensorflow as tf


def convert_to_tensors(train_example):
    bottleneck_label_pair = tf.io.parse_single_example(train_example, image_feature_description)
    bottleneck_vector = tf.ensure_shape(tf.io.parse_tensor(bottleneck_label_pair['bottleneck_vector'], out_type=tf.float32), (1280, ))
    class_label = tf.ensure_shape(tf.io.parse_tensor(bottleneck_label_pair['label'], out_type=tf.uint8), (101, ))
    return bottleneck_vector, class_label


def calculate_loss(model, x, y, training):
    y_ = model(x, training=training)
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true=y, y_pred=y_)


def calculate_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = calculate_loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train():

    bottleneck_tfrecord_DS = tf.data.TFRecordDataset('bottlenecks/testing.tfrecords')

    global image_feature_description
    image_feature_description = {
        'bottleneck_vector': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }

    bottleneck_whole_DS = bottleneck_tfrecord_DS.map(convert_to_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    bottleneck_train_DS = bottleneck_whole_DS.take(70700)
    bottleneck_test_DS = bottleneck_whole_DS.skip(70700)
    bottleneck_val_DS = bottleneck_test_DS.skip(15150)
    bottleneck_test_DS = bottleneck_test_DS.take(15150)

    bottleneck_train_DS = bottleneck_train_DS.batch(64).prefetch(tf.data.experimental.AUTOTUNE)
    bottleneck_val_DS = bottleneck_val_DS.batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    bottleneck_test_DS = bottleneck_test_DS.batch(1).prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1280, )),
        tf.keras.layers.Dense(101, activation='softmax')])

    optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)

    num_epochs = 50

    for epoch in range(num_epochs):
        train_loss_avg = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        epoch_start_time = time.time()

        for x, y in bottleneck_train_DS:
            loss_value, gradients = calculate_gradients(model, x, y)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss_avg(loss_value)
            train_accuracy(y, model(x, training=True))

        epoch_end_time = time.time()

        print("Epoch {:03d}: Time Taken: {:.3f} s".
              format(epoch + 1, epoch_end_time - epoch_start_time))
        print("Training Loss: {:.3f}, Training Accuracy: {:.3%}".
              format(train_loss_avg.result(), train_accuracy.result()))
        print('---------------------------------------------------------')

    validation_accuracy = tf.keras.metrics.CategoricalAccuracy()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for x, y in bottleneck_val_DS:
        validation_accuracy(y, model(x, training=False))

    print("Final Validation Accuracy: {:.3%}".format(validation_accuracy.result()))

    for x, y in bottleneck_test_DS:
        test_accuracy(y, model(x, training=False))

    print("Final Test Accuracy: {:.3%}".format(test_accuracy.result()))


train()
