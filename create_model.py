import tensorflow as tf

class CustomModel(tf.keras.Model):
    def __init__(self, input_dim, num_classes, classifier_head):
        super(CustomModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.classifier_head = classifier_head
        self.classifier_layers = {}

        for layer_name, params in self.classifier_head.items():
            layer_type = layer_name.split('_', 1)[1]
            assign_layer = getattr(self, layer_type)
            if layer_name.split('_', 1)[0] == '1':
                self.classifier_layers[layer_name] = assign_layer(input_shape = self.input_dim, **params)
            else:
                self.classifier_layers[layer_name] = assign_layer(**params)

    def conv2d(self, filters, kernel_size, activation, input_shape = None):
        if input_shape is None:
            return tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, activation = activation)
        return tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, activation = activation, input_shape = input_shape)

    def maxpooling2d(self, pool_size, input_shape = None):
        if input_shape is None:
            return tf.keras.layers.MaxPooling2D(pool_size = pool_size)
        return tf.keras.layers.MaxPooling2D(pool_size = pool_size, input_shape = input_shape)

    def flatten(self, input_shape = None):
        if input_shape is None:
            return tf.keras.layers.Flatten()
        return tf.keras.layers.Flatten(input_shape = input_shape)

    def dense(self, units, activation, input_shape = None):
        if input_shape is None:
            return tf.keras.layers.Dense(units = units, activation = activation)
        return tf.keras.layers.Dense(units = units, activation = activation, input_shape = input_shape)

    def call(self, inputs):
        # Define our forward pass using layers created in the __init__ function
        output_tensor = inputs

        for _, layer_object in self.classifier_layers.items():
            output_tensor = layer_object(output_tensor)

        return output_tensor
  
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)
    
    def get_summary(self):
        input_tensor = tf.keras.Input(shape = self.input_dim)
        output_tensor = tf.keras.Model(inputs = [input_tensor], outputs = self.call(input_tensor))
        return output_tensor