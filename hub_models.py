import tensorflow as tf
import tensorflow_hub as hub


def get_hub_url(architecture):

    hub_url = f'https://tfhub.dev/google/imagenet/{architecture}/feature_vector/4'

    return hub_url


def get_input_image_size(module_layer):

    input_image_size = list(module_layer._func.__call__.concrete_functions[0].
                            structured_input_signature[0][0].shape)[-3:]

    return input_image_size


def get_output_tensor_shape(module_layer, input_image_size):

    output_tensor_shape = module_layer(
        tf.zeros([1] + input_image_size))[0].numpy().shape

    return output_tensor_shape


def get_model_shapes(architecture):

    hub_url = get_hub_url(architecture)
    module_layer = hub.KerasLayer(hub_url, trainable=False)

    input_image_size = get_input_image_size(module_layer)
    output_tensor_shape = get_output_tensor_shape(module_layer,
                                                  input_image_size)

    return input_image_size, output_tensor_shape


def get_hub_model(architecture):

    hub_url = get_hub_url(architecture)

    return hub.load(hub_url)
