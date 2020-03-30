def get_hub_url(architecture):

    url_list = {
        'mobilenet_v2_100_224': 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4',
        'inception_v3': 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4'
    }

    return url_list[architecture]