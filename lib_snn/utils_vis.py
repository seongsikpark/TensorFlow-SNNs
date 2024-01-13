
from config import config
conf = config.flags

#
input_size_default = {
    'ImageNet': 224,
    'CIFAR10': 32,
    'CIFAR100': 32,
    'CIFAR10_DVS': 32,
}

#
input_sizes_imagenet = {
    'Xception': 299,
    'InceptionV3': 299,
    'InceptionResNetV2': 299,
    'NASNetLarge': 331,
    'EfficientNetB1': 240,
    'EfficientNetB2': 260,
    'EfficientNetB4': 380,
    'EfficientNetB5': 456,
    'EfficientNetB6': 528,
    'EfficientNetB7': 600,
}

input_sizes_cifar = {
    'VGG16': 32,
}

input_sizes_cifar_dvs = {
    'VGG16': 32,
}

#
input_size_sel ={
    'ImageNet': input_sizes_imagenet,
    'CIFAR10': input_sizes_cifar,
    'CIFAR100': input_sizes_cifar,
    'CIFAR10_DVS': input_sizes_cifar_dvs,
}


#
def image_shape_vis(model_name, dataset_name):
    #
    input_size = input_size_sel[dataset_name].get(model_name,input_size_default[dataset_name])
    #input_size = input_sizes.get(model_name,224)

    #
    if conf.input_data_time_dim:
        image_shape = (conf.time_dim_size, input_size, input_size, 2)
        #image_shape = (conf.time_dim_size, input_size, input_size, 1)
    else:
        image_shape = (input_size, input_size, 3)

    return image_shape
