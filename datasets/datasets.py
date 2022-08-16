
import tensorflow as tf

import datasets

from models.imagenet_input_preprocessor import preprocessor_input_imagenet
#from tensorflow.python.keras.applications.imagenet_utils import preprocess_input as preprocess_input_others

preprocess_input_others = tf.keras.applications.imagenet_utils.preprocess_input

def load(model_name,dataset_name,batch_size,input_size,train_type,train,conf,num_parallel_call):

    dataset_sel = {
        #'ImageNet': datasets.imagenet,
        #'CIFAR10': datasets.cifar,
        #'CIFAR100': datasets.cifar,
        'ImageNet': datasets.image_cls,
        'CIFAR10': datasets.image_cls,
        'CIFAR100': datasets.image_cls,
    }
    dataset = dataset_sel[dataset_name]

    # num_class
    num_class_sel = {
        'ImageNet': 1000,
        'CIFAR10': 10,
        'CIFAR100': 100,
    }
    num_class = num_class_sel[dataset_name]

    #
    # input shape
    if dataset_name == 'ImageNet':
        # include_top = True

        if 'MobileNet' in model_name:
            input_size_pre_crop_ratio = 1
        elif 'EfficientNet' in model_name:
            input_size_pre_crop_ratio = 1
        else:
            input_size_pre_crop_ratio = 256 / 224

        # TODO: set
        if train:
            input_prec_mode = None  # should be modified
        else:
            input_prec_mode = 'caffe'  # keras pre-trained model

        preprocessor_input = preprocessor_input_imagenet[model_name]
    else:
        # CIFAR-10
        # TODO:
        if train_type == 'transfer':
            input_size_pre_crop_ratio = 256 / 224
            # TODO: check it - transfer learning with torch mode
            input_prec_mode = 'caffe'
        elif train_type == 'scratch':
            input_size = 32
            input_size_pre_crop_ratio = 36 / 32
            input_prec_mode = 'torch'
        else:
            assert False, 'not supported train type {}'.format(train_type)

        preprocessor_input = preprocess_input_others


    train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num = dataset.load(dataset_name,
                                               batch_size,input_size, input_size_pre_crop_ratio, num_class, train,
                                               num_parallel_call, conf, input_prec_mode, preprocessor_input)

    return train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class
