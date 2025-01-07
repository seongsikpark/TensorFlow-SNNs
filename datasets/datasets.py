
import tensorflow as tf

import flags
import lib_snn
import datasets

from config import config

from models.imagenet_input_preprocessor import preprocessor_input_imagenet
#from tensorflow.python.keras.applications.imagenet_utils import preprocess_input as preprocess_input_others

preprocess_input_others = tf.keras.applications.imagenet_utils.preprocess_input


# label - one-hot

#def load(model_name,dataset_name,batch_size,input_size,train_type,train,conf,num_parallel_call):
def load():

    model_name = config.model_name
    dataset_name = config.dataset_name
    batch_size = config.batch_size
    train_type = config.train_type
    train = config.train

    dataset_sel = {
        #'ImageNet': datasets.imagenet,
        #'CIFAR10': datasets.cifar,
        #'CIFAR100': datasets.cifar,
        'ImageNet': datasets.image_cls,
        'CIFAR10': datasets.image_cls,
        'CIFAR100': datasets.image_cls,
        #'CIFAR10_DVS': datasets.cifar10_dvs.CIFAR10DVS,
        'CIFAR10_DVS': datasets.cifar10_dvs,
    }
    if dataset_name=='CIFAR10_DVS':
        #dataset = dataset_sel[dataset_name]()
        dataset = dataset_sel[dataset_name]
    else:
        dataset = dataset_sel[dataset_name]

    # TODO: depreciated
    # num_class
    num_class_sel = {
        'ImageNet': 1000,
        'CIFAR10': 10,
        'CIFAR100': 100,
        'CIFAR10_DVS': 10,
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
        # CIFAR
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

        #input_size = lib_snn.utils_vis.image_shape_vis(model_name,dataset_name)[0]


    if dataset_name=='CIFAR10_DVS':
        # TODO
        input_size = lib_snn.utils_vis.image_shape_vis(model_name,dataset_name)[0]
        train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num = dataset.load()

    else:
        input_size = lib_snn.utils_vis.image_shape_vis(model_name,dataset_name)[0]
        train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num = dataset.load(dataset_name,
                                                   batch_size,input_size, input_size_pre_crop_ratio, num_class, train,
                                                   input_prec_mode, preprocessor_input)

    train_steps_per_epoch = train_ds.cardinality().numpy()


    # data-based weight normalization (DNN-to-SNN conversion)
    if config.flags.f_write_stat and config.flags.f_stat_train_mode:
        test_ds = train_ds

    return train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch
