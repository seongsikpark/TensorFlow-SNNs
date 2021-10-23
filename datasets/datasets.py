

import datasets

def load(dataset_name,batch_size,input_size,train_type,train,conf,num_parallel_call):

    dataset_sel = {
        'ImageNet': datasets.imagenet,
        'CIFAR10': datasets.cifar,
        'CIFAR100': datasets.cifar,
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
        input_size_pre_crop_ratio = 256 / 224
        # TODO: set
        if train:
            input_prec_mode = None  # should be modified
        else:
            input_prec_mode = 'caffe'  # keras pre-trained model

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


    train_ds, valid_ds, test_ds = dataset.load(dataset_name,
                                               batch_size,input_size, input_size_pre_crop_ratio, num_class, train,
                                               num_parallel_call, conf, input_prec_mode)

    return train_ds, valid_ds, test_ds, num_class
