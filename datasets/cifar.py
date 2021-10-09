
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds

from datasets.augmentation_cifar import resize_with_crop
from datasets.augmentation_cifar import resize_with_crop_aug
from datasets.augmentation_cifar import mixup
from datasets.augmentation_cifar import cutmix

f_cross_valid = False

def load(dataset_name,input_size,input_size_pre_crop_ratio,num_class,train,num_parallel,conf,input_prec_mode):

    dataset_name = dataset_name.lower()

    num_class = num_class
    batch_size_train = conf.batch_size
    input_size = input_size
    input_size_pre_crop_ratio = input_size_pre_crop_ratio

    if f_cross_valid:
        #train_ds = tfds.load('cifar10',
        train_ds = tfds.load(dataset_name,
                             split=[f'train[:{k}%]+train[{ k +10}%:]' for k in range(0 ,100 ,10)],
                             shuffle_files=True,
                             as_supervised=True)

        valid_ds = tfds.load(dataset_name,
                             split=[f'train[{k}%:{ k +10}%]' for k in range(0 ,100 ,10)],
                             shuffle_files=True,
                             as_supervised=True)

    elif conf.data_aug_mix=='mixup' or conf.data_aug_mix=='cutmix':
        train_ds_1 = tfds.load(dataset_name,
                               split='train',
                               shuffle_files=True,
                               as_supervised=True)

        train_ds_2 = tfds.load(dataset_name,
                               split='train',
                               shuffle_files=True,
                               as_supervised=True)

        train_ds = tf.data.Dataset.zip((train_ds_1 ,train_ds_2))

        valid_ds = tfds.load(dataset_name,
                             split='test',
                             as_supervised=True)
    else:
        train_ds, valid_ds = tfds.load(dataset_name,
                                       split=['train' ,'test'],
                                       shuffle_files=True,
                                       as_supervised=True)
    test_ds = tfds.load(dataset_name,
                        split='test',
                        as_supervised=True)

    #input_prec_mode = 'torch'

    # data augmentation
    # Preprocess input
    if train:
        if conf.data_aug_mix == 'mixup':
            train_ds = train_ds.map(lambda train_ds_1, train_ds_2: mixup(train_ds_1, train_ds_2, 1.0, input_prec_mode),
                                    num_parallel_calls=num_parallel)
            # train_ds=train_ds.map(lambda train_ds_1, train_ds_2: eager_mixup(train_ds_1,train_ds_2,alpha=0.2),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if conf.data_aug_mix == 'cutmix':
            train_ds = train_ds.map(
                lambda train_ds_1, train_ds_2: cutmix(train_ds_1, train_ds_2, input_size, input_size_pre_crop_ratio,
                                                      num_class, 1.0, input_prec_mode),
                num_parallel_calls=num_parallel)
            # train_ds = train_ds.map(lambda train_ds_1, train_ds_2: eager_cutmix(train_ds_1, train_ds_2, alpha=0.2),
            #                        num_parallel_calls=num_parallel)
        else:
            #train_ds = train_ds.map(resize_with_crop_aug, num_parallel_calls=num_parallel)
            train_ds = train_ds.map(
                lambda image, label: resize_with_crop_aug(image, label, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode),
                num_parallel_calls=num_parallel)

        train_ds = train_ds.batch(batch_size_train)
        train_ds = train_ds.prefetch(num_parallel)

    # valid_ds=valid_ds.map(resize_with_crop_cifar,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # valid_ds=valid_ds.map(resize_with_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # valid_ds=valid_ds.map(resize_with_crop,num_parallel_calls=num_parallel)
    valid_ds = valid_ds.map(
        lambda image, label: resize_with_crop(image, label, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode),
        num_parallel_calls=num_parallel)
    # valid_ds=valid_ds.batch(batch_size_inference)
    valid_ds = valid_ds.batch(batch_size_train)
    valid_ds = valid_ds.prefetch(num_parallel)

    return train_ds, valid_ds, test_ds

