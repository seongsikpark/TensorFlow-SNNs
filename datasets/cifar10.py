
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds


f_cross_valid = False

def load(conf):

    if f_cross_valid:
        train_ds = tfds.load('cifar10',
                             split=[f'train[:{k}%]+train[{ k +10}%:]' for k in range(0 ,100 ,10)],
                             shuffle_files=True,
                             as_supervised=True)

        valid_ds = tfds.load('cifar10',
                             split=[f'train[{k}%:{ k +10}%]' for k in range(0 ,100 ,10)],
                             shuffle_files=True,
                             as_supervised=True)

    elif conf.mixup:
        train_ds_1 = tfds.load('cifar10',
                               split='train',
                               shuffle_files=True,
                               as_supervised=True)

        train_ds_2 = tfds.load('cifar10',
                               split='train',
                               shuffle_files=True,
                               as_supervised=True)

        train_ds = tf.data.Dataset.zip((train_ds_1 ,train_ds_2))

        valid_ds = tfds.load('cifar10',
                             split='test',
                             as_supervised=True)
    else:
        train_ds, valid_ds = tfds.load('cifar10',
                                       split=['train' ,'test'],
                                       shuffle_files=True,
                                       as_supervised=True)

    test_ds = tfds.load('cifar10',
                        split='test',
                        as_supervised=True)

    global input_size_pre_crop_ratio
    # input_size_pre_crop_ratio = 40/32
    input_size_pre_crop_ratio = 256/224

    return train_ds, valid_ds, test_ds

