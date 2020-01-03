from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from . import cifar10


def load(conf):
    print("load CIFAR100 dataset")
    (img_train,label_train), (img_test, label_test) = tf.contrib.keras.datasets.cifar100.load_data()

    #print(type(img_train))
    img_train = img_train.astype(float)
    img_test = img_test.astype(float)

    #print(img_train[0])
    print(tf.reduce_min(img_train))
    print(tf.reduce_min(img_test))

    img_train = img_train / 255.0
    img_test = img_test / 255.0

    #print(type(img_train))

    #print(img_train[0])

    #img_train[:,:,:,0] = (img_train[:,:,:,0]-np.mean(img_train[:,:,:,0]))/np.std(img_train[:,:,:,0])
    #img_train[:,:,:,1] = (img_train[:,:,:,1]-np.mean(img_train[:,:,:,1]))/np.std(img_train[:,:,:,1])
    #img_train[:,:,:,2] = (img_train[:,:,:,2]-np.mean(img_train[:,:,:,2]))/np.std(img_train[:,:,:,2])

    #img_test[:,:,:,0] = (img_test[:,:,:,0]-np.mean(img_test[:,:,:,0]))/np.std(img_test[:,:,:,0])
    #img_test[:,:,:,1] = (img_test[:,:,:,1]-np.mean(img_test[:,:,:,1]))/np.std(img_test[:,:,:,1])
    #img_test[:,:,:,2] = (img_test[:,:,:,2]-np.mean(img_test[:,:,:,2]))/np.std(img_test[:,:,:,2])


    print(tf.reduce_min(img_train))
    print(tf.reduce_min(img_test))

    label_test=label_test[:conf.num_test_dataset]
    img_test=img_test[:conf.num_test_dataset,:,:,:]

    train_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,100))))

    val_dataset = tf.data.Dataset.from_tensor_slices((img_test,tf.squeeze(tf.one_hot(label_test,100))))
    val_dataset = val_dataset.map(cifar10.preprocess_test, num_parallel_calls=2)
    val_dataset = val_dataset.prefetch(10*conf.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((img_test,tf.squeeze(tf.one_hot(label_test,100))))
    test_dataset = test_dataset.map(cifar10.preprocess_test, num_parallel_calls=2)
    test_dataset = test_dataset.prefetch(10*conf.batch_size)


    # for stat of train dataset
    if conf.f_stat_train_mode:
        test_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,100))))
        test_dataset = test_dataset.map(cifar10.preprocess_test)

    if conf.f_train_time_const:
        label_train=label_train[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset]
        img_train=img_train[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset,:,:,:]

        test_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,100))))
        test_dataset = test_dataset.map(cifar10.preprocess_test, num_parallel_calls=2)


    print(train_dataset)
    val_dataset = val_dataset.batch(conf.batch_size)
    test_dataset = test_dataset.batch(conf.batch_size)

    return train_dataset, val_dataset, test_dataset


def train_data_augmentation(train_dataset, batch_size):
    #train_dataset_p = train_dataset_p.shuffle(50000).batch(conf.batch_size)
    train_dataset_p = train_dataset.shuffle(10000)
    train_dataset_p = train_dataset_p.prefetch(2*batch_size)

    #train_dataset_p = train_dataset_p.prefetch(100)

    train_dataset_p = train_dataset_p.map(cifar10.preprocess_train, num_parallel_calls=8)
    train_dataset_p = train_dataset_p.batch(batch_size)


    #train_dataset_p = train_dataset_p.apply(tf.data.experimental.map_and_batch(
    #                    preprocess_cifar10_train,conf.batch_size,num_parallel_batches=8))

    #train_dataset_p = train_dataset_p.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))

    return train_dataset_p