from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def load(conf):
    print("load CIFAR10 dataset")
    #(img_train,label_train), (img_test, label_test) = tf.contrib.keras.datasets.cifar10.load_data()
    (img_train,label_train), (img_test, label_test) = tf.keras.datasets.cifar10.load_data()

    #print(type(img_train))
    img_train = img_train.astype(float)
    img_test = img_test.astype(float)

    #print(img_train[0])
    #print(tf.reduce_min(img_train))
    #print(tf.reduce_min(img_test))

    img_train = img_train / 255.0
    img_test = img_test / 255.0

    #img_train[:,:,:,0] = (img_train[:,:,:,0]-np.mean(img_train[:,:,:,0]))/np.std(img_train[:,:,:,0])
    #img_train[:,:,:,1] = (img_train[:,:,:,1]-np.mean(img_train[:,:,:,1]))/np.std(img_train[:,:,:,1])
    #img_train[:,:,:,2] = (img_train[:,:,:,2]-np.mean(img_train[:,:,:,2]))/np.std(img_train[:,:,:,2])

    #img_test[:,:,:,0] = (img_test[:,:,:,0]-np.mean(img_test[:,:,:,0]))/np.std(img_test[:,:,:,0])
    #img_test[:,:,:,1] = (img_test[:,:,:,1]-np.mean(img_test[:,:,:,1]))/np.std(img_test[:,:,:,1])
    #img_test[:,:,:,2] = (img_test[:,:,:,2]-np.mean(img_test[:,:,:,2]))/np.std(img_test[:,:,:,2])


    if conf.f_data_std:
        cifar_mean = [0.485, 0.456, 0.406]
        cifar_std = [0.229, 0.224, 0.225]

        img_train[:,:,:,0] = (img_train[:,:,:,0]-cifar_mean[0])/cifar_std[0]
        img_train[:,:,:,1] = (img_train[:,:,:,1]-cifar_mean[1])/cifar_std[1]
        img_train[:,:,:,2] = (img_train[:,:,:,2]-cifar_mean[2])/cifar_std[2]

        img_test[:,:,:,0] = (img_test[:,:,:,0]-cifar_mean[0])/cifar_std[0]
        img_test[:,:,:,1] = (img_test[:,:,:,1]-cifar_mean[1])/cifar_std[1]
        img_test[:,:,:,2] = (img_test[:,:,:,2]-cifar_mean[2])/cifar_std[2]


    #print(tf.reduce_min(img_train))
    #print(tf.reduce_min(img_test))

    #label_test=label_test[:conf.num_test_dataset]
    #img_test=img_test[:conf.num_test_dataset,:,:,:]


    #num_train_dataset = 45000
    num_train_dataset = 5000
    num_val_dataset = 5000
    num_test_dataset = conf.num_test_dataset


    img_train = img_train[:num_train_dataset,:,:,:]
    label_train = label_train[:num_train_dataset,:]


    #img_val = img_train[num_train_dataset:,:,:,:]
    #label_val = label_train[num_train_dataset:,:]

    img_val = img_train[:num_val_dataset,:,:,:]
    label_val = label_train[:num_val_dataset,:]

    label_test=label_test[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset]
    img_test=img_test[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset,:,:,:]

    train_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,10))))

    #val_dataset = tf.data.Dataset.from_tensor_slices((img_test,tf.squeeze(tf.one_hot(label_test,10))))
    val_dataset = tf.data.Dataset.from_tensor_slices((img_val,tf.squeeze(tf.one_hot(label_val,10))))
    val_dataset = val_dataset.map(preprocess_test, num_parallel_calls=2)
    val_dataset = val_dataset.prefetch(10*conf.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((img_test,tf.squeeze(tf.one_hot(label_test,10))))
    test_dataset = test_dataset.map(preprocess_test, num_parallel_calls=2)
    test_dataset = test_dataset.prefetch(10*conf.batch_size)


    # for stat of train dataset
    if conf.f_stat_train_mode:
        test_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,10))))
        test_dataset = test_dataset.map(preprocess_test, num_parallel_calls=2)

    if conf.f_train_time_const:
        label_train=label_train[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset]
        img_train=img_train[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset,:,:,:]

        test_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,10))))
        test_dataset = test_dataset.map(preprocess_test, num_parallel_calls=2)


    print(train_dataset)

    val_dataset = val_dataset.batch(conf.batch_size)
    test_dataset = test_dataset.batch(conf.batch_size)





    return train_dataset, val_dataset, test_dataset, num_train_dataset, num_val_dataset, num_test_dataset


def preprocess_train(img, label):
    img_p = tf.image.convert_image_dtype(img,tf.float32)
    #img_p = tf.image.resize_image_with_crop_or_pad(img_p,cifar_10_crop_size,cifar_10_crop_size)

    #
    #img_p = tf.image.random_brightness(img_p,max_delta=20)
    #img_p = tf.image.random_brightness(img_p,max_delta=23)
    #img_p = tf.image.random_brightness(img_p,max_delta=83)
    #img_p = tf.image.random_brightness(img_p,max_delta=103)


    #img_p = tf.image.resize_image_with_crop_or_pad(img_p,36,36)
    img_p = tf.image.resize_with_crop_or_pad(img_p,36,36)
    img_p = tf.image.random_crop(img_p,[32,32,3])


    #img_p = tf.cast(img,tf.float32)
    #img_p = tf.image.convert_image_dtype(img,tf.float32)
    #img_p = tf.image.rgb_to_hsv(img_p)

    img_p = tf.image.random_flip_left_right(img_p)


    #
    #img_p = tf.image.random_contrast(img_p,lower=0.2,upper=1.8)
    #img_p = tf.image.random_contrast(img_p,lower=0.1,upper=2.0)

    #
    #img_p = tf.image.per_image_standardization(img_p)
    #img_p.set_shape([cifar_10_crop_size,cifar_10_crop_size,3])
    img_p.set_shape([32,32,3])

    return img_p, label

def preprocess_test(img, label):
    img_p = tf.image.convert_image_dtype(img,tf.float32)
    #img_p = tf.image.resize_image_with_crop_or_pad(img_p,cifar_10_crop_size,cifar_10_crop_size)

    #img_p = tf.cast(img,tf.float32)
    #img_p = tf.image.convert_image_dtype(img,tf.float32)
    #img_p = tf.image.rgb_to_hsv(img_p)
    #img_p = tf.image.resize_image_with_crop_or_pad(img_p,cifar_10_crop_size,cifar_10_crop_size)

    # test 180711
    #img_p = tf.image.per_image_standardization(img_p)

    #img_p.set_shape([cifar_10_crop_size,cifar_10_crop_size,3])
    img_p.set_shape([32,32,3])

    return img_p, label

def train_data_augmentation(train_dataset, batch_size):
    #train_dataset_p = train_dataset_p.shuffle(50000).batch(conf.batch_size)
    train_dataset_p = train_dataset.shuffle(10000)
    train_dataset_p = train_dataset_p.prefetch(2*batch_size)

    #train_dataset_p = train_dataset_p.prefetch(100)

    train_dataset_p = train_dataset_p.map(preprocess_train, num_parallel_calls=8)
    train_dataset_p = train_dataset_p.batch(batch_size)


    #train_dataset_p = train_dataset_p.apply(tf.data.experimental.map_and_batch(
    #                    preprocess_cifar10_train,conf.batch_size,num_parallel_batches=8))

    #train_dataset_p = train_dataset_p.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))

    return train_dataset_p
