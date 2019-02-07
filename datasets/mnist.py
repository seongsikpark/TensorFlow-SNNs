from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load(conf):
    print("load MNIST dataset")
    #data = input_data.read_data_sets(data_dir, one_hot=True)
    data = input_data.read_data_sets("MNIST-data/", one_hot=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((data.train.images,tf.cast(data.train.labels,tf.float32)))
    train_dataset = train_dataset.shuffle(60000).batch(conf.batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((data.validation.images,tf.cast(data.validation.labels,tf.float32)))
    test_dataset = tf.data.Dataset.from_tensor_slices((data.test.images[:conf.num_test_dataset], tf.cast(data.test.labels[:conf.num_test_dataset],tf.float32)))


    #print(data.test.images.shape)

    print(train_dataset)


    val_dataset = val_dataset.batch(conf.batch_size)
    test_dataset = test_dataset.batch(conf.batch_size)

    return train_dataset, val_dataset, test_dataset

def train_data_augmentation(train_dataset, batch_sizu):
    return train_dataset