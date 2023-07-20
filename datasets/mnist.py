from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#from tensorflow.examples.tutorials.mnist import input_data

# TODO: TF-V1
#def load(conf):
#    print("load MNIST dataset")
#    #data = input_data.read_data_sets(data_dir, one_hot=True)
#    data = input_data.read_data_sets("MNIST-data/", one_hot=True)
#
#    train_dataset = tf.data.Dataset.from_tensor_slices((data.train.images,tf.cast(data.train.labels,tf.float32)))
#    train_dataset = train_dataset.shuffle(60000).batch(conf.batch_size)
#
#    val_dataset = tf.data.Dataset.from_tensor_slices((data.validation.images,tf.cast(data.validation.labels,tf.float32)))
#    test_dataset = tf.data.Dataset.from_tensor_slices((data.test.images[:conf.num_test_dataset], tf.cast(data.test.labels[:conf.num_test_dataset],tf.float32)))
#
#
#    #print(data.test.images.shape)
#
#    num_train_dataset = len(data.train.images)
#    num_val_dataset = len(data.validation.images)
#    num_test_dataset = conf.num_test_dataset
#
#
#    print(train_dataset)
#    print('# of dataset: train ({}), val ({}), test ({})'.format(num_train_dataset,num_val_dataset,num_test_dataset))
#
#
#    if conf.f_train_tk:
#        test_dataset = tf.data.Dataset.from_tensor_slices((data.train.images[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset], tf.cast(data.train.labels[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset],tf.float32)))
#
#
#    val_dataset = val_dataset.batch(conf.batch_size)
#    test_dataset = test_dataset.batch(conf.batch_size)
#
#    return train_dataset, val_dataset, test_dataset, num_train_dataset, num_val_dataset, num_test_dataset
#
#def train_data_augmentation(train_dataset, batch_size):
#    return train_dataset



def load(conf):
    print("load MNIST dataset")

    (img_train, label_train), (img_test, label_test) = tf.keras.datasets.mnist.load_data()

    #
    # img_train = img_train.reshape(60000,784).astype('float32') / 255
    # img_test = img_test.reshape(10000,784).astype('float32') / 255
    img_train = img_train.reshape(60000,28,28).astype('float32') / 255
    img_test = img_test.reshape(10000,28,28).astype('float32') / 255

    label_train = tf.one_hot(label_train.astype('float32'),10)
    label_test = tf.one_hot(label_test.astype('float32'),10)

    #
    num_val_dataset = 5000
    num_train_dataset = 60000-num_val_dataset
    # num_test_dataset = conf.num_test_dataset
    num_test_dataset = conf.num_test_data

    #
    img_val = img_train[-num_val_dataset:]
    label_val = label_train[-num_val_dataset:]

    img_train = img_train[:-num_val_dataset]
    label_train = label_train[:-num_val_dataset]

    #
    train_dataset = tf.data.Dataset.from_tensor_slices((img_train,label_train))
    train_dataset = train_dataset.shuffle(60000).batch(conf.batch_size)

    #val_dataset = tf.data.Dataset.from_tensor_slices((data.validation.images,tf.cast(data.validation.labels,tf.float32)))
    val_dataset = tf.data.Dataset.from_tensor_slices((img_val,label_val))
    # test_dataset = tf.data.Dataset.from_tensor_slices((img_test[:conf.num_test_dataset], label_test[:conf.num_test_dataset]))
    test_dataset = tf.data.Dataset.from_tensor_slices((img_test[:conf.num_test_data], label_test[:conf.num_test_data]))

    print(train_dataset)
    print('# of dataset: train ({}), val ({}), test ({})'.format(num_train_dataset,num_val_dataset,num_test_dataset))


    if conf.f_train_tk:
        # test_dataset = tf.data.Dataset.from_tensor_slices((img_train[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset], label_train[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset],tf.float32))
        test_dataset = tf.data.Dataset.from_tensor_slices((img_train[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_data], label_train[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset],tf.float32))


    val_dataset = val_dataset.batch(conf.batch_size)
    test_dataset = test_dataset.batch(conf.batch_size)

    return train_dataset, val_dataset, test_dataset, num_train_dataset, num_val_dataset, num_test_dataset

def train_data_augmentation(train_dataset, batch_size):
    return train_dataset
