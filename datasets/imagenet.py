"""Collection of ImageNet utils
''' from tensornets/datasets/imagenet.py
"""
from __future__ import absolute_import

import os
import numpy as np

from os.path import isfile, join

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from utils_tensornets import *

import math

import random

from scipy.io import loadmat


#def imagenet_load(data_dir, resize_wh, crop_wh, crops):
def imagenet_load(conf):
    data_dir = conf.data_path_imagenet
    resize_wh = 256
    ##resize_wh = 224
    #crop_wh = 224
    crop_wh = conf.input_size
    crops = 1
    return load_batch(
        data_dir, 'val', batch_size=conf.batch_size,
        resize_wh=resize_wh,
        crop_locs=10 if crops == 10 else 4,
        crop_wh=crop_wh,
        total_num= conf.num_test_dataset
    )



def load(conf):
    print('ImageNet load')

    #
    #train_dataset = ImageNetDataset(conf, 'train')
    #valid_dataset = ImageNetDataset(conf, 'val')
    #test_dataset = ImageNetDataset(conf, 'val')

    #print(test_dataset)

    # tensornets
    #data_imgs, data_labels = imagenet.get_files(conf.data_path_imagenet,'val', None)

    #dataset = tf.data.Dataset.from_tensor_slices((data_imgs, data_labels))
    #dataset = dataset.map(imagenet.load_and_crop, resize_wh=256, crop_locs=4, crop_wh=224).batch(conf.batch_size)
    #dataset = dataset.map(imagenet.load_and_crop).batch(conf.batch_size)

    #(data_imgs, data_labels) = tf.data.Dataset.from_generator\
    test_dataset = tf.data.Dataset.from_generator\
        (
            #lambda: imagenet_load(data_dir=conf.data_path_imagenet,resize_wh=256,crop_wh=224,crops=1),
            lambda: imagenet_load(conf),
            #(tf.float32, tf.int32),
            (tf.float32, tf.float32),
            #((None,224,224,3),(None,1000)),
            #((None,224,224,3),(None,1000)),
            #((None,224,224,3),(None,1000)),
            ((None,conf.input_size,conf.input_size,3),(None,conf.num_class)),
        )

    test_dataset = test_dataset.map(keras_imagenet_preprocess, num_parallel_calls=2)
    test_dataset = test_dataset.prefetch(1)

    train_dataset = test_dataset
    valid_dataset = test_dataset

    return train_dataset, valid_dataset, test_dataset


def get_files(data_dir, data_name, max_rows=None):
    """Reads a \`data_name.txt\` (e.g., \`val.txt\`) from
    http://www.image-net.org/challenges/LSVRC/2012/
    """
    files, labels = np.split(
        np.genfromtxt("%s/%s.txt" % (data_dir, data_name),
                      dtype=np.str, max_rows=max_rows),
        [1], axis=1)

    #print(files)
    #print(np.genfromtxt("%s/%s.txt" % (data_dir, data_name),dtype=np.str, max_rows=max_rows))
    #print(np.loadtxt("%s/%s.txt" % (data_dir, data_name),dtype=np.str))

    files = files.flatten()
    labels = np.asarray(labels.flatten(), dtype=np.int)
    return files, labels


def get_labels(data_dir, data_name, max_rows=None):
    _, labels = get_files(data_dir, data_name, max_rows)
    return labels


def load_batch(data_dir, data_name, batch_size, resize_wh,
         crop_locs, crop_wh, total_num=None):
    #from tensornets.utils import crop, load_img

    print('total_num')
    print(total_num)

    files, labels = get_files(data_dir, data_name, total_num)
    #files, labels = get_files(data_dir, data_name, 10000)

    if total_num is None:
        total_num = len(labels)

    #print(total_num)
    #print(batch_size)

    num_batch=int(math.ceil(float(total_num)/float(batch_size)))

    #for batch_start in range(0, total_num, batch_size):
    #for batch_idx in range(0,num_batch):

    batch_idx = 0

    while True:

        if batch_idx >= num_batch:
            return

        #print('batch_idx')
        #print(batch_idx)

        batch_start = batch_idx*batch_size
        #batch_end = min((batch_idx+1)*batch_size-1,total_num)
        batch_end = min((batch_idx+1)*batch_size,total_num)

        #print('start')
        #print(batch_start)
        #print('end')
        #print(batch_end)

        #print(files[batch_start:batch_end])
        batch_size = batch_end-batch_start

        #print(resize_wh)
        #print(crop_wh)

        data_spec = [batch_size, 1, crop_wh, crop_wh, 3]
        if isinstance(crop_locs, list):
            data_spec[1] = len(crop_locs)
        elif crop_locs == 10:
            data_spec[1] = 10
        #X = np.zeros(data_spec, np.float32)
        X = np.zeros(data_spec, np.float64)
        #print(data_spec)

        #for (k, f) in enumerate(files[batch_start:batch_start+batch_size]):
        for (k,f) in enumerate(files[batch_start:batch_end]):
            filename = os.path.join("%s/ILSVRC2012_img_val" % data_dir, f)
            #if os.path.isfile(filename):
            #try:
            os.path.isfile(filename)
            img = load_img(filename, target_size=resize_wh)
            X[k] = crop(img, crop_wh, crop_locs)

            #X.append(crop(img, crop_wh, crop_locs))
            #x = crop(img, crop_wh, crop_locs)

            #except Exception as ex:
            #    print('exception: ',ex)
            #    raise

            #yield tfe.Variable(X), tfe.Variable(labels[k])
            #yield X, labels[k]
            #yield x, tf.constant((,tf.one_hot(labels[k],1000)))

        yield X.reshape((-1, crop_wh, crop_wh, 3)), tf.one_hot(labels[batch_start:batch_end],1000)
        batch_idx += 1


        #yield tfe.Variable(X.reshape((-1, crop_wh, crop_wh, 3))), tfe.Variable(labels[batch_start:batch_start+batch_size])
        #yield tfe.Variable(X), tfe.Variable(labels[batch_start:batch_start+batch_size])

        del X

# Copied from keras, Tensornet
def keras_imagenet_preprocess(x, labels):
    #x = x.copy()
    x = x[:, :, :, ::-1]

    x = tf.subtract(x, [103.939, 116.779, 123.68])
    #x = tf.subtract(x, [123.68, 116.779, 103.939])
    return x, labels


# from tf github: models.research.inception.inception,data.build_imagenet_data.py
def _find_image_files_train_2012(data_dir, labels_file):
    print('Determinig list of input files and labels from %s.' % data_dir)
    challenge_synsets = [l.split()[1] for l in tf.gfile.FastGFile(labels_file,'r').readlines()]


    labels = []
    filenames = []
    synsets = []

    # Leave label index 0 for empty as a backgound class
    label_index = 1

    data_dir = os.path.join(data_dir,'ILSVRC2012_img_train')
    # construct the list of JPEG files and labels
    for synset in challenge_synsets:
        jpeg_file_path = '%s/%s/*.JPEG' %(data_dir,synset)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        synsets.extend([synset] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (label_index, len(challenge_synsets)))
        label_index += 1

    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    synsets = [synsets[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s' % (len(filenames),len(challenge_synsets), data_dir))

    #
    #with open('ILVRC2012_training_filename','w') as f:
    #    for item in filenames:
    #        f.write('%s\n' % item)
    #
    #with open('ILVRC2012_training_synsets','w') as f:
    #    for item in synsets:
    #        f.write('%s\n' % item)
    #
    #with open('ILVRC2012_training_labels','w') as f:
    #    for item in labels:
    #        f.write('%d\n' % item)

    #print(labels)

    return filenames, synsets, labels


def _build_synset_lookup(imagenet_metadata_file):
    metadata = loadmat(imagenet_metadata_file, struct_as_record=False)

    synsets = np.squeeze(metadata['synsets'])
    print(synsets)


    ids = np.squeeze(np.array([x.ILSVRC2012_ID for x in synsets]))
    wnids = np.squeeze(np.array([x.WNID for x in synsets]))
    word = np.squeeze(np.array([x.words for x in synsets]))
    num_children = np.squeeze(np.array([x.num_children for x in synsets]))
    children = [np.squeeze(x.children).astype(np.int) for x in synsets]

    print(ids)
    print(wnids)
    print(word)
    print(num_children)
    print(children)






#def load_train_dataset()