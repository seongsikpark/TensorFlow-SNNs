

global input_size
#global input_size_pre_crop_ratio
global model_name
global num_class

########################################
# configuration
########################################

# Parallel CPU
NUM_PARALLEL_CALL = 15


#
#train=True
train=False

load_model=True
#load_model=False

#
#overwrite_train_model =True
overwrite_train_model=False

epoch = 10000
root_model = './models'

# model
model_name = 'VGG16'

# dataset
dataset_name = 'CIFAR10'
#dataset_name='ImageNet'



#
root_tensorboard = './tensorboard/'

#
lmb = 1.0E-8

# data augmentation
# mixup
en_mixup=True
#en_mixup=False

if dataset_name == 'ImageNet':
    num_class = 1000
elif dataset_name == 'CIFAR10':
    num_class = 10
else:
    assert False





import datetime
import shutil

import tensorflow as tf




from tensorflow.keras.applications import imagenet_utils

#from tensorflow.keras.preprocessing import img_to_array, load_img

#
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#
import tqdm

import os
import matplotlib as plt

import numpy as np
import argparse
import cv2

# configuration
from config import flags

# snn library
import lib_snn

#
import datasets
from datasets import augmentation as daug
from datasets.augmentation import resize_with_crop
from datasets.augmentation import resize_with_crop_aug
from datasets.augmentation import mixup


# models
#from models.vgg16 import VGG16
from models.vgg16_keras import VGG16
#from tensorflow.keras.applications.vgg16 import VGG16


#
#import test
#import train

#
import models.input_preprocessor as preprocessor

#
#tf.config.functions_run_eagerly()

#
gpu_number=0
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)




# training types
#train_type='finetuning' # not supported yet
train_type='trasfer'
#train_type='scratch'
#train_type='pretranined'

#
#model_name='Xception'
#model_name='VGG16'
#model_name='VGG19'
#model_name='ResNet50'
#model_name='ResNet101'
#model_name='ResNet152'
#model_name='ResNet50V2'
#model_name='ResNet101V2'
#model_name='ResNet152V2'
#model_name='InceptionV3'
#model_name='InceptionResNetV2'
#model_name='MobileNet'
#model_name='MobileNetV2'
#model_name='DenseNet121'
#model_name='DenseNet169'
#model_name='DenseNet201'
#model_name='NASNetMobile'
#model_name='NASNetLarge'
#model_name='EfficientNetB0'
#model_name='EfficientNetB1'
#model_name='EfficientNetB2'
#model_name='EfficientNetB3'
#model_name='EfficientNetB4'
#model_name='EfficientNetB5'
#model_name='EfficientNetB6'
#model_name='EfficientNetB7'




#
conf = flags.FLAGS



GPU = 'RTX_3090'
# NVIDIA TITAN V (12GB)
if GPU=='NVIDIA_TITAN_V':
    input_sizes = {
        'Xception': 299,
        'InceptionV3': 299,
        'InceptionResNetV2': 299,
        'NASNetLarge': 331,
        'EfficientNetB1': 240,
        'EfficientNetB2': 260,
        'EfficientNetB4': 380,
        'EfficientNetB5': 456,
        'EfficientNetB6': 528,
        'EfficientNetB7': 600,
    }

    batch_size_inference_sel ={
        'NASNetLarge': 128,
        'EfficientNetB4': 128,
        'EfficientNetB5': 128,
        'EfficientNetB6': 64,
        'EfficientNetB7': 64,
    }


input_sizes = {
    'Xception': 299,
    'InceptionV3': 299,
    'InceptionResNetV2': 299,
    'NASNetLarge': 331,
    'EfficientNetB1': 240,
    'EfficientNetB2': 260,
    'EfficientNetB4': 380,
    'EfficientNetB5': 456,
    'EfficientNetB6': 528,
    'EfficientNetB7': 600,
}

batch_size_inference_sel ={
    'NASNetLarge': 128,
    'EfficientNetB4': 128,
    'EfficientNetB5': 128,
    'EfficientNetB6': 64,
    'EfficientNetB7': 64,
}

batch_size_train_sel = {
    #'VGG16': 256,
    'VGG16': 512,
    #'VGG16': 1024,
    #'VGG16': 2048,
}

dataset_sel = {
    'CIFAR10': datasets.cifar10,
    'ImageNet': datasets.imagenet,
}


# TODO: integrate input size selector
input_size = input_sizes.get(model_name,224)
#batch_size_inference = batch_size_inference_sel.get(model_name,256)
batch_size_inference = conf.batch_size
batch_size_train = conf.batch_size
#batch_size_train = batch_size_train_sel.get(model_name,256)

# input shape
if dataset_name == 'ImageNet':
    include_top = True
    input_size_pre_crop_ratio = 256/224
else:
    # CIFAR-10
    # TODO:
    include_top = False

    assert False, 'selecting input size and ratio depending on transfer learning or scratch mode'
    input_size=224
    input_size_pre_crop_ratio = 256/224


    #input_size=32


#
image_shape = (input_size, input_size, 3)


# dataset load
dataset = dataset_sel[dataset_name]
train_ds, valid_ds, test_ds = dataset.load(conf)


# models
model_sel = {
    'VGG16': VGG16,
}

model = model_sel[model_name]


#
pretrained_model = model(input_shape=image_shape, conf=conf, include_top=include_top, weights='imagenet')
#pretrained_model = model(input_shape=image_shape, include_top=include_top, weights='imagenet',classifier_activation=None)
#pretrained_model = VGG16(include_top=True, weights='imagenet')
#pretrained_model = VGG19(include_top=True, weights='imagenet')
#pretrained_model = ResNet50(include_top=True, weights='imagenet')
#pretrained_model = ResNet101(include_top=True, weights='imagenet')

#pretrained_model.trainable = False


#
#if train:
#    print('< Train mode >')
#    assert False
#
#else:
#    print('< Test mode >')
#    test.run(model_name,dataset_name,num_class,input_size,input_size_pre_crop_ratio)


#assert False
#
# only inference
if dataset_name == 'ImageNet':

    pretrained_model.compile(optimizer='adam',
                             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                             #metrics=['accuracy'])
                             #metrics=[tf.keras.metrics.sparse_top_k_categorical_accuracy])
                             metrics=[tf.keras.metrics.categorical_accuracy, \
                                      tf.keras.metrics.top_k_categorical_accuracy])

    # Preprocess input
    #ds=ds.map(resize_with_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #valid_ds=valid_ds.map(daug.resize_with_crop,num_parallel_calls=NUM_PARALLEL_CALL)
    valid_ds=valid_ds.map(lambda image, label: resize_with_crop(image, label, input_size, input_size_pre_crop_ratio, num_class),
                          num_parallel_calls=NUM_PARALLEL_CALL)
    #valid_ds=valid_ds.map(eager_resize_with_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_ds=valid_ds.batch(batch_size_inference)
    valid_ds=valid_ds.prefetch(NUM_PARALLEL_CALL)

    #ds=ds.take(1)

    #t1 = valid_ds.take(1)
    #assert False

    result = pretrained_model.evaluate(valid_ds,workers=NUM_PARALLEL_CALL)
    #result = pretrained_model.predict(valid_ds,workers=NUM_PARALLEL_CALL)
elif dataset_name == 'CIFAR10':

    # Preprocess input
    if train:
        if en_mixup:
            train_ds=train_ds.map(lambda train_ds_1, train_ds_2: mixup(train_ds_1,train_ds_2,alpha=0.2),num_parallel_calls=NUM_PARALLEL_CALL)
            #train_ds=train_ds.map(lambda train_ds_1, train_ds_2: eager_mixup(train_ds_1,train_ds_2,alpha=0.2),num_parallel_calls=tf.data.experimental.AUTOTUNE)
            train_ds=train_ds.batch(batch_size_train)
            #train_ds=train_ds.prefetch(tf.data.experimental.AUTOTUNE)
            train_ds = train_ds.prefetch(NUM_PARALLEL_CALL)

        else:
            #train_ds=train_ds.map(resize_with_crop_aug,num_parallel_calls=tf.data.experimental.AUTOTUNE)
            train_ds=train_ds.map(resize_with_crop_aug,num_parallel_calls=NUM_PARALLEL_CALL)
            train_ds=train_ds.batch(batch_size_train)
            #train_ds=train_ds.prefetch(tf.data.experimental.AUTOTUNE)
            train_ds = train_ds.prefetch(NUM_PARALLEL_CALL)



    #valid_ds=valid_ds.map(resize_with_crop_cifar,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #valid_ds=valid_ds.map(resize_with_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #valid_ds=valid_ds.map(resize_with_crop,num_parallel_calls=NUM_PARALLEL_CALL)
    valid_ds=valid_ds.map(lambda image, label: resize_with_crop(image, label, input_size, input_size_pre_crop_ratio, num_class),
        num_parallel_calls=NUM_PARALLEL_CALL)
    #valid_ds=valid_ds.batch(batch_size_inference)
    valid_ds=valid_ds.batch(batch_size_train)
    #valid_ds=valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds=valid_ds.prefetch(NUM_PARALLEL_CALL)

    #test_ds=test_ds.map(resize_with_crop,num_parallel_calls=NUM_PARALLEL_CALL)
    #test_ds=test_ds.batch(batch_size_train)
    #test_ds=test_ds.prefetch(NUM_PARALLEL_CALL)



    #
    pretrained_model.trainable=False
    model = tf.keras.Sequential()

    #train = True
    # data augmentation
    if train:
        #model.add(tf.keras.layers.GaussianNoise(0.1))
        model.add(tf.keras.layers.experimental.preprocessing.RandomZoom((-0.1,0.1)))
        model.add(tf.keras.layers.experimental.preprocessing.RandomRotation((-0.03,0.03)))

    model.add(pretrained_model)
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(lmb), name='fc1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(lmb), name='fc2'))
    #model.add(tf.keras.layers.Dense(1024, activation='relu', name='fc2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name='predictions'))


    #x = pretrained_model(train_ds)
    #x = tf.keras.layers.Flatten(name='flatten')(x)
    #x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    #x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
    #output = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)



    #metric_accuracy = tf.keras.metrics.sparse_categorical_accuracy(name='accuracy')
    #metric_accuracy_top5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(name='accuracy_top5')

    metric_accuracy = tf.keras.metrics.categorical_accuracy
    metric_accuracy_top5 = tf.keras.metrics.top_k_categorical_accuracy

    metric_accuracy.name = 'acc'
    metric_accuracy_top5.name = 'acc-5'

    model.compile(optimizer='adam',
                             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                             #metrics=['accuracy'])
                             metrics=[metric_accuracy, metric_accuracy_top5])



    batch_size=batch_size_train

    exp_set_name = model_name+'_'+dataset_name
    #dir_model = './'+exp_set_name
    dir_model = os.path.join(root_model,exp_set_name)

    #file_name='checkpoint-epoch-{}-batch-{}.h5'.format(epoch,batch_size)
    #config_name='ep-{epoch:04d}_bat-{}_lmb-{:.1E}'.format(batch_size,lmb)
    #config_name='bat-{}_lmb-{:.1E}'.format(batch_size,lmb)
    if en_mixup:
        config_name='bat-{}_lmb-{:.1E}_mu'.format(batch_size,lmb)
    else:
        config_name='bat-{}_lmb-{:.1E}'.format(batch_size,lmb)

    filepath = os.path.join(dir_model,config_name)



    ########################################
    #
    ########################################

    if load_model:
        # get latest saved model
        latest_model=lib_snn.util.get_latest_saved_model(filepath)
        load_path = os.path.join(filepath,latest_model)
        model = tf.keras.models.load_model(load_path)

        if not latest_model.startswith('ep-'):
            assert False, 'the dir name of latest model should start with ''ep-'''
        init_epoch = int(latest_model.split('-')[1])
    else:
        init_epoch = 0


    if train:
        print('Train mode')
        # remove dir - train model
        if not load_model:
            if overwrite_train_model:
                if os.path.isdir(filepath):
                    shutil.rmtree(filepath)

        #path_tensorboard = root_tensorboard+exp_set_name
        #path_tensorboard = root_tensorboard+filepath
        path_tensorboard = os.path.join(root_tensorboard,exp_set_name)
        path_tensorboard = os.path.join(path_tensorboard,config_name)

        if os.path.isdir(path_tensorboard):
            date_cur = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
            path_dest_tensorboard = path_tensorboard+'_'+date_cur
            print('tensorboard data already exists')
            print('move {} to {}'.format(path_tensorboard,path_dest_tensorboard))

            shutil.move(path_tensorboard,path_dest_tensorboard)

        #
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath+'/ep-{epoch:04d}',
                save_best_only=True,
                monitor='val_acc',
                #period=1,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(log_dir=path_tensorboard,update_freq='epoch'),
            lib_snn.callbacks.ManageSavedModels(filepath=filepath)
        ]

        train_histories = model.fit(train_ds,epochs=epoch,initial_epoch=init_epoch,validation_data=valid_ds,callbacks=callbacks)
        #train_results = training_model.fit(train_ds,epochs=3,validation_data=valid_ds)

        #assert False

        #result = pretrained_model.evaluate(ds)
    else:
        print('Test mode')
        result = model.evaluate(test_ds)
        #result = model.predict(test_ds)

        print(result)