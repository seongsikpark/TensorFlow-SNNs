

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
train=True
#train=False

#load_model=True
load_model=False

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



import re
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
from datasets.augmentation import cutmix



# models
#from models.vgg16 import VGG16
from models.vgg16_keras_toh5 import VGG16 as VGG16_KERAS
from models.vgg16_tr import VGG16_TR
from models.vgg16 import VGG16
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

# TODO: gpu mem usage - parameterize
# GPU mem usage
gpu = tf.config.experimental.list_physical_devices('GPU')
if gpu:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpu[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    except RuntimeError as e:
        print(e)


# training types
#train_type='finetuning' # not supported yet
#train_type='transfer'
train_type='scratch'
#train_type='scratch-4k-4k'
#train_type='scratch-4k-2k'
#train_type='scratch-4k-1k'
#train_type='scratch-4k-0.5k'
#train_type='scratch-2k-2k'
#train_type='scratch-2k-1k'
#train_type='scratch-2k-0.5k'
#train_type='scratch-1k-1k'
#train_type='scratch-1k-0.5k'
#train_type='scratch-0.5k-0.5k'

#n_dim_classifier=None
#n_dim_classifier=(4096,4096)
#n_dim_classifier=(4096,2048)
n_dim_classifier=(4096,1024)
#n_dim_classifier=(4096,512)
#n_dim_classifier=(2048,2048)
#n_dim_classifier=(2048,1024)
#n_dim_classifier=(2048,512)
#n_dim_classifier=(1024,1024)
#n_dim_classifier=(1024,512)
#n_dim_classifier=(512,512)

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



########################################
# DO NOT TOUCH
########################################
# data augmentation - mix

if conf.data_aug_mix == 'mixup':
    en_mixup = True
    en_cutmix = False
elif conf.data_aug_mix == 'cutmix':
    en_mixup = False
    en_cutmix = True
else:
    en_mixup = False
    en_cutmix = False



if dataset_name == 'ImageNet':
    num_class = 1000
elif dataset_name == 'CIFAR10':
    num_class = 10
else:
    assert False


# l2-norm
#lmb = 1.0E-10
lmb = conf.lmb


#



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
    #include_top = True
    input_size_pre_crop_ratio = 256/224
    # TODO: set
    if train:
        input_prec_mode = None # should be modified
    else:
        input_prec_mode = 'caffe' # keras pre-trained model

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



#
image_shape = (input_size, input_size, 3)


# dataset load
dataset = dataset_sel[dataset_name]
train_ds, valid_ds, test_ds = dataset.load(
            input_size,input_size_pre_crop_ratio,num_class,train,NUM_PARALLEL_CALL,conf,input_prec_mode)


# models
model_sel_tr = {
    'VGG16': VGG16_TR,
}

model_sel_sc = {
    'VGG16': VGG16,
}




# model compile
metric_accuracy = tf.keras.metrics.categorical_accuracy
metric_accuracy_top5 = tf.keras.metrics.top_k_categorical_accuracy

# TODO: move to configuration

metric_name_acc = 'acc'
metric_name_acc_top5 = 'acc-5'
monitor_cri = 'val_' + metric_name_acc

metric_accuracy.name = metric_name_acc
metric_accuracy_top5.name = metric_name_acc_top5

#model.compile(optimizer='adam',
              #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              ## metrics=['accuracy'])
              #metrics=[metric_accuracy, metric_accuracy_top5])

batch_size = batch_size_train

exp_set_name = model_name + '_' + dataset_name
# dir_model = './'+exp_set_name
dir_model = os.path.join(root_model, exp_set_name)

# TODO: functionalize
# file_name='checkpoint-epoch-{}-batch-{}.h5'.format(epoch,batch_size)
# config_name='ep-{epoch:04d}_bat-{}_lmb-{:.1E}'.format(batch_size,lmb)
# config_name='bat-{}_lmb-{:.1E}'.format(batch_size,lmb)

config_name = 'bat-{}_lmb-{:.1E}'.format(batch_size, lmb)
#config_name = 'bat-512_lmb-{:.1E}'.format(lmb)

if train_type=='transfer':
    pass
    config_name += '_tr'
elif train_type=='scratch':
    config_name += '_sc'
    if n_dim_classifier is not None:
        if model_name == 'VGG16':
            config_name = config_name+'-'+str(n_dim_classifier[0])+'-'+str(n_dim_classifier[1])
else:
    assert False

if en_mixup:
    config_name += '_mu'
elif en_cutmix:
    config_name += '_cm'

filepath = os.path.join(dir_model, config_name)




########################################
#
########################################

if load_model:
    # get latest saved model
    #latest_model = lib_snn.util.get_latest_saved_model(filepath)

    assert False, 'not yet implemented'
    latest_model = 'ep-1085'
    load_weight = os.path.join(filepath, latest_model)
    pre_model = tf.keras.models.load_model(load_weight)
    #print(pre_model.evaluate(valid_ds))
    #assert False

    #latest_model = lib_snn.util.get_latest_saved_model(filepath)
    #load_weight = os.path.join(filepath, latest_model)



    #model.load_weights(load_path)
    #tf.keras.models.save_model(model,filepath+'/ttt')
    #model.save_weights(filepath+'/weight_1.h5')

    #if not latest_model.startswith('ep-'):
        #assert False, 'the dir name of latest model should start with ''ep-'''
    #init_epoch = int(re.split('-|\.',latest_model)[1])

    include_top = True
    add_top = False
else:
    if train_type == 'transfer':
        load_weight = 'imagenet'
        include_top = False
        add_top = True

        model = model_sel_tr[model_name]

    else:
        load_weight = None
        include_top = True
        add_top = False

        model = model_sel_sc[model_name]
init_epoch = 0



#
#pretrained_model = model(input_shape=image_shape, conf=conf, include_top=include_top, weights='imagenet', train=train)
model = model(input_shape=image_shape, conf=conf, include_top=include_top,
              weights=load_weight, classes=num_class, n_dim_classifier=n_dim_classifier)
#model = model(input_shape=image_shape, conf=conf, include_top=False, weights=load_weight, train=train, add_top=True)
#model = model(input_shape=image_shape, conf=conf, include_top=include_top, train=train, add_top=add_top)
#pretrained_model = model(input_shape=image_shape, include_top=include_top, weights='imagenet',classifier_activation=None)
#pretrained_model = VGG16(include_top=True, weights='imagenet')
#pretrained_model = VGG19(include_top=True, weights='imagenet')
#pretrained_model = ResNet50(include_top=True, weights='imagenet')
#pretrained_model = ResNet101(include_top=True, weights='imagenet')

# TODO: move to parameter
run_eagerly=False
#run_eagerly=True

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[metric_accuracy, metric_accuracy_top5], run_eagerly=run_eagerly)

#assert False

if train:
    print('Train mode')
    # remove dir - train model
    if not load_model:
        if overwrite_train_model:
            if os.path.isdir(filepath):
                shutil.rmtree(filepath)

    # path_tensorboard = root_tensorboard+exp_set_name
    # path_tensorboard = root_tensorboard+filepath
    path_tensorboard = os.path.join(root_tensorboard, exp_set_name)
    path_tensorboard = os.path.join(path_tensorboard, config_name)

    if os.path.isdir(path_tensorboard):
        date_cur = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
        path_dest_tensorboard = path_tensorboard + '_' + date_cur
        print('tensorboard data already exists')
        print('move {} to {}'.format(path_tensorboard, path_dest_tensorboard))

        shutil.move(path_tensorboard, path_dest_tensorboard)

    #
    if load_model:
        print('Evaluate pretrained model')
        assert monitor_cri == 'val_acc', 'currently only consider monitor criterion - val_acc'
        idx_monitor_cri = model.metrics_names.index('acc')
        result = model.evaluate(valid_ds)
        best = result[idx_monitor_cri]
        print('previous best result - {}'.format(best))
    else:
        best = None

    #model.save_weights(filepath+'ep-1085',save_format='h5')

    #model.trainable=True
    #model.save_weights(filepath+'/test.h5',save_format='h5')
    #assert False

    #
    callbacks = [
        # tf.keras.callbacks.ModelCheckpoint(
        lib_snn.callbacks.ModelCheckpointResume(
            #filepath=filepath + '/ep-{epoch:04d}',
            filepath=filepath + '/ep-{epoch:04d}',
            save_weight_only=True,
            save_best_only=True,
            # monitor='val_acc',
            monitor=monitor_cri,
            # period=1,
            verbose=1,
            best=best
        ),
        tf.keras.callbacks.TensorBoard(log_dir=path_tensorboard, update_freq='epoch'),
        lib_snn.callbacks.ManageSavedModels(filepath=filepath)
    ]

    train_histories = model.fit(train_ds, epochs=epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                                callbacks=callbacks)

assert False
# set weights by layer from transfer learning model
# keep
if False:
    model = model(input_shape=image_shape, conf=conf, include_top=include_top, weights=None, train=train,
                  add_top=add_top, classes=num_class)
    model.model.get_layer('conv1').set_weights(pre_model.get_layer('vgg16').get_layer('block1_conv1').get_weights())
    model.model.get_layer('conv1_1').set_weights(pre_model.get_layer('vgg16').get_layer('block1_conv2').get_weights())
    model.model.get_layer('conv2').set_weights(pre_model.get_layer('vgg16').get_layer('block2_conv1').get_weights())
    model.model.get_layer('conv2_1').set_weights(pre_model.get_layer('vgg16').get_layer('block2_conv2').get_weights())
    model.model.get_layer('conv3').set_weights(pre_model.get_layer('vgg16').get_layer('block3_conv1').get_weights())
    model.model.get_layer('conv3_1').set_weights(pre_model.get_layer('vgg16').get_layer('block3_conv2').get_weights())
    model.model.get_layer('conv3_2').set_weights(pre_model.get_layer('vgg16').get_layer('block3_conv3').get_weights())
    model.model.get_layer('conv4').set_weights(pre_model.get_layer('vgg16').get_layer('block4_conv1').get_weights())
    model.model.get_layer('conv4_1').set_weights(pre_model.get_layer('vgg16').get_layer('block4_conv2').get_weights())
    model.model.get_layer('conv4_2').set_weights(pre_model.get_layer('vgg16').get_layer('block4_conv3').get_weights())
    model.model.get_layer('conv5').set_weights(pre_model.get_layer('vgg16').get_layer('block5_conv1').get_weights())
    model.model.get_layer('conv5_1').set_weights(pre_model.get_layer('vgg16').get_layer('block5_conv2').get_weights())
    model.model.get_layer('conv5_2').set_weights(pre_model.get_layer('vgg16').get_layer('block5_conv3').get_weights())
    w_bn_list = pre_model.get_layer('fc1').get_weights()
    w_bn_list += pre_model.get_layer('batch_normalization').get_weights()
    model.model.get_layer('fc1').set_weights(w_bn_list)
    w_bn_list = pre_model.get_layer('fc2').get_weights()
    w_bn_list += pre_model.get_layer('batch_normalization_1').get_weights()
    model.model.get_layer('fc2').set_weights(w_bn_list)
    model.model.get_layer('predictions').set_weights(pre_model.get_layer('predictions').get_weights())

    model.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=[metric_accuracy, metric_accuracy_top5],run_eagerly=True)

    result = model.evaluate(valid_ds)




assert False
#assert False

#
#pretrained_model = model(input_shape=image_shape, conf=conf, include_top=False, weights='imagenet', train=train)
#pretrained_model.trainable = False

pretrained_model = VGG16_KERAS(input_shape=image_shape, conf=conf, include_top=False, weights=load_weight, classes=num_class)




assert False
model = tf.keras.Sequential()

# train = True
# data augmentation
if train:
    # model.add(tf.keras.layers.GaussianNoise(0.1))
    model.add(tf.keras.layers.experimental.preprocessing.RandomZoom((-0.1, 0.1)))
    model.add(tf.keras.layers.experimental.preprocessing.RandomRotation((-0.03, 0.03)))

model.add(pretrained_model)
model.add(tf.keras.layers.Flatten(name='vgg16/flatten'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(lmb), name='vgg16/fc1'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(lmb), name='fc2'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax', name='predictions'))

model.load_weights(load_weight)


assert False

if train:
    print('Train mode')
    # remove dir - train model
    if not load_model:
        if overwrite_train_model:
            if os.path.isdir(filepath):
                shutil.rmtree(filepath)

    # path_tensorboard = root_tensorboard+exp_set_name
    # path_tensorboard = root_tensorboard+filepath
    path_tensorboard = os.path.join(root_tensorboard, exp_set_name)
    path_tensorboard = os.path.join(path_tensorboard, config_name)

    if os.path.isdir(path_tensorboard):
        date_cur = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
        path_dest_tensorboard = path_tensorboard + '_' + date_cur
        print('tensorboard data already exists')
        print('move {} to {}'.format(path_tensorboard, path_dest_tensorboard))

        shutil.move(path_tensorboard, path_dest_tensorboard)

    #
    if load_model:
        print('Evaluate pretrained model')
        assert monitor_cri == 'val_acc', 'currently only consider monitor criterion - val_acc'
        idx_monitor_cri = model.metrics_names.index('acc')
        result = model.evaluate(valid_ds)
        best = result[idx_monitor_cri]
        print('previous best result - {}'.format(best))
    else:
        best = None

    #model.save_weights(filepath+'ep-1085',save_format='h5')

    #model.trainable=True
    #model.save_weights(filepath+'/test.h5',save_format='h5')
    #assert False

    #
    callbacks = [
        # tf.keras.callbacks.ModelCheckpoint(
        lib_snn.callbacks.ModelCheckpointResume(
            filepath=filepath + '/ep-{epoch:04d}',
            save_weight_only=True,
            save_best_only=True,
            # monitor='val_acc',
            monitor=monitor_cri,
            # period=1,
            verbose=1,
            best=best
        ),
        tf.keras.callbacks.TensorBoard(log_dir=path_tensorboard, update_freq='epoch'),
        lib_snn.callbacks.ManageSavedModels(filepath=filepath)
    ]

    train_histories = model.fit(train_ds, epochs=epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                                callbacks=callbacks)
    # train_results = training_model.fit(train_ds,epochs=3,validation_data=valid_ds)

    # assert False

    # result = pretrained_model.evaluate(ds)
else:
    print('Test mode')
    result = model.evaluate(valid_ds)
    # result = model.predict(test_ds)

    print(result)



assert False

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

    # metric_accuracy = tf.keras.metrics.sparse_categorical_accuracy(name='accuracy')
    # metric_accuracy_top5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(name='accuracy_top5')





    if train:
        print('Train mode')
        # remove dir - train model
        if not load_model:
            if overwrite_train_model:
                if os.path.isdir(filepath):
                    shutil.rmtree(filepath)

        # path_tensorboard = root_tensorboard+exp_set_name
        # path_tensorboard = root_tensorboard+filepath
        path_tensorboard = os.path.join(root_tensorboard, exp_set_name)
        path_tensorboard = os.path.join(path_tensorboard, config_name)

        if os.path.isdir(path_tensorboard):
            date_cur = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
            path_dest_tensorboard = path_tensorboard + '_' + date_cur
            print('tensorboard data already exists')
            print('move {} to {}'.format(path_tensorboard, path_dest_tensorboard))

            shutil.move(path_tensorboard, path_dest_tensorboard)

        #
        if load_model:
            print('Evaluate pretrained model')

            assert monitor_cri == 'val_acc', 'currently only consider monitor criterion - val_acc'

            idx_monitor_cri = model.metrics_names.index('acc')

            result = model.evaluate(test_ds)

            best = result[idx_monitor_cri]

            print('previous best result - {}'.format(best))
        else:
            best = None

        # assert False
        #
        callbacks = [
            # tf.keras.callbacks.ModelCheckpoint(
            lib_snn.callbacks.ModelCheckpointResume(
                filepath=filepath + '/ep-{epoch:04d}',
                save_best_only=True,
                # monitor='val_acc',
                monitor=monitor_cri,
                # period=1,
                verbose=1,
                best=best
            ),
            tf.keras.callbacks.TensorBoard(log_dir=path_tensorboard, update_freq='epoch'),
            lib_snn.callbacks.ManageSavedModels(filepath=filepath)
        ]

        train_histories = model.fit(train_ds, epochs=epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                                    callbacks=callbacks)
        # train_results = training_model.fit(train_ds,epochs=3,validation_data=valid_ds)

        # assert False

        # result = pretrained_model.evaluate(ds)
    else:
        print('Test mode')
        result = model.evaluate(test_ds)
        # result = model.predict(test_ds)

        print(result)
