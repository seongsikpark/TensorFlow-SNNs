


########
# HP Tune
########

def model_builder(hp):
    #
    # pretrained_model = model(input_shape=image_shape, conf=conf, include_top=include_top, weights='imagenet', train=train)

    # model_top = model_top(input_shape=image_shape, conf=conf, include_top=include_top,
    # weights=load_weight, classes=num_class, n_dim_classifier=n_dim_classifier,name=model_name)

    hp_lmb = hp.Choice('lmb', values = [5e-4, 1e-4, 5e-5, 1e-5])
    hp_learning_rate = hp.Choice('learning_rate', values = [0.1, 0.01, 0.2])

    model_top = model_top_glb(input_shape=image_shape, conf=conf, include_top=include_top,
                              weights=load_weight, classes=num_class, name=model_name, lmb=hp_lmb)

    # model = model(input_shape=image_shape, conf=conf, include_top=False, weights=load_weight, train=train, add_top=True)
    # model = model(input_shape=image_shape, conf=conf, include_top=include_top, train=train, add_top=add_top)
    # pretrained_model = model(input_shape=image_shape, include_top=include_top, weights='imagenet',classifier_activation=None)
    # pretrained_model = VGG16(include_top=True, weights='imagenet')
    # pretrained_model = VGG19(include_top=True, weights='imagenet')
    # pretrained_model = ResNet50(include_top=True, weights='imagenet')
    # pretrained_model = ResNet101(include_top=True, weights='imagenet')

    # TODO: move to parameter
    run_eagerly = False
    # run_eagerly=True

    # TODO: parameterize
    lr_schedule_first_decay_step = train_steps_per_epoch * 10  # in iteration

    if lr_schedule == 'COS':
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(hp_learning_rate, train_steps_per_epoch * 300)
    elif lr_schedule == 'COSR':
        learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(hp_learning_rate, lr_schedule_first_decay_step)
    elif lr_schedule == 'STEP':
        # learning_rate = lib_snn.optimizers.LRSchedule_step(learning_rate, 100*100, 0.1)
        learning_rate = lib_snn.optimizers.LRSchedule_step(hp_learning_rate, train_steps_per_epoch * 100, 0.1)
    elif lr_schedule == 'STEP_WUP':
        learning_rate = lib_snn.optimizers.LRSchedule_step_wup(hp_learning_rate, train_steps_per_epoch * 100, 0.1,
                                                               train_steps_per_epoch * 30)
    else:
        assert False

    if opt == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD')
    else:
        assert False

    # TODO: model_top wrapper use check - from scratch, transfer learning

    model = model_top.model

    if load_model:
        model.load_weights(load_weight)
        # model.load_weights(load_weight,by_name=True)

    # opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[metric_accuracy, metric_accuracy_top5], run_eagerly=run_eagerly)


    return model



#global input_size
#global input_size_pre_crop_ratio
global model_name

########################################
# configuration
########################################

# Parallel CPU
#NUM_PARALLEL_CALL = 7
NUM_PARALLEL_CALL = 15



#
train=True
#train=False

#load_model=True
load_model=False

#
#overwrite_train_model =True
overwrite_train_model=False

#epoch = 20000
#epoch = 20472
train_epoch = 300
#train_epoch = 10
root_model = './models'

# model
model_name = 'VGG16'
model_name = 'ResNet18'
model_name = 'ResNet20'
#model_name = 'ResNet34'
#model_name = 'ResNet50'

# dataset
dataset_name = 'CIFAR10'
#dataset_name = 'CIFAR100'
#dataset_name='ImageNet'

#
learning_rate = 0.1
#learning_rate = 0.01

#
opt='SGD'

#
lr_schedule = 'COS'     # COSine
#lr_schedule = 'COSR'    # COSine with Restart
#lr_schedule = 'STEP'    # STEP wise
#lr_schedule = 'STEP_WUP'    # STEP wise, warmup


#
root_tensorboard = './tensorboard/'



import re
import datetime
import shutil

import tensorflow as tf

# HP tune
#import kerastuner as kt
import keras_tuner as kt



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


# lr schedule



# models
#from models.vgg16 import VGG16
from models.vgg16_keras_toh5 import VGG16 as VGG16_KERAS
from models.vgg16_tr import VGG16_TR
from models.vgg16 import VGG16
from models.resnet import ResNet18
from models.resnet import ResNet20
from models.resnet import ResNet34
from models.resnet import ResNet50
from models.resnet import ResNet101
from models.resnet import ResNet152
#from tensorflow.keras.applications.vgg16 import VGG16


#
#import test
#import train

#
#import models.input_preprocessor as preprocessor

#
#tf.config.functions_run_eagerly()

#
gpu_number=0
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)

# TODO: gpu mem usage - parameterize
# GPU mem usage
#if False:
#gpu_mem = 6144
gpu_mem = 10240
if False:
#if True:
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if gpu:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpu[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem)])
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
#n_dim_classifier=(4096,1024)
#n_dim_classifier=(4096,512)
#n_dim_classifier=(2048,2048)
#n_dim_classifier=(2048,1024)
#n_dim_classifier=(2048,512)
#n_dim_classifier=(1024,1024)
#n_dim_classifier=(1024,512)
n_dim_classifier=(512,512)

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

#
assert conf.data_format == 'channels_last', 'not support "{}", only support channels_last'.format(conf.data_format)

########################################
# DO NOT TOUCH
########################################
# data augmentation - mix




# l2-norm
#lmb = 1.0E-10
lmb = conf.lmb




GPU = 'RTX_3090'
# NVIDIA TITAN V (12GB)
if GPU=='NVIDIA_TITAN_V':
    batch_size_inference_sel ={
        'NASNetLarge': 128,
        'EfficientNetB4': 128,
        'EfficientNetB5': 128,
        'EfficientNetB6': 64,
        'EfficientNetB7': 64,
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

# TODO:
dataset_sel = {
    'ImageNet': datasets.imagenet,
    'CIFAR10': datasets.cifar,
    'CIFAR100': datasets.cifar,
}


#
input_size_default = {
    'ImageNet': 244,
    'CIFAR10': 32,
    'CIFAR100': 32,
}

#
input_sizes_imagenet = {
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

input_sizes_cifar = {
    'VGG16': 32,
}

#
input_size_sel ={
    'ImageNet': input_sizes_imagenet,
    'CIFAR10': input_sizes_cifar,
    'CIFAR100': input_sizes_cifar,
}

# TODO: integrate input size selector
input_size = input_size_sel[dataset_name].get(model_name,input_size_default[dataset_name])
#input_size = input_sizes.get(model_name,224)


#batch_size_inference = batch_size_inference_sel.get(model_name,256)
batch_size_inference = conf.batch_size
batch_size_train = conf.batch_size
#batch_size_train = batch_size_train_sel.get(model_name,256)



#
image_shape = (input_size, input_size, 3)


# dataset load
#dataset = dataset_sel[dataset_name]
#train_ds, valid_ds, test_ds = dataset.load(dataset_name,input_size,input_size_pre_crop_ratio,num_class,train,NUM_PARALLEL_CALL,conf,input_prec_mode)
train_ds, valid_ds, test_ds, num_class = datasets.datasets.load(dataset_name,input_size,train_type,train,conf,NUM_PARALLEL_CALL)

#assert False
train_steps_per_epoch = train_ds.cardinality().numpy()

# models
model_sel_tr = {
    'VGG16': VGG16_TR,
}

model_sel_sc = {
    'VGG16': VGG16,
    'ResNet18': ResNet18,
    'ResNet20': ResNet20,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
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


# TODO: configuration & file naming
exp_set_name = model_name + '_' + dataset_name
# dir_model = './'+exp_set_name
dir_model = os.path.join(root_model, exp_set_name)


# TODO: functionalize
# file_name='checkpoint-epoch-{}-batch-{}.h5'.format(epoch,batch_size)
# config_name='ep-{epoch:04d}_bat-{}_lmb-{:.1E}'.format(batch_size,lmb)
# config_name='bat-{}_lmb-{:.1E}'.format(batch_size,lmb)

#config_name = 'bat-{}_opt-{}_lr-{:.0E}_lmb-{:.0E}'.format(batch_size,opt,learning_rate,lmb)
config_name = 'bat-{}_opt-{}_lr-{}-{:.0E}_lmb-{:.0E}'.format(batch_size,opt,lr_schedule,learning_rate,lmb)

#config_name = 'bat-{}_lmb-{:.0E}'.format(batch_size, lmb)
#config_name = 'bat-512_lmb-{:.1E}'.format(lmb)

if train_type=='transfer':
    config_name += '_tr'
elif train_type=='scratch':
    config_name += '_sc'
    if n_dim_classifier is not None:
        if model_name == 'VGG16':
            config_name = config_name+'-'+str(n_dim_classifier[0])+'-'+str(n_dim_classifier[1])
else:
    assert False

if conf.data_aug_mix == 'mixup':
    en_mixup = True
    en_cutmix = False
elif conf.data_aug_mix == 'cutmix':
    en_mixup = False
    en_cutmix = True
else:
    en_mixup = False
    en_cutmix = False

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

    #assert False, 'not yet implemented'
    #latest_model = 'ep-1085'
    latest_model = lib_snn.util.get_latest_saved_model(filepath)
    load_weight = os.path.join(filepath, latest_model)
    #pre_model = tf.keras.models.load_model(load_weight)
    #print(pre_model.evaluate(valid_ds))
    #assert False

    #latest_model = lib_snn.util.get_latest_saved_model(filepath)
    #load_weight = os.path.join(filepath, latest_model)



    #model.load_weights(load_path)
    #tf.keras.models.save_model(model,filepath+'/ttt')
    #model.save_weights(filepath+'/weight_1.h5')

    if not latest_model.startswith('ep-'):
        assert False, 'the dir name of latest model should start with ''ep-'''
    init_epoch = int(re.split('-|\.',latest_model)[1])

    include_top = True
    add_top = False

    if train_type == 'transfer':
        model_top = model_sel_tr[model_name]
    elif train_type == 'scratch':
        model_top = model_sel_sc[model_name]
    else:
        assert False
else:
    if train_type == 'transfer':
        load_weight = 'imagenet'
        include_top = False
        add_top = True

        model_top = model_sel_tr[model_name]

    elif train_type == 'scratch':
        load_weight = None
        include_top = True
        add_top = False

        model_top = model_sel_sc[model_name]
    else:
        assert False

    init_epoch = 0



# for HP tune
model_top_glb = model_top

#tuner = kt.Hyperband(model_builder,
tuner=kt.RandomSearch(model_builder,
                     objective='val_acc',
                     max_trials=12,
                     #max_epochs = 300,
                     #factor=3,
                     #overwrite=True,
                     directory='Keras_Tune',
                     project_name='keras_tune_random',
                     #directory='test_hp_dir',
                     #project_name='test_hp')
                      )

#tuner.results_summary()
#assert False


#
#if train:
    #print('Train mode')
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
    result = model.evaluate(valid_ds)
    idx_monitor_cri = model.metrics_names.index('acc')
    best = result[idx_monitor_cri]
    print('previous best result - {}'.format(best))
else:
    best = Non#e

#model.save_weights(filepath+'ep-1085',save_format='h5')

#model.trainable=True
#model.save_weights(filepath+'/test.h5',save_format='h5')
#assert False

#
callbacks = [
    # tf.keras.callbacks.ModelCheckpoint(
    lib_snn.callbacks.ModelCheckpointResume(
        #filepath=filepath + '/ep-{epoch:04d}',
        #filepath=filepath + '/ep-{epoch:04d}.ckpt',
        filepath=filepath + '/ep-{epoch:04d}.hdf5',
        save_weight_only=True,
        #save_weight_only=False,
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

tuner.search(train_ds, epochs=train_epoch, initial_epoch=init_epoch, validation_data=valid_ds,
             callbacks=callbacks)

#train_histories = model.fit(train_ds, epochs=epoch, initial_epoch=init_epoch, validation_data=valid_ds,
#                            callbacks=callbacks)


########################################
#
########################################


