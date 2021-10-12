




#global input_size
#global input_size_pre_crop_ratio
import collections

global model_name

########################################
# configuration
########################################

# GPU setting
#
GPU_NUMBER=0

GPU_PARALLEL_RUN = 1
#GPU_PARALLEL_RUN = 2
#GPU_PARALLEL_RUN = 3

# RTX3090
if GPU_PARALLEL_RUN == 1:
    gpu_mem = -1
    NUM_PARALLEL_CALL = 15
elif GPU_PARALLEL_RUN == 2:
    gpu_mem = 10240
    NUM_PARALLEL_CALL = 7
elif GPU_PARALLEL_RUN == 3:
    gpu_mem = 6144
    NUM_PARALLEL_CALL = 5
else:
    assert False



# exp set name
#exp_set_name = 'HPTune-RS'
#exp_set_name = 'HPTune-TEST'
#exp_set_name = 'HPTune-GRID'
#exp_set_name = 'HPTune-GRID'
#exp_set_name = 'CODE_TEST'
exp_set_name = 'Train_SC'

# hyperparamter tune mode
#hp_tune = True
hp_tune = False

#
train=True
#train=False

#load_model=True
load_model=False

#
#overwrite_train_model =True
overwrite_train_model=False

#
overwrite_tensorboard = True

#epoch = 20000
#epoch = 20472
train_epoch = 300
#train_epoch = 1


# learning rate schedule - step_decay
step_decay_epoch = 100

#
root_hp_tune = './hp_tune'

#
root_model = './models'

# model
#model_name = 'VGG16'
#model_name = 'ResNet18'
#model_name = 'ResNet20'
#model_name = 'ResNet32'
#model_name = 'ResNet34'
#model_name = 'ResNet50'
model_name = 'ResNet18V2'
#model_name = 'ResNet20V2'

# dataset
dataset_name = 'CIFAR10'
#dataset_name = 'CIFAR100'
#dataset_name='ImageNet'

#
learning_rate = 0.2
#learning_rate = 0.01

#
opt='SGD'

#
#lr_schedule = 'COS'     # COSine
#lr_schedule = 'COSR'    # COSine with Restart
lr_schedule = 'STEP'    # STEP wise
#lr_schedule = 'STEP_WUP'    # STEP wise, warmup


#
root_tensorboard = './tensorboard/'



import re
import datetime
import shutil

from functools import partial

import tensorflow as tf

# HP tune
#import kerastuner as kt
import keras_tuner as kt
#import tensorboard.plugins.hparams import api as hp



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

#from models.vgg16_tr import VGG16_TR
#from models.vgg16 import VGG16
#from models.resnet import ResNet18
#from models.resnet import ResNet20
#from models.resnet import ResNet32
#from models.resnet import ResNet34
#from models.resnet import ResNet50
#from models.resnet import ResNet101
#from models.resnet import ResNet152
#from models.resnet import ResNet18V2
#from models.resnet import ResNet20V2

from models.models import model_sel

#
#from lib_snn.hp_tune_model import model_builder




#
#import test
#import train

#
#import models.input_preprocessor as preprocessor

#
#tf.config.functions_run_eagerly()


os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_NUMBER)

# TODO: gpu mem usage - parameterize
# GPU mem usage
#if False:
#if False:
#if True:
if gpu_mem != -1:
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

#
f_hp_tune = train and hp_tune

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


#
initial_channels_sel= {
    'VGG16': 64,
}
initial_channels = initial_channels_sel.get(model_name,64)



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

################
# name set
################

# TODO: configuration & file naming
#exp_set_name = model_name + '_' + dataset_name
model_dataset_name = model_name + '_' + dataset_name

# dir_model = './'+exp_set_name
#dir_model = os.path.join(root_model, exp_set_name)
dir_model = os.path.join(root_model, model_dataset_name)


# hyperparameter tune name
#hp_tune_name = exp_set_name+'_'+model_dataset_name+'_ep-'+str(train_epoch)
hp_tune_name = exp_set_name

# TODO: functionalize
# file_name='checkpoint-epoch-{}-batch-{}.h5'.format(epoch,batch_size)
# config_name='ep-{epoch:04d}_bat-{}_lmb-{:.1E}'.format(batch_size,lmb)
# config_name='bat-{}_lmb-{:.1E}'.format(batch_size,lmb)

#config_name = 'bat-{}_opt-{}_lr-{:.0E}_lmb-{:.0E}'.format(batch_size,opt,learning_rate,lmb)
config_name = 'ep-{}_bat-{}_opt-{}_lr-{}-{:.0E}_lmb-{:.0E}'.format(train_epoch,batch_size,opt,lr_schedule,learning_rate,lmb)

#config_name = 'bat-{}_lmb-{:.0E}'.format(batch_size, lmb)
#config_name = 'bat-512_lmb-{:.1E}'.format(lmb)

if train_type=='transfer':
    config_name += '_tr'
elif train_type=='scratch':
    config_name += '_sc'
    #if n_dim_classifier is not None:
        #if model_name == 'VGG16':
            #config_name = config_name+'-'+str(n_dim_classifier[0])+'-'+str(n_dim_classifier[1])
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

model_top = model_sel(model_name,train_type)

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

    #if train_type == 'transfer':
        #model_top = model_sel_tr[model_name]
    #elif train_type == 'scratch':
        #model_top = model_sel_sc[model_name]
    #else:
        #assert False
else:
    if train_type == 'transfer':
        load_weight = 'imagenet'
        include_top = False
        add_top = True

        #model_top = model_sel_tr[model_name]

    elif train_type == 'scratch':
        load_weight = None
        include_top = True
        add_top = False

        #model_top = model_sel_sc[model_name]
    else:
        assert False

    init_epoch = 0



# for HP tune
model_top_glb = model_top

#
# model builder
if f_hp_tune:

    # TODO: move to config.py
    #hp_model_builder = model_builder
    hps = collections.OrderedDict()
    hps['dataset'] = [dataset_name]
    hps['model'] = [model_name]
    hps['opt'] = [opt]
    hps['lr_schedule'] = [lr_schedule]
    hps['train_epoch'] = [train_epoch]
    hps['step_decay_epoch'] = [step_decay_epoch]

    # main to hp_tune, need to seperate configuration
    hp_tune_args = collections.OrderedDict()
    hp_tune_args['model_top'] = model_top
    hp_tune_args['image_shape'] = image_shape
    hp_tune_args['conf'] = conf
    hp_tune_args['include_top'] = include_top
    hp_tune_args['load_weight'] = load_weight
    hp_tune_args['num_class'] = num_class
    hp_tune_args['metric_accuracy'] = metric_accuracy
    hp_tune_args['metric_accuracy_top_5'] = metric_accuracy_top5
    hp_tune_args['train_steps_per_epoch'] = train_steps_per_epoch

    #hp_model_builder = partial(model_builder, hp, hps)
    hp_model_builder = lib_snn.hp_tune_model.CustomHyperModel(hp_tune_args, hps)


    #tuner = kt.Hyperband(model_builder,
    #tuner=kt.RandomSearch(model_builder,
    tuner=lib_snn.hp_tune.GridSearch(hp_model_builder,
                         objective='val_acc',
                         #max_trials=12,
                         #max_epochs = 300,
                         #factor=3,
                         overwrite=True,
                         directory=root_hp_tune,
                         project_name=hp_tune_name,
                         #directory='test_hp_dir',
                         #project_name='test_hp')
                          )

    #tuner.results_summary()
    #assert False
else:
    model = lib_snn.model_builder.model_builder(
        model_top, image_shape, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
        train_epoch, train_steps_per_epoch,
        opt, learning_rate,
        lr_schedule, step_decay_epoch,
        metric_accuracy, metric_accuracy_top5)



if load_model:
    model.load_weights(load_weight)
    # model.load_weights(load_weight,by_name=True)


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

if f_hp_tune:
    path_tensorboard = os.path.join(root_tensorboard, hp_tune_name)

else:
    path_tensorboard = os.path.join(root_tensorboard, exp_set_name)
    path_tensorboard = os.path.join(path_tensorboard, model_dataset_name)
    path_tensorboard = os.path.join(path_tensorboard, config_name)

if not overwrite_tensorboard:
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
    best = None

#model.save_weights(filepath+'ep-1085',save_format='h5')

#model.trainable=True
#model.save_weights(filepath+'/test.h5',save_format='h5')
#assert False

#
#with tf.summary.create_file_writer('path_tensorboard/hptune').as_default():
    #hp.hparams_config(
        #hparams
    #)


########
# Callbacks
########
# model checkpoint save and resume
cb_model_checkpoint = lib_snn.callbacks.ModelCheckpointResume(
    # filepath=filepath + '/ep-{epoch:04d}',
    # filepath=filepath + '/ep-{epoch:04d}.ckpt',
    filepath=filepath + '/ep-{epoch:04d}.hdf5',
    save_weight_only=True,
    save_best_only=True,
    monitor=monitor_cri,
    verbose=1,
    best=best,
    log_dir=path_tensorboard,
    # tensorboard_writer=cb_tensorboard._writers['train']
)
cb_manage_saved_model = lib_snn.callbacks.ManageSavedModels(filepath=filepath)
cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=path_tensorboard, update_freq='epoch')



#
if train:
    if hp_tune:
        print('HP Tune mode')

        callbacks = [cb_tensorboard]

        tuner.search(train_ds, epochs=train_epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                     callbacks=callbacks)
    else:
        print('Train mode')

        callbacks = [
            cb_model_checkpoint,
            cb_tensorboard,
            cb_manage_saved_model
        ]

        train_histories = model.fit(train_ds, epochs=train_epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                                    callbacks=callbacks)
else:
    print('Test mode')
    result = model.evaluate(valid_ds)
    # result = model.predict(test_ds)

    print(result)


