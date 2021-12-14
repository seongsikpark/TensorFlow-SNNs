



import re
import datetime
import shutil

from functools import partial

import os

# TF logging setup
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_probability as tfp

#
from absl import app
from absl import flags

# HP tune
#import kerastuner as kt
import keras_tuner as kt
#import tensorboard.plugins.hparams import api as hp


#
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#
#import tqdm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
#np.set_printoptions(precision=4)
#np.set_printoptions(linewidth=np.inf)
#import tensorflow.experimental.numpy as tnp
#tnp.set_printoptions(linewidth=np.inf)

#import argparse
#import cv2

# configuration
import config
from config import conf

# snn library
import lib_snn

#
import datasets
#global input_size
#global input_size_pre_crop_ratio
import collections


# TODO: check use
global model_name


#
from lib_snn.sim import glb_plot
from lib_snn.sim import glb_plot_1
from lib_snn.sim import glb_plot_2

#
#conf = flags.FLAGS

########################################
# configuration
########################################

# logging - ignore warning
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# GPU setting
#
GPU_NUMBER=1

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
#exp_set_name = 'DNN-to-SNN'

# hyperparamter tune mode
#hp_tune = True
hp_tune = False


#
#train=True
#train=False
train=conf.train

#load_model=True
load_model=False

#
#save_model = False
save_model = True

#
#overwrite_train_model =True
overwrite_train_model=False

#
overwrite_tensorboard = True

#epoch = 20000
#epoch = 20472
train_epoch = 300
#train_epoch = 1000
#train_epoch = 1


# learning rate schedule - step_decay
step_decay_epoch = 100
#step_decay_epoch = 200


# TODO: move to config
#
root_hp_tune = './hp_tune'

#
#root_model = './models_trained'
root_model = './models_trained_test'

# model
#model_name = 'VGG16'
#model_name = 'ResNet18'
#model_name = 'ResNet20'
#model_name = 'ResNet32'
#model_name = 'ResNet34'
#model_name = 'ResNet50'
#model_name = 'ResNet18V2'
#model_name = 'ResNet20V2'
model_name = conf.model

# dataset
#dataset_name = 'CIFAR10'
#dataset_name = 'CIFAR100'
#dataset_name='ImageNet'
dataset_name = conf.dataset

#
#learning_rate = 0.2
#learning_rate = 0.01
learning_rate = conf.learning_rate

#
opt='SGD'

#
#lr_schedule = 'COS'     # COSine
#lr_schedule = 'COSR'    # COSine with Restart
lr_schedule = 'STEP'    # STEP wise
#lr_schedule = 'STEP_WUP'    # STEP wise, warmup


#
#root_tensorboard = './tensorboard/'
root_tensorboard = conf.root_tensorboard





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


# TODO: batch size calulation unification
#batch_size_inference = batch_size_inference_sel.get(model_name,256)
batch_size_train = conf.batch_size
if train:
    batch_size_inference = batch_size_train
else:
    if conf.full_test:
        batch_size_inference = conf.batch_size_inf
    else:
        if conf.batch_size_inf > conf.num_test_data:
            batch_size_inference = conf.num_test_data
        else:
            batch_size_inference = conf.batch_size_inf
    #batch_size_train = batch_size_train_sel.get(model_name,256)


#
if train:
    batch_size = batch_size_train
else:
    batch_size = batch_size_inference


#
image_shape = (input_size, input_size, 3)


# dataset load
#dataset = dataset_sel[dataset_name]
#train_ds, valid_ds, test_ds = dataset.load(dataset_name,input_size,input_size_pre_crop_ratio,num_class,train,NUM_PARALLEL_CALL,conf,input_prec_mode)
train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class =\
    datasets.datasets.load(dataset_name,batch_size,input_size,train_type,train,conf,NUM_PARALLEL_CALL)


# data-based weight normalization (DNN-to-SNN conversion)
if conf.f_write_stat and conf.f_stat_train_mode:
    test_ds = train_ds

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

################
# name set
################

#
if conf.load_best_model:
    root_model = conf.root_model_best


# TODO: configuration & file naming
#exp_set_name = model_name + '_' + dataset_name
model_dataset_name = model_name + '_' + dataset_name

# path_model = './'+exp_set_name
#path_model = os.path.join(root_model, exp_set_name)
path_model = os.path.join(root_model, model_dataset_name)


# hyperparameter tune name
#hp_tune_name = exp_set_name+'_'+model_dataset_name+'_ep-'+str(train_epoch)
hp_tune_name = exp_set_name

# TODO: functionalize
# file_name='checkpoint-epoch-{}-batch-{}.h5'.format(epoch,batch_size)
# config_name='ep-{epoch:04d}_bat-{}_lmb-{:.1E}'.format(batch_size,lmb)
# config_name='bat-{}_lmb-{:.1E}'.format(batch_size,lmb)

#config_name = 'bat-{}_opt-{}_lr-{:.0E}_lmb-{:.0E}'.format(batch_size,opt,learning_rate,lmb)
config_name = 'ep-{}_bat-{}_opt-{}_lr-{}-{:.0E}_lmb-{:.0E}'.format(train_epoch,batch_size_train,opt,lr_schedule,learning_rate,lmb)

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

#
if train:
    filepath = os.path.join(path_model, config_name)
else:
    if conf.load_best_model:
        filepath = path_model
    else:
        filepath = os.path.join(path_model, config_name)




########################################
#
########################################

model_top = model_sel(model_name,train_type)

if load_model:
    # get latest saved model
    #latest_model = lib_snn.util.get_latest_saved_model(filepath)

    latest_model = lib_snn.util.get_latest_saved_model(filepath)
    load_weight = os.path.join(filepath, latest_model)
    print('load weight: '+load_weight)
    #pre_model = tf.keras.models.load_model(load_weight)

    #latest_model = lib_snn.util.get_latest_saved_model(filepath)
    #load_weight = os.path.join(filepath, latest_model)


    if not latest_model.startswith('ep-'):
        assert False, 'the name of latest model should start with ''ep-'''
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



# TODO: move to parameter
# eager mode
if train:
    eager_mode=False
else:
    if conf.f_write_stat:
        eager_mode=True
        #eager_mode=False
    else:
        eager_mode=False

if conf.debug_mode:
    # TODO: parameterize - debug mode
    eager_mode=True


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
    hp_tune_args['batch_size'] = batch_size
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
        eager_mode, model_top, batch_size, image_shape, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
        train_epoch, train_steps_per_epoch,
        opt, learning_rate,
        lr_schedule, step_decay_epoch,
        metric_accuracy, metric_accuracy_top5)

#
if conf.nn_mode=='SNN' and conf.dnn_to_snn:
    print('DNN-to-SNN mode')
    nn_mode_ori = conf.nn_mode
    conf.nn_mode='ANN'
    model_ann = lib_snn.model_builder.model_builder(
        eager_mode, model_top, batch_size, image_shape, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
        train_epoch, train_steps_per_epoch,
        opt, learning_rate,
        lr_schedule, step_decay_epoch,
        metric_accuracy, metric_accuracy_top5)
    conf.nn_mode=nn_mode_ori

    #model_ann.set_en_snn('ANN')

    model_ann.load_weights(load_weight)

    print('-- model_ann - load done')
    model.load_weights_dnn_to_snn(model_ann)

    #del(model_ann)


elif load_model:
    model.load_weights(load_weight)
    #model.load_weights(load_weight, by_name=True, skip_mismatch=True)
    #model.load_weights_custom(load_weight)
    #model.load_weights(load_weight, by_name=True)
    # model.load_weights(load_weight,by_name=

if conf.nn_mode=='ANN':
    model_ann=None


#ann_kernel={}
#snn_kernel={}
#
#ann_bias={}
#snn_bias={}
#
#ann_bn={}
#snn_bn={}
#
#print('loaded kernel')
#for layer in model_ann.layers:
#    if hasattr(layer,'kernel'):
#        print('{} - {}'.format(layer.name,tf.reduce_sum(layer.kernel)))
#        ann_kernel[layer.name] = tf.reduce_sum(layer.kernel)
#
#
#print('loaded bias')
#for layer in model_ann.layers:
#    if hasattr(layer,'bias'):
#        print('{} - {}'.format(layer.name,tf.reduce_sum(layer.bias)))
#        ann_bias[layer.name] = tf.reduce_sum(layer.bias)
#
#print('loaded bn')
#for layer in model_ann.layers:
#    if hasattr(layer, 'bn') and layer.bn is not None:
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.beta)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.gamma)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.moving_mean)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.moving_variance)))
#        ann_bn[layer.name] = tf.reduce_sum(layer.bn.beta)
#
#
#print('loaded kernel')
#for layer in model.layers:
#    if hasattr(layer,'kernel'):
#        print('{} - {}'.format(layer.name,tf.reduce_sum(layer.kernel)))
#        snn_kernel[layer.name] = tf.reduce_sum(layer.kernel)
#
#
#print('loaded bias')
#for layer in model.layers:
#    if hasattr(layer,'bias'):
#        print('{} - {}'.format(layer.name,tf.reduce_sum(layer.bias)))
#        snn_bias[layer.name] = tf.reduce_sum(layer.bias)
#
#print('loaded bn')
#for layer in model.layers:
#    if hasattr(layer, 'bn') and layer.bn is not None:
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.beta)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.gamma)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.moving_mean)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.moving_variance)))
#        snn_bn[layer.name] = tf.reduce_sum(layer.bn.beta)
#
#
##for layer in model.layers:
#for layer_name in snn_kernel.keys():
#    assert snn_kernel[layer_name]==ann_kernel[layer_name]
#
#for layer_name in snn_bias.keys():
#    assert snn_bias[layer_name]==ann_bias[layer_name]
#
#for layer_name in snn_bn.keys():
#    assert snn_bn[layer_name]==ann_bn[layer_name]

#assert False


#
#model.make_test_function = lib_snn.training.make_test_function(model)

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



########
# Callbacks
########

#
if train and load_model:
    print('Evaluate pretrained model')
    assert monitor_cri == 'val_acc', 'currently only consider monitor criterion - val_acc'
    result = model.evaluate(valid_ds)
    idx_monitor_cri = model.metrics_names.index('acc')
    best = result[idx_monitor_cri]
    print('previous best result - {}'.format(best))
else:
    best = None

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

#cb_dnntosnn = lib_snn.callbacks.DNNtoSNN()
cb_libsnn = lib_snn.callbacks.SNNLIB(conf,path_model,test_ds_num,model_ann)
cb_libsnn_ann = lib_snn.callbacks.SNNLIB(conf,path_model,test_ds_num)

#
callbacks_train = [cb_tensorboard]
if save_model:
    callbacks_train.append(cb_model_checkpoint)
    callbacks_train.append(cb_manage_saved_model)

callbacks_test = []
# TODO: move to parameters
#dnn_to_snn = True
#if dnn_to_snn:
    #callbacks_test.append(cb_dnntosnn)

callbacks_test = [cb_libsnn]
callbacks_test_ann = [cb_libsnn_ann]

#
if train:
    if hp_tune:
        print('HP Tune mode')

        #callbacks = [cb_tensorboard]

        tuner.search(train_ds, epochs=train_epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                     callbacks=callbacks_train)
    else:
        print('Train mode')

        #callbacks = [
            #cb_model_checkpoint,
            #cb_manage_saved_model,
            #cb_tensorboard
        #]

        model.summary()

        train_histories = model.fit(train_ds, epochs=train_epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                                    callbacks=callbacks_train)
else:
    print('Test mode')

    #dnn_snn_compare=True
    #dnn_snn_compare=False

    #compare_control_snn = False
    compare_control_snn = True

    act_based_calibration = conf.calibration_bias_ICML_21 or conf.calibration_vmem_ICML_21 or conf.calibration_weight_post
    #if (conf.nn_mode=='SNN') and (dnn_snn_compare or conf.calibration_bias_ICML_21 or conf.calibration_vmem_ICML_21) :
    #if (conf.nn_mode == 'SNN') and (dnn_snn_compare or act_based_calibration):
    if (conf.nn_mode == 'SNN') and (compare_control_snn or act_based_calibration):

        cb_libsnn_ann.run_for_calibration = True

        #
        #compare_control_snn = True
        if (not conf.full_test) and compare_control_snn and conf.verbose_visual:
            #cb_libsnn_ann.run_for_compare_post_calib = True
            lib_snn.sim.set_for_visual_debug(True)
            model_ann.evaluate(test_ds, callbacks=callbacks_test_ann)
            #cb_libsnn_ann.run_for_compare_post_calib=False
            lib_snn.sim.set_for_visual_debug(False)


        #if conf.calibration_bias_ICML_21 or conf.calibration_vmem_ICML_21:
        if act_based_calibration:
            # TODO: random sampling
            #test_ds_one_batch = tf.data.experimental.get_single_element(test_ds)
            #test_ds_one_batch = tf.data.Dataset.from_tensors(test_ds_one_batch)
            images_one_batch, labels_one_batch = next(iter(train_ds))
            #images_one_batch, labels_one_batch = next(iter(test_ds))
            #print(tf.reduce_mean(images_one_batch))
        #else:
            #images_one_batch, labels_one_batch = next(iter(test_ds))

        ds_one_batch = tf.data.Dataset.from_tensors((images_one_batch,labels_one_batch))
        ds_ann = ds_one_batch


        #
        nn_mode_ori = conf.nn_mode
        #conf.nn_mode = 'ANN'
        result_ann = model_ann.evaluate(ds_ann, callbacks=callbacks_test_ann)
        conf.nn_mode = nn_mode_ori

    #
    # calibration with activations
    # calibration ICML-21
    #
    #if (conf.nn_mode == 'SNN') and (conf.calibration_bias_ICML_21 or conf.calibration_vmem_ICML_21):
    if (conf.nn_mode == 'SNN') and (act_based_calibration):
        # pre
        cb_libsnn.run_for_calibration = True
        glb_plot.mark='ro'
        glb_plot_1.mark='ro'
        glb_plot_2.mark='ro'

        #
        compare_control_snn=True
        if (not conf.full_test) and compare_control_snn and conf.verbose_visual:
            lib_snn.sim.set_for_visual_debug(True)
            model.evaluate(test_ds, callbacks=callbacks_test)
            lib_snn.sim.set_for_visual_debug(False)

        # run
        model.evaluate(ds_one_batch, callbacks=callbacks_test)

        # post
        cb_libsnn.run_for_calibration = False
        glb_plot.mark = 'bo'
        glb_plot_1.mark = 'bo'
        glb_plot_2.mark = 'bo'

    #
    # run
    #
    lib_snn.sim.set_for_visual_debug(True)
    result = model.evaluate(test_ds, callbacks=callbacks_test)
    lib_snn.sim.set_for_visual_debug(False)

    #result = model.evaluate(test_ds)
    # result = model.predict(test_ds)

    print(result)


    #
#    ## compare control model
#    compare_control_snn_model = True
#    if compare_control_snn_model:
#
#        cb_libsnn_ctrl = lib_snn.callbacks.SNNLIB(conf, path_model, test_ds_num)
#        cb_libsnn_ctrl.run_for_calibration = True
#
#
#
#        #if dnn_snn_compare:
#        model_ann.evaluate(test_ds)
#
#        lib_snn.proc.dnn_snn_compare_func(cb_libsnn)

    #
    #for layer in model.layers_w_neuron:
    #    print('{} - {}'.format(layer.name,tf.reduce_sum(layer.act.spike_count_int)))

    # ANN for comparison

    #print(result_ann)

#
if False:
    zeros_input = tf.zeros([1,32,32,3])
    zeros_output = tf.zeros([1,10])
    #zeros_output = tf.constant([0,0,0,0,0,0,0,0,1,0])
    result = model.evaluate(x=zeros_input,y=zeros_output,callbacks=callbacks_test)
    #result = model.evaluate(test_ds,callbacks=callbacks_test)
    print(result)

    #for layer in model.layers:
    #    if hasattr(layer,'record_output'):
    #        print('{} - {}'.format(layer.name,tf.reduce_mean(layer.record_output)))

# debug - compare activation
if False:
    result_ann = model_ann.evaluate(test_ds, callbacks=callbacks_test_ann)

    for layer in model.layers:
        if hasattr(layer,'record_output'):
            snn = layer.record_output
            ann = model_ann.get_layer(layer.name).record_output

            #assert snn == ann
            if snn is not None:
                if tf.reduce_mean(snn) != tf.reduce_mean(ann):
                    print(ann)
                    print(layer.name)
                    print(tf.reduce_mean(snn))
                    print(tf.reduce_mean(ann))
                    assert False



#if __name__=="__main__":
#    # logging.set_verbosity(logging.INfO)
#    config.configurations()
#    app.run(main)