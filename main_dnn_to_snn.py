

import re
import datetime
import shutil

#from functools import partial

#
import os

#
import tensorflow as tf

# TF logging setup
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import keras_tuner
#import tensorflow_probability as tfp
#from tensorflow.python.keras.engine import data_adapter

#
#from tensorflow.python.keras.utils import data_utils
#from keras.utils.vis_utils import plot_model


#
#from absl import app
#from absl import flags

# HP tune
#import kerastuner as kt
#import keras_tuner as kt
#import tensorboard.plugins.hparams import api as hp


#
import tensorflow_datasets as tfds

#import config_common
from config import config

tfds.disable_progress_bar()

#

#import matplotlib.pyplot as plt

#import numpy as np
##np.set_printoptions(precision=4)
##np.set_printoptions(linewidth=np.inf)
##import tensorflow.experimental.numpy as tnp
##tnp.set_printoptions(linewidth=np.inf)


# configuration
import config_snn_training
#from config import conf
#from absl import flags
#conf = flags.FLAGS
#from config import config as conf

# snn library
import lib_snn

#
import datasets
#global input_size
#global input_size_pre_crop_ratio
import collections




#
#from lib_snn.sim import glb_plot
#from lib_snn.sim import glb_plot_1
#from lib_snn.sim import glb_plot_2
#from lib_snn.sim import glb_plot_3

from lib_snn.sim import glb_ig_attributions
from lib_snn.sim import glb_rand_vth
from lib_snn.sim import glb_vth_search_err
from lib_snn.sim import glb_vth_init
from lib_snn.sim import glb_bias_comp


# ImageNet utils
from models import imagenet_utils


########################################
# configuration
########################################

if False:
    # logging - ignore warning
    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # distribute strategy
    devices = tf.config.list_physical_devices('GPU')
    if len(devices)==1:
        dist_strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
    else:
        devices = ['/gpu:{}'.format(i) for i in range(len(devices))]
        dist_strategy = tf.distribute.MirroredStrategy(devices=devices)
    #dist_strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])
    #dist_strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')


    #
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

dist_strategy = lib_snn.utils.set_gpu()

# CPU
#NUM_PARALLEL_CALL=16

# exp set name
#exp_set_name = conf.exp_set_name
exp_set_name = config.flags.exp_set_name

#
#train= (conf.mode=='train') or (conf.mode=='load_and_train')
train = config.train


# TODO: parameterize
#load_model=True
#load_model=False
#load_model = (conf.mode=='inference') or (conf.mode=='load_and_train')
load_model = config.load_model

#
#save_model = False
save_model = True

#
#overwrite_train_model =True
overwrite_train_model=False

#
overwrite_tensorboard = True

#
tf.config.experimental.enable_tensor_float_32_execution(config.flags.tf32_mode)

#train_epoch = conf.train_epoch
train_epoch = config.flags.train_epoch

# learning rate schedule - step_decay
#step_decay_epoch = conf.step_decay_epoch



# TODO: move to config
#
root_hp_tune = './hp_tune'

#
# model
#model_name = conf.model
#model_name = config.model_name

# dataset
#dataset_name = conf.dataset
dataset_name = config.dataset_name

#
#learning_rate = 0.2
#learning_rate = 0.01
#learning_rate = conf.learning_rate

#
#opt = conf.optimizer
#opt='SGD'
#opt='ADAM'

#
# lr schedule
#lr_schedule = 'COS'     # COSine
#lr_schedule = 'COSR'    # COSine with Restart
#lr_schedule = 'STEP'    # STEP wise
#lr_schedule = 'STEP_WUP'    # STEP wise, warmup
#lr_schedule = conf.lr_schedule

#
#root_tensorboard = './tensorboard/'
root_tensorboard = config.flags.root_tensorboard


# models
##from models.vgg16 import VGG16
#from models.vgg16_keras_toh5 import VGG16 as VGG16_KERAS
#from models.models import model_sel


# training types
#train_type='finetuning' # not supported yet
#train_type='transfer'
#train_type='scratch'
train_type = config.flags.train_type


#
assert config.flags.data_format == 'channels_last', \
    'not support "{}", only support channels_last'.format(config.flags.data_format)

########################################
# DO NOT TOUCH
########################################

#
#f_hp_tune_train = train and conf.hp_tune
#f_hp_tune_load = (not train) and conf.hp_tune
f_hp_tune_train = config.f_hp_tune_train
f_hp_tune_load = config.f_hp_tune_load

if False:
    # TODO: batch size calculation unification
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

    if not conf.full_test:
        assert (conf.num_test_data%batch_size_inference)==0

    #
    if train:
        batch_size = batch_size_train
    else:
        batch_size = batch_size_inference

batch_size = config.batch_size
batch_size_train = config.batch_size_train
batch_size_inference = config.batch_size_inference

################
# name set
################
#
filepath_save, filepath_load, config_name = lib_snn.utils.set_file_path()


########################################
# load dataset
########################################
train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch = \
    datasets.datasets.load()



#
with dist_strategy.scope():
    model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch)

    ########################################
    # load model
    ########################################
    model_ann=None
    if config.flags.nn_mode=='SNN' and config.flags.dnn_to_snn:
        print('DNN-to-SNN mode')
        nn_mode_ori = config.flags.nn_mode
        config.flags.nn_mode='ANN'
        model_ann = lib_snn.model_builder.model_builder(
            eager_mode, model_top, batch_size, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
            train_epoch, train_steps_per_epoch,
            opt, learning_rate,
            lr_schedule, step_decay_epoch,
            metric_accuracy, metric_accuracy_top5, dataset_name)
        conf.nn_mode=nn_mode_ori

        model_ann.nn_mode = 'ANN'

        #model_ann.set_en_snn('ANN')

        if dataset_name=='ImageNet':
            #imagenet_utils.load_weights(model_name,model_ann)
            model_ann.load_weights(load_weight)
        else:
            model_ann.load_weights(load_weight)
            #model_ann.load_weights(load_weight,by_name=True)

        print('-- model_ann - load done')
        model.load_weights_dnn_to_snn(model_ann)
        #model.load_weights_dnn_to_snn(model_ann,by_name=True)


    elif load_model:
        model.load_weights(load_weight)

    #
    #if train:
        #print('Train mode')
    # remove dir - train model
    if not load_model:
        if overwrite_train_model:
            if os.path.isdir(filepath_save):
                shutil.rmtree(filepath_save)

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
    monitor_cri = config.monitor_cri
    if train and load_model and (not f_hp_tune_train):
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
        filepath=filepath_save + '/ep-{epoch:04d}.hdf5',
        save_weight_only=True,
        save_best_only=True,
        monitor=monitor_cri,
        verbose=1,
        best=best,
        log_dir=path_tensorboard,
        # tensorboard_writer=cb_tensorboard._writers['train']
    )
    cb_manage_saved_model = lib_snn.callbacks.ManageSavedModels(filepath=filepath_save)
    cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=path_tensorboard, update_freq='epoch')

    cb_libsnn = lib_snn.callbacks.SNNLIB(config.flags,filepath_load,test_ds_num,model_ann)
    cb_libsnn_ann = lib_snn.callbacks.SNNLIB(config.flags,filepath_load,test_ds_num)

    #
    callbacks_train = []
    if save_model:
        callbacks_train.append(cb_model_checkpoint)
        callbacks_train.append(cb_manage_saved_model)
    callbacks_train.append(cb_libsnn)
    callbacks_train.append(cb_tensorboard)

    callbacks_test = []
    # TODO: move to parameters

    callbacks_test = [cb_libsnn]
    callbacks_test_ann = [cb_libsnn_ann]

    init_epoch = config.init_epoch

    #
    if train:
        print('Train mode')

        model.summary()
        train_steps_per_epoch = train_ds_num/batch_size
        train_histories = model.fit(train_ds, epochs=train_epoch, steps_per_epoch=train_steps_per_epoch,
                                    initial_epoch=init_epoch, validation_data=valid_ds, callbacks=callbacks_train)
    else:
        print('Test mode')

        result = model.evaluate(ds_one_batch, callbacks=callbacks_test)

