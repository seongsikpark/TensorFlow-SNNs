

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
from absl import flags
conf = flags.FLAGS

# snn library
import lib_snn

#
import datasets
#global input_size
#global input_size_pre_crop_ratio
import collections


# TODO: check use
#global model_name


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
NUM_PARALLEL_CALL=16

# exp set name
exp_set_name = conf.exp_set_name

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
tf.config.experimental.enable_tensor_float_32_execution(conf.tf32_mode)

#train_epoch = conf.train_epoch
train_epoch = config.train_epoch

# learning rate schedule - step_decay
#step_decay_epoch = conf.step_decay_epoch



# TODO: move to config
#
root_hp_tune = './hp_tune'

#
# model
#model_name = conf.model
model_name = config.model_name

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
root_tensorboard = conf.root_tensorboard


# models
##from models.vgg16 import VGG16
#from models.vgg16_keras_toh5 import VGG16 as VGG16_KERAS
#from models.models import model_sel


# training types
#train_type='finetuning' # not supported yet
#train_type='transfer'
#train_type='scratch'
train_type = conf.train_type


#
assert conf.data_format == 'channels_last', 'not support "{}", only support channels_last'.format(conf.data_format)

########################################
# DO NOT TOUCH
########################################

#
#f_hp_tune_train = train and conf.hp_tune
#f_hp_tune_load = (not train) and conf.hp_tune
f_hp_tune_train = config.f_hp_tune_train
f_hp_tune_load = config.f_hp_tune_load

# l2-norm
#lmb = 1.0E-10
#lmb = conf.lmb





#
#initial_channels_sel= {
    #'VGG16': 64,
#}
#initial_channels = initial_channels_sel.get(model_name,64)


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

# model compile
if False:
    metric_accuracy = tf.keras.metrics.categorical_accuracy
    metric_accuracy_top5 = tf.keras.metrics.top_k_categorical_accuracy

    # TODO: move to configuration
    metric_name_acc = 'acc'
    metric_name_acc_top5 = 'acc-5'
    monitor_cri = 'val_' + metric_name_acc

    metric_accuracy.name = metric_name_acc
    metric_accuracy_top5.name = metric_name_acc_top5

################
# name set
################

# TODO: configuration & file naming
model_dataset_name = model_name + '_' + dataset_name

# hyperparameter tune name
hp_tune_name = exp_set_name

#filepath_save, filepath_load, config_name = lib_snn.util.set_file_path(batch_size)
filepath_save, filepath_load, config_name = lib_snn.utils.set_file_path(batch_size_train)

##
use_bn_dict = collections.OrderedDict()
use_bn_dict['VGG16_ImageNet'] = False

#
try:
    conf.use_bn = use_bn_dict[model_dataset_name]
except KeyError:
    pass

########################################
# load dataset
########################################
# dataset load
#train_ds, valid_ds, test_ds = dataset.load(dataset_name,input_size,input_size_pre_crop_ratio,num_class,train,NUM_PARALLEL_CALL,conf,input_prec_mode)
#train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class = datasets.datasets.load(model_name, dataset_name,batch_size,input_size,train_type,train,conf,NUM_PARALLEL_CALL)
train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch = \
    datasets.datasets.load(model_name, dataset_name,batch_size,train_type,train,conf,NUM_PARALLEL_CALL)


# data-based weight normalization (DNN-to-SNN conversion)
if conf.f_write_stat and conf.f_stat_train_mode:
    test_ds = train_ds

#assert False
#train_steps_per_epoch = train_ds.cardinality().numpy()

########################################
# load model
########################################

if False:
    # TODO: integration - ImageNet
    if load_model and (not f_hp_tune_load):
        #if conf.dataset == 'ImageNet':
        if False:
            # ImageNet pretrained model
            load_weight = 'imagenet'
            include_top = True
            add_top = False
        else:
            # get latest saved model
            #latest_model = lib_snn.util.get_latest_saved_model(filepath)

            latest_model = lib_snn.util.get_latest_saved_model(filepath_load)
            load_weight = os.path.join(filepath_load, latest_model)
            #print('load weight: '+load_weight)
            #pre_model = tf.keras.models.load_model(load_weight)

            #latest_model = lib_snn.util.get_latest_saved_model(filepath)
            #load_weight = os.path.join(filepath, latest_model)


            if not latest_model.startswith('ep-'):
                #assert False, 'the name of latest model should start with ''ep-'''
                print('the name of latest model should start with ep-')

                load_weight = tf.train.latest_checkpoint(filepath_load)

                # TODO:
                init_epoch = 0
            else:
                print('load weight: '+load_weight)

                if conf.mode=='inference':
                    init_epoch = int(re.split('-|\.',latest_model)[1])
                elif conf.mode=='load_and_train':
                    init_epoch = 0
                else:
                    assert False


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


if False:
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






with dist_strategy.scope():

    #model_top = model_sel(model_name,train_type)
    # for HP tune
    #model_top_glb = model_top

    #
    # model builder
    if f_hp_tune_train or f_hp_tune_load:

        # TODO: move to config.py
        #hp_model_builder = model_builder
        hps = collections.OrderedDict()
        hps['dataset'] = [dataset_name]
        hps['model'] = [model_name]
        #hps['opt'] = [opt]
        #hps['lr_schedule'] = [lr_schedule]
        #hps['train_epoch'] = [train_epoch]
        #hps['step_decay_epoch'] = [step_decay_epoch]

        # main to hp_tune, need to seperate configuration
        hp_tune_args = collections.OrderedDict()
        #hp_tune_args['model_top'] = model_top
        hp_tune_args['batch_size'] = batch_size
        #hp_tune_args['image_shape'] = image_shape
        hp_tune_args['conf'] = conf
        hp_tune_args['include_top'] = include_top
        hp_tune_args['load_weight'] = load_weight
        hp_tune_args['num_class'] = num_class
        #hp_tune_args['metric_accuracy'] = metric_accuracy
        #hp_tune_args['metric_accuracy_top_5'] = metric_accuracy_top5
        hp_tune_args['train_steps_per_epoch'] = train_steps_per_epoch
        hp_tune_args['dist_strategy'] = dist_strategy

        #hp_model_builder = partial(model_builder, hp, hps)
        hp_model_builder = lib_snn.hp_tune_model.CustomHyperModel(hp_tune_args, hps)

        search_func = lib_snn.hp_tune.GridSearch
        #search_func = keras_tuner.RandomSearch
        search_max_trials = 20

        #tuner = kt.Hyperband(model_builder,
        #tuner=kt.RandomSearch(model_builder,
        #tuner=lib_snn.hp_tune.GridSearch(hp_model_builder,
        tuner=search_func(hp_model_builder,
                             objective='val_acc',
                             max_trials=search_max_trials,
                             #max_epochs = 300,
                             #factor=3,
                             #overwrite=True,
                             directory=root_hp_tune,
                             project_name=hp_tune_name,
                             #directory='test_hp_dir',
                             #project_name='test_hp')
                              )

        #tuner.results_summary()
        #assert False
    else:
        if False: # old
            model = lib_snn.model_builder.model_builder(
                eager_mode, model_top, batch_size, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
                train_epoch, train_steps_per_epoch,
                opt, learning_rate,
                lr_schedule, step_decay_epoch,
                metric_accuracy, metric_accuracy_top5, dataset_name, dist_strategy)
        else:
            model = lib_snn.model_builder.model_builder(
                num_class,
                train_steps_per_epoch)


        #
        # test
        #train_ds = dist_strategy.experimental_distribute_dataset(train_ds)
        #print('fit start')
        #model.fit(train_ds,epochs=1,steps_per_epoch=100)





    ########################################
    # load model
    ########################################
    model_ann=None
    if conf.nn_mode=='SNN' and conf.dnn_to_snn:
        print('DNN-to-SNN mode')
        nn_mode_ori = conf.nn_mode
        conf.nn_mode='ANN'
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
        if f_hp_tune_load:
            tuner.reload()
            best_model = tuner.get_best_models()[0]
            print(tuner)
            print(tuner.directory)
            #print(tuner.load_model(0))
            print(tuner.get_best_models()[0])
            print(tuner.get_best_hyperparameters(num_trials=1)[0].values)
            print(tuner.results_summary(num_trials=2))

            print('best trial')
            best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
            print(best_trial.trial_id)

            print('best model evaluate')
            best_model.evaluate(test_ds)

            print('test model evaluate')
            #test_model = tuner.load_model(tuner.oracle.get_best_trials(num_trials=2)[1])
            test_model = tuner.get_best_models(num_models=2)[1]
            test_model.evaluate(test_ds)

            assert False

        elif not f_hp_tune_train:
            if dataset_name == 'ImageNet':
                #imagenet_utils.load_weights(model_name, model)
                model.load_weights(load_weight)
            else:
                model.load_weights(load_weight)
            #model.load_weights(load_weight,by_name=True,skip_mismatch=True)
            #model.load_weights_custom(load_weight)
            #model.load_weights(load_weight, by_name=True)

    #if conf.nn_mode=='ANN' or (conf.nn_mode=='SNN' and train):
    #    model_ann=None


    #
    #if train:
        #print('Train mode')
    # remove dir - train model
    if not load_model:
        if overwrite_train_model:
            if os.path.isdir(filepath_save):
                shutil.rmtree(filepath_save)

    # path_tensorboard = root_tensorboard+exp_set_name
    # path_tensorboard = root_tensorboard+filepath

    if f_hp_tune_train:
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

    #cb_dnntosnn = lib_snn.callbacks.DNNtoSNN()
    #cb_libsnn = lib_snn.callbacks.SNNLIB(conf,path_model_load,test_ds_num,model_ann)
    #cb_libsnn_ann = lib_snn.callbacks.SNNLIB(conf,path_model_load,test_ds_num)
    cb_libsnn = lib_snn.callbacks.SNNLIB(conf,filepath_load,test_ds_num,model_ann)
    cb_libsnn_ann = lib_snn.callbacks.SNNLIB(conf,filepath_load,test_ds_num)

    #
    #callbacks_train = [cb_tensorboard]
    callbacks_train = []
    if save_model:
        callbacks_train.append(cb_model_checkpoint)
        callbacks_train.append(cb_manage_saved_model)
    callbacks_train.append(cb_libsnn)
    callbacks_train.append(cb_tensorboard)

    callbacks_test = []
    # TODO: move to parameters
    #dnn_to_snn = True
    #if dnn_to_snn:
        #callbacks_test.append(cb_dnntosnn)

    callbacks_test = [cb_libsnn]
    callbacks_test_ann = [cb_libsnn_ann]

    init_epoch = config.init_epoch

    #
    if train:
        if conf.hp_tune:
            print('HP Tune mode')

            tuner.search(train_ds, epochs=train_epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                         callbacks=callbacks_train)
        else:
            print('Train mode')

            model.summary()
            train_steps_per_epoch = train_ds_num/batch_size
            train_histories = model.fit(train_ds, epochs=train_epoch, steps_per_epoch=train_steps_per_epoch,
                                        initial_epoch=init_epoch, validation_data=valid_ds, callbacks=callbacks_train)
            #train_histories = model.fit(train_ds, epochs=train_epoch, initial_epoch=init_epoch, validation_data=valid_ds, callbacks=callbacks_train)
    else:
        print('Test mode')

        #
        #dnn_snn_compare=True
        #dnn_snn_compare=False

        result = model.evaluate(ds_one_batch, callbacks=callbacks_test)

