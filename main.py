from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import sys
sys.path.insert(0,'./')
#sys.path.insert(0,'/home/sspark/Projects/05_SNN/')

import cProfile
import pprofile

from datetime import datetime


#en_gpu=False
en_gpu=True

gpu_number=0
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)

#
# 0: all messages
# 1: INFO not printed
# 2: INFO, WARNING not printed
# 3: INFO, WARNING, ERROR not printed
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

if en_gpu==False:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf

#
import tensorflow_probability as tfp
tfd = tfp.distributions

#
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS=False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#import tensorflow.contrib.eager as tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import ops

#from tensorflow.python.platform import gfile
#from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

builder = option_builder.ProfileOptionBuilder

#
import train
import test

import train_snn

# models
from models import models
#from models import mlp
#from models import cnn
import model_cnn_mnist
#import model_cnn_mnist_ori as model_cnn_mnist
# TODO: convert TF V2
#import resnet


# dataset
from datasets import mnist
from datasets import cifar10
from datasets import cifar100
from datasets import imagenet  # from tensornets

# imagenet
from data import ImageNetDataset


# cifar-100

import shutil

#
import pprint
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d


from tqdm import tqdm

import math

import csv
import collections
import re

import gc

now = datetime.now()

#

#
np.set_printoptions(threshold=np.inf, linewidth=1000, precision=4)

#
pp = pprint.PrettyPrinter().pprint

#
#flags = tf.compat.v1.app.flags
flags = tf.compat.v1.app.flags
tf.compat.v1.app.flags.DEFINE_string('date','','date')

tf.compat.v1.app.flags.DEFINE_integer('epoch', 300, 'Number os epochs')
tf.compat.v1.app.flags.DEFINE_string('gpu_fraction', '1/3', 'define the gpu fraction used')
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 100, '')
tf.compat.v1.app.flags.DEFINE_string('activation', 'ReLU', '')
tf.compat.v1.app.flags.DEFINE_string('optim_type', 'adam', '[exp_decay, adam]')

tf.compat.v1.app.flags.DEFINE_string('nn_mode', 'SNN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')
#tf.compat.v1.app.flags.DEFINE_string('nn_mode', 'ANN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')

tf.compat.v1.app.flags.DEFINE_string('output_dir', './tensorboard', 'Directory to write TensorBoard summaries')
tf.compat.v1.app.flags.DEFINE_string('checkpoint_dir', './models_ckpt', 'Directory to save checkpoints')
tf.compat.v1.app.flags.DEFINE_string('checkpoint_load_dir', './models_ckpt', 'Directory to load checkpoints')
tf.compat.v1.app.flags.DEFINE_bool('en_load_model', False, 'Enable to load model')

#
tf.compat.v1.app.flags.DEFINE_boolean('en_train', False, 'enable training')

#
tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.7, 'initial value of vth')
tf.compat.v1.app.flags.DEFINE_float('n_in_init_vth', 0.7, 'initial value of vth of n_in')
tf.compat.v1.app.flags.DEFINE_float('n_init_vinit', 0.0, 'initial value of vinit')
tf.compat.v1.app.flags.DEFINE_float('n_init_vrest', 0.0, 'initial value of vrest')

# exponetial decay
'''
tf.compat.v1.app.flags.DEFINE_float('init_lr', 0.1, '')
tf.compat.v1.app.flags.DEFINE_float('decay_factor', 0.1, '')
tf.compat.v1.app.flags.DEFINE_integer('num_epoch_per_decay', 350, '')
'''
# adam optimizer
#tf.compat.v1.app.flags.DEFINE_float('init_lr', 1e-5, '')
# SGD
tf.compat.v1.app.flags.DEFINE_float('lr', 0.1, 'learning rate')
tf.compat.v1.app.flags.DEFINE_float('momentum', 0.9, 'momentum')
# ADAM
#tf.compat.v1.app.flags.DEFINE_float('lr', 0.0001, '')

# l2 norm
tf.compat.v1.app.flags.DEFINE_float('lamb',0.0001, 'lambda')

tf.compat.v1.app.flags.DEFINE_float('lr_decay', 0.1, '')
tf.compat.v1.app.flags.DEFINE_integer('lr_decay_step', 50, '')

tf.compat.v1.app.flags.DEFINE_integer('time_step', 10, 'time steps per sample in SNN')


tf.compat.v1.app.flags.DEFINE_integer('idx_test_dataset_s', 0, 'start index of test dataset')
tf.compat.v1.app.flags.DEFINE_integer('num_test_dataset', 10000, 'number of test datset')
tf.compat.v1.app.flags.DEFINE_integer('size_test_batch', 1, 'size of test batch') # not used now

tf.compat.v1.app.flags.DEFINE_string('pooling', 'max', 'max or avg, only for CNN')

tf.compat.v1.app.flags.DEFINE_integer('save_interval', 10, 'save interval of model')

tf.compat.v1.app.flags.DEFINE_bool('en_remove_output_dir', False, 'enable removing output dir')


#
tf.compat.v1.app.flags.DEFINE_boolean('use_bias', True, 'use bias')
tf.compat.v1.app.flags.DEFINE_string('regularizer', 'L2', 'L2 or L1 regularizer')


tf.compat.v1.app.flags.DEFINE_string('model_name', 'snn_train_mlp_mnist', 'model name')

tf.compat.v1.app.flags.DEFINE_string('n_type', 'LIF', 'LIF or IF: neuron type')

#
tf.compat.v1.app.flags.DEFINE_string('dataset', 'MNIST', 'dataset')
tf.compat.v1.app.flags.DEFINE_string('ann_model', 'MLP', 'neural network model')

#
tf.compat.v1.app.flags.DEFINE_boolean('verbose',True, 'verbose mode')
tf.compat.v1.app.flags.DEFINE_boolean('verbose_visual',True, 'verbose visual mode')

#
tf.compat.v1.app.flags.DEFINE_integer('time_step_save_interval',10,'snn test save interval')

#
tf.compat.v1.app.flags.DEFINE_bool('f_fused_bn',False,'f_fused_bn')

#
tf.compat.v1.app.flags.DEFINE_bool('f_stat_train_mode',False,'f_stat_train_mode')
tf.compat.v1.app.flags.DEFINE_bool('f_real_value_input_snn',False,'f_real_value_input_snn')
tf.compat.v1.app.flags.DEFINE_bool('f_vth_conp',False,'f_vth_conp')
tf.compat.v1.app.flags.DEFINE_bool('f_spike_max_pool',False,'f_spike_max_pool')
tf.compat.v1.app.flags.DEFINE_bool('f_w_norm_data',False,'f_w_norm_data')
tf.compat.v1.app.flags.DEFINE_bool('f_ws',False,'wieghted synapse')
tf.compat.v1.app.flags.DEFINE_integer('p_ws',8,'period of wieghted synapse')

tf.compat.v1.app.flags.DEFINE_integer('num_class',10,'number_of_class (do not touch)')

tf.compat.v1.app.flags.DEFINE_string('input_spike_mode','POISSON','input spike mode - POISSON, WEIGHTED_SPIKE, ROPOSED')
tf.compat.v1.app.flags.DEFINE_string('neural_coding','RATE','neural coding - RATE, WEIGHTED_SPIKE, PROPOSED')

tf.compat.v1.app.flags.DEFINE_bool('f_positive_vmem',False,'positive vmem')
tf.compat.v1.app.flags.DEFINE_bool('f_tot_psp',False,'accumulate total psp')

tf.compat.v1.app.flags.DEFINE_bool('f_isi',False,'isi stat')
tf.compat.v1.app.flags.DEFINE_bool('f_refractory',False,'refractory mode')

tf.compat.v1.app.flags.DEFINE_bool('f_comp_act',False,'compare activation')
tf.compat.v1.app.flags.DEFINE_bool('f_entropy',False,'entropy test')
tf.compat.v1.app.flags.DEFINE_bool('f_write_stat',False,'write stat')
tf.compat.v1.app.flags.DEFINE_string('act_save_mode','channel','activation save mode')
tf.compat.v1.app.flags.DEFINE_bool('f_save_result',True,'save result to xlsx file')

# data.py - imagenet data
tf.compat.v1.app.flags.DEFINE_string('data_path_imagenet', './imagenet', 'data path imagenet')
tf.compat.v1.app.flags.DEFINE_integer('k_pathces', 5, 'patches for test (random crop)')
tf.compat.v1.app.flags.DEFINE_integer('input_size', 28, 'input image width / height')


#
tf.compat.v1.app.flags.DEFINE_string('path_stat','./stat/', 'path stat')
tf.compat.v1.app.flags.DEFINE_string('prefix_stat','act_n_train', 'prefix of stat file name')


#
tf.compat.v1.app.flags.DEFINE_bool('f_data_std', True, 'data_standardization')


# pruning
tf.compat.v1.app.flags.DEFINE_bool('f_pruning_channel', False, 'purning - channel')


tf.compat.v1.app.flags.DEFINE_string('path_result_root','./result/', 'path result root')

# temporal coding
#tf.compat.v1.app.flags.DEFINE_float('tc',10.0,'time constant for temporal coding')
#tf.compat.v1.app.flags.DEFINE_float('time_window',20.0,'time window of each layer for temporal coding')
#tf.compat.v1.app.flags.DEFINE_float('time_fire_start',20.0,'time fire start (integration time before starting fire) for temporal coding')
#tf.compat.v1.app.flags.DEFINE_float('time_fire_duration',20.0,'time fire duration for temporal coding')
tf.compat.v1.app.flags.DEFINE_integer('tc',10,'time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_integer('time_window',20,'time window of each layer for temporal coding')
#tf.compat.v1.app.flags.DEFINE_integer('time_fire_start',20,'time fire start (integration time before starting fire) for temporal coding')
#tf.compat.v1.app.flags.DEFINE_integer('time_fire_duration',20,'time fire duration for temporal coding')
tf.compat.v1.app.flags.DEFINE_float('time_fire_start',20,'time fire start (integration time before starting fire) for temporal coding')
tf.compat.v1.app.flags.DEFINE_float('time_fire_duration',20,'time fire duration for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_record_first_spike_time',False,'flag - recording first spike time of each neuron')
tf.compat.v1.app.flags.DEFINE_bool('f_visual_record_first_spike_time',False,'flag - visual recording first spike time of each neuron')
tf.compat.v1.app.flags.DEFINE_bool('f_train_time_const',False,'flag - enable to train time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_train_time_const_outlier',True,'flag - enable to outlier roubst train time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_load_time_const',False,'flag - load time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_string('time_const_init_file_name','./temporal_coding/time_const','temporal coding file name - time_const, time_delay`')
tf.compat.v1.app.flags.DEFINE_integer('time_const_num_trained_data',0,'number of trained data - time constant')
tf.compat.v1.app.flags.DEFINE_integer('time_const_save_interval',10000,'save interval - training time constant')
tf.compat.v1.app.flags.DEFINE_integer('epoch_train_time_const',1,'epoch - training time constant')

tf.compat.v1.app.flags.DEFINE_bool('f_tc_based',False,'flag - tau based')
tf.compat.v1.app.flags.DEFINE_integer('n_tau_fire_start',4,'n tau - fire start')
tf.compat.v1.app.flags.DEFINE_integer('n_tau_fire_duration',4,'n tau - fire duration')
tf.compat.v1.app.flags.DEFINE_integer('n_tau_time_window',4,'n tau - time window')


#
tf.compat.v1.app.flags.DEFINE_enum('snn_output_type',"VMEM", ["SPIKE", "VMEM", "FIRST_SPIKE_TIME"], "snn output type")

# SNN trianing w/ TTFS coding
tf.compat.v1.app.flags.DEFINE_integer("init_first_spike_time_n",-1,"init_first_spike_time = init_first_spike_n x time_windw")

# surrogate training model
tf.compat.v1.app.flags.DEFINE_bool("f_surrogate_training_model", True, "flag - surrogate training model (DNN)")

#
tf.compat.v1.app.flags.DEFINE_bool("f_overwrite_train_model", False, "overwrite trained model")

#
tf.compat.v1.app.flags.DEFINE_bool("f_validation_snn", False, "validation on SNN")



#
conf = flags.FLAGS

data_path_imagenet='/home/sspark/Datasets/ILSVRC2012'
conf.data_path_imagenet = data_path_imagenet


#
conf.time_fire_start = 1.5


if conf.model_name == 'vgg_cifar_ro_0':
    conf.f_data_std = False

def main(_):
    #pr = cProfile.Profile()
    #pr=pprofile.Profile()
    #pr.enable()

    #print(device_lib.list_local_devices())
    print('main start')

    # remove output dir
    if conf.en_remove_output_dir:
        print('remove output dir: %s' % conf.output_dir)
        shutil.rmtree(conf.output_dir,ignore_errors=True)

    #tfe.enable_eager_execution()
    #tf.compat.v1.enable_eager_execution()
    #tf.compat.v1.disable_eager_execution()


    #tfe.enable_eager_execution(tfe.DEVICE_PLACEMENT_SILENT)
    #tf.set_random_seed(1)

    # profiler
    #profiler = Profiler()


    # print flags
    for k, v in flags.FLAGS.flag_values_dict().items():
        print(k+': '+str(v))

    #(device, data_format) = ('/gpu:0', 'channels_first')
    #if tfe.num_gpus() <= 0:
    #    (device, data_format) = ('/cpu:0', 'channels_last')
    data_format = 'channels_last'

    if en_gpu==True:
        (device, data_format) = ('/gpu:0', data_format)
    else:
        (device, data_format) = ('/cpu:0', data_format)

    print ('Using device %s, and data format %s.' %(device, data_format))

    #with tf.device('/cpu:0'):
    with tf.device('/gpu:0'):
        # load dataset
#        if conf.dataset=='MNIST':
#            (train_dataset, val_dataset, test_dataset) = mnist.load_data(conf.batch_size, conf.num_test_dataset)
#        elif conf.dataset=='CIFAR-10':
#            (train_dataset, val_dataset, test_dataset) = cifar10.load_data(conf.batch_size, conf.num_test_dataset, conf.f_stat_train_mode)
#        elif conf.dataset=='CIFAR-100':
#            (train_dataset, val_dataset, test_dataset) = cifar100.load_data(conf.batch_size, conf.num_test_dataset, conf.f_stat_train_mode)
#        elif conf.dataset=='ImageNet':
#            print('only test dataset supported yet')
#            (train_dataset, val_dataset, test_dataset) = imagenet.load(conf)

        dataset_type= {
            'MNIST': mnist,
            'CIFAR-10': cifar10,
            'CIFAR-100': cifar100,
            'ImageNet': imagenet
        }
        dataset = dataset_type[conf.dataset]

        (train_dataset, val_dataset, test_dataset, num_train_dataset, num_val_dataset, num_test_dataset) = dataset.load(conf)


    #print('> Dataset info.')
    #print(train_dataset)
    #print(test_dataset)



    if conf.ann_model=='MLP':
        if conf.dataset=='MNIST':
            #model = models.MNISTModel_MLP(data_format,conf)

            #model = mlp.mlp_mnist(data_format,conf)
            print("not implemented mode: MLP-MNIST")
            assert(False)

            #if conf.nn_mode=='ANN':
            #    train_func = train.train_ann_one_epoch_mnist
            #else:
            #    train_func = train.train_snn_one_epoch_mnist
    elif conf.ann_model=='CNN':
        if conf.dataset=='MNIST':
            #model = models.MNISTModel_CNN(data_format,conf)
            model = model_cnn_mnist.MNISTModel_CNN(data_format,conf)
            #model = cnn.cnn_mnist(data_format,conf)


            #if conf.nn_mode=='ANN':
            #    #train_func = train.train_ann_one_epoch_mnist_cnn
            #    train_func = train.train_ann_one_epoch_mnist_cnn
            #else:
            #    train_func = train.train_snn_one_epoch_mnist_cnn

        elif conf.dataset=='CIFAR-10':
            model = models.CIFARModel_CNN(data_format,conf)
        elif conf.dataset=='CIFAR-100':
            model = models.CIFARModel_CNN(data_format,conf)
    elif conf.ann_model=='VGG16':
        if conf.dataset=='CIFAR-10':
            model = models.CIFARModel_CNN(data_format,conf)
        elif conf.dataset=='CIFAR-100':
            model = models.CIFARModel_CNN(data_format,conf)
    elif conf.ann_model=='ResNet18':
        if conf.dataset=='CIFAR-10' or conf.dataset=='CIFAR-100':
            #model = resnet.Resnet18(data_format,conf)
            print("not converted to TF V2 yet - resnet")
            assert(False)
    elif conf.ann_model=='ResNet50':
        if conf.dataset=='ImageNet':
            #model = resnet.Resnet50(data_format,conf)
            print("not converted to TF V2 yet - resnet")
            assert(False)
    else:
        print('not supported model name: '+conf.ann_model)
        os._exit(0)


    #
    save_target_acc_sel = {
        'MNIST': 90.0,
        #'MNIST': 99.0,
        'CIFAR-10': 92.0,
        'CIFAR-100': 68.0
    }
    save_target_acc = save_target_acc_sel[conf.dataset]


    #
    val_snn_target_acc_sel = {
        'MNIST': 99.4,
        #'MNIST': 99.99,
        'CIFAR-10': 92.0,
        'CIFAR-100': 68.0
    }
    val_snn_target_acc = val_snn_target_acc_sel[conf.dataset]





    if(conf.nn_mode=="ANN"):
        if(conf.f_surrogate_training_model):
            train_func = train_snn.train_one_epoch(conf.neural_coding)
        else:
            train_func = train.train_one_epoch(conf.nn_mode)
    elif(conf.nn_mode=="SNN"):
        train_func = train_snn.train_one_epoch(conf.neural_coding)
    else:
        print("error in nn_mode: %s"%(conf.nn_mode))
        assert(False)



    en_train_sel = {
        'ANN': conf.en_train,
        'SNN': conf.en_train
    }

    en_train = en_train_sel[conf.nn_mode]

    #
    #lr=tfe.Variable(conf.lr)
    lr=tf.Variable(conf.lr)
    momentum = conf.momentum
    #optimizer = tf.train.GradientDescentOptimizer(lr)
    #optimizer = tf.train.MomentumOptimizer(lr,momentum)
    #optimizer = tf.train.AdamOptimizer(lr)
    optimizer = tf.keras.optimizers.Adam(lr)


    if conf.output_dir:
        output_dir = os.path.join(conf.output_dir,conf.model_name+'_'+conf.nn_mode)
        output_dir = os.path.join(output_dir,now.strftime("%Y%m%d-%H%M"))

        #if conf.nn_mode == 'SNN':
        #    output_dir = os.path.join(output_dir,conf.n_type+'_time_step_'+str(conf.time_step)+'_vth_'+str(conf.n_init_vth))

        train_dir = os.path.join(output_dir,'train')
        val_dir = os.path.join(output_dir, 'val')
        if conf.f_validation_snn:
            val_snn_dir = os.path.join(output_dir, 'val_snn')
        test_dir = os.path.join(output_dir, 'test')

        if not os.path.isdir(output_dir):
            #tf.io.gfile.MakeDirs(output_dir)
            tf.io.gfile.makedirs(output_dir)
    else:
        train_dir = None
        val_dir = None
        val_snn_dir = None
        test_dir = None

    summary_writer = tf.summary.create_file_writer(train_dir,flush_millis=100)
    val_summary_writer = tf.summary.create_file_writer(val_dir,flush_millis=100,name='val')

    if conf.f_validation_snn:
        val_snn_summary_writer = tf.summary.create_file_writer(val_snn_dir,flush_millis=100,name='val_snn')

    # TODO: TF-V1
    #test_summary_writer = tf.contrib.summary.create_file_writer(test_dir,flush_millis=100,name='test')
    test_summary_writer = tf.summary.create_file_writer(test_dir,flush_millis=100,name='test')
    checkpoint_dir = os.path.join(conf.checkpoint_dir,conf.model_name)
    checkpoint_load_dir = os.path.join(conf.checkpoint_load_dir,conf.model_name)

    print('model load path: %s' % checkpoint_load_dir)
    print('model save path: %s' % checkpoint_dir)

    if en_train:
        # force to overwrite train model
        if not conf.en_load_model:
            print('remove pre-trained model: {}'.format(checkpoint_dir))
            shutil.rmtree(checkpoint_dir,ignore_errors=True)

        if not os.path.isdir(checkpoint_dir):
            #os.mkdir(checkpoint_dir)
            #tf.io.gfile.MakeDirs(checkpoint_dir)
            tf.io.gfile.makedirs(checkpoint_dir)

    if not os.path.isdir(checkpoint_load_dir):
        print('there is no load dir: %s' % checkpoint_load_dir)
        sys.exit(1)

    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')





    # epoch
    #global_epoch = tf.Variable(name='global_epoch', initial_value=tf.zeros(shape=[]),dtype=tf.int32,trainable=False)
    global_epoch = tf.Variable(name='global_epoch', initial_value=tf.zeros(shape=[]),dtype=tf.float32,trainable=False)

    with tf.device(device):
        #if conf.en_train:
        if en_train:
            print('Train Phase >')

            acc_val_target_best = 0.0
            acc_val_best = 0.0
            acc_val_snn_best = 0.0

            #with tfe.restore_variables_on_create(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)):

            if conf.dataset!='ImageNet':
                train_dataset_p = dataset.train_data_augmentation(train_dataset, conf.batch_size)


            #print(list(train_dataset_p.as_numpy_iterator()))
            #print(train_dataset_p.next())

            #for element in train_dataset_p:
            #    print(element)
            #    assert(False)

            #sample=next(train_dataset_p.__iter__())

            #images_0, _ = tfe.Iterator(train_dataset_p).get_next()
            #images_0, _ = tf.compat.v1.data.Iterator(train_dataset_p).get_next()
            #images_0 = tf.constant(0.0,dtype=tf.float32,shape)
            images_0 = next(train_dataset_p.__iter__())[0]


            model(images_0,False)

            if conf.en_load_model:
                #tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_load_dir))

                #restore_variables = (model.trainable_weights + optimizer.variables() + [epoch])
                restore_variables = (model.trainable_weights + optimizer.variables() + [global_epoch])
                #restore_variables = (model.trainable_weights)

                print('load model')
                print(tf.train.latest_checkpoint(checkpoint_load_dir))

                load_layer = model.load_layer_ann_checkpoint
                #load_layer = tf.train.Checkpoint(conv1=model.conv1, conv2=model.conv2, fc1=model.fc1, conv1_bn=model.conv1_bn, conv2_bn=model.conv2_bn)
                load_model = tf.train.Checkpoint(model=load_layer, optimizer=optimizer, global_epoch=global_epoch)
                load_model.restore(tf.train.latest_checkpoint(checkpoint_dir))


                #saver = tfe.Saver(restore_variables)
                #saver = tf.compat.v1.train.Saver(restore_variables)
                #saver.restore(None, tf.train.latest_checkpoint(checkpoint_load_dir))

                #checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_load_dir))

                print('load model done')
                epoch_start=int(global_epoch.numpy())

                #
                #loss_val, acc_val, _ = test.test(model, val_dataset, conf, f_val=True)
                #acc_val_best = acc_val

            else:
                epoch_start = 0





            # TODO: tmp for kl loss test


            # for 3d plot
            #fig_tmp=plt.figure()
            #axs_tmp=[]
            #axs_tmp.append(fig_tmp.add_subplot(1,2,1, projection='3d'))
            #axs_tmp.append(fig_tmp.add_subplot(1,2,2, projection='3d'))

            # for 2d plot
            fig_tmp, axs_tmp = plt.subplots(1,2)

            #
            hist_prev=None
            hist_prev_dec=None

            #ax_tmp = plt.axes(projection='3d')




            #

            #with tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_load_dir)):
            #for epoch in range(1,11):
            for epoch in range(epoch_start,epoch_start+conf.epoch+1):
                #global_step = tf.train.get_or_create_global_step()
                #start = time.time()

                #if conf.dataset=='CIFAR-10' or conf.dataset=='CIFAR-100':
                #    #with tf.device('/cpu:0'):
                #    #    train_dataset_p = train_data_augmentation(train_dataset)
                #    train_dataset_p = train_data_augmentation(train_dataset)

                with summary_writer.as_default():
                    #
#                    if epoch < 5 :
#                        lr.assign(0.1/5*(epoch+1))
#                    elif epoch > 5 and epoch < 90:
#                        lr.assign(0.1)
#                    elif epoch > 90 and epoch < 135:
#                        lr.assign(0.01)
#                    elif epoch > 135 and epoch < 180:
#                        lr.assign(0.01)
#                    elif epoch > 180 and epoch < 250:
#                        lr.assign(0.001)
#                    elif epoch > 250:
#                        lr.assign(0.0001)

#                    if epoch == 0:
#                        lr.assign(0.01)
#                    elif epoch > 0 and epoch < 150 :
#                    #if epoch < 90 :
#                        lr.assign(0.01)
#                    elif epoch > 150 and epoch < 250 :
#                        lr.assign(0.001)
#                    elif epoch > 250:
#                        lr.assign(0.0001)

                    # for resnet, cifar10
                    #if epoch == 0:
                    #    lr.assign(0.01)
                    #elif epoch > 0 and epoch < 80 :
                    #    lr.assign(0.1)
                    #elif epoch > 80 and epoch < 135 :
                    #    lr.assign(0.01)
                    #elif epoch > 135 and epoch < 180 :
                    #    lr.assign(0.001)
                    #elif epoch > 180 and epoch < 280 :
                    #    lr.assign(0.0001)
                    #elif epoch > 280:
                    #    lr.assign(0.00001)



                    # learning rate decay
                    #if (epoch%conf.lr_decay_step==0) and (epoch!=0):
                    #    lr.assign(lr.numpy()*conf.lr_decay)
                    #    #print('lr_decay) lr: %',optimizer._learning_rate.numpy())

                    #if epoch > 100 and epoch < 250:
                    #    lr.assign(0.01)
                    #elif epoch > 250 and epoch < 350:
                    #    lr.assign(0.001)
                    #elif epoch > 350 and epoch < 450:
                    #    lr.assign(0.0001)
                    #elif epoch > 450:
                    #    lr.assign(0.00001)


                    #loss_train, acc_train = train.train_one_epoch(model, optimizer, train_dataset_p)
                    #loss_train, acc_train = train.train_snn_one_epoch(model, optimizer, train_dataset_p, conf)
                    #loss_train, acc_train = train.train_ann_one_epoch_mnist(model, optimizer, train_dataset_p, conf)
                    #loss_train, acc_train = train_func(model, optimizer, train_dataset_p, conf)
                    if conf.f_surrogate_training_model:
                        model.epoch = epoch
                        loss_and_acc_train = train_func(model, optimizer, train_dataset_p, epoch)

                        loss_train, loss_pred_train, loss_enc_st_train, loss_max_enc_st_train,\
                        loss_min_enc_st_train, loss_max_tk_rep, acc_train = loss_and_acc_train

                        # TODO: remove
                        # tmp for extract model graph
                        #gr=tf.autograph.to_graph(model)
                        #print(gr)
                        #sess = tf.compat.v1.Session()
                        #print(sess.graph)
                        #tf.io.write_graph(sess.graph_def,'./cnn_mnist','cnn_mnist_graph.pbtxt')




                        #all_variables = (model.trainable_variables + optimizer.variables() + [global_step])

                        #for v in all_variables:
                        #    print(v.name)

                        #print('trainable')
                        #for v in model.trainable_variables:
                        #    print(v.name)


                        # TODO: parameterize
                        if conf.f_surrogate_training_model:
                            #save_epoch=model.list_tk['in'].epoch_start_train_tk
                            #save_epoch=100
                            save_epoch=1
                        else:
                            #save_epoch=100
                            save_epoch=1
                        #save_epoch=0

                        #
                        #with tf.summary.always_record_summaries():
                        tf.summary.scalar('loss', loss_train, step=epoch)
                        tf.summary.scalar('loss_pred', loss_pred_train, step=epoch)
                        tf.summary.scalar('loss_enc_st', loss_enc_st_train, step=epoch)
                        tf.summary.scalar('loss_max_enc_st', loss_max_enc_st_train, step=epoch)
                        tf.summary.scalar('loss_min_enc_st', loss_min_enc_st_train, step=epoch)
                        tf.summary.scalar('loss_max_tk_rep', loss_max_tk_rep, step=epoch)
                        tf.summary.scalar('accuracy', acc_train, step=epoch)

                        #for l_name in model.layer_name[:-1]:
                        for l_name, tk in model.list_tk.items():
                            #scalar_name = 'tc_dec_avg_'+l_name
                            #tf.contrib.summary.scalar(scalar_name, tf.reduce_mean(model.list_tk[l_name].tc_dec), step=epoch)
                            #scalar_name = 'td_dec_avg_'+l_name
                            #tf.contrib.summary.scalar(scalar_name, tf.reduce_mean(model.list_tk[l_name].td_dec), step=epoch)

                            scalar_name = 'tc_avg_'+l_name
                            #tf.summary.scalar(scalar_name, tf.reduce_mean(model.list_tk[l_name].tc), step=epoch)
                            tf.summary.scalar(scalar_name, tf.reduce_mean(tk.tc), step=epoch)
                            scalar_name = 'td_avg_'+l_name
                            #tf.summary.scalar(scalar_name, tf.reduce_mean(model.list_tk[l_name].td), step=epoch)
                            tf.summary.scalar(scalar_name, tf.reduce_mean(tk.td), step=epoch)

                            #scalar_name = 'ta_avg_'+l_name
                            #tf.contrib.summary.scalar(scalar_name, tf.reduce_mean(model.list_tk[l_name].ta), step=epoch)



                        # TODO: tmp for kl loss test -> tensorboard histogram

                        pre_enc = model.list_tk['conv1'].in_enc
                        enc_st = model.list_tk['conv1'].out_enc
                        dec_st = model.list_tk['conv1'].out_dec

                        #
                        #enc_st[0,0,0,0]=tf.zeros(shape=[1])
                        #dec_st[0,0,0,0]=tf.zeros(shape=[1])
                        #dec_st[0,0,0,0].assign(0.0)


                        #dist = tfd.Beta(0.8,2)
                        #dist_sample = dist.sample(enc_st.shape)
                        #dist_sample = tf.multiply(dist_sample,model.conf.time_window)
                        #print(type(dist_sample))
                        #fig, axs = plt.subplots(1,2)

                        #axs_tmp[0].hist(tf.reshape(dist_sample[0,:,:,:],shape=-1))
                        #axs_tmp[1].hist(tf.reshape(enc_st[0,:,:,:],shape=-1))
                        #plt.hist(tf.reshape(dist_sample[0,:,:,:],shape=-1))
                        #plt.hist(tf.reshape(enc_st[0,:,:,:],shape=-1),bins=100,range=(0,40))


                        # plot - start here
                        #
#                        f_plot_training_snn_ttfs=True
#                        if f_plot_training_snn_ttfs:
#                            hist, bins = np.histogram(tf.reshape(enc_st[0,:,:,:],shape=-1),bins=40,range=(0,40),normed=True)
#                            hist=np.expand_dims(hist,axis=1)
#
#                            if hist_prev is None:
#                                print(type(hist_prev))
#                                hist_prev=hist
#                            else:
#                                hist_prev=np.append(hist_prev,hist,axis=1)
#
#                            hist_accu = hist_prev
#                            axs_tmp[0].pcolormesh(hist_accu)
#
#                            #
#                            hist_dec, bins = np.histogram(tf.reshape(dec_st[0,:,:,:],shape=-1),bins=30,range=(0.0,1.2),normed=True)
#
#                            hist_dec=np.expand_dims(hist_dec,axis=1)
#
#                            if hist_prev_dec is None:
#                                hist_prev_dec=hist_dec
#                            else:
#                                hist_prev_dec=np.append(hist_prev_dec,hist_dec,axis=1)
#
#                            hist_accu_dec = hist_prev_dec
#                            axs_tmp[1].pcolormesh(hist_accu_dec)


                        # 3d plot
                        #hist, bins = np.histogram(tf.reshape(enc_st[0,:,:,:],shape=-1),bins=40,range=(0,40))
                        #xs = (bins[:-1]+bins[1:])/2
                        #axs_tmp[0].bar(xs, hist, zs=epoch, zdir='y')
                        #
                        #hist, bins = np.histogram(tf.reshape(dec_st[0,:,:,:],shape=-1),bins=100,range=(0.0,2.5))
                        #xs = (bins[:-1]+bins[1:])/2
                        #axs_tmp[1].bar(xs, hist, zs=epoch, zdir='y', width=bins[1])


                        #plt.draw()
                        #plt.pause(0.0000000000000001)

                        #value_range=[0.0, 30.0]
                        #hist = tf.histogram_fixed_width_bins(enc_st, value_range, nbins=30)
                        #print(hist)


                        #tf.summary.histogram("pre_enc", pre_enc, step=epoch, buckets=1000)
                        #tf.summary.histogram("enc_st", enc_st, step=epoch, buckets=1000)
                        #tf.summary.histogram("dec_st", dec_st, step=epoch, buckets=1000)


                    else:
                        loss_train, acc_train = train_func(model, optimizer, train_dataset_p)

                        save_epoch=1

                        #
                        #with tf.summary.always_record_summaries():
                        tf.summary.scalar('loss', loss_train, step=epoch)
                        tf.summary.scalar('accuracy', acc_train, step=epoch)

                #end = time.time()
                #print('\nTrain time for epoch #%d (global step %d): %f' % (epoch, global_step.numpy(), end-start))


                #
                f_save_model = False
                with val_summary_writer.as_default():
                    loss_val, acc_val, _ = test.test(model, val_dataset, num_val_dataset, conf, f_val=True)
                    #loss_val, acc_val, _ = test.test(model, test_dataset, conf, f_val=True)

                    #with tf.summary.always_record_summaries():
                    tf.summary.scalar('loss', loss_val, step=epoch)
                    tf.summary.scalar('accuracy', acc_val, step=epoch)



                    if acc_val_best < acc_val:
                        acc_val_best = acc_val
                        f_save_model = True

                    #
                    acc_val_target = acc_val
                    acc_val_target_best = acc_val_best


                    #
                    f_val_snn_start = (acc_val_best > val_snn_target_acc) and (epoch%5==0)
                    #f_val_snn_start = acc_val_best > 99.0
                    if conf.f_validation_snn and f_val_snn_start:

                        with val_snn_summary_writer.as_default():
                            #loss_val_snn, acc_val_snn, _ = test.test(model, val_dataset, conf, f_val=False, f_val_snn=True)
                            loss_val_snn, acc_val_snn, _ = test.test(model, val_dataset, num_val_dataset, conf, f_val=True, f_val_snn=True)

                            #with tf.summary.always_record_summaries():
                            tf.summary.scalar('loss', loss_val_snn, step=epoch)
                            tf.summary.scalar('accuracy', acc_val_snn, step=epoch)
                            tf.summary.scalar('spikes', model.total_spike_count_int[-1,-1]/num_val_dataset, step=epoch)

                        acc_val_target = acc_val_snn

                        f_save_model = False
                        if acc_val_snn_best < acc_val_snn:
                            acc_val_snn_best = acc_val_snn
                            spikes_best = model.total_spike_count_int[-1,-1]/num_val_dataset
                            f_save_model = True

                        acc_val_target_best=acc_val_snn_best
                    #
                    #if acc_val_target_best < acc_val_target:
                    #    acc_val_target_best = acc_val_target
                    if f_save_model:

                        if epoch > epoch_start+save_epoch:
                            # TODO: parameterize
                            #f_save_model = acc_val_target_best > 99.0
                            f_save_model = acc_val_target_best > save_target_acc
                            #f_save_model = acc_val_best > 10.0


                            #if acc_val_best > 90.0:
                            if f_save_model:
                                print('save model')

                                #if conf.f_surrogate_training_model:
                                #    for l_name in model.layer_name:
                                #        s_name_tc = 'tc_avg_'+l_name
                                #        s_name_td = 'td_avg_'+l_name
                                #        s_name_ta = 'ta_avg_'+l_name
                                #
                                #        s_tc = tf.reduce_mean(model.list_tk[l_name].tc)
                                #        s_td = tf.reduce_mean(model.list_tk[l_name].td)
                                #        s_ta = tf.reduce_mean(model.list_tk[l_name].ta)
                                #
                                #        print('{}: {}, {}: {}, {}: {}'.format(s_name_tc,s_tc,s_name_td,s_td,s_name_ta,s_ta))

                                global_epoch.assign(epoch)

                                #print(type(global_step))

                                # save model
                                #if epoch%conf.save_interval==0:
                                all_variables = (model.variables + optimizer.variables() + [global_epoch])
                                #all_variables = (model.trainable_variables + optimizer.variables() + [global_step])

                                #print(all_variables)
                                #print(model)


                                #print([v.name for v in all_variables])
                                #tfe.Saver(all_variables).save(checkpoint_prefix, global_step=global_step)

                                #tf.compat.v1.train.Saver(all_variables).save(checkpoint_prefix, global_step=global_epoch)
                                #tf.compat.v1.train.Saver(all_variables).save(None, checkpoint_prefix, global_step=global_epoch)

                                checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, global_epoch=global_epoch)
                                #checkpoint = tf.train.Checkpoint(model=model.trainable_variables, optimizer=optimizer, global_epoch=global_epoch)
                                checkpoint.save(file_prefix=checkpoint_prefix)

                                #tfe.Saver(all_variables).save(checkpoint_prefix, global_step=epoch)
                                #print(all_variables)
                               #print('save model > global_step: %d'%(global_step.numpy()))



                print('[%3d] train(loss: %.3f, acc: %.3f), valid(loss: %.3f, acc: %.3f, best: %.3f)'%(epoch,loss_train,acc_train,loss_val,acc_val,acc_val_best))

                if conf.f_validation_snn and f_val_snn_start:
                    print('valid_snn(loss: %.3f, acc: %.3f, best: %.3f, spikes: t %e, c1 %e, c2 %e spikes_best: %e)'%(loss_val_snn,acc_val_snn,acc_val_snn_best,model.total_spike_count_int[-1,-1]/num_val_dataset,model.total_spike_count_int[-1,0]/num_val_dataset,model.total_spike_count_int[-1,1]/num_val_dataset,spikes_best))


                # test test
                #loss_test, acc_test, _ = test.test(model, test_dataset, conf, f_val=False)
                #print('[%3d] test(loss: %.3f, acc: %.3f)'%(epoch,loss_test,acc_test))


                #gc.collect()


        print(' Test Phase >')
        #print(model.variables)
        #test(model, test_dataset)
        #print(test_dataset.range(0))

        print(test_dataset)

        #if conf.en_train == False:
        if en_train == False:
            if conf.dataset=='ImageNet':
                #images_0, _ = test_dataset.next()  # tensornets
                #images_0, _ = tfe.Iterator(test_dataset).get_next()
                #images_0, _ = tf.compat.v1.data.Iterator(test_dataset).get_next()
                images_0 = next(test_dataset.__iter__())[0]
            else:
                #images_0, _ = tfe.Iterator(test_dataset).get_next()
                #images_0, _ = tf.compat.v1.data.Iterator(test_dataset).get_next()
                images_0 = next(test_dataset.__iter__())[0]
            model(images_0,False)
            #print('image 0 done')

            #restore_variables = [v for v in model.trainable_variables if ('neuron' not in v.name) and ('dummy' not in v.name)]
            #restore_variables = [v for v in model.variables if ('neuron' not in v.name) and ('dummy' not in v.name)]


            #print([v.name for v in model.variables])
            #print([v.name for v in restore_variables])

            #restore_variables = model.trainable_weights
            #restore_variables = (model.trainable_weights + optimizer.variables() + [epoch])
            #restore_variables = (model.trainable_weights + optimizer.variables())
            #restore_variables = optimizer.variables()


            #
            #print([v.name for v in restore_variables])


            if conf.ann_model=='ResNet50':
                #print('restore variables')
                var_dict=collections.OrderedDict()
                for var in restore_variables:
                    v_name=var.name[:-2]
                    v_name=re.sub('kernel','weights',v_name)
                    v_name=re.sub('bias','biases',v_name)
                    v_name=re.sub('_block','/block',v_name)
                    if v_name != 'Variable':    # inserted Variable in SNN mode
                        var_dict[v_name] = var
                        #print(v_name)
                #print([v.name for v in restore_variables])
                #print(var_list)
                #print(var_dict['resnet50/conv1/conv/kernel'])

                #restore_var_list = []
                #restore_var_list = restore_var_list.append(var_dict['resnet50/conv1/conv/kernel'])
                #print(restore_var_list)

                # TODO: TF-V1
                #saver = tfe.Saver(var_dict)
                #saver = tf.compat.v1.train.Saver(var_dict)
            else:
                # TODO: TF-V1
                #saver = tfe.Saver(restore_variables)
                #saver = tf.compat.v1.train.Saver(restore_variables)

                #checkpoint = tf.train.Checkpoint(restore_variables)
                #checkpoint = tf.train.Checkpoint(model=model)
                #kernel = model.conv1.kernel
                #kernel=model.conv2.kernel
                #bias=model.conv1.bias
                #bias=model.conv2.bias
                #load_kernel = tf.train.Checkpoint(kernel=kernel, bias=bias)
                #load_bias = tf.train.Checkpoint(bias=bias)

                #
                #conv1 = tf.train.Checkpoint(kernel=model.conv1.kernel, bias=model.conv1.bias)
                #conv2 = tf.train.Checkpoint(kernel=model.conv2.kernel, bias=model.conv2.bias)
                #net = tf.train.Checkpoint(conv2d_1=conv1)
                #load_layer = tf.train.Checkpoint(model=net)

                #load_layer = tf.train.Checkpoint(conv2=model.conv2)

                load_layer = model.load_layer_ann_checkpoint


                #k = tf.train.Checkpoint(kernel=model.conv1.kernel)
                #load_layer = tf.train.Checkpoint(conv2d=k)

                #load_layer = tf.train.Checkpoint(conv1=model.conv1, conv2=model.conv2, fc1=model.fc1, conv1_bn=model.conv1_bn, conv2_bn=model.conv2_bn)
                #load_model = tf.train.Checkpoint(model=load_layer)

                # checkpoint TF V1 and restore in TF V2
                #load_model = tf.train.Checkpoint(mnist_model_cnn=load_layer)

                # checkpoint load - save and restore in TF V2
                #print(model.trainable_variables)
                #load_model = tf.train.Checkpoint(model=model)
                #load_model = tf.train.Checkpoint(model=model.trainable_variables, optimizer=optimizer, global_epoch=global_epoch)
                #load_model = tf.train.Checkpoint(model=model, optimizer=optimizer, global_epoch=global_epoch)
                load_model = tf.train.Checkpoint(model=load_layer, optimizer=optimizer, global_epoch=global_epoch)
                #load_model = tf.train.Checkpoint(conv1=model.conv1, conv2=model.conv2, fc1=model.fc1, conv1_bn=model.conv1_bn, conv2_bn=model.conv2_bn)
                #load_model = tf.train.Checkpoint(model=load_layer, optimizer=optimizer, global_epoch=global_epoch)




            print(load_model)
            print('load model')
            #print(tf.train.latest_checkpoint(checkpoint_dir))
            print(tf.train.list_variables(checkpoint_dir))
            #print(tf.train.list_variables(checkpoint_dir)[1])
            #print(tf.train.list_variables(checkpoint_dir)[1][0])



            #
            #restore_variables = [v for v in model.variables if ('conv1' in v.name)]
            #print([v.name for v in restore_variables])

            #~restore = tf.train.Checkpoint(model=model)
            #restore=tf.train.Checkpoint()
            #restore.listed=[]
            #restore.listed.append(restore_variables)
            #print(restore.listed)
            #restore.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

            #restore.restore(tf.train.latest_checkpoint(checkpoint_dir))


            # start here - restore code
            #load_model.restore(tf.train.latest_checkpoint(checkpoint_dir)).assert_consumed()
            status = load_model.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
            #status = load_model.restore(tf.train.latest_checkpoint(checkpoint_dir))
            #status.assert_existing_objects_matched()


#            # temporary for old version TF-V1 cnn mnist
#            conv1_k = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/conv2d/kernel")
#            conv1_b = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/conv2d/bias")
#            model.list_layer['conv1'].kernel = conv1_k
#            model.list_layer['conv1'].bias = conv1_b
#
#            conv2_k = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/conv2d_1/kernel")
#            conv2_b = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/conv2d_1/bias")
#            model.list_layer['conv2'].kernel = conv2_k
#            model.list_layer['conv2'].bias = conv2_b
#
#            fc1_k = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/dense/kernel")
#            fc1_b = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/dense/bias")
#            model.list_layer['fc1'].kernel = fc1_k
#            model.list_layer['fc1'].bias = fc1_b
#
#            conv1_bn_b = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/batch_normalization/beta")
#            conv1_bn_g = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/batch_normalization/gamma")
#            conv1_bn_mm = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/batch_normalization/moving_mean")
#            conv1_bn_mv = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/batch_normalization/moving_variance")
#            model.list_layer['conv1_bn'].beta = conv1_bn_b
#            model.list_layer['conv1_bn'].gamma = conv1_bn_g
#            model.list_layer['conv1_bn'].moving_mean = conv1_bn_mm
#            model.list_layer['conv1_bn'].moving_variance = conv1_bn_mv
#
#
#            conv2_bn_b = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/batch_normalization_1/beta")
#            conv2_bn_g = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/batch_normalization_1/gamma")
#            conv2_bn_mm = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/batch_normalization_1/moving_mean")
#            conv2_bn_mv = tf.train.load_variable(tf.train.latest_checkpoint(checkpoint_dir),"mnist_model_cnn/batch_normalization_1/moving_variance")
#            model.list_layer['conv2_bn'].beta = conv2_bn_b
#            model.list_layer['conv2_bn'].gamma = conv2_bn_g
#            model.list_layer['conv2_bn'].moving_mean = conv2_bn_mm
#            model.list_layer['conv2_bn'].moving_variance = conv2_bn_mv




            #print([v.name for v in model.variables])


            #assert(False)

            #print(tf.get_variable("mnist_model_mlp/dense_2/kernel"))

            # temporally off
            #saver = tfe.Saver(restore_variables)
            #saver = tfe.Saver(restore_var_list)
            #print(model.variables)
            #saver = tfe.Saver(
            #    var_list=
            #    #{'resnet50/conv1/conv/weights': kernel}
            #    {'resnet50/conv1/conv/weights': }
            #)

            #ckpt_var_list = []

            #(ckpt_var_list, _) = tf.contrib.framework.list_variables(checkpoint_dir)
            #ckpt_var_list, ckpt_shape_list = tf.contrib.framework.list_variables(checkpoint_dir)
            #print(ckpt_var_list)


            #for (var, shape) in tf.contrib.framework.list_variables(checkpoint_dir):
            #    ckpt_var_list.append(var)
            #    print(var)
            #    print(shape)
            #    saver = tfe.Saver(var)
            #    saver.restore(checkpoint_dir)

            #print(ckpt_var_list)
            #print(len(ckpt_var_list))
            #print(len(restore_variables))

            # restore
            #saver.restore(tf.train.latest_checkpoint(checkpoint_dir))
            #saver.restore(None, tf.train.latest_checkpoint(checkpoint_dir))
            #load_model.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
            #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).assert_consumed()
            #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).assert_nontrivial_match()

            #tf.contrib.framework.init_from_checkpoint(
            #    checkpoint_dir,
            #    {'*/weights' : '*/kernel'}
            #)

            #print(tf.get_variable('resnet50/conv1/kernel:0'))

            print('load model done')

            #print('print model')
            #for var in restore_variables:
            #    print(var)

        # for profile
#        with context.eager_mode():
#            prof_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#            run_metadata = tf.RunMetadata()
#            dump_dir = './'
#            prof_dump = os.path.join(dump_dir,'dump')
#            opts = builder(
#                builder.time_and_memory()
#            ).with_file_output(prof_dump).build()
#            context.enable_run_metadata()
#
#
#            # test
#            with test_summary_writer.as_default():
#                loss_test, acc_test, acc_test_top5 = test.test(model, test_dataset, conf)
#                if conf.dataset == 'ImageNet':
#                    print('loss_test: %f, acc_test: %f, acc_test_top5: %f'%(loss_test,acc_test,acc_test_top5))
#                else:
#                    print('loss_test: %f, acc_test: %f'%(loss_test,acc_test))
#
#            profiler = model_analyzer.Profiler()
#            profiler.add_step(0, context.export_run_metadata())
#            context.disable_run_metadata()
#            profiler.profile_operations(opts)
#
#            with gfile.Open(prof_dump, 'r') as f:
#                out_str = f.read()
#
#                print(out_str)



            #if conf.f_train_time_const:
            #    loss_train, acc_train, acc_train_top5 = test.test(model, train_dataset, conf)
            #    if conf.dataset == 'ImageNet':
            #        print('loss_test: %f, acc_test: %f, acc_test_top5: %f'%(loss_test,acc_test,acc_test_top5))
            #    else:
            #        print('loss_test: %f, acc_test: %f'%(loss_test,acc_test))


            if conf.f_train_time_const:

                for epoch in range(conf.epoch_train_time_const):
                    print("epoch: {:d}".format(epoch))
                    with test_summary_writer.as_default():
                        loss_test, acc_test, acc_test_top5 = test.test(model, test_dataset, conf, epoch=epoch)
                        if conf.dataset == 'ImageNet':
                            print('loss_test: %f, acc_test: %f, acc_test_top5: %f'%(loss_test,acc_test,acc_test_top5))
                        else:
                            print('loss_test: %f, acc_test: %f'%(loss_test,acc_test))


            else:
                #
                with test_summary_writer.as_default():
                    loss_test, acc_test, acc_test_top5 = test.test(model, test_dataset, num_test_dataset, conf)
                    #loss_test, acc_test, acc_test_top5 = test.test(model, val_dataset, num_val_dataset, conf)
                    if conf.dataset == 'ImageNet':
                        print('loss_test: %f, acc_test: %f, acc_test_top5: %f'%(loss_test,acc_test,acc_test_top5))
                    else:
                        print('loss_test: %f, acc_test: %f'%(loss_test,acc_test))



                    #loss_val_snn, acc_val_snn, _ = test.test(model, val_dataset, conf, f_val=True, f_val_snn=True)
                    #print('valid_snn(loss: %.3f, acc: %.3f, best: %.3f, spikes: t %e, c1 %e, c2 %e spikes_best: %e)'%(loss_val_snn,acc_val_snn,acc_val_snn_best,model.total_spike_count_int[-1,-1],model.total_spike_count_int[-1,0],model.total_spike_count_int[-1,1],spikes_best))


                    #if conf.f_surrogate_training_model:
                    #    for l_name in model.layer_name:
                    #        s_name_tc = 'tc_avg_'+l_name
                    #        s_name_td = 'td_avg_'+l_name
                    #        s_name_ta = 'ta_avg_'+l_name
                    #
                    #        s_tc = tf.reduce_mean(model.list_tk[l_name].tc)
                    #        s_td = tf.reduce_mean(model.list_tk[l_name].td)
                    #        s_ta = tf.reduce_mean(model.list_tk[l_name].ta)
                    #
                    #        print('{}: {}, {}: {}, {}: {}'.format(s_name_tc,s_tc,s_name_td,s_td,s_name_ta,s_ta))



                    #
                    #plt.hist(model.neuron_list['conv1'].stat_ws.numpy().flatten())
        print('end')

        #os._exit(0)


    #pr.disable()

    #pr.print_stats(sort='time')
    #pr.print_stats()
    #pr_dump_file_name='dump.prof'
    #pr.dump_stats(pr_dump_file_name)






if __name__ == '__main__':
    tf.compat.v1.app.run()

