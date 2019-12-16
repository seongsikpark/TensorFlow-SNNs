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


#en_gpu=False
en_gpu=True

#gpu_number=0

#os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)

if en_gpu==False:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf

import tensorflow.contrib.eager as tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import ops

from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

builder = option_builder.ProfileOptionBuilder

#
import train
import test

# models
from models import models
from models import mlp
from models import cnn
import model_cnn_mnist
import resnet


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

from tqdm import tqdm

import math

import csv
import collections
import re

import gc



#

#
np.set_printoptions(threshold=np.inf, linewidth=1000, precision=4)

#
pp = pprint.PrettyPrinter().pprint

#
flags = tf.app.flags
tf.app.flags.DEFINE_string('date','','date')

tf.app.flags.DEFINE_integer('epoch', 300, 'Number os epochs')
tf.app.flags.DEFINE_string('gpu_fraction', '1/3', 'define the gpu fraction used')
tf.app.flags.DEFINE_integer('batch_size', 100, '')
tf.app.flags.DEFINE_string('activation', 'ReLU', '')
tf.app.flags.DEFINE_string('optim_type', 'adam', '[exp_decay, adam]')

tf.app.flags.DEFINE_string('nn_mode', 'SNN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')
#tf.app.flags.DEFINE_string('nn_mode', 'ANN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')

tf.app.flags.DEFINE_string('output_dir', './tensorboard', 'Directory to write TensorBoard summaries')
tf.app.flags.DEFINE_string('checkpoint_dir', './models_ckpt', 'Directory to save checkpoints')
tf.app.flags.DEFINE_string('checkpoint_load_dir', './models_ckpt', 'Directory to load checkpoints')
tf.app.flags.DEFINE_bool('en_load_model', False, 'Enable to load model')

#
tf.app.flags.DEFINE_boolean('en_train', False, 'enable training')

#
tf.app.flags.DEFINE_float('n_init_vth', 0.7, 'initial value of vth')
tf.app.flags.DEFINE_float('n_in_init_vth', 0.7, 'initial value of vth of n_in')
tf.app.flags.DEFINE_float('n_init_vinit', 0.0, 'initial value of vinit')
tf.app.flags.DEFINE_float('n_init_vrest', 0.0, 'initial value of vrest')

# exponetial decay
'''
tf.app.flags.DEFINE_float('init_lr', 0.1, '')
tf.app.flags.DEFINE_float('decay_factor', 0.1, '')
tf.app.flags.DEFINE_integer('num_epoch_per_decay', 350, '')
'''
# adam optimizer
#tf.app.flags.DEFINE_float('init_lr', 1e-5, '')
# SGD
tf.app.flags.DEFINE_float('lr', 0.1, 'learning rate')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum')
# ADAM
#tf.app.flags.DEFINE_float('lr', 0.0001, '')

# l2 norm
tf.app.flags.DEFINE_float('lamb',0.0001, 'lambda')

tf.app.flags.DEFINE_float('lr_decay', 0.1, '')
tf.app.flags.DEFINE_integer('lr_decay_step', 50, '')

tf.app.flags.DEFINE_integer('time_step', 10, 'time steps per sample in SNN')


tf.app.flags.DEFINE_integer('idx_test_dataset_s', 0, 'start index of test dataset')
tf.app.flags.DEFINE_integer('num_test_dataset', 10000, 'number of test datset')
tf.app.flags.DEFINE_integer('size_test_batch', 1, 'size of test batch') # not used now

tf.app.flags.DEFINE_string('pooling', 'max', 'max or avg, only for CNN')

tf.app.flags.DEFINE_integer('save_interval', 10, 'save interval of model')

tf.app.flags.DEFINE_bool('en_remove_output_dir', False, 'enable removing output dir')


#
tf.app.flags.DEFINE_boolean('use_bias', True, 'use bias')
tf.app.flags.DEFINE_string('regularizer', 'L2', 'L2 or L1 regularizer')


tf.app.flags.DEFINE_string('model_name', 'snn_train_mlp_mnist', 'model name')

tf.app.flags.DEFINE_string('n_type', 'LIF', 'LIF or IF: neuron type')

#
tf.app.flags.DEFINE_string('dataset', 'MNIST', 'dataset')
tf.app.flags.DEFINE_string('ann_model', 'MLP', 'neural network model')

#
tf.app.flags.DEFINE_boolean('verbose',True, 'verbose mode')
tf.app.flags.DEFINE_boolean('verbose_visual',True, 'verbose visual mode')

#
tf.app.flags.DEFINE_integer('time_step_save_interval',10,'snn test save interval')

#
tf.app.flags.DEFINE_bool('f_fused_bn',False,'f_fused_bn')

#
tf.app.flags.DEFINE_bool('f_stat_train_mode',False,'f_stat_train_mode')
tf.app.flags.DEFINE_bool('f_real_value_input_snn',False,'f_real_value_input_snn')
tf.app.flags.DEFINE_bool('f_vth_conp',False,'f_vth_conp')
tf.app.flags.DEFINE_bool('f_spike_max_pool',False,'f_spike_max_pool')
tf.app.flags.DEFINE_bool('f_w_norm_data',False,'f_w_norm_data')
tf.app.flags.DEFINE_bool('f_ws',False,'wieghted synapse')
tf.app.flags.DEFINE_integer('p_ws',8,'period of wieghted synapse')

tf.app.flags.DEFINE_integer('num_class',10,'number_of_class (do not touch)')

tf.app.flags.DEFINE_string('input_spike_mode','POISSON','input spike mode - POISSON, WEIGHTED_SPIKE, ROPOSED')
tf.app.flags.DEFINE_string('neural_coding','RATE','neural coding - RATE, WEIGHTED_SPIKE, PROPOSED')

tf.app.flags.DEFINE_bool('f_positive_vmem',False,'positive vmem')
tf.app.flags.DEFINE_bool('f_tot_psp',False,'accumulate total psp')

tf.app.flags.DEFINE_bool('f_isi',False,'isi stat')
tf.app.flags.DEFINE_bool('f_refractory',False,'refractory mode')

tf.app.flags.DEFINE_bool('f_comp_act',False,'compare activation')
tf.app.flags.DEFINE_bool('f_entropy',False,'entropy test')
tf.app.flags.DEFINE_bool('f_write_stat',False,'write stat')
tf.app.flags.DEFINE_string('act_save_mode','channel','activation save mode')
tf.app.flags.DEFINE_bool('f_save_result',True,'save result to xlsx file')

# data.py - imagenet data
tf.app.flags.DEFINE_string('data_path_imagenet', './imagenet', 'data path imagenet')
tf.app.flags.DEFINE_integer('k_pathces', 5, 'patches for test (random crop)')
tf.app.flags.DEFINE_integer('input_size', 28, 'input image width / height')


#
tf.app.flags.DEFINE_string('path_stat','./stat/', 'path stat')
tf.app.flags.DEFINE_string('prefix_stat','act_n_train', 'prefix of stat file name')


#
tf.app.flags.DEFINE_bool('f_data_std', True, 'data_standardization')


# pruning
tf.app.flags.DEFINE_bool('f_pruning_channel', False, 'purning - channel')


tf.app.flags.DEFINE_string('path_result_root','./result/', 'path result root')

# temporal coding
#tf.app.flags.DEFINE_float('tc',10.0,'time constant for temporal coding')
#tf.app.flags.DEFINE_float('time_window',20.0,'time window of each layer for temporal coding')
#tf.app.flags.DEFINE_float('time_fire_start',20.0,'time fire start (integration time before starting fire) for temporal coding')
#tf.app.flags.DEFINE_float('time_fire_duration',20.0,'time fire duration for temporal coding')
tf.app.flags.DEFINE_integer('tc',10,'time constant for temporal coding')
tf.app.flags.DEFINE_integer('time_window',20,'time window of each layer for temporal coding')
tf.app.flags.DEFINE_integer('time_fire_start',20,'time fire start (integration time before starting fire) for temporal coding')
tf.app.flags.DEFINE_integer('time_fire_duration',20,'time fire duration for temporal coding')
tf.app.flags.DEFINE_bool('f_record_first_spike_time',False,'flag - recording first spike time of each neuron')
tf.app.flags.DEFINE_bool('f_visual_record_first_spike_time',False,'flag - visual recording first spike time of each neuron')
tf.app.flags.DEFINE_bool('f_train_time_const',False,'flag - enable to train time constant for temporal coding')
tf.app.flags.DEFINE_bool('f_train_time_const_outlier',True,'flag - enable to outlier roubst train time constant for temporal coding')
tf.app.flags.DEFINE_bool('f_load_time_const',False,'flag - load time constant for temporal coding')
tf.app.flags.DEFINE_string('time_const_init_file_name','./temporal_coding/time_const','temporal coding file name - time_const, time_delay`')
tf.app.flags.DEFINE_integer('time_const_num_trained_data',0,'number of trained data - time constant')
tf.app.flags.DEFINE_integer('time_const_save_interval',10000,'save interval - training time constant')
tf.app.flags.DEFINE_integer('epoch_train_time_const',1,'epoch - training time constant')

tf.app.flags.DEFINE_bool('f_tc_based',False,'flag - tau based')
tf.app.flags.DEFINE_integer('n_tau_fire_start',4,'n tau - fire start')
tf.app.flags.DEFINE_integer('n_tau_fire_duration',4,'n tau - fire duration')
tf.app.flags.DEFINE_integer('n_tau_time_window',4,'n tau - time window')


#
conf = flags.FLAGS

data_path_imagenet='/home/sspark/Datasets/ILSVRC2012'
conf.data_path_imagenet = data_path_imagenet


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

    tfe.enable_eager_execution()
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

        (train_dataset, val_dataset, test_dataset) = dataset.load(conf)

    #print('> Dataset info.')
    #print(train_dataset)
    #print(test_dataset)



    #train_func_sel = {
    #    'ANN': train.train_ann_one_epoch,
    #    'SNN': train.train_snn_one_epoch
    #}
    #
    #train_func = train_func_sel[conf.nn_mode]

    train_func = train.train_one_epoch

    if conf.ann_model=='MLP':
        if conf.dataset=='MNIST':
            #model = models.MNISTModel_MLP(data_format,conf)
            model = mlp.mlp_mnist(data_format,conf)
            if conf.nn_mode=='ANN':
                train_func = train.train_ann_one_epoch_mnist
            else:
                train_func = train.train_snn_one_epoch_mnist
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
    elif conf.ann_model=='ResNet18':
        if conf.dataset=='CIFAR-10' or conf.dataset=='CIFAR-100':
            model = resnet.Resnet18(data_format,conf)
    elif conf.ann_model=='ResNet50':
        if conf.dataset=='ImageNet':
            model = resnet.Resnet50(data_format,conf)
    else:
        print('not supported model name: '+self.ann_model)
        os._exit(0)



    en_train_sel = {
        'ANN': conf.en_train,
        'SNN': conf.en_train
    }

    en_train = en_train_sel[conf.nn_mode]

    #
    lr=tfe.Variable(conf.lr)
    momentum = conf.momentum
    #optimizer = tf.train.GradientDescentOptimizer(lr)
    #optimizer = tf.train.MomentumOptimizer(lr,momentum)
    optimizer = tf.train.AdamOptimizer(lr)


    if conf.output_dir:
        output_dir = os.path.join(conf.output_dir,conf.model_name+'_'+conf.nn_mode)

        if conf.nn_mode == 'SNN':
            output_dir = os.path.join(output_dir,conf.n_type+'_time_step_'+str(conf.time_step)+'_vth_'+str(conf.n_init_vth))

        train_dir = os.path.join(output_dir,'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')

        if not os.path.isdir(output_dir):
            tf.gfile.MakeDirs(output_dir)
    else:
        train_dir = None
        val_dir = None
        test_dir = None

    summary_writer = tf.contrib.summary.create_file_writer(train_dir,flush_millis=100)
    val_summary_writer = tf.contrib.summary.create_file_writer(val_dir,flush_millis=100,name='val')
    test_summary_writer = tf.contrib.summary.create_file_writer(test_dir,flush_millis=100,name='test')
    checkpoint_dir = os.path.join(conf.checkpoint_dir,conf.model_name)
    checkpoint_load_dir = os.path.join(conf.checkpoint_load_dir,conf.model_name)

    print('model load path: %s' % checkpoint_load_dir)
    print('model save path: %s' % checkpoint_dir)

    if not os.path.isdir(checkpoint_dir):
        #os.mkdir(checkpoint_dir)
        tf.gfile.MakeDirs(checkpoint_dir)

    if not os.path.isdir(checkpoint_load_dir):
        print('there is no load dir: %s' % checkpoint_load_dir)
        sys.exit(1)

    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')



    with tf.device(device):
        #if conf.en_train:
        if en_train:
            print('Train Phase >')

            acc_val_best = 0.0
            #with tfe.restore_variables_on_create(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)):

            if conf.dataset!='ImageNet':
                train_dataset_p = dataset.train_data_augmentation(train_dataset, conf.batch_size)

            images_0, _ = tfe.Iterator(train_dataset_p).get_next()
            #images_0, _ = tfe.Iterator(test_dataset).get_next()

            model(images_0,False)

            if conf.en_load_model:
                #tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_load_dir))

                #restore_variables = (model.trainable_weights + optimizer.variables() + [epoch])
                restore_variables = (model.trainable_weights + optimizer.variables())
                #restore_variables = (model.trainable_weights)

                print(restore_variables)

                print('load model')
                print(tf.train.latest_checkpoint(checkpoint_load_dir))

                saver = tfe.Saver(restore_variables)
                saver.restore(tf.train.latest_checkpoint(checkpoint_load_dir))


            epoch_start = 0


            #

            #with tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_load_dir)):
            #for epoch in range(1,11):
            for epoch in range(epoch_start,conf.epoch+1):
                global_step = tf.train.get_or_create_global_step()
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


                    loss_train, acc_train = train.train_one_epoch(model, optimizer, train_dataset_p)
                    #loss_train, acc_train = train.train_snn_one_epoch(model, optimizer, train_dataset_p, conf)
                    #loss_train, acc_train = train.train_ann_one_epoch_mnist(model, optimizer, train_dataset_p, conf)
                    #loss_train, acc_train = train_func(model, optimizer, train_dataset_p, conf)


                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar('loss', loss_train, step=epoch)
                        tf.contrib.summary.scalar('accuracy', acc_train, step=epoch)

                #end = time.time()
                #print('\nTrain time for epoch #%d (global step %d): %f' % (epoch, global_step.numpy(), end-start))

                #
                with val_summary_writer.as_default():
                    loss_val, acc_val, _ = test.test(model, val_dataset, conf, f_val=True)

                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar('loss', loss_val, step=epoch)
                        tf.contrib.summary.scalar('accuracy', acc_val, step=epoch)

                    #if acc_val_best < acc_val and epoch > 50:
                    if acc_val_best < acc_val:
                        acc_val_best = acc_val

                        if acc_val_best > 90.0:
                            print('save model')

                            # save model
                            #if epoch%conf.save_interval==0:
                            all_variables = (model.variables + optimizer.variables() + [global_step])
                            #print([v.name for v in all_variables])
                            #tfe.Saver(all_variables).save(checkpoint_prefix, global_step=global_step)
                            tfe.Saver(all_variables).save(checkpoint_prefix, global_step=epoch)
                            #print(all_variables)
                           #print('save model > global_step: %d'%(global_step.numpy()))



                print('[%3d] train(loss: %.3f, acc: %.3f), valid(loss: %.3f, acc: %.3f, best: %.3f)'%(epoch,loss_train,acc_train,loss_val,acc_val,acc_val_best))

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
                images_0, _ = tfe.Iterator(test_dataset).get_next()
            else:
                images_0, _ = tfe.Iterator(test_dataset).get_next()
            model(images_0,False)
            #print('image 0 done')

            restore_variables = [v for v in model.variables if 'neuron' not in v.name]
            #restore_variables = model.trainable_weights
            #restore_variables = (model.trainable_weights + optimizer.variables() + [epoch])
            #restore_variables = (model.trainable_weights + optimizer.variables())
            #restore_variables = optimizer.variables()



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

                saver = tfe.Saver(var_dict)
            else:
                saver = tfe.Saver(restore_variables)

            print('load model')
            print(tf.train.latest_checkpoint(checkpoint_dir))

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
            #
            saver.restore(tf.train.latest_checkpoint(checkpoint_dir))

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
                    loss_test, acc_test, acc_test_top5 = test.test(model, test_dataset, conf)
                    if conf.dataset == 'ImageNet':
                        print('loss_test: %f, acc_test: %f, acc_test_top5: %f'%(loss_test,acc_test,acc_test_top5))
                    else:
                        print('loss_test: %f, acc_test: %f'%(loss_test,acc_test))

        print('end')

        #os._exit(0)


    #pr.disable()

    #pr.print_stats(sort='time')
    #pr.print_stats()
    #pr_dump_file_name='dump.prof'
    #pr.dump_stats(pr_dump_file_name)






if __name__ == '__main__':
    tf.app.run()

