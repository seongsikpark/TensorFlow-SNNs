import tensorflow as tf
#import tensorflow.contrib.eager as tfe

#from tensorflow.contrib.layers.python.layers import initializers
#from tensorflow.contrib.layers.python.layers import regularizers

import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers


from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops

import tensorflow_probability as tfp
tfd = tfp.distributions

#
import util
import lib_snn
import sys
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import math
import csv
import collections

from scipy import stats
from scipy import sparse

from operator import itemgetter

from functools import partial


import pandas as pd

#import tfplot
import threading

#
# noinspection PyUnboundLocalVariable
#class CIFARModel_CNN(tfe.Network):
#class CIFARModel_CNN(tf.keras.layers):
class CIFARModel_CNN(tf.keras.layers.Layer):
    def __init__(self, data_format, conf):
        super(CIFARModel_CNN, self).__init__(name='')

        self.data_format = data_format
        self.conf = conf
        self.num_class = self.conf.num_class

        self.f_1st_iter = True
        self.f_load_model_done = False
        self.verbose = conf.verbose
        self.f_debug_visual = conf.verbose_visual
        self.f_done_preproc = False

        self.kernel_size = 3
        self.fanin_conv = self.kernel_size*self.kernel_size
        #self.fanin_conv = self.kernel_size*self.kernel_size/9

        self.ts=conf.time_step

        self.count_accuracy_time_point=0
        self.accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
        self.accuracy_time_point.append(conf.time_step)
        #self.num_accuracy_time_point = int(math.ceil(float(conf.time_step)/float(conf.time_step_save_interval))
        self.num_accuracy_time_point = len(self.accuracy_time_point)


        if self.f_debug_visual:
            #self.debug_visual_threads = []
            self.debug_visual_axes = []
            self.debug_visual_list_neuron = collections.OrderedDict()

        #
        self.f_skip_bn = self.conf.f_fused_bn


        #
        self.epoch = -1

        #
        self.layer_name=[
            # TODO: 'in' check - why commented out 'in'?
            'in',
            'conv1',
            'conv1_1',
            'conv2',
            'conv2_1',
            'conv3',
            'conv3_1',
            'conv3_2',
            'conv4',
            'conv4_1',
            'conv4_2',
            'conv5',
            'conv5_1',
            'conv5_2',
            'fc1',
            'fc2',
            'fc3',
        ]

        self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])
        self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])

        self.total_residual_vmem=np.zeros(len(self.layer_name)+1)

        if self.conf.f_isi:
            self.total_isi=np.zeros(self.conf.time_step)

            type_spike_amp_kind = {
                'RATE': 1,
                'WEIGHTED_SPIKE': self.conf.p_ws,
                'BURST': int(5-np.log2(self.conf.n_init_vth))
            }

            self.spike_amp_kind = type_spike_amp_kind[self.conf.neural_coding]+1
            self.total_spike_amp=np.zeros(self.spike_amp_kind)

            type_spike_amp_bin = {
                'RATE': np.power(0.5,range(0,self.spike_amp_kind+1)),
                'WEIGHTED_SPIKE': np.power(0.5,range(0,self.spike_amp_kind+1)),
                'BURST': 32*np.power(0.5,range(0,self.spike_amp_kind+1))
            }

            #self.spike_amp_bin=type_spike_amp_bin[self.conf.neural_coding]
            #print(range(0,self.spike_amp_kind+2))
            #print(np.power(0.5,range(0,self.spike_amp_kind+2)))
            #print(np.power(0,5,range(0,self.spike_amp_kind)[::-1]))
            #self.spike_amp_bin=np.power(np.array(self.spike_amp_kind).fill(0.5),range(1,self.spike_amp_kind+1))
            #self.spike_amp_bin=np.power(0.5,range(0,self.spike_amp_kind+2))
            self.spike_amp_bin=type_spike_amp_bin[self.conf.neural_coding]
            self.spike_amp_bin=self.spike_amp_bin[::-1]
            self.spike_amp_bin[0]=0.0

            #print(self.spike_amp_bin)

        if self.conf.f_comp_act:
            self.total_comp_act=np.zeros([self.conf.time_step,len(self.layer_name)+1])


        self.output_layer_isi=np.zeros(self.num_class)
        self.output_layer_last_spike_time=np.zeros(self.num_class)

        # nomarlization factor
        self.norm=collections.OrderedDict()
        self.norm_b=collections.OrderedDict()




        if self.data_format == 'channels_first':
            self._input_shape = [-1,3,32,32]    # CIFAR-10
            #self._input_shape = [-1,3,cifar_10_crop_size,cifar_10_crop_size]    # CIFAR-10
        else:
            assert self.data_format == 'channels_last'
            self._input_shape = [-1,32,32,3]
            #self._input_shape = [-1,cifar_10_crop_size,cifar_10_crop_size,3]

        if conf.nn_mode == 'ANN':
            use_bias = conf.use_bias
        else :
            #use_bias = False
            use_bias = conf.use_bias

        #activation = tf.nn.relu
        activation = None
        padding = 'same'

        self.run_mode = {
            'ANN': self.call_ann if not self.conf.f_surrogate_training_model else self.call_ann_surrogate_training,
            'SNN': self.call_snn if not self.conf.f_pruning_channel else self.call_snn_pruning
        }

        self.run_mode_load_model = {
            'ANN': self.call_ann if not self.conf.f_surrogate_training_model else self.call_ann_surrogate_training,
            'SNN': self.call_snn
        }

        #regularizer_type = {
        #    'L1': regularizers.l1_regularizer(conf.lamb),
        #    'L2': regularizers.l2_regularizer(conf.lamb)
        #}

        regularizer_type = {
            'L1': regularizers.l1(conf.lamb),
            'L2': regularizers.l2(conf.lamb)
        }


        kernel_regularizer = regularizer_type[self.conf.regularizer]
        #kernel_initializer = initializers.xavier_initializer(True)
        kernel_initializer = initializers.GlorotUniform()
        #kernel_initializer = initializers.variance_scaling_initializer(factor=2.0,mode='FAN_IN')    # MSRA init. = He init

        self.list_layer=collections.OrderedDict()
        self.list_layer['conv1'] = tf.keras.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv1_1'] = tf.keras.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv1_1_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv2'] = tf.keras.layers.Conv2D(128,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv2_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv2_1'] = tf.keras.layers.Conv2D(128,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv2_1_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv3'] = tf.keras.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv3_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv3_1'] = tf.keras.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv3_1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv3_2'] = tf.keras.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv3_2_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv4'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv4_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv4_1'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv4_1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv4_2'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv4_2_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv5'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv5_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv5_1'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv5_1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv5_2'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv5_2_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['fc1'] = tf.keras.layers.Dense(512,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer)

        self.list_layer['fc1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['fc2'] = tf.keras.layers.Dense(512,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer)
        self.list_layer['fc2_bn'] = tf.keras.layers.BatchNormalization()

        #self.list_layer['fc3'] = tf.keras.layers.Dense(self.num_class,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer))
        self.list_layer['fc3'] = tf.keras.layers.Dense(self.num_class,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer)
        self.list_layer['fc3_bn'] = tf.keras.layers.BatchNormalization()

        self.dropout_conv = tf.keras.layers.Dropout(0.3)
        self.dropout_conv2 = tf.keras.layers.Dropout(0.4)
        self.dropout = tf.keras.layers.Dropout(0.5)


        # remove later
        self.conv1=self.list_layer['conv1']
        self.conv1_bn=self.list_layer['conv1_bn']
        self.conv1_1=self.list_layer['conv1_1']
        self.conv1_1_bn=self.list_layer['conv1_1_bn']
        self.conv2=self.list_layer['conv2']
        self.conv2_bn=self.list_layer['conv2_bn']
        self.conv2_1=self.list_layer['conv2_1']
        self.conv2_1_bn=self.list_layer['conv2_1_bn']
        self.conv3=self.list_layer['conv3']
        self.conv3_bn=self.list_layer['conv3_bn']
        self.conv3_1=self.list_layer['conv3_1']
        self.conv3_1_bn=self.list_layer['conv3_1_bn']
        self.conv3_2=self.list_layer['conv3_2']
        self.conv3_2_bn=self.list_layer['conv3_2_bn']
        self.conv4=self.list_layer['conv4']
        self.conv4_bn=self.list_layer['conv4_bn']
        self.conv4_1=self.list_layer['conv4_1']
        self.conv4_1_bn=self.list_layer['conv4_1_bn']
        self.conv4_2=self.list_layer['conv4_2']
        self.conv4_2_bn=self.list_layer['conv4_2_bn']
        self.conv5=self.list_layer['conv5']
        self.conv5_bn=self.list_layer['conv5_bn']
        self.conv5_1=self.list_layer['conv5_1']
        self.conv5_1_bn=self.list_layer['conv5_1_bn']
        self.conv5_2=self.list_layer['conv5_2']
        self.conv5_2_bn=self.list_layer['conv5_2_bn']
        self.fc1=self.list_layer['fc1']
        self.fc1_bn=self.list_layer['fc1_bn']
        self.fc2=self.list_layer['fc2']
        self.fc2_bn=self.list_layer['fc2_bn']
        self.fc3=self.list_layer['fc3']
        self.fc3_bn=self.list_layer['fc3_bn']

        pooling_type= {
            'max': tf.keras.layers.MaxPooling2D((2,2),(2,2),padding='SAME',data_format=data_format),
            'avg': tf.keras.layers.AveragePooling2D((2,2),(2,2),padding='SAME',data_format=data_format)
        }

        self.pool2d = pooling_type[self.conf.pooling]

        #self.pool2d_arg ='max': tf.nn.max_poolwith_argmax((2,2),(2,2),padding='SAME',data_format=data_format)),

        self.act_relu = tf.nn.relu


        self.in_shape = [self.conf.batch_size]+self._input_shape[1:]

        #self.shape_out_conv1 = util.cal_output_shape_Conv2D(self.data_format,input_shape_one_sample,64,3,1)
        self.shape_out_conv1 = util.cal_output_shape_Conv2D(self.data_format,self.in_shape,64,self.kernel_size,1)
        self.shape_out_conv1_1 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv1,64,self.kernel_size,1)
        self.shape_out_conv1_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv1_1,2,2)

        self.shape_out_conv2 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv1_p,128,self.kernel_size,1)
        self.shape_out_conv2_1 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv2,128,self.kernel_size,1)
        self.shape_out_conv2_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv2_1,2,2)

        self.shape_out_conv3 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv2_p,256,self.kernel_size,1)
        self.shape_out_conv3_1 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv3,256,self.kernel_size,1)
        self.shape_out_conv3_2 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv3_1,256,self.kernel_size,1)
        self.shape_out_conv3_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv3_2,2,2)

        self.shape_out_conv4 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv3_p,512,self.kernel_size,1)
        self.shape_out_conv4_1 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv4,512,self.kernel_size,1)
        self.shape_out_conv4_2 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv4_1,512,self.kernel_size,1)
        self.shape_out_conv4_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv4_2,2,2)

        self.shape_out_conv5 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv4_p,512,self.kernel_size,1)
        self.shape_out_conv5_1 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv5,512,self.kernel_size,1)
        self.shape_out_conv5_2 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv5_1,512,self.kernel_size,1)
        self.shape_out_conv5_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv5_2,2,2)

        self.shape_out_fc1 = tensor_shape.TensorShape([self.conf.batch_size,512]).as_list()
        self.shape_out_fc2 = tensor_shape.TensorShape([self.conf.batch_size,512]).as_list()
        self.shape_out_fc3 = tensor_shape.TensorShape([self.conf.batch_size,self.num_class]).as_list()


        self.dict_shape=collections.OrderedDict()
        self.dict_shape['conv1']=self.shape_out_conv1
        self.dict_shape['conv1_1']=self.shape_out_conv1_1
        self.dict_shape['conv1_p']=self.shape_out_conv1_p
        self.dict_shape['conv2']=self.shape_out_conv2
        self.dict_shape['conv2_1']=self.shape_out_conv2_1
        self.dict_shape['conv2_p']=self.shape_out_conv2_p
        self.dict_shape['conv3']=self.shape_out_conv3
        self.dict_shape['conv3_1']=self.shape_out_conv3_1
        self.dict_shape['conv3_2']=self.shape_out_conv3_2
        self.dict_shape['conv3_p']=self.shape_out_conv3_p
        self.dict_shape['conv4']=self.shape_out_conv4
        self.dict_shape['conv4_1']=self.shape_out_conv4_1
        self.dict_shape['conv4_2']=self.shape_out_conv4_2
        self.dict_shape['conv4_p']=self.shape_out_conv4_p
        self.dict_shape['conv5']=self.shape_out_conv5
        self.dict_shape['conv5_1']=self.shape_out_conv5_1
        self.dict_shape['conv5_2']=self.shape_out_conv5_2
        self.dict_shape['conv5_p']=self.shape_out_conv5_p
        self.dict_shape['fc1']=self.shape_out_fc1
        self.dict_shape['fc2']=self.shape_out_fc2
        self.dict_shape['fc3']=self.shape_out_fc3


        self.dict_shape_one_batch=collections.OrderedDict()
#        self.dict_shape_one_batch['conv1']=[1,]+self.shape_out_conv1[1:]
#        self.dict_shape_one_batch['conv1_1']=[1,]+self.shape_out_conv1_1[1:]
#        self.dict_shape_one_batch['conv1_p']=[1,]+self.shape_out_conv1_p[1:]
#        self.dict_shape_one_batch['conv2']=[1,]+self.shape_out_conv2[1:]
#        self.dict_shape_one_batch['conv2_1']=[1,]+self.shape_out_conv2_1[1:]
#        self.dict_shape_one_batch['conv2_p']=[1,]+self.shape_out_conv2_p[1:]
#        self.dict_shape_one_batch['conv3']=[1,]+self.shape_out_conv3[1:]
#        self.dict_shape_one_batch['conv3_1']=[1,]+self.shape_out_conv3_1[1:]
#        self.dict_shape_one_batch['conv3_2']=[1,]+self.shape_out_conv3_2[1:]
#        self.dict_shape_one_batch['conv3_p']=[1,]+self.shape_out_conv3_p[1:]
#        self.dict_shape_one_batch['conv4']=[1,]+self.shape_out_conv4[1:]
#        self.dict_shape_one_batch['conv4_1']=[1,]+self.shape_out_conv4_1[1:]
#        self.dict_shape_one_batch['conv4_2']=[1,]+self.shape_out_conv4_2[1:]
#        self.dict_shape_one_batch['conv4_p']=[1,]+self.shape_out_conv4_p[1:]
#        self.dict_shape_one_batch['conv5']=[1,]+self.shape_out_conv5[1:]
#        self.dict_shape_one_batch['conv5_1']=[1,]+self.shape_out_conv5_1[1:]
#        self.dict_shape_one_batch['conv5_2']=[1,]+self.shape_out_conv5_2[1:]
#        self.dict_shape_one_batch['conv5_p']=[1,]+self.shape_out_conv5_p[1:]
#        self.dict_shape_one_batch['fc2']=[1,]+self.shape_out_fc2[1:]
#        self.dict_shape_one_batch['fc3']=[1,]+self.shape_out_fc3[1:]
        for l_name in self.layer_name:
            self.dict_shape_one_batch[l_name] = tensor_shape.TensorShape([1,]+self.dict_shape[l_name][1:])


        #
        self.dict_stat_r=collections.OrderedDict()  # read
        self.dict_stat_w=collections.OrderedDict()  # write


        #
        if self.conf.f_entropy:
            self.dict_stat_w['conv1']=np.zeros([self.conf.time_step,]+self.shape_out_conv1[1:])
            self.dict_stat_w['conv1_1']=np.zeros([self.conf.time_step,]+self.shape_out_conv1_1[1:])
            self.dict_stat_w['conv2']=np.zeros([self.conf.time_step,]+self.shape_out_conv2[1:])
            self.dict_stat_w['conv2_1']=np.zeros([self.conf.time_step,]+self.shape_out_conv2_1[1:])
            self.dict_stat_w['conv3']=np.zeros([self.conf.time_step,]+self.shape_out_conv3[1:])
            self.dict_stat_w['conv3_1']=np.zeros([self.conf.time_step,]+self.shape_out_conv3_1[1:])
            self.dict_stat_w['conv3_2']=np.zeros([self.conf.time_step,]+self.shape_out_conv3_2[1:])
            self.dict_stat_w['conv4']=np.zeros([self.conf.time_step,]+self.shape_out_conv4[1:])
            self.dict_stat_w['conv4_1']=np.zeros([self.conf.time_step,]+self.shape_out_conv4_1[1:])
            self.dict_stat_w['conv4_2']=np.zeros([self.conf.time_step,]+self.shape_out_conv4_2[1:])
            self.dict_stat_w['conv5']=np.zeros([self.conf.time_step,]+self.shape_out_conv5[1:])
            self.dict_stat_w['conv5_1']=np.zeros([self.conf.time_step,]+self.shape_out_conv5_1[1:])
            self.dict_stat_w['conv5_2']=np.zeros([self.conf.time_step,]+self.shape_out_conv5_2[1:])
            self.dict_stat_w['fc1']=np.zeros([self.conf.time_step,]+self.shape_out_fc1[1:])
            self.dict_stat_w['fc2']=np.zeros([self.conf.time_step,]+self.shape_out_fc2[1:])
            self.dict_stat_w['fc3']=np.zeros([self.conf.time_step,]+self.shape_out_fc3[1:])

            self.arr_length = [2,3,4,5,8,10]
            self.total_entropy=np.zeros([len(self.arr_length),len(self.layer_name)+1])

        #
        if self.conf.f_write_stat or self.conf.f_comp_act:

            # TODO: check
            #
            self.layer_name_write_stat=[
                #'in',
                #'conv1',
                #'conv1_1',
                #'conv2',
                #'conv2_1',
                #'conv3',
                #'conv3_1',
                #'conv3_2',
                #'conv4',
                #'conv4_1',
                #'conv4_2',
                'conv5',
                'conv5_1',
                'conv5_2',
                'fc1',
                'fc2',
                'fc3',
            ]

            self.f_1st_iter_stat = True

#            self.dict_stat_w['conv1']=np.zeros([1,]+self.shape_out_conv1[1:])
#            self.dict_stat_w['conv1_1']=np.zeros([1,]+self.shape_out_conv1_1[1:])
#            self.dict_stat_w['conv2']=np.zeros([1,]+self.shape_out_conv2[1:])
#            self.dict_stat_w['conv2_1']=np.zeros([1,]+self.shape_out_conv2_1[1:])
#            self.dict_stat_w['conv3']=np.zeros([1,]+self.shape_out_conv3[1:])
#            self.dict_stat_w['conv3_1']=np.zeros([1,]+self.shape_out_conv3_1[1:])
#            self.dict_stat_w['conv3_2']=np.zeros([1,]+self.shape_out_conv3_2[1:])
#            self.dict_stat_w['conv4']=np.zeros([1,]+self.shape_out_conv4[1:])
#            self.dict_stat_w['conv4_1']=np.zeros([1,]+self.shape_out_conv4_1[1:])
#            self.dict_stat_w['conv4_2']=np.zeros([1,]+self.shape_out_conv4_2[1:])
#            self.dict_stat_w['conv5']=np.zeros([1,]+self.shape_out_conv5[1:])
#            self.dict_stat_w['conv5_1']=np.zeros([1,]+self.shape_out_conv5_1[1:])
#            self.dict_stat_w['conv5_2']=np.zeros([1,]+self.shape_out_conv5_2[1:])
#            self.dict_stat_w['fc1']=np.zeros([1,]+self.shape_out_fc1[1:])
#            self.dict_stat_w['fc2']=np.zeros([1,]+self.shape_out_fc2[1:])
#            self.dict_stat_w['fc3']=np.zeros([1,]+self.shape_out_fc3[1:])

            #self.dict_stat_w['fc1']=tf.Variable(initial_value=tf.zeros(self.dict_shape['fc1']),trainable=None)

            for l_name in self.layer_name_write_stat:
                self.dict_stat_w[l_name]=tf.Variable(initial_value=tf.zeros(self.dict_shape[l_name]),trainable=None)


        self.conv_p=collections.OrderedDict()
        self.conv_p['conv1_p']=np.empty(self.dict_shape['conv1_p'],dtype=np.float32)
        self.conv_p['conv2_p']=np.empty(self.dict_shape['conv2_p'],dtype=np.float32)
        self.conv_p['conv3_p']=np.empty(self.dict_shape['conv3_p'],dtype=np.float32)
        self.conv_p['conv4_p']=np.empty(self.dict_shape['conv4_p'],dtype=np.float32)
        self.conv_p['conv5_p']=np.empty(self.dict_shape['conv5_p'],dtype=np.float32)


        #
        if (self.conf.nn_mode=='SNN' and self.conf.f_train_time_const) or (self.conf.nn_mode=='ANN' and self.conf.f_write_stat):
            self.dnn_act_list=collections.OrderedDict()

        # neurons
        if self.conf.nn_mode == 'SNN' or self.conf.f_validation_snn:
            print('Neuron setup')

            #self.input_shape_snn = [1] + self._input_shape[1:]
            self.input_shape_snn = [self.conf.batch_size] + self._input_shape[1:]
            #self.input_shape_snn = self._input_shape

            print('Input shape snn: '+str(self.input_shape_snn))

            #self.conf.n_init_vth = (1.0/np.power(2,1))
            #self.conf.n_init_vth = (1.0/np.power(2,2))
            #self.conf.n_init_vth = (1.0/np.power(2,3))
            #self.conf.n_init_vth = (1.0/np.power(2,4))
            #self.conf.n_init_vth = (1.0/np.power(2,5))
            #self.conf.n_init_vth = (1.0/np.power(2,7))
            #self.conf.n_init_vth = (1.0/np.power(2,8))

            #vth = self.conf.n_init_vth
            #vinit = self.conf.n_init_vinit
            #vrest = self.conf.n_init_vrest
            #time_step = self.conf.time_step
            n_type = self.conf.n_type
            nc = self.conf.neural_coding

            self.list_neuron=collections.OrderedDict()

            #self.list_neuron['in'] = lib_snn.Neuron(self.input_shape_snn,'IN',1,self.conf)
            self.list_neuron['in'] = lib_snn.Neuron(self.input_shape_snn,'IN',1,self.conf,nc,0,'in')


            self.list_neuron['conv1'] = lib_snn.Neuron(self.shape_out_conv1,n_type,self.fanin_conv,self.conf,nc,1,'conv1')
            self.list_neuron['conv1_1'] = lib_snn.Neuron(self.shape_out_conv1_1,n_type,self.fanin_conv,self.conf,nc,2,'conv1_1')


            self.list_neuron['conv2'] = lib_snn.Neuron(self.shape_out_conv2,n_type,self.fanin_conv,self.conf,nc,3,'conv2')
            self.list_neuron['conv2_1'] = lib_snn.Neuron(self.shape_out_conv2_1,n_type,self.fanin_conv,self.conf,nc,4,'conv2_1')

            self.list_neuron['conv3'] = lib_snn.Neuron(self.shape_out_conv3,n_type,self.fanin_conv,self.conf,nc,5,'conv3')
            self.list_neuron['conv3_1'] = lib_snn.Neuron(self.shape_out_conv3_1,n_type,self.fanin_conv,self.conf,nc,6,'conv3_1')
            self.list_neuron['conv3_2'] = lib_snn.Neuron(self.shape_out_conv3_2,n_type,self.fanin_conv,self.conf,nc,7,'conv3_2')

            #nc = 'BURST'
            #self.conf.n_init_vth=0.125

            self.list_neuron['conv4'] = lib_snn.Neuron(self.shape_out_conv4,n_type,self.fanin_conv,self.conf,nc,8,'conv4')
            self.list_neuron['conv4_1'] = lib_snn.Neuron(self.shape_out_conv4_1,n_type,self.fanin_conv,self.conf,nc,9,'conv4_1')
            self.list_neuron['conv4_2'] = lib_snn.Neuron(self.shape_out_conv4_2,n_type,self.fanin_conv,self.conf,nc,10,'conv4_2')

            #self.conf.n_init_vth=0.125/2.0

            self.list_neuron['conv5'] = lib_snn.Neuron(self.shape_out_conv5,n_type,self.fanin_conv,self.conf,nc,11,'conv5')
            self.list_neuron['conv5_1'] = lib_snn.Neuron(self.shape_out_conv5_1,n_type,self.fanin_conv,self.conf,nc,12,'conv5_1')
            self.list_neuron['conv5_2'] = lib_snn.Neuron(self.shape_out_conv5_2,n_type,self.fanin_conv,self.conf,nc,13,'conv5_2')

            self.list_neuron['fc1'] = lib_snn.Neuron(self.shape_out_fc1,n_type,512,self.conf,nc,14,'fc1')
            self.list_neuron['fc2'] = lib_snn.Neuron(self.shape_out_fc2,n_type,512,self.conf,nc,15,'fc2')
            #self.list_neuron['fc3'] = lib_snn.Neuron(self.shape_out_fc3,n_type,512,self.conf)
            self.list_neuron['fc3'] = lib_snn.Neuron(self.shape_out_fc3,'OUT',512,self.conf,nc,16,'fc3')


            # modify later
            self.n_in = self.list_neuron['in']

            self.n_conv1 = self.list_neuron['conv1']
            self.n_conv1_1 = self.list_neuron['conv1_1']
            #self.n_conv1_1 = tf.contrib.eager.defun(self.list_neuron['conv1_1'])

            self.n_conv2 = self.list_neuron['conv2']
            self.n_conv2_1 = self.list_neuron['conv2_1']

            self.n_conv3 = self.list_neuron['conv3']
            self.n_conv3_1 = self.list_neuron['conv3_1']
            self.n_conv3_2 = self.list_neuron['conv3_2']

            self.n_conv4 = self.list_neuron['conv4']
            self.n_conv4_1 = self.list_neuron['conv4_1']
            self.n_conv4_2 = self.list_neuron['conv4_2']

            self.n_conv5 = self.list_neuron['conv5']
            self.n_conv5_1 = self.list_neuron['conv5_1']
            self.n_conv5_2 = self.list_neuron['conv5_2']

            self.n_fc1 = self.list_neuron['fc1']
            self.n_fc2 = self.list_neuron['fc2']
            self.n_fc3 = self.list_neuron['fc3']

            #
            self.snn_output_layer = self.n_fc3

            self.snn_output = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_layer.dim)),dtype=tf.float32,trainable=False)
            #
            if self.conf.f_train_time_const:
                self.dnn_act_list=collections.OrderedDict()


            if self.conf.neural_coding=='TEMPORAL' and self.conf.f_load_time_const:
            #if self.conf.neural_coding=='TEMPORAL':

                file_name = self.conf.time_const_init_file_name
                file_name = file_name + '/'+self.conf.model_name
                #if self.conf.f_tc_based:
                if False:
                    file_name = file_name+'/tc-'+str(self.conf.tc)+'_tw-'+str(self.conf.n_tau_time_window)+'_tau_itr-'+str(self.conf.time_const_num_trained_data)
                else:
                    file_name = file_name+'/tc-'+str(self.conf.tc)+'_tw-'+str(self.conf.time_window)+'_itr-'+str(self.conf.time_const_num_trained_data)

                if conf.f_train_time_const_outlier:
                    file_name+="_outlier"

                print('load trained time constant: file_name: {:s}'.format(file_name))

                file = open(file_name,'r')
                lines = csv.reader(file)

                for line in lines:
                    if not line:
                        continue

                    print(line)

                    type = line[0]
                    name = line[1]
                    val = float(line[2])

                    if (type=='tc') :

                        self.list_neuron[name].set_time_const_init_fire(val)

                        if not ('in' in name):
                            self.list_neuron[name].set_time_const_init_integ(self.list_neuron[name_prev].time_const_init_fire)

                        name_prev = name

                    elif (type=='td'):

                        self.list_neuron[name].set_time_delay_init_fire(val)

                        if not ('in' in name):
                            self.list_neuron[name].set_time_delay_init_integ(self.list_neuron[name_prev].time_delay_init_fire)

                        name_prev = name

                    else:
                        print("not supported temporal coding type")
                        assert(False)


                file.close()


            # SNN training w/ TTFS coding
            # surrogate mode training


            #
            self.spike_count = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.n_fc3.dim)),dtype=tf.float32,trainable=False)
            #self.spike_count = tf.contrib.eager.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.n_fc3.dim)),dtype=tf.float32,trainable=False)
        #
        self.cmap=matplotlib.cm.get_cmap('viridis')
        #self.normalize=matplotlib.colors.Normalize(vmin=min(self.n_fc3.vmem),vmax=max(self.n_fc3.vmem))

        # pruning
        if self.conf.f_pruning_channel:
            self.idx_pruning_channel = collections.OrderedDict()
            self.f_idx_pruning_channel = collections.OrderedDict()
            self.conv_pruning_channel = collections.OrderedDict()
            self.kernel_pruning_channel = collections.OrderedDict()
            self.th_idx_pruning_channel = 0.5


        # surrogate DNN model for SNN training w/ TTFS coding
        if self.conf.f_surrogate_training_model:

            self.list_tk=collections.OrderedDict()

            self.init_tc = self.conf.tc

            init_tc = self.init_tc
            init_act_target_range=0.5
            #init_act_target_range_in=0.5
            #init_act_target_range_in=5.0
            init_act_target_range_in=self.conf.td

            #init_td=init_tc*np.log(init_act_target_range)
            self.init_td_in=init_tc*np.log(init_act_target_range_in)
            init_td_in = self.init_td_in

            # TODO: removed
            init_ta=10.0

            #
            self.list_tk['in'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv1'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv1_1'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv2'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv2_1'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv3'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv3_1'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv3_2'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv4'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv4_1'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv4_2'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv5'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv5_1'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['conv5_2'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['fc1'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            self.list_tk['fc2'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window,self.conf)
            #self.list_tk['fc3'] = lib_snn.Temporal_kernel([], [], init_tc, init_td_in, init_ta,self.conf.time_window)



            #
            self.enc_st_n_tw = self.conf.enc_st_n_tw

            #self.enc_st_target_end = self.conf.time_window*10
            self.enc_st_target_end = self.conf.time_window*self.enc_st_n_tw

            # TODO: parameterize with other file (e.g., train_snn.py)
            #f_loss_dist = True

            # TODO: function
            self.f_loss_enc_spike_dist = False
            self.f_loss_enc_spike_bn = False
            self.f_loss_enc_spike_bn_only = False   # loss aginst only BN parameters
            self.f_loss_enc_spike_bn_only_new = False   # debug version
            self.f_loss_enc_spike_bn_only_new_2 = False   # squred
            self.f_loss_enc_spike_bn_only_new_lin = False   # linear approx
            self.f_loss_enc_spike_bn_only_new_new = False   # debug version - new new

            if self.conf.d_loss_enc_spike == 'bn':
                self.f_loss_enc_spike_dist = False
                self.f_loss_enc_spike_bn = True
                self.f_loss_enc_spike_bn_only = False
            elif self.conf.d_loss_enc_spike == 'bno':
                self.f_loss_enc_spike_dist = False
                self.f_loss_enc_spike_bn = True
                self.f_loss_enc_spike_bn_only = True
            elif self.conf.d_loss_enc_spike == 'bnon':
                self.f_loss_enc_spike_dist = False
                self.f_loss_enc_spike_bn = True
                self.f_loss_enc_spike_bn_only = True
                self.f_loss_enc_spike_bn_only_new = True
            elif self.conf.d_loss_enc_spike == 'bnon2':
                self.f_loss_enc_spike_dist = False
                self.f_loss_enc_spike_bn = True
                self.f_loss_enc_spike_bn_only = True
                self.f_loss_enc_spike_bn_only_new_2 = True
            elif self.conf.d_loss_enc_spike == 'bnonl':
                self.f_loss_enc_spike_dist = False
                self.f_loss_enc_spike_bn = True
                self.f_loss_enc_spike_bn_only = True
                self.f_loss_enc_spike_bn_only_new_lin = True
            elif self.conf.d_loss_enc_spike == 'bnonn':
                self.f_loss_enc_spike_dist = False
                self.f_loss_enc_spike_bn = True
                self.f_loss_enc_spike_bn_only = True
                self.f_loss_enc_spike_bn_only_new_new = True
            else:
                self.f_loss_enc_spikes_dist = self.conf.f_loss_enc_spike
                self.f_loss_enc_spike_bn = False
                self.f_loss_enc_spike_bn_only = False

            # TODO: function
            if self.f_loss_enc_spike_dist:

                #alpha = 0.1
                #beta = 0.9

                alpha = self.conf.beta_dist_a
                beta = self.conf.beta_dist_b


                if 'b' in self.conf.d_loss_enc_spike:
                    self.dist = tfd.Beta(alpha,beta)
                elif 'g' in self.conf.d_loss_enc_spike:
                    self.dist = tfd.Gamma(alpha,beta)
                elif 'h' in self.conf.d_loss_enc_spike:
                    self.dist = tfd.Horseshoe(alpha)
                else:
                    assert False, 'not supported distribution {}'.format(self.conf.d_loss_enc_spike)

                self.dist_beta_sample = collections.OrderedDict()

            #
            self.train_tk_strategy = self.conf.train_tk_strategy.split('-')[0]
            if self.train_tk_strategy != 'N':
                self.train_tk_strategy_coeff = (int)(self.conf.train_tk_strategy.split('-')[1])
                self.train_tk_strategy_coeff_x3 = self.train_tk_strategy_coeff*3

            #
            self.t_train_tk_reg = self.conf.t_train_tk_reg.split('-')[0]
            self.t_train_tk_reg_mode = self.conf.t_train_tk_reg.split('-')[1]


        # model loading V2
        self.load_layer_ann_checkpoint = self.load_layer_ann_checkpoint_func()

    #
    def dist_beta_sample_func(self):
        for l_name, tk in self.list_tk.items():
            enc_st = tf.reshape(tk.out_enc, [-1])

            samples = self.dist.sample(enc_st.shape)
            #samples = tf.divide(samples,tf.reduce_max(samples))
            samples = tf.multiply(samples,self.enc_st_target_end)
            self.dist_beta_sample[l_name] = tf.histogram_fixed_width(samples, [0,self.enc_st_target_end], nbins=self.enc_st_target_end)

    #
    def load_layer_ann_checkpoint_func(self):
        if self.conf.f_surrogate_training_model:
            load_layer_ann_checkpoint = tf.train.Checkpoint(
                conv1=self.conv1,
                conv1_bn=self.conv1_bn,
                conv1_1=self.conv1_1,
                conv1_1_bn=self.conv1_1_bn,
                conv2=self.conv2,
                conv2_bn=self.conv2_bn,
                conv2_1=self.conv2_1,
                conv2_1_bn=self.conv2_1_bn,
                conv3=self.conv3,
                conv3_bn=self.conv3_bn,
                conv3_1=self.conv3_1,
                conv3_1_bn=self.conv3_1_bn,
                conv3_2=self.conv3_2,
                conv3_2_bn=self.conv3_2_bn,
                conv4=self.conv4,
                conv4_bn=self.conv4_bn,
                conv4_1=self.conv4_1,
                conv4_1_bn=self.conv4_1_bn,
                conv4_2=self.conv4_2,
                conv4_2_bn=self.conv4_2_bn,
                conv5=self.conv5,
                conv5_bn=self.conv5_bn,
                conv5_1=self.conv5_1,
                conv5_1_bn=self.conv5_1_bn,
                conv5_2=self.conv5_2,
                conv5_2_bn=self.conv5_2_bn,
                fc1=self.fc1,
                fc1_bn=self.fc1_bn,
                fc2=self.fc2,
                fc2_bn=self.fc2_bn,
                fc3=self.fc3,
                fc3_bn=self.fc3_bn,
                list_tk=self.list_tk
            )
        else:
            load_layer_ann_checkpoint = tf.train.Checkpoint(
                conv1=self.conv1,
                conv1_bn=self.conv1_bn,
                conv1_1=self.conv1_1,
                conv1_1_bn=self.conv1_1_bn,
                conv2=self.conv2,
                conv2_bn=self.conv2_bn,
                conv2_1=self.conv2_1,
                conv2_1_bn=self.conv2_1_bn,
                conv3=self.conv3,
                conv3_bn=self.conv3_bn,
                conv3_1=self.conv3_1,
                conv3_1_bn=self.conv3_1_bn,
                conv3_2=self.conv3_2,
                conv3_2_bn=self.conv3_2_bn,
                conv4=self.conv4,
                conv4_bn=self.conv4_bn,
                conv4_1=self.conv4_1,
                conv4_1_bn=self.conv4_1_bn,
                conv4_2=self.conv4_2,
                conv4_2_bn=self.conv4_2_bn,
                conv5=self.conv5,
                conv5_bn=self.conv5_bn,
                conv5_1=self.conv5_1,
                conv5_1_bn=self.conv5_1_bn,
                conv5_2=self.conv5_2,
                conv5_2_bn=self.conv5_2_bn,
                fc1=self.fc1,
                fc1_bn=self.fc1_bn,
                fc2=self.fc2,
                fc2_bn=self.fc2_bn,
                fc3=self.fc3,
                fc3_bn=self.fc3_bn
            )

        return load_layer_ann_checkpoint



    #
    def call_legacy(self, inputs, f_training, epoch=-1, f_val_snn=False):


        if self.f_load_model_done:
            if (self.conf.nn_mode=='SNN' and self.conf.f_pruning_channel==True):

                tw_sampling = 20
                ret_val = self.call_snn(inputs,f_training,tw_sampling,epoch)
                ret_val = self.run_mode[self.conf.nn_mode](inputs,f_training,self.conf.time_step,epoch)
            else:
                if (self.conf.nn_mode=='SNN' and self.conf.neural_coding=="TEMPORAL"):
                    # inference - temporal coding
                    #ret_val = self.call_snn_temporal(inputs,f_training,self.conf.time_step)

                    #
                    self.run_mode['ANN'](inputs,f_training,self.conf.time_step,epoch)

                    ret_val = self.run_mode[self.conf.nn_mode](inputs,f_training,self.conf.time_step,epoch)

                    # training time constant
                    if self.conf.f_train_time_const:
                        self.train_time_const()

                else:
                    # inference - rate, phase burst coding
                    ret_val = self.run_mode[self.conf.nn_mode](inputs,f_training,self.conf.time_step,epoch)
        else:

            if self.conf.nn_mode=='SNN' and self.conf.f_surrogate_training_model:
                ret_val = self.call_ann_surrogate_training(inputs,f_training,self.conf.time_step,epoch)

            ret_val = self.run_mode_load_model[self.conf.nn_mode](inputs,f_training,self.conf.time_step,epoch)
            self.f_load_model_done=True
        return ret_val



    ###########################################################################
    ## processing
    ###########################################################################

    def reset_per_run_snn(self):
        self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])
        self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])


    def reset_per_sample_snn(self):
        self.reset_neuron()
        #self.snn_output = np.zeros((self.num_accuracy_time_point,)+self.n_fc1.get_spike_count().numpy().shape)
        self.snn_output.assign(tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_layer.dim)))
        self.count_accuracy_time_point=0


    #def reset_neuron(self):
        #self.n_in.reset()
        #self.n_conv1.reset()
        #self.n_conv2.reset()
        #self.n_fc1.reset()


    def preproc(self, inputs, f_training, f_val_snn=False):
        preproc_sel= {
            'ANN': self.preproc_ann,
            'SNN': self.preproc_snn
        }

        if f_val_snn:
            self.preproc_snn(inputs,f_training)
        else:
            preproc_sel[self.conf.nn_mode](inputs, f_training)



    def preproc_snn(self,inputs,f_training):
        # reset for sample
        self.reset_per_sample_snn()

        if self.f_done_preproc == False:
            self.f_done_preproc = True
            #self.print_model_conf()
            self.reset_per_run_snn()
            self.preproc_ann_to_snn()

            # snn validation mode
            if self.conf.f_surrogate_training_model:
                self.load_temporal_kernel_para()

        if self.conf.f_comp_act:
            self.save_ann_act(inputs,f_training)

        # gradient-based optimization of TC and td in temporal coding (TTFS)
        if (self.conf.neural_coding=="TEMPORAL" and self.conf.f_train_time_const):
            self.call_ann(inputs,f_training)

    def preproc_ann(self, inputs, f_training):
        if self.f_done_preproc == False:
            self.f_done_preproc=True
            self.print_model_conf()
            self.preproc_ann_norm()

            # surrogate DNN model for training SNN with temporal information
            if self.conf.f_surrogate_training_model:
                self.preproc_surrogate_training_model()

        self.f_skip_bn=self.conf.f_fused_bn


    def preproc_ann_to_snn(self):
        if self.conf.verbose:
            print('preprocessing: ANN to SNN')

        if self.conf.f_fused_bn or ((self.conf.nn_mode=='ANN')and(self.conf.f_validation_snn)):
            self.fused_bn()


        #print(np.max(self.list_layer.values()[0].kernel))
        #self.temporal_norm()
        #print(np.max(self.list_layer.values()[0].kernel))


        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()

        #if self.conf.f_comp_act:
        #    self.load_act_after_w_norm()

        #self.print_act_after_w_norm()

    def preproc_surrogate_training_model(self):
        if self.f_loss_enc_spike_dist:
            self.dist_beta_sample_func()



    #
    # TODO: input neuron ?
    def load_temporal_kernel_para(self):
        if self.conf.verbose:
            print('preprocessing: load_temporal_kernel_para')

        for l_name in self.layer_name:

            if l_name != self.layer_name[-1]:
                self.list_neuron[l_name].set_time_const_fire(self.list_tk[l_name].tc)
                self.list_neuron[l_name].set_time_delay_fire(self.list_tk[l_name].td)

            if not ('in' in l_name):
                #self.list_neuron[l_name].set_time_const_integ(self.list_tk[l_name_prev].tc_dec)
                #self.list_neuron[l_name].set_time_delay_integ(self.list_tk[l_name_prev].td_dec)
                self.list_neuron[l_name].set_time_const_integ(self.list_tk[l_name_prev].tc)
                self.list_neuron[l_name].set_time_delay_integ(self.list_tk[l_name_prev].td)

            l_name_prev = l_name

        # encoding decoding kernerl seperate
        #assert(False)




    #
    def preproc_ann_norm(self):
        if self.conf.f_fused_bn:
            self.fused_bn()

        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()

    #
    def call(self, inputs, f_training, epoch=-1, f_val_snn=False):


        if self.f_load_model_done:

            # pre-processing
            self.preproc(inputs,f_training,f_val_snn)

            # run
            if (self.conf.nn_mode=='SNN' and self.conf.neural_coding=="TEMPORAL" and self.conf.f_train_time_const):
                # inference - temporal coding
                #ret_val = self.call_snn_temporal(inputs,f_training,self.conf.time_step)

                #
                if self.conf.f_train_time_const:
                    self.run_mode['ANN'](inputs,f_training,self.conf.time_step,epoch)

                #
                ret_val = self.run_mode[self.conf.nn_mode](inputs,f_training,self.conf.time_step,epoch)

                # training time constant
                if self.conf.f_train_time_const:
                    self.train_time_const()
            else:
                # inference - rate, phase burst coding
                if f_val_snn:
                    ret_val = self.call_snn(inputs,f_training,self.conf.time_step,epoch)
                else:
                    ret_val = self.run_mode[self.conf.nn_mode](inputs,f_training,self.conf.time_step,epoch)


            # post-processing
            #self.postproc(f_val_snn)
        else:

            if self.conf.nn_mode=='SNN' and self.conf.f_surrogate_training_model:
                ret_val = self.call_ann_surrogate_training(inputs,False,self.conf.time_step,epoch)

            # validation on SNN
            if self.conf.en_train and self.conf.f_validation_snn:
                ret_val = self.call_snn(inputs,False,1,0)

            #ret_val = self.run_mode_load_model[self.conf.nn_mode](inputs,f_training,self.conf.time_step,epoch)
            ret_val = self.run_mode_load_model[self.conf.nn_mode](inputs,False,self.conf.time_step,epoch)

            self.f_load_model_done=True

        return ret_val


    #
    def fused_bn(self):
        print('fused_bn')
        self.conv_bn_fused(self.conv1, self.conv1_bn, 1.0)
        self.conv_bn_fused(self.conv1_1, self.conv1_1_bn, 1.0)
        self.conv_bn_fused(self.conv2, self.conv2_bn, 1.0)
        self.conv_bn_fused(self.conv2_1, self.conv2_1_bn, 1.0)
        self.conv_bn_fused(self.conv3, self.conv3_bn, 1.0)
        self.conv_bn_fused(self.conv3_1, self.conv3_1_bn, 1.0)
        self.conv_bn_fused(self.conv3_2, self.conv3_2_bn, 1.0)
        self.conv_bn_fused(self.conv4, self.conv4_bn, 1.0)
        self.conv_bn_fused(self.conv4_1, self.conv4_1_bn, 1.0)
        self.conv_bn_fused(self.conv4_2, self.conv4_2_bn, 1.0)
        self.conv_bn_fused(self.conv5, self.conv5_bn, 1.0)
        self.conv_bn_fused(self.conv5_1, self.conv5_1_bn, 1.0)
        self.conv_bn_fused(self.conv5_2, self.conv5_2_bn, 1.0)
        self.fc_bn_fused(self.fc1, self.fc1_bn, 1.0)
        self.fc_bn_fused(self.fc2, self.fc2_bn, 1.0)
        if ('bn' in self.conf.model_name) or ('ro' in self.conf.model_name):
            self.fc_bn_fused(self.fc3, self.fc3_bn, 1.0)

    def defused_bn(self):
        #print('defused_bn')
        self.conv_bn_defused(self.conv1, self.conv1_bn, 1.0)
        self.conv_bn_defused(self.conv1_1, self.conv1_1_bn, 1.0)
        self.conv_bn_defused(self.conv2, self.conv2_bn, 1.0)
        self.conv_bn_defused(self.conv2_1, self.conv2_1_bn, 1.0)
        self.conv_bn_defused(self.conv3, self.conv3_bn, 1.0)
        self.conv_bn_defused(self.conv3_1, self.conv3_1_bn, 1.0)
        self.conv_bn_defused(self.conv3_2, self.conv3_2_bn, 1.0)
        self.conv_bn_defused(self.conv4, self.conv4_bn, 1.0)
        self.conv_bn_defused(self.conv4_1, self.conv4_1_bn, 1.0)
        self.conv_bn_defused(self.conv4_2, self.conv4_2_bn, 1.0)
        self.conv_bn_defused(self.conv5, self.conv5_bn, 1.0)
        self.conv_bn_defused(self.conv5_1, self.conv5_1_bn, 1.0)
        self.conv_bn_defused(self.conv5_2, self.conv5_2_bn, 1.0)
        self.fc_bn_defused(self.fc1, self.fc1_bn, 1.0)
        self.fc_bn_defused(self.fc2, self.fc2_bn, 1.0)
        if ('bn' in self.conf.model_name) or ('ro' in self.conf.model_name):
            self.fc_bn_defused(self.fc3, self.fc3_bn, 1.0)




    #
    def w_norm_layer_wise(self):
        f_norm=np.max
        #f_norm=np.mean

        for idx_l, l in enumerate(self.layer_name):
            if idx_l==0:
                self.norm[l]=f_norm(self.dict_stat_r[l])
            else:
                self.norm[l]=f_norm(list(self.dict_stat_r.values())[idx_l])/f_norm(list(self.dict_stat_r.values())[idx_l-1])

            self.norm_b[l]=f_norm(self.dict_stat_r[l])

        if self.conf.f_vth_conp:
            for idx_l, l in enumerate(self.layer_name):
                #self.list_neuron[l].set_vth(np.broadcast_to(self.conf.n_init_vth*1.0 + 0.1*self.dict_stat_r[l]/f_norm(self.dict_stat_r[l]),self.dict_shape[l]))
                self.list_neuron[l].set_vth(np.broadcast_to(self.dict_stat_r[l]/f_norm(self.dict_stat_r[l]),self.dict_shape[l]))
                #self.list_neuron[l].set_vth(np.broadcast_to(self.dict_stat_r[l]/np.broadcast_to(f_norm(self.dict_stat_r[l]),self.dict_stat_r[l].shape)   ,self.dict_shape[l]))

        #self.print_act_d()
        # print
        for k, v in self.norm.items():
            print(k +': '+str(v))

        for k, v in self.norm_b.items():
            print(k +': '+str(v))

        if self.conf.noise_en:
            if self.conf.noise_robust_en:
                #layer_const = 0.55  # noise del 0.0
                #layer_const = 0.50648  # noise del 0.0
                #layer_const = 0.65 # noise del 0.0
                #layer_const = 0.55 # noise del 0.0
                #layer_const = 0.6225 # noise del 0.0
                #layer_const = 0.50648  # noise del 0.0, n: 2
                #layer_const = 0.45505  # noise del 0.0  n: 3
                #layer_const = 0.42866  # noise del 0.0  n: 4
                #layer_const = 0.41409  # noise del 0.0  n: 5
                #layer_const = 0.40572  # noise del 0.0  n: 6
                #layer_const = 0.40081  # noise del 0.0  n: 7
                #layer_const = 0.45505*1.2 # noise del 0.0 - n 3
                #layer_const = 1.0  # noise del 0.0
                #layer_const = 0.55  # noise del 0.01
                #layer_const = 0.6  # noise del 0.1
                #layer_const = 0.65  # noise del 0.2
                #layer_const = 0.78   # noise del 0.4
                #layer_const = 0.7   # noise del 0.6
                #layer_const=1.0


                layer_const = 1.0
                bias_const = 1.0

                if self.conf.neural_coding == 'TEMPORAL':
                    if self.conf.noise_robust_spike_num==0:
                        layer_const = 1.0
                    elif self.conf.noise_robust_spike_num==1:
                        layer_const = 0.50648
                    elif self.conf.noise_robust_spike_num==2:
                        layer_const = 0.50648
                    elif self.conf.noise_robust_spike_num==3:
                        layer_const = 0.45505
                    elif self.conf.noise_robust_spike_num==4:
                        layer_const = 0.42866
                    elif self.conf.noise_robust_spike_num==5:
                        layer_const = 0.41409
                    elif self.conf.noise_robust_spike_num==6:
                        layer_const = 0.40572
                    elif self.conf.noise_robust_spike_num==7:
                        layer_const = 0.40081
                    elif self.conf.noise_robust_spike_num==10:
                        layer_const = 0.39508
                    elif self.conf.noise_robust_spike_num==15:
                        layer_const = 0.39360
                    elif self.conf.noise_robust_spike_num==20:
                        layer_const = 0.39348
                    else:
                        assert False
                else:
                    if self.conf.noise_robust_spike_num==0:
                        layer_const = 1.0
                    else:
                        assert False

                #layer_const = 1.0

                # compenstation - p
                if self.conf.noise_robust_comp_pr_en:
                    if self.conf.noise_type=="DEL":
                        layer_const = layer_const / (1.0-self.conf.noise_pr)
                    elif self.conf.noise_type=="JIT" or self.conf.noise_type=="JIT-A" or self.conf.noise_type=="SYN":
                        layer_const = layer_const
                        #layer_const = layer_const / (1.0-self.conf.noise_pr/4.0)
                    else:
                        assert False
            else:
                layer_const = 1.0
                bias_const = 1.0



            if self.conf.input_spike_mode=='REAL':
                self.conv1.kernel = self.conv1.kernel/self.norm['conv1']
                self.conv1.bias = self.conv1.bias/self.norm_b['conv1']
                self.conv1_1.kernel = self.conv1_1.kernel/self.norm['conv1_1']
                self.conv1_1.bias = self.conv1_1.bias/self.norm_b['conv1_1']
            else:
                self.conv1.kernel = self.conv1.kernel/self.norm['conv1']*layer_const
                self.conv1.bias = self.conv1.bias/self.norm_b['conv1']*bias_const
                self.conv1_1.kernel = self.conv1_1.kernel/self.norm['conv1_1']*layer_const
                self.conv1_1.bias = self.conv1_1.bias/self.norm_b['conv1_1']*bias_const


#            if self.conf.noise_type=="SYN":
#                rand_2 = tf.random.normal(shape=self.conv2.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_2_1 = tf.random.normal(shape=self.conv2_1.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_3 = tf.random.normal(shape=self.conv3.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_3_1 = tf.random.normal(shape=self.conv3_1.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_3_2 = tf.random.normal(shape=self.conv3_2.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_4 = tf.random.normal(shape=self.conv4.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_4_1 = tf.random.normal(shape=self.conv4_1.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_4_2 = tf.random.normal(shape=self.conv4_2.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_5 = tf.random.normal(shape=self.conv5.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_5_1 = tf.random.normal(shape=self.conv5_1.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_5_2 = tf.random.normal(shape=self.conv5_2.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_fc1 = tf.random.normal(shape=self.fc1.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_fc2 = tf.random.normal(shape=self.fc2.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#                rand_fc3 = tf.random.normal(shape=self.fc3.kernel.shape,mean=0.0,stddev=self.conf.noise_pr)
#
#                noise_w_2 = tf.add(1.0,rand_2)
#                noise_w_2_1 = tf.add(1.0,rand_2_1)
#                noise_w_3 = tf.add(1.0,rand_3)
#                noise_w_3_1 = tf.add(1.0,rand_3_1)
#                noise_w_3_2 = tf.add(1.0,rand_3_2)
#                noise_w_4 = tf.add(1.0,rand_4)
#                noise_w_4_1 = tf.add(1.0,rand_4_1)
#                noise_w_4_2 = tf.add(1.0,rand_4_2)
#                noise_w_5 = tf.add(1.0,rand_5)
#                noise_w_5_1 = tf.add(1.0,rand_5_1)
#                noise_w_5_2 = tf.add(1.0,rand_5_2)
#                noise_w_fc1 = tf.add(1.0,rand_fc1)
#                noise_w_fc2 = tf.add(1.0,rand_fc2)
#                noise_w_fc3 = tf.add(1.0,rand_fc3)
#            else:
#                noise_w_2 = 1.0
#                noise_w_2_1 = 1.0
#                noise_w_3 = 1.0
#                noise_w_3_1 = 1.0
#                noise_w_3_2 = 1.0
#                noise_w_4 = 1.0
#                noise_w_4_1 = 1.0
#                noise_w_4_2 = 1.0
#                noise_w_5 = 1.0
#                noise_w_5_1 = 1.0
#                noise_w_5_2 = 1.0
#                noise_w_fc1 = 1.0
#                noise_w_fc2 = 1.0
#                noise_w_fc3 = 1.0

        else:
            layer_const = 1.0
            bias_const = 1.0

            noise_w_2 = 1.0
            noise_w_2_1 = 1.0
            noise_w_3 = 1.0
            noise_w_3_1 = 1.0
            noise_w_3_2 = 1.0
            noise_w_4 = 1.0
            noise_w_4_1 = 1.0
            noise_w_4_2 = 1.0
            noise_w_5 = 1.0
            noise_w_5_1 = 1.0
            noise_w_5_2 = 1.0
            noise_w_fc1 = 1.0
            noise_w_fc2 = 1.0
            noise_w_fc3 = 1.0


        depth_const = 1.0
        self.conv2.kernel = self.conv2.kernel/self.norm['conv2']*layer_const*depth_const*noise_w_2
        self.conv2.bias = self.conv2.bias/self.norm_b['conv2']*bias_const
        self.conv2_1.kernel = self.conv2_1.kernel/self.norm['conv2_1']*layer_const*depth_const*noise_w_2_1
        self.conv2_1.bias = self.conv2_1.bias/self.norm_b['conv2_1']*bias_const

        depth_const = 1.0
        self.conv3.kernel = self.conv3.kernel/self.norm['conv3']*layer_const*depth_const*noise_w_3
        self.conv3.bias = self.conv3.bias/self.norm_b['conv3']*bias_const
        self.conv3_1.kernel = self.conv3_1.kernel/self.norm['conv3_1']*layer_const*depth_const*noise_w_3_1
        self.conv3_1.bias = self.conv3_1.bias/self.norm_b['conv3_1']*bias_const
        self.conv3_2.kernel = self.conv3_2.kernel/self.norm['conv3_2']*layer_const*depth_const*noise_w_3_2
        self.conv3_2.bias = self.conv3_2.bias/self.norm_b['conv3_2']*bias_const

        depth_const = 1.0
        self.conv4.kernel = self.conv4.kernel/self.norm['conv4']*layer_const*depth_const*noise_w_4
        self.conv4.bias = self.conv4.bias/self.norm_b['conv4']*bias_const
        self.conv4_1.kernel = self.conv4_1.kernel/self.norm['conv4_1']*layer_const*depth_const*noise_w_4_1
        self.conv4_1.bias = self.conv4_1.bias/self.norm_b['conv4_1']*bias_const
        self.conv4_2.kernel = self.conv4_2.kernel/self.norm['conv4_2']*layer_const*depth_const*noise_w_4_2
        self.conv4_2.bias = self.conv4_2.bias/self.norm_b['conv4_2']*bias_const

        depth_const = 1.0
        self.conv5.kernel = self.conv5.kernel/self.norm['conv5']*layer_const*depth_const*noise_w_5
        self.conv5.bias = self.conv5.bias/self.norm_b['conv5']*bias_const
        self.conv5_1.kernel = self.conv5_1.kernel/self.norm['conv5_1']*layer_const*depth_const*noise_w_5_1
        self.conv5_1.bias = self.conv5_1.bias/self.norm_b['conv5_1']*bias_const
        self.conv5_2.kernel = self.conv5_2.kernel/self.norm['conv5_2']*layer_const*depth_const*noise_w_5_2
        self.conv5_2.bias = self.conv5_2.bias/self.norm_b['conv5_2']*bias_const

        depth_const = 1.0
        self.fc1.kernel = self.fc1.kernel/self.norm['fc1']*layer_const*depth_const*noise_w_fc1
        self.fc1.bias = self.fc1.bias/self.norm_b['fc1']*bias_const
        self.fc2.kernel = self.fc2.kernel/self.norm['fc2']*layer_const*depth_const*noise_w_fc2
        self.fc2.bias = self.fc2.bias/self.norm_b['fc2']*bias_const
        self.fc3.kernel = self.fc3.kernel/self.norm['fc3']*layer_const*depth_const*noise_w_fc3
        self.fc3.bias = self.fc3.bias/self.norm_b['fc3']*bias_const

    #
    def data_based_w_norm(self):
        #stat_file='./stat/dist_act_trainset_'+self.conf.model_name
        #stat_file='./stat/dist_act_neuron_trainset_'+self.conf.model_name
        #stat_file='./stat/act_n_trainset_test_'+self.conf.model_name

        # TODO: check
        #f_new = False
        f_new = True
        if self.conf.model_name=='vgg_cifar100_ro_0':
            f_new = False

        if f_new:
            path_stat=self.conf.path_stat
        else:
            path_stat='./stat/'

        f_name_stat_pre=self.conf.prefix_stat

        #f_name_stat='act_n_train'

        #stat_conf=['max','mean','max_999','max_99','max_98']

        f_stat=collections.OrderedDict()
        r_stat=collections.OrderedDict()

        # choose one
        #stat='max'
        #stat='mean'
        stat='max_999'
        #stat='max_99'
        #stat='max_98'
        #stat='max_95'
        #stat='max_90'

        for idx_l, l in enumerate(self.layer_name):
            key=l+'_'+stat

            f_name_stat = f_name_stat_pre+'_'+key

            # TODO: check
            if f_new:
                f_name=os.path.join(path_stat,f_name_stat)
                f_stat[key]=open(f_name,'r')
            else:
                f_stat[key]=open(path_stat+f_name_stat+'_'+self.conf.model_name,'r')


            #print(f_name)
            r_stat[key]=csv.reader(f_stat[key])

            for row in r_stat[key]:
                self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.dict_shape[l][1:])

            #if self.conf.f_ws:
                #self.dict_stat_r['conv1']=self.dict_stat_r['conv1']/(1-1/np.power(2,8))

            print(np.shape(self.dict_stat_r[l]))


        self.w_norm_layer_wise()



    #
    def load_act_after_w_norm(self):
        #path_stat='./stat/'
        path_stat=self.conf.path_stat
        #f_name_stat='act_n_train_after_w_norm_max_999'

        f_stat=collections.OrderedDict()
        r_stat=collections.OrderedDict()

        # choose one
        #stat='max'
        stat='max_999'
        #stat='mean'
        #stat='min'
        #stat='max_75'
        #stat='max_25'

        f_name_stat_pre=self.conf.prefix_stat

        for idx_l, l in enumerate(self.layer_name):
            key=l+'_'+stat

            f_name_stat = f_name_stat_pre+'_'+key

            f_name=os.path.join(path_stat,f_name_stat)
            f_stat[key]=open(f_name,'r')
            #f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'r')
            r_stat[key]=csv.reader(f_stat[key])

            for row in r_stat[key]:
                self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.dict_shape[l][1:])

                #print(self.dict_stat_r[l])


    def print_act_after_w_norm(self):
        self.load_act_after_w_norm()

        self.print_act_d()


    def temporal_norm(self):
        print('Temporal normalization')
        for key, value in self.list_layer.items():
            if self.conf.f_fused_bn:
                if not ('bn' in key):
                    value.kernel=value.kernel/self.ts
                    value.bias=value.bias/self.ts
            else:
                value.kernel=value.kernel/self.ts
                value.bias=value.bias/self.ts


    #
    def preproc_ann_norm(self):
        if self.conf.f_fused_bn:
            self.fused_bn()

        #self.print_model()

        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()

        #self.print_model()


    def call_ann(self,inputs,f_training, tw, epoch):
        #print(type(inputs))
        #if self.f_1st_iter == False and self.conf.nn_mode=='ANN':
        if self.f_1st_iter == False:
            # TODO: check c1 or c2
            # c1
            if self.f_done_preproc == False:
                self.f_done_preproc=True
                self.print_model_conf()
                self.preproc_ann_norm()

            # c2
            #assert False
            #if self.f_done_preproc == False:
                #self.f_done_preproc=True
                #self.print_model_conf()
                #self.preproc_ann_norm()


            self.f_skip_bn=self.conf.f_fused_bn
        else:
            self.f_skip_bn=False

        x = tf.reshape(inputs,self._input_shape)

        a_in = x

        s_conv1 = self.conv1(a_in)
#        if self.f_1st_iter:
#            s_conv1 = self.conv1(x)
#        else:
#            #s_conv1 = self.conv1(x)
#            s_conv1 = tf.contrib.layers.conv2d(
#                x,64,3,
#                activation_fn=None,
#                weights_initializer=tf.constant_initializer(self.conv1.kernel.numpy()),
#                biases_initializer=tf.constant_initializer(self.conv1.bias.numpy())
#                #trainable=f_training
#            )

        if self.f_skip_bn:
            s_conv1_bn = s_conv1
        else:
            s_conv1_bn = self.conv1_bn(s_conv1,training=f_training)

        a_conv1 = tf.nn.relu(s_conv1_bn)
        if f_training:
            a_conv1 = self.dropout_conv(a_conv1,training=f_training)
        s_conv1_1 = self.conv1_1(a_conv1)

        #pred = tf.reduce_mean(self.conv1_1.kernel,[0,1])

        if self.f_skip_bn:
            s_conv1_1_bn = s_conv1_1
        else:
            s_conv1_1_bn = self.conv1_1_bn(s_conv1_1,training=f_training)
        a_conv1_1 = tf.nn.relu(s_conv1_1_bn)
        p_conv1_1 = self.pool2d(a_conv1_1)
        #if f_training:
        #    x = self.dropout_conv(x,training=f_training)

        s_conv2 = self.conv2(p_conv1_1)
        if self.f_skip_bn:
            s_conv2_bn = s_conv2
        else:
            s_conv2_bn = self.conv2_bn(s_conv2,training=f_training)
        a_conv2 = tf.nn.relu(s_conv2_bn)
        if f_training:
           a_conv2 = self.dropout_conv2(a_conv2,training=f_training)
        s_conv2_1 = self.conv2_1(a_conv2)
        if self.f_skip_bn:
            s_conv2_1_bn = s_conv2_1
        else:
            s_conv2_1_bn = self.conv2_1_bn(s_conv2_1,training=f_training)
        a_conv2_1 = tf.nn.relu(s_conv2_1_bn)
        p_conv2_1 = self.pool2d(a_conv2_1)
        #if f_training:
        #   x = self.dropout_conv2(x,training=f_training)

        s_conv3 = self.conv3(p_conv2_1)
        if self.f_skip_bn:
            s_conv3_bn = s_conv3
        else:
            s_conv3_bn = self.conv3_bn(s_conv3,training=f_training)
        a_conv3 = tf.nn.relu(s_conv3_bn)
        if f_training:
           a_conv3 = self.dropout_conv2(a_conv3,training=f_training)
        s_conv3_1 = self.conv3_1(a_conv3)
        if self.f_skip_bn:
            s_conv3_1_bn = s_conv3_1
        else:
            s_conv3_1_bn = self.conv3_1_bn(s_conv3_1,training=f_training)
        a_conv3_1 = tf.nn.relu(s_conv3_1_bn)
        if f_training:
           a_conv3_1 = self.dropout_conv2(a_conv3_1,training=f_training)
        s_conv3_2 = self.conv3_2(a_conv3_1)
        if self.f_skip_bn:
            s_conv3_2_bn = s_conv3_2
        else:
            s_conv3_2_bn = self.conv3_2_bn(s_conv3_2,training=f_training)
        a_conv3_2 = tf.nn.relu(s_conv3_2_bn)
        p_conv3_2 = self.pool2d(a_conv3_2)
        #if f_training:
        #   x = self.dropout_conv2(x,training=f_training)

        s_conv4 = self.conv4(p_conv3_2)
        if self.f_skip_bn:
            s_conv4_bn = s_conv4
        else:
            s_conv4_bn = self.conv4_bn(s_conv4,training=f_training)
        a_conv4 = tf.nn.relu(s_conv4_bn)
        if f_training:
           a_conv4 = self.dropout_conv2(a_conv4,training=f_training)
        s_conv4_1 = self.conv4_1(a_conv4)
        if self.f_skip_bn:
            s_conv4_1_bn = s_conv4_1
        else:
            s_conv4_1_bn = self.conv4_1_bn(s_conv4_1,training=f_training)
        a_conv4_1 = tf.nn.relu(s_conv4_1_bn)
        if f_training:
           a_conv4_1 = self.dropout_conv2(a_conv4_1,training=f_training)
        s_conv4_2 = self.conv4_2(a_conv4_1)
        if self.f_skip_bn:
            s_conv4_2_bn = s_conv4_2
        else:
            s_conv4_2_bn = self.conv4_2_bn(s_conv4_2,training=f_training)
        a_conv4_2 = tf.nn.relu(s_conv4_2_bn)
        p_conv4_2 = self.pool2d(a_conv4_2)
        #if f_training:
        #   x = self.dropout_conv2(x,training=f_training)

        s_conv5 = self.conv5(p_conv4_2)
        if self.f_skip_bn:
            s_conv5_bn = s_conv5
        else:
            s_conv5_bn = self.conv5_bn(s_conv5,training=f_training)
        a_conv5 = tf.nn.relu(s_conv5_bn)
        if f_training:
           a_conv5 = self.dropout_conv2(a_conv5,training=f_training)
        s_conv5_1 = self.conv5_1(a_conv5)
        if self.f_skip_bn:
            s_conv5_1_bn = s_conv5_1
        else:
            s_conv5_1_bn = self.conv5_1_bn(s_conv5_1,training=f_training)
        a_conv5_1 = tf.nn.relu(s_conv5_1_bn)
        if f_training:
           a_conv5_1 = self.dropout_conv2(a_conv5_1,training=f_training)
        s_conv5_2 = self.conv5_2(a_conv5_1)
        if self.f_skip_bn:
            s_conv5_2_bn = s_conv5_2
        else:
            s_conv5_2_bn = self.conv5_2_bn(s_conv5_2,training=f_training)
        a_conv5_2 = tf.nn.relu(s_conv5_2_bn)
        p_conv5_2 = self.pool2d(a_conv5_2)

        s_flat = tf.compat.v1.layers.flatten(p_conv5_2)

        if f_training:
           s_flat = self.dropout(s_flat,training=f_training)

        s_fc1 = self.fc1(s_flat)
        if self.f_skip_bn:
            s_fc1_bn = s_fc1
        else:
            s_fc1_bn = self.fc1_bn(s_fc1,training=f_training)
        a_fc1 = tf.nn.relu(s_fc1_bn)
        if f_training:
           a_fc1 = self.dropout(a_fc1,training=f_training)

        s_fc2 = self.fc2(a_fc1)
        if self.f_skip_bn:
            s_fc2_bn = s_fc2
        else:
            s_fc2_bn = self.fc2_bn(s_fc2,training=f_training)
        a_fc2 = tf.nn.relu(s_fc2_bn)
        if f_training:
           a_fc2 = self.dropout(a_fc2,training=f_training)

        s_fc3 = self.fc3(a_fc2)
        if self.f_skip_bn:
            s_fc3_bn = s_fc3
        else:
            #if ('bn' in self.conf.model_name) or ('ro' in self.conf.model_name) :
                #s_fc3_bn = self.fc3_bn(s_fc3,training=f_training)
            #else:
                #s_fc3_bn = s_fc3
            s_fc3_bn = self.fc3_bn(s_fc3,training=f_training)
        #a_fc3 = s_fc3_bn
        #if 'ro' in self.conf.model_name:
         #   a_fc3 = tf.nn.relu(s_fc3_bn)
        #else:
            #a_fc3 = s_fc3_bn
        a_fc3 = s_fc3_bn

        #if f_training:
        #   x = self.dropout(x,training=f_training)





        if self.conf.f_comp_act and (not self.f_1st_iter):
            self.dict_stat_w['conv1']=a_conv1.numpy()
            self.dict_stat_w['conv1_1']=a_conv1_1.numpy()
            self.dict_stat_w['conv2']=a_conv2.numpy()
            self.dict_stat_w['conv2_1']=a_conv2_1.numpy()
            self.dict_stat_w['conv3']=a_conv3.numpy()
            self.dict_stat_w['conv3_1']=a_conv3_1.numpy()
            self.dict_stat_w['conv3_2']=a_conv3_2.numpy()
            self.dict_stat_w['conv4']=a_conv4.numpy()
            self.dict_stat_w['conv4_1']=a_conv4_1.numpy()
            self.dict_stat_w['conv4_2']=a_conv4_2.numpy()
            self.dict_stat_w['conv5']=a_conv5.numpy()
            self.dict_stat_w['conv5_1']=a_conv5_1.numpy()
            self.dict_stat_w['conv5_2']=a_conv5_2.numpy()
            self.dict_stat_w['fc1']=a_fc1.numpy()
            self.dict_stat_w['fc2']=a_fc2.numpy()
            self.dict_stat_w['fc3']=a_fc3.numpy()



        a_out = a_fc3
        #print(a_out)
        #print(tf.reduce_max(a_out))
        #print(a_fc2)
        #print(s_fc3)
        #print(self.fc3.kernel)
        #print(self.fc3.bias)
        #print(a_fc2)
        #print(a_fc3)

#        if not self.f_1st_iter:
#
#            fig, axs = plt.subplots(4,5)
#            axs=axs.ravel()
#
#            axs[0].hist(x.numpy().flatten())
#            axs[1].hist(s_conv1_bn.numpy().flatten())
#            axs[2].hist(s_conv1_1_bn.numpy().flatten())
#            axs[3].hist(s_conv2_bn.numpy().flatten())
#            axs[4].hist(s_conv2_1_bn.numpy().flatten())
#            axs[5].hist(s_conv3_bn.numpy().flatten())
#            axs[6].hist(s_conv3_1_bn.numpy().flatten())
#            axs[7].hist(s_conv3_2_bn.numpy().flatten())
#            axs[8].hist(s_conv4_bn.numpy().flatten())
#            axs[9].hist(s_conv4_1_bn.numpy().flatten())
#            axs[10].hist(s_conv4_2_bn.numpy().flatten())
#            axs[11].hist(s_conv5_bn.numpy().flatten())
#            axs[12].hist(s_conv5_1_bn.numpy().flatten())
#            axs[13].hist(s_conv5_2_bn.numpy().flatten())
#            axs[14].hist(s_fc1_bn.numpy().flatten())
#            axs[15].hist(s_fc2_bn.numpy().flatten())
#            axs[16].hist(s_fc3_bn.numpy().flatten())
#
#            plt.show()
#
#            assert False



        if self.f_1st_iter and self.conf.nn_mode=='ANN':
            print('1st iter')
            self.f_1st_iter = False
            self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)


        if not self.f_1st_iter and (self.conf.f_train_time_const or self.conf.f_write_stat):
            #print("training time constant for temporal coding in SNN")

            self.dnn_act_list['in'] = a_in
            self.dnn_act_list['conv1']   = a_conv1
            self.dnn_act_list['conv1_1'] = a_conv1_1

            self.dnn_act_list['conv2']   = a_conv2
            self.dnn_act_list['conv2_1'] = a_conv2_1

            self.dnn_act_list['conv3']   = a_conv3
            self.dnn_act_list['conv3_1'] = a_conv3_1
            self.dnn_act_list['conv3_2'] = a_conv3_2

            self.dnn_act_list['conv4']   = a_conv4
            self.dnn_act_list['conv4_1'] = a_conv4_1
            self.dnn_act_list['conv4_2'] = a_conv4_2

            self.dnn_act_list['conv5']   = a_conv5
            self.dnn_act_list['conv5_1'] = a_conv5_1
            self.dnn_act_list['conv5_2'] = a_conv5_2

            self.dnn_act_list['fc1'] = a_fc1
            self.dnn_act_list['fc2'] = a_fc2
            self.dnn_act_list['fc3'] = a_fc3


                    # write stat
        #print(self.f_1st_iter)
        #print(self.dict_stat_w["fc1"])
        if (self.conf.f_write_stat) and (not self.f_1st_iter):
            self.write_act()

            #self.dict_stat_w['conv1']=np.append(self.dict_stat_w['conv1'],a_conv1.numpy(),axis=0)
            #self.dict_stat_w['conv1_1']=np.append(self.dict_stat_w['conv1_1'],a_conv1_1.numpy(),axis=0)
            #self.dict_stat_w['conv2']=np.append(self.dict_stat_w['conv2'],a_conv2.numpy(),axis=0)
            #self.dict_stat_w['conv2_1']=np.append(self.dict_stat_w['conv2_1'],a_conv2_1.numpy(),axis=0)
            #self.dict_stat_w['conv3']=np.append(self.dict_stat_w['conv3'],a_conv3.numpy(),axis=0)
            #self.dict_stat_w['conv3_1']=np.append(self.dict_stat_w['conv3_1'],a_conv3_1.numpy(),axis=0)
            #self.dict_stat_w['conv3_2']=np.append(self.dict_stat_w['conv3_2'],a_conv3_2.numpy(),axis=0)
            #self.dict_stat_w['conv4']=np.append(self.dict_stat_w['conv4'],a_conv4.numpy(),axis=0)
            #self.dict_stat_w['conv4_1']=np.append(self.dict_stat_w['conv4_1'],a_conv4_1.numpy(),axis=0)
            #self.dict_stat_w['conv4_2']=np.append(self.dict_stat_w['conv4_2'],a_conv4_2.numpy(),axis=0)
            #self.dict_stat_w['conv5']=np.append(self.dict_stat_w['conv5'],a_conv5.numpy(),axis=0)
            #self.dict_stat_w['conv5_1']=np.append(self.dict_stat_w['conv5_1'],a_conv5_1.numpy(),axis=0)
            #self.dict_stat_w['conv5_2']=np.append(self.dict_stat_w['conv5_2'],a_conv5_2.numpy(),axis=0)
            #self.dict_stat_w['fc1']=np.append(self.dict_stat_w['fc1'],a_fc1.numpy(),axis=0)
            #self.dict_stat_w['fc2']=np.append(self.dict_stat_w['fc2'],a_fc2.numpy(),axis=0)
            #self.dict_stat_w['fc3']=np.append(self.dict_stat_w['fc3'],a_fc3.numpy(),axis=0)

            # test bn activation distribution
            #self.dict_stat_w['fc1']=np.append(self.dict_stat_w['fc1'],s_fc1_bn.numpy(),axis=0)

            #self.dict_stat_w['fc1']=np.append(self.dict_stat_w['fc1'],s_fc1_bn.numpy(),axis=0)


        return a_out


    # TODO: move other spot
    #
    def write_act(self):

        for l_name in self.layer_name_write_stat:
            dict_stat_w = self.dict_stat_w[l_name]

            if self.f_1st_iter_stat:
                dict_stat_w.assign(self.dnn_act_list[l_name])
            else:
                self.dict_stat_w[l_name]=tf.concat([dict_stat_w,self.dnn_act_list[l_name]], 0)


        #print(self.dict_stat_w[l_name].shape)

        if self.f_1st_iter_stat:
            self.f_1st_iter_stat = False





    # surrogate DNN model for training SNN w/ TTFS coding
    def call_ann_surrogate_training(self,inputs,f_training,tw,epoch):
        #print(epoch)
        #print(type(inputs))
        #if self.f_1st_iter == False and self.conf.nn_mode=='ANN':
        if self.f_1st_iter == False:
            #if self.f_done_preproc == False:
                #self.f_done_preproc=True
                #self.print_model_conf()
                #self.preproc_ann_norm()
            self.f_skip_bn=self.conf.f_fused_bn
        else:
            self.f_skip_bn=False

        x = tf.reshape(inputs,self._input_shape)

        #a_in = x

        #pr = 0.1
        #pr = 0.6
        #pr = 0.9
        #pr = 1.0
        #target_epoch = 600

        target_epoch = self.conf.bypass_target_epoch
        pr_target_epoch = tf.cast(tf.divide(tf.add(epoch,1),target_epoch),tf.float32)
        pr = tf.multiply(tf.subtract(1.0,self.conf.bypass_pr),pr_target_epoch)


        #if epoch==-1 or epoch > 100:
        #if epoch==-1 or tf.random.uniform(shape=(),minval=0,maxval=1)<pr:
        #pr_layer = pr*(5/5)*pr_target_epoch

        #pr_layer = tf.multiply(pr,pr_target_epoch)
        pr_layer = pr
        #print("epoch: {}, target_epoch: {}".format(epoch,target_epoch))
        #print("pr: {}, pr_target_epoch: {}".format(pr_layer,pr_target_epoch))
        #pr_layer = 2.0


        #if self.f_1st_iter:
        if not self.f_load_model_done:
        #if f_training==False or tf.random.uniform(shape=(),minval=0,maxval=1)<pr_layer:
            v_in = x
            t_in = self.list_tk['in'](v_in,'enc', self.epoch, f_training)
            v_in_dec= self.list_tk['in'](t_in, 'dec', self.epoch, f_training)
            a_in = v_in_dec
        else:
            a_in = x

        s_conv1 = self.conv1(a_in)
        if self.f_skip_bn:
            s_conv1_bn = s_conv1
        else:
            s_conv1_bn = self.conv1_bn(s_conv1,training=f_training)


        #if epoch==-1 or epoch > 100:
        #if epoch==-1 or tf.random.uniform(shape=(),minval=0,maxval=1)<pr:
        #if f_training==False or ((f_training) and (tf.random.uniform(shape=(),minval=0,maxval=1)<pr_layer)):
        #if f_training==False or ((f_training) and (rand<pr_layer)):
        rand = tf.random.uniform(shape=(),minval=0,maxval=1)
        if f_training==False or ((f_training==True) and (tf.math.less(rand,pr_layer))):
        #if True:
            v_conv1 = s_conv1_bn
            t_conv1 = self.list_tk['conv1'](v_conv1,'enc',self.epoch,f_training)
            v_conv1_dec = self.list_tk['conv1'](t_conv1,'dec',self.epoch,f_training)
            a_conv1 = v_conv1_dec
        else:
            a_conv1 = tf.nn.relu(s_conv1_bn)

        if f_training:
            a_conv1 = self.dropout_conv(a_conv1,training=f_training)
        s_conv1_1 = self.conv1_1(a_conv1)

        #pred = tf.reduce_mean(self.conv1_1.kernel,[0,1])

        if self.f_skip_bn:
            s_conv1_1_bn = s_conv1_1
        else:
            s_conv1_1_bn = self.conv1_1_bn(s_conv1_1,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv1_1 = s_conv1_1_bn
            t_conv1_1 = self.list_tk['conv1_1'](v_conv1_1,'enc',self.epoch,f_training)
            v_conv1_1_dec = self.list_tk['conv1_1'](t_conv1_1,'dec',self.epoch,f_training)
            a_conv1_1 = v_conv1_1_dec
        else:
            a_conv1_1 = tf.nn.relu(s_conv1_1_bn)

        p_conv1_1 = self.pool2d(a_conv1_1)
        #if f_training:
        #    x = self.dropout_conv(x,training=f_training)

        s_conv2 = self.conv2(p_conv1_1)
        if self.f_skip_bn:
            s_conv2_bn = s_conv2
        else:
            s_conv2_bn = self.conv2_bn(s_conv2,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv2 = s_conv2_bn
            t_conv2 = self.list_tk['conv2'](v_conv2,'enc',self.epoch,f_training)
            v_conv2_dec = self.list_tk['conv2'](t_conv2,'dec',self.epoch,f_training)
            a_conv2 = v_conv2_dec
        else:
            a_conv2 = tf.nn.relu(s_conv2_bn)

        if f_training:
           a_conv2 = self.dropout_conv2(a_conv2,training=f_training)
        s_conv2_1 = self.conv2_1(a_conv2)
        if self.f_skip_bn:
            s_conv2_1_bn = s_conv2_1
        else:
            s_conv2_1_bn = self.conv2_1_bn(s_conv2_1,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv2_1 = s_conv2_1_bn
            t_conv2_1 = self.list_tk['conv2_1'](v_conv2_1,'enc',self.epoch,f_training)
            v_conv2_1_dec = self.list_tk['conv2_1'](t_conv2_1,'dec',self.epoch,f_training)
            a_conv2_1 = v_conv2_1_dec
        else:
            a_conv2_1 = tf.nn.relu(s_conv2_1_bn)

        p_conv2_1 = self.pool2d(a_conv2_1)
        #if f_training:
        #   x = self.dropout_conv2(x,training=f_training)

        s_conv3 = self.conv3(p_conv2_1)
        if self.f_skip_bn:
            s_conv3_bn = s_conv3
        else:
            s_conv3_bn = self.conv3_bn(s_conv3,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv3 = s_conv3_bn
            t_conv3 = self.list_tk['conv3'](v_conv3,'enc',self.epoch,f_training)
            v_conv3_dec = self.list_tk['conv3'](t_conv3,'dec',self.epoch,f_training)
            a_conv3 = v_conv3_dec
        else:
            a_conv3 = tf.nn.relu(s_conv3_bn)

        if f_training:
           a_conv3 = self.dropout_conv2(a_conv3,training=f_training)
        s_conv3_1 = self.conv3_1(a_conv3)
        if self.f_skip_bn:
            s_conv3_1_bn = s_conv3_1
        else:
            s_conv3_1_bn = self.conv3_1_bn(s_conv3_1,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv3_1 = s_conv3_1_bn
            t_conv3_1 = self.list_tk['conv3_1'](v_conv3_1,'enc',self.epoch,f_training)
            v_conv3_1_dec = self.list_tk['conv3_1'](t_conv3_1,'dec',self.epoch,f_training)
            a_conv3_1 = v_conv3_1_dec
        else:
            a_conv3_1 = tf.nn.relu(s_conv3_1_bn)

        if f_training:
           a_conv3_1 = self.dropout_conv2(a_conv3_1,training=f_training)
        s_conv3_2 = self.conv3_2(a_conv3_1)
        if self.f_skip_bn:
            s_conv3_2_bn = s_conv3_2
        else:
            s_conv3_2_bn = self.conv3_2_bn(s_conv3_2,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv3_2 = s_conv3_2_bn
            t_conv3_2 = self.list_tk['conv3_2'](v_conv3_2,'enc',self.epoch,f_training)
            v_conv3_2_dec = self.list_tk['conv3_2'](t_conv3_2,'dec',self.epoch,f_training)
            a_conv3_2 = v_conv3_2_dec
        else:
            a_conv3_2 = tf.nn.relu(s_conv3_2_bn)

        p_conv3_2 = self.pool2d(a_conv3_2)
        #if f_training:
        #   x = self.dropout_conv2(x,training=f_training)

        s_conv4 = self.conv4(p_conv3_2)
        if self.f_skip_bn:
            s_conv4_bn = s_conv4
        else:
            s_conv4_bn = self.conv4_bn(s_conv4,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv4 = s_conv4_bn
            t_conv4 = self.list_tk['conv4'](v_conv4,'enc',self.epoch,f_training)
            v_conv4_dec = self.list_tk['conv4'](t_conv4,'dec',self.epoch,f_training)
            a_conv4 = v_conv4_dec
        else:
            a_conv4 = tf.nn.relu(s_conv4_bn)

        if f_training:
           a_conv4 = self.dropout_conv2(a_conv4,training=f_training)
        s_conv4_1 = self.conv4_1(a_conv4)
        if self.f_skip_bn:
            s_conv4_1_bn = s_conv4_1
        else:
            s_conv4_1_bn = self.conv4_1_bn(s_conv4_1,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv4_1 = s_conv4_1_bn
            t_conv4_1 = self.list_tk['conv4_1'](v_conv4_1,'enc',self.epoch,f_training)
            v_conv4_1_dec = self.list_tk['conv4_1'](t_conv4_1,'dec',self.epoch,f_training)
            a_conv4_1 = v_conv4_1_dec
        else:
            a_conv4_1 = tf.nn.relu(s_conv4_1_bn)

        if f_training:
           a_conv4_1 = self.dropout_conv2(a_conv4_1,training=f_training)
        s_conv4_2 = self.conv4_2(a_conv4_1)
        if self.f_skip_bn:
            s_conv4_2_bn = s_conv4_2
        else:
            s_conv4_2_bn = self.conv4_2_bn(s_conv4_2,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv4_2 = s_conv4_2_bn
            t_conv4_2 = self.list_tk['conv4_2'](v_conv4_2,'enc',self.epoch,f_training)
            v_conv4_2_dec = self.list_tk['conv4_2'](t_conv4_2,'dec',self.epoch,f_training)
            a_conv4_2 = v_conv4_2_dec
        else:
            a_conv4_2 = tf.nn.relu(s_conv4_2_bn)

        p_conv4_2 = self.pool2d(a_conv4_2)
        #if f_training:
        #   x = self.dropout_conv2(x,training=f_training)

        s_conv5 = self.conv5(p_conv4_2)
        if self.f_skip_bn:
            s_conv5_bn = s_conv5
        else:
            s_conv5_bn = self.conv5_bn(s_conv5,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv5 = s_conv5_bn
            t_conv5 = self.list_tk['conv5'](v_conv5,'enc',self.epoch,f_training)
            v_conv5_dec = self.list_tk['conv5'](t_conv5,'dec',self.epoch,f_training)
            a_conv5 = v_conv5_dec
        else:
            a_conv5 = tf.nn.relu(s_conv5_bn)

        if f_training:
           a_conv5 = self.dropout_conv2(a_conv5,training=f_training)
        s_conv5_1 = self.conv5_1(a_conv5)
        if self.f_skip_bn:
            s_conv5_1_bn = s_conv5_1
        else:
            s_conv5_1_bn = self.conv5_1_bn(s_conv5_1,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv5_1 = s_conv5_1_bn
            t_conv5_1 = self.list_tk['conv5_1'](v_conv5_1,'enc',self.epoch,f_training)
            v_conv5_1_dec = self.list_tk['conv5_1'](t_conv5_1,'dec',self.epoch,f_training)
            a_conv5_1 = v_conv5_1_dec
        else:
            a_conv5_1 = tf.nn.relu(s_conv5_1_bn)

        if f_training:
           a_conv5_1 = self.dropout_conv2(a_conv5_1,training=f_training)
        s_conv5_2 = self.conv5_2(a_conv5_1)
        if self.f_skip_bn:
            s_conv5_2_bn = s_conv5_2
        else:
            s_conv5_2_bn = self.conv5_2_bn(s_conv5_2,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_conv5_2 = s_conv5_2_bn
            t_conv5_2 = self.list_tk['conv5_2'](v_conv5_2,'enc',self.epoch,f_training)
            v_conv5_2_dec = self.list_tk['conv5_2'](t_conv5_2,'dec',self.epoch,f_training)
            a_conv5_2 = v_conv5_2_dec
        else:
            a_conv5_2 = tf.nn.relu(s_conv5_2_bn)

        #print(tf.reduce_max(a_conv5_2))
        p_conv5_2 = self.pool2d(a_conv5_2)

        s_flat = tf.compat.v1.layers.flatten(p_conv5_2)

        if f_training:
           s_flat = self.dropout(s_flat,training=f_training)

        s_fc1 = self.fc1(s_flat)
        if self.f_skip_bn:
            s_fc1_bn = s_fc1
        else:
            s_fc1_bn = self.fc1_bn(s_fc1,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_fc1 = s_fc1_bn
            t_fc1 = self.list_tk['fc1'](v_fc1,'enc',self.epoch,f_training)
            v_fc1_dec = self.list_tk['fc1'](t_fc1,'dec',self.epoch,f_training)
            a_fc1 = v_fc1_dec
        else:
            a_fc1 = tf.nn.relu(s_fc1_bn)

        if f_training:
           a_fc1 = self.dropout(a_fc1,training=f_training)

        s_fc2 = self.fc2(a_fc1)
        if self.f_skip_bn:
            s_fc2_bn = s_fc2
        else:
            s_fc2_bn = self.fc2_bn(s_fc2,training=f_training)

        rand = tf.random.uniform(shape=(), minval=0, maxval=1)
        if f_training == False or ((f_training) and (tf.math.less(rand, pr_layer))):
        #if True:
            v_fc2 = s_fc2_bn
            t_fc2 = self.list_tk['fc2'](v_fc2,'enc',self.epoch,f_training)
            v_fc2_dec = self.list_tk['fc2'](t_fc2,'dec',self.epoch,f_training)
            a_fc2 = v_fc2_dec
        else:
            a_fc2 = tf.nn.relu(s_fc2_bn)
        if f_training:
           a_fc2 = self.dropout(a_fc2,training=f_training)


        s_fc3 = self.fc3(a_fc2)
        if self.f_skip_bn:
            s_fc3_bn = s_fc3
        else:
            if ('bn' in self.conf.model_name) or ('ro' in self.conf.model_name) :
                s_fc3_bn = self.fc3_bn(s_fc3,training=f_training)
            else:
                s_fc3_bn = s_fc3
        #a_fc3 = s_fc3_bn
        if 'ro' in self.conf.model_name:
            a_fc3 = tf.nn.relu(s_fc3_bn)
        else:
            a_fc3 = s_fc3_bn


        # print - activation histogram
        #if not self.f_1st_iter:
        if False:

            fig, axs = plt.subplots(4,5)
            axs=axs.ravel()

            list_hist_count=[]
            list_hist_bins=[]
            list_hist_bars=[]
            list_beta=[]
            list_x_M=[]
            list_x_m=[]

            #
            counts, bins, bars = axs[0].hist(x.numpy().flatten(),bins=1000)
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(0)
            list_x_M.append(0)
            list_x_m.append(0)

            #
            counts, bins, bars = axs[1].hist(s_conv1_bn.numpy().flatten(),bins=1000)
            layer_name = 'conv1'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[1].vlines(beta,0, np.max(counts), color='k')
            axs[1].vlines(x_m,0, np.max(counts), color='r')
            axs[1].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #
            counts, bins, bars = axs[2].hist(s_conv1_1_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv1_1'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[2].vlines(beta,0, np.max(counts), color='k')
            axs[2].vlines(x_m,0, np.max(counts), color='r')
            axs[2].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #
            counts, bins, bars = axs[3].hist(s_conv2_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv2'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[3].vlines(beta,0, np.max(counts), color='k')
            axs[3].vlines(x_m,0, np.max(counts), color='r')
            axs[3].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #
            counts, bins, bars = axs[4].hist(s_conv2_1_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv2_1'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[4].vlines(beta,0, np.max(counts), color='k')
            axs[4].vlines(x_m,0, np.max(counts), color='r')
            axs[4].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #
            counts, bins, bars = axs[5].hist(s_conv3_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv3'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[5].vlines(beta,0, np.max(counts), color='k')
            axs[5].vlines(x_m,0, np.max(counts), color='r')
            axs[5].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #


            counts, bins, bars = axs[6].hist(s_conv3_1_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv3_1'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[6].vlines(beta,0, np.max(counts), color='k')
            axs[6].vlines(x_m,0, np.max(counts), color='r')
            axs[6].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #


            counts, bins, bars = axs[7].hist(s_conv3_2_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv3_2'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[7].vlines(beta,0, np.max(counts), color='k')
            axs[7].vlines(x_m,0, np.max(counts), color='r')
            axs[7].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #


            counts, bins, bars = axs[8].hist(s_conv4_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv4'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[8].vlines(beta,0, np.max(counts), color='k')
            axs[8].vlines(x_m,0, np.max(counts), color='r')
            axs[8].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #


            counts, bins, bars = axs[9].hist(s_conv4_1_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv4_1'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[9].vlines(beta,0, np.max(counts), color='k')
            axs[9].vlines(x_m,0, np.max(counts), color='r')
            axs[9].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #


            counts, bins, bars = axs[10].hist(s_conv4_2_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv4_2'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[10].vlines(beta,0, np.max(counts), color='k')
            axs[10].vlines(x_m,0, np.max(counts), color='r')
            axs[10].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #


            counts, bins, bars = axs[11].hist(s_conv5_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv5'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[11].vlines(beta,0, np.max(counts), color='k')
            axs[11].vlines(x_m,0, np.max(counts), color='r')
            axs[11].vlines(x_M,0, np.max(counts), color='m')

            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #

            counts, bins, bars = axs[12].hist(s_conv5_1_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv5_1'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[12].vlines(beta,0, np.max(counts), color='k')
            axs[12].vlines(x_m,0, np.max(counts), color='r')
            axs[12].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #


            counts, bins, bars = axs[13].hist(s_conv5_2_bn.numpy().flatten(),bins=1000)

            layer_name = 'conv5_2'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[13].vlines(beta,0, np.max(counts), color='k')
            axs[13].vlines(x_m,0, np.max(counts), color='r')
            axs[13].vlines(x_M,0, np.max(counts), color='m')

            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #

            counts, bins, bars = axs[14].hist(s_fc1_bn.numpy().flatten(),bins=1000)

            layer_name = 'fc1'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[14].vlines(beta,0, np.max(counts), color='k')
            axs[14].vlines(x_m,0, np.max(counts), color='r')
            axs[14].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #


            counts, bins, bars = axs[15].hist(s_fc2_bn.numpy().flatten(),bins=1000)

            layer_name = 'fc2'
            beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
            x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
            axs[15].vlines(beta,0, np.max(counts), color='k')
            axs[15].vlines(x_m,0, np.max(counts), color='r')
            axs[15].vlines(x_M,0, np.max(counts), color='m')
            list_hist_count.append(counts)
            list_hist_bins.append(bins)
            list_hist_bars.append(bars)
            list_beta.append(beta.numpy())
            list_x_M.append(x_M.numpy()[0])
            list_x_m.append(x_m.numpy()[0])

            #


            counts, bins, bars = axs[16].hist(s_fc3_bn.numpy().flatten(),bins=1000)



            #
            #output_xlsx_name='bn_act_hist_SM-2.xlsx'
            #output_xlsx_name='bn_act_hist_SR.xlsx'
            #output_xlsx_name='bn_act_hist_TR.xlsx'
            output_xlsx_name='bn_act_hist_TB.xlsx'
            df=pd.DataFrame(list_hist_count).T
            #            #df=pd.DataFrame({'loss_prec': list_loss_prec, 'loss_min': list_loss_min, 'loss_max': list_loss_max})
            df.to_excel(output_xlsx_name,sheet_name='count')

            with pd.ExcelWriter(output_xlsx_name,mode='a') as writer:
                df=pd.DataFrame(list_hist_bins).T
                df.to_excel(writer,sheet_name='bins')

                df=pd.DataFrame(list_hist_bars).T
                df.to_excel(writer,sheet_name='bars')

                print(list_beta)
                df=pd.DataFrame(list_beta).T
                df.to_excel(writer,sheet_name='beta')

                df=pd.DataFrame(list_x_M).T
                df.to_excel(writer,sheet_name='x_M')

                df=pd.DataFrame(list_x_m).T
                df.to_excel(writer,sheet_name='x_m')


            # for data write
#            #
##            col_x = x.numpy().flatten()
##            col_conv1_bn   = s_conv1_bn.numpy().flatten()
##            col_conv1_1_bn = s_conv1_1_bn.numpy().flatten()
##            col_conv2_bn   = s_conv2_bn.numpy().flatten()
##            col_conv2_1_bn = s_conv2_1_bn.numpy().flatten()
##            col_conv2_2_bn = s_conv2_2_bn.numpy().flatten()
##            col_conv3_bn   = s_conv3_bn.numpy().flatten()
##            col_conv3_1_bn = s_conv3_1_bn.numpy().flatten()
##            col_conv3_2_bn = s_conv3_2_bn.numpy().flatten()
##            col_conv4_bn   = s_conv4_bn.numpy().flatten()
##            col_conv4_1_bn = s_conv4_1_bn.numpy().flatten()
##            col_conv4_2_bn = s_conv4_2_bn.numpy().flatten()
##            col_conv5_bn   = s_conv5_bn.numpy().flatten()
##            col_conv5_1_bn = s_conv5_1_bn.numpy().flatten()
##            col_conv5_2_bn = s_conv5_2_bn.numpy().flatten()
##            col_fc1_bn     = s_fc1_bn.numpy().flatten()
##            col_fc2_bn     = s_fc2_bn.numpy().flatten()
##            col_fc3_bn     = s_fc3_bn.numpy().flatten()
#
#            #
#            list_df=[]
#            list_df.append(x.numpy().flatten())
#            list_df.append(s_conv1_bn.numpy().flatten())
#            list_df.append(s_conv1_1_bn.numpy().flatten())
#
#            df=pd.DataFrame(list_df)
#            #df=pd.DataFrame({'loss_prec': list_loss_prec, 'loss_min': list_loss_min, 'loss_max': list_loss_max})
#            df.to_excel('test.xlsx')
#
#            print(n)
#            print(bins)
#            print(patches)
#
#            print(tf.math.reduce_mean(self.list_layer['conv1_bn'].gamma))
#            print(tf.math.reduce_mean(self.list_layer['conv1_bn'].beta))
#            print(self.list_tk['conv1'].tc)
#            print(self.list_tk['conv1'].td)
#            x_M = tf.math.exp(tf.math.divide(self.list_tk['conv1'].td,self.list_tk['conv1'].tc))
#            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk['conv1'].tc)))
#            print(x_M)
#            print(x_m)

            plt.show()

            assert False



        if self.conf.f_comp_act and (not self.f_1st_iter):
            self.dict_stat_w['conv1']=a_conv1.numpy()
            self.dict_stat_w['conv1_1']=a_conv1_1.numpy()
            self.dict_stat_w['conv2']=a_conv2.numpy()
            self.dict_stat_w['conv2_1']=a_conv2_1.numpy()
            self.dict_stat_w['conv3']=a_conv3.numpy()
            self.dict_stat_w['conv3_1']=a_conv3_1.numpy()
            self.dict_stat_w['conv3_2']=a_conv3_2.numpy()
            self.dict_stat_w['conv4']=a_conv4.numpy()
            self.dict_stat_w['conv4_1']=a_conv4_1.numpy()
            self.dict_stat_w['conv4_2']=a_conv4_2.numpy()
            self.dict_stat_w['conv5']=a_conv5.numpy()
            self.dict_stat_w['conv5_1']=a_conv5_1.numpy()
            self.dict_stat_w['conv5_2']=a_conv5_2.numpy()
            self.dict_stat_w['fc1']=a_fc1.numpy()
            self.dict_stat_w['fc2']=a_fc2.numpy()
            self.dict_stat_w['fc3']=a_fc3.numpy()



        a_out = a_fc3
        #print(a_out)
        #print(a_fc2)
        #print(s_fc3)
        #print(self.fc3.kernel)
        #print(self.fc3.bias)
        #print(a_fc2)
        #print(a_fc3)


        # debugging
#        act = a_fc2
#        print('act {}: min - {}, max - {}, avg - {}'.format('fc3',tf.reduce_min(act),tf.reduce_max(act),tf.reduce_mean(act)))
#
#        try:
#            t_fc2
#        except NameError:
#            pass
#        else:
#            act = t_fc2
#            print('enc {}: min - {}, max - {}, avg - {}'.format('fc3',tf.reduce_min(act),tf.reduce_max(act),tf.reduce_mean(act)))
#            plt.hist(act.numpy().flatten(),bins=self.enc_st_target_end)
#            #plt.hist(act.numpy().flatten(),bins=[0,200])
#            plt.xlim([0,self.conf.time_window])
#            #plt.xlim([0,200])
#
#            #print(self.enc_st_target_end)
#            #assert False
#
#            plt.draw()
#            plt.pause(0.0000000001)

        if self.f_1st_iter and self.conf.nn_mode=='ANN':
            print('1st iter')
            self.f_1st_iter = False
            self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)

            #
#            if self.conf.f_surrogate_training_model:
#                # TODO: init init range as the sum of initial weights
#
#
#                target_range = 1.0
#                self.list_tk['in'].set_init_td_by_target_range(target_range)
#
#                fig, axs = plt.subplots(4,5)
#                axs=axs.ravel()
#
#                for idx, n_layer in enumerate(self.layer_name):
#                    if not n_layer=='in':
#                        print(n_layer)
#
#                        layer = self.list_layer[n_layer]
#                        l_type = type(layer)
#
#                        print(l_type)
#                        print(tf.reduce_sum(layer.kernel))
#                        print(tf.reduce_mean(layer.kernel))
#
#                        # herehere
#
#                        axs[idx].hist(layer.kernel.numpy().flatten())
#
#
#                #plt.hist(self.conv1.kernel.numpy().flatten())
#
#                plt.show()
#
#                assert False



        if not self.f_1st_iter and self.conf.f_train_time_const:
            print("training time constant for temporal coding in SNN")

            self.dnn_act_list['in'] = a_in
            self.dnn_act_list['conv1']   = a_conv1
            self.dnn_act_list['conv1_1'] = a_conv1_1

            self.dnn_act_list['conv2']   = a_conv2
            self.dnn_act_list['conv2_1'] = a_conv2_1

            self.dnn_act_list['conv3']   = a_conv3
            self.dnn_act_list['conv3_1'] = a_conv3_1
            self.dnn_act_list['conv3_2'] = a_conv3_2

            self.dnn_act_list['conv4']   = a_conv4
            self.dnn_act_list['conv4_1'] = a_conv4_1
            self.dnn_act_list['conv4_2'] = a_conv4_2

            self.dnn_act_list['conv5']   = a_conv5
            self.dnn_act_list['conv5_1'] = a_conv5_1
            self.dnn_act_list['conv5_2'] = a_conv5_2

            self.dnn_act_list['fc1'] = a_fc1
            self.dnn_act_list['fc2'] = a_fc2
            self.dnn_act_list['fc3'] = a_fc3


        return a_out







    #
    def print_model_conf(self):
        # print model configuration
        print('Input   N: '+str(self.in_shape))

        print('Conv1   S: '+str(self.conv1.kernel.get_shape()))
        print('Conv1   N: '+str(self.shape_out_conv1))
        print('Conv1_1 S: '+str(self.conv1_1.kernel.get_shape()))
        print('Conv1_1 N: '+str(self.shape_out_conv1_1))
        print('Pool1   N: '+str(self.shape_out_conv1_p))

        print('Conv2   S: '+str(self.conv2.kernel.get_shape()))
        print('Conv2   N: '+str(self.shape_out_conv2))
        print('Conv2_1 S: '+str(self.conv2_1.kernel.get_shape()))
        print('Conv2_1 N: '+str(self.shape_out_conv2_1))
        print('Pool2   N: '+str(self.shape_out_conv2_p))

        print('Conv3   S: '+str(self.conv3.kernel.get_shape()))
        print('Conv3   N: '+str(self.shape_out_conv3))
        print('Conv3_1 S: '+str(self.conv3_1.kernel.get_shape()))
        print('Conv3_1 N: '+str(self.shape_out_conv3_1))
        print('Conv3_2 S: '+str(self.conv3_2.kernel.get_shape()))
        print('Conv3_2 N: '+str(self.shape_out_conv3_2))
        print('Pool3   N: '+str(self.shape_out_conv3_p))

        print('Conv4   S: '+str(self.conv4.kernel.get_shape()))
        print('Conv4   N: '+str(self.shape_out_conv4))
        print('Conv4_1 S: '+str(self.conv4_1.kernel.get_shape()))
        print('Conv4_1 N: '+str(self.shape_out_conv4_1))
        print('Conv4_1 S: '+str(self.conv4_2.kernel.get_shape()))
        print('Conv4_2 N: '+str(self.shape_out_conv4_2))
        print('Pool4   N: '+str(self.shape_out_conv4_p))

        print('Conv5   S: '+str(self.conv5.kernel.get_shape()))
        print('Conv5   N: '+str(self.shape_out_conv5))
        print('Conv5_1 S: '+str(self.conv5_1.kernel.get_shape()))
        print('Conv5_1 N: '+str(self.shape_out_conv5_1))
        print('Conv5_1 S: '+str(self.conv5_1.kernel.get_shape()))
        print('Conv5_2 N: '+str(self.shape_out_conv5_2))
        print('Pool5   N: '+str(self.shape_out_conv5_p))

        print('Fc1     S: '+str(self.fc1.kernel.get_shape()))
        print('Fc1     N: '+str(self.shape_out_fc1))
        print('Fc2     S: '+str(self.fc2.kernel.get_shape()))
        print('Fc2     N: '+str(self.shape_out_fc2))
        print('Fc3     S: '+str(self.fc3.kernel.get_shape()))
        print('Fc3     N: '+str(self.shape_out_fc3))


    def print_act_d(self):
        print('print activation')

        fig, axs = plt.subplots(6,3)

        axs=axs.ravel()

        #for idx_l, (name_l,stat_l) in enumerate(self.dict_stat_r):
        for idx, (key, value) in enumerate(self.dict_stat_r.items()):
            axs[idx].hist(value.flatten())

        plt.show()


    def print_model(self):
        print('print model')

        plt.subplot(631)
        h_conv1=plt.hist(self.conv1.kernel.numpy().flatten())
        plt.vlines(self.conv1.bias.numpy(),0,h_conv1[0].max())
        plt.subplot(632)
        h_conv1_1=plt.hist(self.conv1_1.kernel.numpy().flatten())
        plt.vlines(self.conv1_1.bias.numpy(),0,h_conv1_1[0].max())

        plt.subplot(634)
        h_conv2=plt.hist(self.conv2.kernel.numpy().flatten())
        plt.vlines(self.conv2.bias.numpy(),0,h_conv2[0].max())
        plt.subplot(635)
        h_conv2_1=plt.hist(self.conv2_1.kernel.numpy().flatten())
        plt.vlines(self.conv2_1.bias.numpy(),0,h_conv2_1[0].max())

        plt.subplot(637)
        h_conv3=plt.hist(self.conv3.kernel.numpy().flatten())
        plt.vlines(self.conv3.bias.numpy(),0,h_conv3[0].max())
        plt.subplot(638)
        h_conv3_1=plt.hist(self.conv3_1.kernel.numpy().flatten())
        plt.vlines(self.conv3_1.bias.numpy(),0,h_conv3_1[0].max())
        plt.subplot(639)
        h_conv3_2=plt.hist(self.conv3_2.kernel.numpy().flatten())
        plt.vlines(self.conv3_2.bias.numpy(),0,h_conv3_2[0].max())

        plt.subplot2grid((6,3),(3,0))
        h_conv4=plt.hist(self.conv4.kernel.numpy().flatten())
        plt.vlines(self.conv4.bias.numpy(),0,h_conv4[0].max())
        plt.subplot2grid((6,3),(3,1))
        h_conv4_1=plt.hist(self.conv4_1.kernel.numpy().flatten())
        plt.vlines(self.conv4_1.bias.numpy(),0,h_conv4_1[0].max())
        plt.subplot2grid((6,3),(3,2))
        h_conv4_2=plt.hist(self.conv4_2.kernel.numpy().flatten())
        plt.vlines(self.conv4_2.bias.numpy(),0,h_conv4_2[0].max())

        plt.subplot2grid((6,3),(4,0))
        h_conv5=plt.hist(self.conv5.kernel.numpy().flatten())
        plt.vlines(self.conv5.bias.numpy(),0,h_conv5[0].max())
        plt.subplot2grid((6,3),(4,1))
        h_conv5_1=plt.hist(self.conv5_1.kernel.numpy().flatten())
        plt.vlines(self.conv5_1.bias.numpy(),0,h_conv5_1[0].max())
        plt.subplot2grid((6,3),(4,2))
        h_conv5_2=plt.hist(self.conv5_2.kernel.numpy().flatten())
        plt.vlines(self.conv5_2.bias.numpy(),0,h_conv5_2[0].max())

        plt.subplot2grid((6,3),(5,0))
        h_fc1=plt.hist(self.fc1.kernel.numpy().flatten())
        plt.vlines(self.fc1.bias.numpy(),0,h_fc1[0].max())
        plt.subplot2grid((6,3),(5,1))
        h_fc2=plt.hist(self.fc2.kernel.numpy().flatten())
        plt.vlines(self.fc2.bias.numpy(),0,h_fc2[0].max())
        plt.subplot2grid((6,3),(5,2))
        h_fc3=plt.hist(self.fc3.kernel.numpy().flatten())
        plt.vlines(self.fc3.bias.numpy(),0,h_fc3[0].max())

        #print(self.fc2_bn.beta.numpy())
        #print(self.fc2_bn.gamma)
        #print(self.fc2_bn.beta)

        plt.show()

    def conv_bn_fused(self, conv, bn, time_step):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        #conv.kernel = conv.kernel*math_ops.cast(inv,conv.kernel.dtype)/time_step
        conv.kernel = conv.kernel*math_ops.cast(inv,conv.kernel.dtype)
        #conv.bias = (conv.bias*gamma/var_e+beta-gamma*mean/var_e)/time_step
        #conv.bias = ((conv.bias-mean)*inv+beta)/time_step
        conv.bias = ((conv.bias-mean)*inv+beta)

        #print(gamma)
        #print(beta)

        #return kernel, bias

    #
    def conv_bn_defused(self, conv, bn, time_step):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        conv.kernel = conv.kernel/inv
        conv.bias = (conv.bias-beta)/inv+mean


    def fc_bn_fused(self, conv, bn, time_step):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        #conv.kernel = conv.kernel*gamma/var_e/time_step
        conv.kernel = conv.kernel*math_ops.cast(inv,conv.kernel.dtype)
        conv.bias = ((conv.bias-mean)*inv+beta)

        #return kernel, bias

    #
    def fc_bn_defused(self, conv, bn, time_step):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        conv.kernel = conv.kernel/inv
        conv.bias = (conv.bias-beta)/inv+mean



#    def preproc_ann_to_snn(self):
#        print('preprocessing: ANN to SNN')
#        if self.conf.f_fused_bn:
#            self.fused_bn()
#
#        #self.print_model()
#        #print(np.max(list(self.list_layer.values())[0].kernel))
#        #self.temporal_norm()
#        #print(np.max(list(self.list_layer.values())[0].kernel))
#
#
#        #self.print_model()
#        # weight normalization - data based
#        if self.conf.f_w_norm_data:
#            self.data_based_w_norm()
#
#        #if self.conf.f_comp_act:
#        #    self.load_act_after_w_norm()
#
#        #self.print_act_after_w_norm()

    def _model_based_norm(self, layer, axis):
        w_in_sum = tf.reduce_sum(tf.maximum(layer.kernel,0),axis=axis)
        w_in_max = tf.reduce_max(w_in_sum)
        layer.kernel = layer.kernel/w_in_max

    def model_based_norm(self):
        print('model based_norm')
        w_in_sum_conv1 = tf.reduce_sum(tf.maximum(self.conv1.kernel,0),axis=[0,1,2])
        #w_in_max_conv1 = tf.reduce_max(w_in_sum_conv1+self.conv1.bias)
        w_in_max_conv1 = tf.reduce_max(w_in_sum_conv1)
        self.conv1.kernel = self.conv1.kernel/w_in_max_conv1
        w_in_sum_conv1_1 = tf.reduce_sum(tf.maximum(self.conv1_1.kernel,0),axis=[0,1,2])
        #w_in_max_conv1_1 = tf.reduce_max(w_in_sum_conv1_1+self.conv1_1.bias)
        w_in_max_conv1_1 = tf.reduce_max(w_in_sum_conv1_1)
        self.conv1_1.kernel = self.conv1_1.kernel/w_in_max_conv1_1

        w_in_sum_conv2 = tf.reduce_sum(tf.maximum(self.conv2.kernel,0),axis=[0,1,2])
        #w_in_max_conv2 = tf.reduce_max(w_in_sum_conv2+self.conv2.bias)
        w_in_max_conv2 = tf.reduce_max(w_in_sum_conv2)
        self.conv2.kernel = self.conv2.kernel/w_in_max_conv2
        w_in_sum_conv2_1 = tf.reduce_sum(tf.maximum(self.conv2_1.kernel,0),axis=[0,1,2])
        #w_in_max_conv2_1 = tf.reduce_max(w_in_sum_conv2_1+self.conv2_1.bias)
        w_in_max_conv2_1 = tf.reduce_max(w_in_sum_conv2_1)
        self.conv2_1.kernel = self.conv2_1.kernel/w_in_max_conv2_1

        w_in_sum_conv3 = tf.reduce_sum(tf.maximum(self.conv3.kernel,0),axis=[0,1,2])
        #w_in_max_conv3 = tf.reduce_max(w_in_sum_conv3+self.conv3.bias)
        w_in_max_conv3 = tf.reduce_max(w_in_sum_conv3)
        self.conv3.kernel = self.conv3.kernel/w_in_max_conv3
        w_in_sum_conv3_1 = tf.reduce_sum(tf.maximum(self.conv3_1.kernel,0),axis=[0,1,2])
        #w_in_max_conv3_1 = tf.reduce_max(w_in_sum_conv3_1+self.conv3_1.bias)
        w_in_max_conv3_1 = tf.reduce_max(w_in_sum_conv3_1)
        self.conv3_1.kernel = self.conv3_1.kernel/w_in_max_conv3_1
        w_in_sum_conv3_2 = tf.reduce_sum(tf.maximum(self.conv3_2.kernel,0),axis=[0,1,2])
        #w_in_max_conv3_2 = tf.reduce_max(w_in_sum_conv3_2+self.conv3_2.bias)
        w_in_max_conv3_2 = tf.reduce_max(w_in_sum_conv3_2)
        self.conv3_2.kernel = self.conv3_2.kernel/w_in_max_conv3_2

        w_in_sum_conv4 = tf.reduce_sum(tf.maximum(self.conv4.kernel,0),axis=[0,1,2])
        #w_in_max_conv4 = tf.reduce_max(w_in_sum_conv4+self.conv4.bias)
        w_in_max_conv4 = tf.reduce_max(w_in_sum_conv4)
        self.conv4.kernel = self.conv4.kernel/w_in_max_conv4
        w_in_sum_conv4_1 = tf.reduce_sum(tf.maximum(self.conv4_1.kernel,0),axis=[0,1,2])
        #w_in_max_conv4_1 = tf.reduce_max(w_in_sum_conv4_1+self.conv4_1.bias)
        w_in_max_conv4_1 = tf.reduce_max(w_in_sum_conv4_1)
        self.conv4_1.kernel = self.conv4_1.kernel/w_in_max_conv4_1
        w_in_sum_conv4_2 = tf.reduce_sum(tf.maximum(self.conv4_2.kernel,0),axis=[0,1,2])
        #w_in_max_conv4_2 = tf.reduce_max(w_in_sum_conv4_2+self.conv4_2.bias)
        w_in_max_conv4_2 = tf.reduce_max(w_in_sum_conv4_2)
        self.conv4_2.kernel = self.conv4_2.kernel/w_in_max_conv4_2

        w_in_sum_conv5 = tf.reduce_sum(tf.maximum(self.conv5.kernel,0),axis=[0,1,2])
        #w_in_max_conv5 = tf.reduce_max(w_in_sum_conv5+self.conv5.bias)
        w_in_max_conv5 = tf.reduce_max(w_in_sum_conv5)
        self.conv5.kernel = self.conv5.kernel/w_in_max_conv5
        w_in_sum_conv5_1 = tf.reduce_sum(tf.maximum(self.conv5_1.kernel,0),axis=[0,1,2])
        #w_in_max_conv5_1 = tf.reduce_max(w_in_sum_conv5_1+self.conv5_1.bias)
        w_in_max_conv5_1 = tf.reduce_max(w_in_sum_conv5_1)
        self.conv5_1.kernel = self.conv5_1.kernel/w_in_max_conv5_1
        w_in_sum_conv5_2 = tf.reduce_sum(tf.maximum(self.conv5_2.kernel,0),axis=[0,1,2])
        #w_in_max_conv5_2 = tf.reduce_max(w_in_sum_conv5_2+self.conv5_2.bias)
        w_in_max_conv5_2 = tf.reduce_max(w_in_sum_conv5_2)
        self.conv5_2.kernel = self.conv5_2.kernel/w_in_max_conv5_2

        w_in_sum_fc1 = tf.reduce_sum(tf.maximum(self.fc1.kernel,0),axis=[0])
        #w_in_max_fc1 = tf.reduce_max(w_in_sum_fc1+self.fc1.bias)
        w_in_max_fc1 = tf.reduce_max(w_in_sum_fc1)
        self.fc1.kernel = self.fc1.kernel/w_in_max_fc1
        w_in_sum_fc2 = tf.reduce_sum(tf.maximum(self.fc2.kernel,0),axis=[0])
        #w_in_max_fc2 = tf.reduce_max(w_in_sum_fc2+self.fc2.bias)
        w_in_max_fc2 = tf.reduce_max(w_in_sum_fc2)
        self.fc2.kernel = self.fc2.kernel/w_in_max_fc2
        w_in_sum_fc3 = tf.reduce_sum(tf.maximum(self.fc3.kernel,0),axis=[0])
        #w_in_max_fc3 = tf.reduce_max(w_in_sum_fc3+self.fc3.bias)
        w_in_max_fc3 = tf.reduce_max(w_in_sum_fc3)
        # w_in_min_fc3 = tf.reduce_min(w_in_sum_fc3)
        #self.fc3.kernel = self.fc3.kernel-kernel_min/2
        self.fc3.kernel = self.fc3.kernel/w_in_max_fc3


        if self.verbose == True:
            print('w_in_max_conv1:' +str(w_in_max_conv1))
            print('w_in_max_conv1_1:' +str(w_in_max_conv1_1))

            print('w_in_max_conv2:' +str(w_in_max_conv2))
            print('w_in_max_conv2_1:' +str(w_in_max_conv2_1))

            print('w_in_max_conv3:' +str(w_in_max_conv3))
            print('w_in_max_conv3_1:' +str(w_in_max_conv3_1))
            print('w_in_max_conv3_2:' +str(w_in_max_conv3_2))

            print('w_in_max_conv4:' +str(w_in_max_conv4))
            print('w_in_max_conv4_1:' +str(w_in_max_conv4_1))
            print('w_in_max_conv4_2:' +str(w_in_max_conv4_2))

            print('w_in_max_conv5:' +str(w_in_max_conv5))
            print('w_in_max_conv5_1:' +str(w_in_max_conv5_1))
            print('w_in_max_conv5_2:' +str(w_in_max_conv5_2))

            print('w_in_max_fc1:' +str(w_in_max_fc1))
            print('w_in_max_fc2:' +str(w_in_max_fc2))
            print('w_in_max_fc3:' +str(w_in_max_fc3))



    def plot(self, x, y, mark):
        #plt.ion()
        #plt.hist(self.n_fc3.vmem)
        plt.plot(x, y, mark)
        plt.draw()
        plt.pause(0.00000001)
        #plt.ioff()


    def scatter(self, x, y, color, axe=None, marker='o'):
        if axe==None:
            plt.scatter(x, y, c=color, s=1, marker=marker)
            plt.draw()
            plt.pause(0.0000000000000001)
        else:
            axe.scatter(x, y, c=color, s=1, marker=marker)
            plt.draw()
            plt.pause(0.0000000000000001)

    def figure_hold(self):
        plt.close("dummy")
        plt.show()



    def visual(self, t):
        #plt.subplot2grid((2,4),(0,0))
        ax=plt.subplot(2,4,1)
        plt.title('w_sum_in (max)')
        self.plot(t,tf.reduce_max(s_fc2).numpy(), 'bo')
        #plt.subplot2grid((2,4),(0,1),sharex=ax)
        plt.subplot(2,4,2,sharex=ax)
        plt.title('vmem (max)')
        self.plot(t,tf.reduce_max(self.n_fc2.vmem).numpy(), 'bo')
        #plt.subplot2grid((2,4),(0,2))
        plt.subplot(2,4,3,sharex=ax)
        plt.title('# spikes (max)')
        self.plot(t,tf.reduce_max(self.n_fc2.get_spike_count()).numpy(), 'bo')
        #self.scatter(np.full(np.shape),tf.reduce_max(self.n_fc2.get_spike_count()).numpy(), 'bo')
        #plt.subplot2grid((2,4),(0,3))
        plt.subplot(2,4,4,sharex=ax)
        plt.title('spike neuron idx')
        plt.grid(True)
        plt.ylim([0,512])
        #plt.ylim([0,int(self.n_fc2.dim[1])])
        plt.xlim([0,tw])
        #self.plot(t,np.where(self.n_fc2.out.numpy()==1),'bo')
        #if np.where(self.n_fc2.out.numpy()==1).size == 0:
        idx_fire=np.where(self.n_fc2.out.numpy()==1)[1]
        if not len(idx_fire)==0:
            #print(np.shape(idx_fire))
            #print(idx_fire)
            #print(np.full(np.shape(idx_fire),t))
            self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')


        #plt.subplot2grid((2,4),(1,0))
        plt.subplot(2,4,5,sharex=ax)
        self.plot(t,tf.reduce_max(s_fc3).numpy(), 'bo')
        #plt.subplot2grid((2,4),(1,1))
        plt.subplot(2,4,6,sharex=ax)
        self.plot(t,tf.reduce_max(self.n_fc3.vmem).numpy(), 'bo')
        #plt.subplot2grid((2,4),(1,2))
        plt.subplot(2,4,7,sharex=ax)
        self.plot(t,tf.reduce_max(self.n_fc3.get_spike_count()).numpy(), 'bo')
        plt.subplot(2,4,8)
        plt.grid(True)
        #plt.ylim([0,self.n_fc3.dim[1]])
        plt.ylim([0,self.num_class])
        plt.xlim([0,tw])
        idx_fire=np.where(self.n_fc3.out.numpy()==1)[1]
        if not len(idx_fire)==0:
            self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')

    def get_total_residual_vmem(self):
        len=self.total_residual_vmem.shape[0]
        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            idx=idx_n-1
            if nn!='in' or nn!='fc3':
                self.total_residual_vmem[idx]+=tf.reduce_sum(tf.abs(n.vmem))
                self.total_residual_vmem[len-1]+=self.total_residual_vmem[idx]

    def get_total_isi(self):
        isi_count=np.zeros(self.conf.time_step)

        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            if nn!='in' or nn!='fc3':
                isi_count_n = np.bincount(np.int32(n.isi.numpy().flatten()))
                isi_count_n.resize(self.conf.time_step)
                isi_count = isi_count + isi_count_n

        return isi_count


    def f_out_isi(self,t):
        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            if nn!='in' or nn!='fc3':
                f_name = './isi/'+nn+'_'+self.conf.model_name+'_'+self.conf.input_spike_mode+'_'+self.conf.neural_coding+'_'+str(self.conf.time_step)+'.csv'

                if t==0:
                    f = open(f_name,'w')
                else:
                    f = open(f_name,'a')

                wr = csv.writer(f)

                array=n.isi.numpy().flatten()

                for i in range(len(array)):
                    if array[i]!=0:
                        wr.writerow((i,n.isi.numpy().flatten()[i]))

                f.close()


    def get_total_spike_amp(self):
        spike_amp=np.zeros(self.spike_amp_kind)
        #print(range(0,self.spike_amp_kind)[::-1])
        #print(np.power(0.5,range(0,self.spike_amp_kind)))

        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            if nn!='in' or nn!='fc3':
                spike_amp_n = np.histogram(n.out.numpy().flatten(),self.spike_amp_bin)
                #spike_amp = spike_amp + spike_amp_n[0]
                spike_amp += spike_amp_n[0]

                #isi_count_n = np.bincount(np.int32(n.isi.numpy().flatten()))
                #isi_count_n.resize(self.conf.time_step)
                #isi_count = isi_count + isi_count_n

        return spike_amp


    def get_total_spike_count(self):
        len=self.total_spike_count.shape[1]
        spike_count = np.zeros([len,])
        spike_count_int = np.zeros([len,])

        #for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            idx=idx_n-1
            if nn!='in':
                spike_count_int[idx]=tf.reduce_sum(n.get_spike_count_int())
                spike_count_int[len-1]+=spike_count_int[idx]
                spike_count[idx]=tf.reduce_sum(n.get_spike_count())
                spike_count[len-1]+=spike_count[idx]

                #print(nn+": "+str(spike_count_int[idx]))


        #print("total: "+str(spike_count_int[len-1])+"\n")

        return [spike_count_int, spike_count]

    #def bias_norm_weighted_spike(self):
        #for k, l in self.list_layer.items():
        #    if not 'bn' in k:
        #        l.bias = l.bias/(1-1/np.power(2,8))
        #        #l.bias = l.bias/8.0
        #self.list_layer['conv1'].bias=self.list_layer['conv1'].bias/8.0

    def bias_norm_proposed_method(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.bias = l.bias*self.conf.n_init_vth
                #l.bias = l.bias/200
                #l.bias = l.bias*0.0

    def bias_enable(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.use_bias = True

    def bias_disable(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.use_bias = False


    def comp_act_rate(self,t):
        self.total_comp_act[t,-1]=0.0
        for idx_l, l in enumerate(self.layer_name):
            if l !='fc3':
                self.total_comp_act[t,idx_l]=np.mean(np.abs(self.list_neuron[l].spike_counter.numpy().flatten()/(float)(t+1)-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,-1]+=self.total_comp_act[t,idx_l]


        #l='conv1'
        #print(self.list_neuron[l].spike_counter.numpy().flatten())
        #print(self.dict_stat_w[l].flatten())

    def comp_act_ws(self,t):
        self.total_comp_act[t,-1]=0.0
        for idx_l, l in enumerate(self.layer_name):
            if l !='fc3':
                #self.total_comp_act[t,idx_l]=np.mean(np.abs(self.list_neuron[l].spike_counter.numpy().flatten()/((float)(t+1)/(float)(self.conf.p_ws))-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,idx_l]=np.mean(np.abs(self.list_neuron[l].spike_counter.numpy().flatten()/((float)(t+1)/(float)(self.conf.p_ws))-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,-1]+=self.total_comp_act[t,idx_l]


    def comp_act_pro(self,t):
        self.comp_act_ws(t)

    def save_ann_act(self,inputs,f_training):
        self.call_ann(inputs,f_training)


    def cal_entropy(self):
        #total_pattern=np.empty(len(self.layer_name))

        for il, length in enumerate(self.arr_length):
            #total_pattern=0
            #total_pattern=np.zeros(1)
            for idx_l, l in enumerate(self.layer_name):
                if l !='fc3':
                    #print(self.dict_stat_w[l].shape)
                    self.dict_stat_w[l][np.nonzero(self.dict_stat_w[l])]=1.0

                    #print(self.dict_stat_w[l])

                    #print(np.array2string(str(self.dict_stat_w[l]),max_line_width=4))
                    #print(self.dict_stat_w[l].shape)

                    num_words = self.dict_stat_w[l].shape[0]/length
                    tmp = np.zeros((num_words,)+(self.dict_stat_w[l].shape[1:]))

                    for idx in range(num_words):
                        for idx_length in range(length):
                            tmp[idx] += self.dict_stat_w[l][idx*length+idx_length]*np.power(2,idx_length)

                    #print(tmp)

                    #plt.hist(tmp.flatten())
                    #plt.show()
                    #print(tmp.shape)

                    #print(np.histogram(tmp.flatten(),density=True))
                    self.total_entropy[il,idx_l] += stats.entropy(np.histogram(tmp.flatten(),density=True)[0])
                    #print(np.histogram(tmp.flatten()))

                    #total_pattern = np.concatenate((total_pattern,tmp.flatten()))
                    #print(total_pattern)

                    #total_pattern=np.append(total_pattern,tmp.flatten())
                    #total_pattern += (tmp.flatten())
                    #total_pattern[idx_l]=tmp.flatten()

                    self.total_entropy[il,-1] += self.total_entropy[il,idx_l]

            #self.total_entropy[il,-1]=np.histogram(total_pattern.flatten(),density=True)[0]
            #self.total_entropy[il,-1]=np.histogram(total_pattern,density=True)[0]

            #print(self.total_entropy[il,-1])



        #l='fc2'
        #print(self.dict_stat_w[l])
        #print(self.dict_stat_w[l].shape)

        #l='conv4'
        #print(self.dict_stat_w[l])
        #print(self.dict_stat_w[l].shape)


    ###########################################
    # bias control
    ###########################################
    # TODO: bias control
    def bias_control(self,t):
        if self.conf.input_spike_mode == 'WEIGHTED_SPIKE' or self.conf.neural_coding == 'WEIGHTED_SPIKE':
            #if self.conf.neural_coding == 'WEIGHTED_SPIKE':
            #if tf.equal(tf.reduce_max(a_in),0.0):
            if (int)(t%self.conf.p_ws) == 0:
                self.bias_enable()
            else:
                self.bias_disable()
        else:
            if self.conf.input_spike_mode == 'BURST':
                if t==0:
                    self.bias_enable()
                else:
                    if tf.equal(tf.reduce_max(a_in),0.0):
                        self.bias_enable()
                    else:
                        self.bias_disable()


        if self.conf.neural_coding == 'TEMPORAL':
            #if (int)(t%self.conf.p_ws) == 0:
            if t == 0:
                self.bias_enable()
            else:
                self.bias_disable()

    def bias_norm_weighted_spike(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
            #if (not 'bn' in k) and (not 'fc1' in k) :
                #l.bias = l.bias/(1-1/np.power(2,8))
                l.bias = l.bias/8.0

    def bias_norm_proposed_method(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.bias = l.bias*self.conf.n_init_vth
                #l.bias = l.bias/200
                #l.bias = l.bias*0.0

    def bias_enable(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.use_bias = True

    def bias_disable(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.use_bias = False

    def bias_restore(self):
        if self.conf.use_bias:
            self.bias_enable()
        else:
            self.bias_disable()


    ######################################################################
    # SNN call
    ######################################################################
    def call_snn(self,inputs,f_training,tw,epoch):


        #
        plt.clf()

        #print(self.list_tk['conv1'].tc)
        #
        for t in range(tw):
            if self.verbose == True:
                print('time: '+str(t))
            #x = tf.reshape(inputs,self._input_shape)

            self.bias_control(t)

            a_in = self.n_in(inputs,t)

            #if self.conf.f_real_value_input_snn:
            #    a_in = inputs
            #else:
            #    a_in = self.n_in(inputs,t)


            ####################
            #
            ####################
            s_conv1 = self.conv1(a_in)
            a_conv1 = self.n_conv1(s_conv1,t)

            s_conv1_1 = self.conv1_1(a_conv1)
            a_conv1_1 = self.n_conv1_1(s_conv1_1,t)

            if self.conf.f_spike_max_pool:
                p_conv1_1 = lib_snn.spike_max_pool(
                    a_conv1_1,
                    self.n_conv1_1.get_spike_count(),
                    self.dict_shape['conv1_p']
                )
            else:
                p_conv1_1 = self.pool2d(a_conv1_1)

            s_conv2 = self.conv2(p_conv1_1)
            a_conv2 = self.n_conv2(s_conv2,t)
            s_conv2_1 = self.conv2_1(a_conv2)
            a_conv2_1 = self.n_conv2_1(s_conv2_1,t)

            if self.conf.f_spike_max_pool:
                p_conv2_1 = lib_snn.spike_max_pool(
                    a_conv2_1,
                    self.n_conv2_1.get_spike_count(),
                    self.dict_shape['conv2_p']
                )
            else:
                p_conv2_1 = self.pool2d(a_conv2_1)

            s_conv3 = self.conv3(p_conv2_1)
            a_conv3 = self.n_conv3(s_conv3,t)
            s_conv3_1 = self.conv3_1(a_conv3)
            a_conv3_1 = self.n_conv3_1(s_conv3_1,t)
            s_conv3_2 = self.conv3_2(a_conv3_1)
            a_conv3_2 = self.n_conv3_2(s_conv3_2,t)

            if self.conf.f_spike_max_pool:
                p_conv3_2 = lib_snn.spike_max_pool(
                    a_conv3_2,
                    self.n_conv3_2.get_spike_count(),
                    self.dict_shape['conv3_p']
                )
            else:
                p_conv3_2 = self.pool2d(a_conv3_2)


            s_conv4 = self.conv4(p_conv3_2)
            a_conv4 = self.n_conv4(s_conv4,t)
            s_conv4_1 = self.conv4_1(a_conv4)
            a_conv4_1 = self.n_conv4_1(s_conv4_1,t)
            s_conv4_2 = self.conv4_2(a_conv4_1)
            a_conv4_2 = self.n_conv4_2(s_conv4_2,t)

            if self.conf.f_spike_max_pool:
                p_conv4_2 = lib_snn.spike_max_pool(
                    a_conv4_2,
                    self.n_conv4_2.get_spike_count(),
                    self.dict_shape['conv4_p']
                )
            else:
                p_conv4_2 = self.pool2d(a_conv4_2)

            s_conv5 = self.conv5(p_conv4_2)
            a_conv5 = self.n_conv5(s_conv5,t)
            s_conv5_1 = self.conv5_1(a_conv5)
            a_conv5_1 = self.n_conv5_1(s_conv5_1,t)
            s_conv5_2 = self.conv5_2(a_conv5_1)
            a_conv5_2 = self.n_conv5_2(s_conv5_2,t)

            if self.conf.f_spike_max_pool:
                p_conv5_2 = lib_snn.spike_max_pool(
                    a_conv5_2,
                    self.n_conv5_2.get_spike_count(),
                    self.dict_shape['conv5_p']
                )
            else:
                p_conv5_2 = self.pool2d(a_conv5_2)

            flat = tf.compat.v1.layers.flatten(p_conv5_2)

            s_fc1 = self.fc1(flat)
            #s_fc1_bn = self.fc1_bn(s_fc1,training=f_training)
            #a_fc1 = self.n_fc1(s_fc1_bn,t)
            a_fc1 = self.n_fc1(s_fc1,t)

            s_fc2 = self.fc2(a_fc1)
            #s_fc2_bn = self.fc2_bn(s_fc2,training=f_training)
            #a_fc2 = self.n_fc2(s_fc2_bn,t)
            a_fc2 = self.n_fc2(s_fc2,t)

            s_fc3 = self.fc3(a_fc2)
            #print('a_fc3')
            a_fc3 = self.n_fc3(s_fc3,t)


            #print(str(t)+" : "+str(self.n_fc3.vmem.numpy()))



            ###
#            idx_print=0,10,10,0
#            print("time: {time:} - input: {input:0.5f}, vmem: {vmem:0.5f}, spk: {spike:f} {spike_bool:}\n"
#                  .format(
#                        time=t,
#                        input=inputs.numpy()[idx_print],
#                        vmem=self.n_in.vmem.numpy()[idx_print],
#                        spike=self.n_in.out.numpy()[idx_print],
#                        spike_bool= (self.n_in.out.numpy()[idx_print]>0.0))
#                  )




            #print('synapse')
            #print(s_conv1[0,15,15,0])
            #print('vmem')
            #print(self.n_conv1.vmem[0,15,15,0])
            #print('vth')
            #print(self.n_conv1.vth[0,15,15,0])
            #print('')


            #
            #if self.f_1st_iter == False:
            #    plt_idx=0
            #    subplot_x=4
            #    subplot_y=4
            #    for layer_name, layer in self.list_layer.items():
            #        if not ('bn' in layer_name):
            #            print('subplot {}, {}'.format(int(plt_idx/subplot_x+1),int(plt_idx%subplot_x+1)))
            #            #plt.subplot(subplot_x*subplot_y,int(plt_idx/subplot_x+1),int(plt_idx%subplot_x+1))
            #            #plt.subplot(int(plt_idx/subplot_x+1),int(plt_idx%subplot_x+1),plt_idx+1)
            #            plt.subplot(subplot_y,subplot_y,plt_idx+1)
            #            plt.hist(np.abs(layer.kernel.numpy().flatten()),bins=100)
            #            plt_idx+=1
            #
            #            print("{} - max: {}, mean: {}".format(layer_name,np.max(np.abs(layer.kernel.numpy())),np.mean(np.abs(layer.kernel.numpy()))))
            #
            #    plt.show()

            if self.f_1st_iter == False and self.f_debug_visual == True:
                #self.visual(t)


                synapse=s_conv1
                neuron=self.n_in

                #synapse=s_fc2
                #neuron=self.n_fc2

                synapse_1=s_conv1
                neuron_1=self.n_conv1

                synapse_2 = s_conv1_1
                neuron_2 = self.n_conv1_1


                #self.debug_visual(synapse, neuron, synapse_1, neuron_1, synapse_2, neuron_2, t)
                self.debug_visual_raster(t)

                #if t==self.ts-1:
                #    plt.figure()
                #    #plt.show()

            ##########
            #
            ##########

            if self.f_1st_iter == False:
                if self.conf.f_comp_act:
                    if self.conf.neural_coding=='RATE':
                        self.comp_act_rate(t)
                    elif self.conf.neural_coding=='WEIGHTED_SPIKE':
                        self.comp_act_ws(t)
                    elif self.conf.neural_coding=='BURST':
                        self.comp_act_pro(t)

                if self.conf.f_isi:
                    self.total_isi += self.get_total_isi()
                    self.total_spike_amp += self.get_total_spike_amp()

                    self.f_out_isi(t)


                if self.conf.f_entropy:
                    #print(a_conv1.numpy().shape)
                    #print(self.n_conv1.out.numpy().shape)
                    #print(self.dict_stat_w['conv1'].shape)
                    #self.dict_stat_w['conv4'][t]=self.n_conv4.out.numpy()
                    #self.dict_stat_w['fc2'][t]=self.n_fc2.out.numpy()

                    for idx_l, l in enumerate(self.layer_name):
                        if l !='fc3':
                            self.dict_stat_w[l][t] = self.list_neuron[l].out.numpy()


                    #print(self.dict_stat_w['conv1'])

                if t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
                    output=self.n_fc3.vmem
                    self.recoding_ret_val()


                    #num_spike_count = tf.cast(tf.reduce_sum(self.spike_count,axis=[2]),tf.int32)
                    #num_spike_count = tf.reduce_sum(self.spike_count,axis=[2])

            #print(t, self.n_fc3.last_spike_time.numpy())
            #print(t, self.n_fc3.isi.numpy())

        if self.conf.f_entropy and (not self.f_1st_iter):
            self.cal_entropy()


        if self.f_1st_iter:
            self.f_1st_iter = False

            self.conv1_bn(s_conv1,training=f_training)
            self.conv1_1_bn(s_conv1_1,training=f_training)

            self.conv2_bn(s_conv2,training=f_training)
            self.conv2_1_bn(s_conv2_1,training=f_training)

            self.conv3_bn(s_conv3,training=f_training)
            self.conv3_1_bn(s_conv3_1,training=f_training)
            self.conv3_2_bn(s_conv3_2,training=f_training)

            self.conv4_bn(s_conv4,training=f_training)
            self.conv4_1_bn(s_conv4_1,training=f_training)
            self.conv4_2_bn(s_conv4_2,training=f_training)

            self.conv5_bn(s_conv5,training=f_training)
            self.conv5_1_bn(s_conv5_1,training=f_training)
            self.conv5_2_bn(s_conv5_2,training=f_training)

            self.fc1_bn(s_fc1,training=f_training)
            self.fc2_bn(s_fc2,training=f_training)
            self.fc3_bn(s_fc3,training=f_training)

            return 0


        else:

            if not self.conf.f_pruning_channel:
                self.get_total_residual_vmem()

                #spike_zero = tf.reduce_sum(self.spike_count,axis=[0,2])
                spike_zero = tf.reduce_sum(self.snn_output,axis=[0,2])
                #spike_zero = tf.count_nonzero(self.spike_count,axis=[0,2])

                if np.any(spike_zero.numpy() == 0.0):
                #if num_spike_count==0.0:
                    print('spike count 0')
                    #print(num_spike_count.numpy())
                    #a = input("press any key to exit")
                    #os.system("Pause")
                    #raw_input("Press any key to exit")
                    #sys.exit(0)

            #plt.hist(self.n_conv1.vmem.numpy().flatten())
            #plt.show()

            if self.conf.f_pruning_channel:
                for idx_l, l in enumerate(self.layer_name):
                    if 'conv' in l:
                        neuron = self.list_neuron[l]
                        kernel_name = self.layer_name[idx_l+1]

                        if 'conv' in kernel_name:
                            self.idx_pruning_channel[kernel_name] =\
                                self.get_pruning_channel_idx(neuron.get_spike_count())


                            n_remain_channel = tf.shape(self.idx_pruning_channel[kernel_name])[-1]
                            n_original_channel = tf.shape(neuron.get_spike_count())[-1]
                            remain_ratio = n_remain_channel/n_original_channel

                            print('%3d / %3d : %.4f'%(n_remain_channel,n_original_channel,remain_ratio))

                            self.f_idx_pruning_channel[kernel_name]=remain_ratio < self.th_idx_pruning_channel
                            #print(self.f_idx_pruning_channel)

            # first_spike_time visualization
            if self.conf.f_record_first_spike_time and self.conf.f_visual_record_first_spike_time:
                print('first spike time')
                _, axes = plt.subplots(4,4)
                idx_plot=0
                for n_name, n in self.list_neuron.items():
                    if not ('fc3' in n_name):
                        #positive = n.first_spike_time > 0
                        #print(n_name+'] min: '+str(tf.reduce_min(n.first_spike_time[positive]))+', mean: '+str(tf.reduce_mean(n.first_spike_time[positive])))
                        #print(tf.reduce_min(n.first_spike_time[positive]))

                        #positive=n.first_spike_time.numpy().flatten() > 0
                        positive=tf.boolean_mask(n.first_spike_time,n.first_spike_time>0)

                        if not tf.equal(tf.size(positive),0):

                            #min=np.min(n.first_spike_time.numpy().flatten()[positive,])
                            #print(positive.shape)
                            #min=np.min(n.first_spike_time.numpy().flatten()[positive])
                            min=tf.reduce_min(positive)
                            #mean=np.mean(n.first_spike_time.numpy().flatten()[positive,])

                            #if self.conf.f_tc_based:
                            #    fire_s=idx_plot*self.conf.time_fire_start
                            #    fire_e=idx_plot*self.conf.time_fire_start+self.conf.time_fire_duration
                            #else:
                            #    fire_s=idx_plot*self.conf.time_fire_start
                            #    fire_e=idx_plot*self.conf.time_fire_start+self.conf.time_fire_duration

                            #fire_s = n.time_start_fire
                            #fire_e = n.time_end_fire

                            fire_s = idx_plot * self.conf.time_fire_start
                            fire_e = idx_plot * self.conf.time_fire_start + self.conf.time_fire_duration

                            axe=axes.flatten()[idx_plot]
                            #axe.hist(n.first_spike_time.numpy().flatten()[positive],bins=range(fire_s,fire_e,1))
                            axe.hist(positive.numpy().flatten(),bins=range(fire_s,fire_e,1))

                            axe.axvline(x=min.numpy(),color='b', linestyle='dashed')

                            axe.axvline(x=fire_s)
                            axe.axvline(x=fire_e)


                        idx_plot+=1



                # file write raw data
                for n_name, n in self.list_neuron.items():
                    if not ('fc3' in n_name):
                        positive=tf.boolean_mask(n.first_spike_time,n.first_spike_time>0).numpy()

                        fname = './spike_time/spike-time'
                        if self.conf.f_load_time_const:
                            fname += '_train-'+str(self.conf.time_const_num_trained_data)+'_tc-'+str(self.conf.tc)+'_tw-'+str(self.conf.time_window)

                        fname += '_'+n_name+'.csv'
                        f = open(fname,'w')
                        wr = csv.writer(f)
                        wr.writerow(positive)
                        f.close()


                plt.show()


        #return self.spike_count
        return self.snn_output

    ######################################################################
    # SNN call - backup
    ######################################################################
    def call_snn_bck(self,inputs,f_training,tw,epoch):


        #
        self.count_accuracy_time_point=0

        # reset for sample
        if self.f_1st_iter == False:
            self.reset_neuron()

            if self.f_done_preproc == False:
                self.f_done_preproc = True
                self.print_model_conf()
                self.preproc_ann_to_snn()

                # for proposed method
                #self.bias_norm_proposed_method()
                #if self.conf.f_ws:
                #    self.bias_norm_weighted_spike()

                #if self.f_debug_visual == True:
                #    self.print_model()

            #spike_count = np.zeros((self.num_accuracy_time_point,)+self.n_fc3.get_spike_count().numpy().shape)
            self.spike_count.assign(tf.zeros((self.num_accuracy_time_point,)+tuple(self.n_fc3.dim)))

            if self.conf.f_comp_act:
                self.save_ann_act(inputs,f_training)
            #print((self.num_accuracy_time_point,)+self.n_fc3.get_spike_count().numpy().shape)

        #
        plt.clf()

        #
        for t in range(tw):
            if self.verbose == True:
                print('time: '+str(t))
            #x = tf.reshape(inputs,self._input_shape)

            a_in = self.n_in(inputs,t)


            #if self.conf.f_real_value_input_snn:
            #    a_in = inputs
            #else:
            #    a_in = self.n_in(inputs,t)

            ####################
            # bias control
            ####################
            if self.conf.input_spike_mode == 'WEIGHTED_SPIKE' or self.conf.neural_coding == 'WEIGHTED_SPIKE':
                #if self.conf.neural_coding == 'WEIGHTED_SPIKE':
                #if tf.equal(tf.reduce_max(a_in),0.0):
                if (int)(t%self.conf.p_ws) == 0:
                    self.bias_enable()
                else:
                    self.bias_disable()
            else:
                if self.conf.input_spike_mode == 'BURST':
                    if t==0:
                        self.bias_enable()
                    else:
                        if tf.equal(tf.reduce_max(a_in),0.0):
                            self.bias_enable()
                        else:
                            self.bias_disable()


            if self.conf.neural_coding == 'TEMPORAL':
                #if (int)(t%self.conf.p_ws) == 0:
                if t == 0:
                    self.bias_enable()
                else:
                    self.bias_disable()


            ####################
            #
            ####################
            s_conv1 = self.conv1(a_in)
            a_conv1 = self.n_conv1(s_conv1,t)

            s_conv1_1 = self.conv1_1(a_conv1)
            a_conv1_1 = self.n_conv1_1(s_conv1_1,t)

            if self.conf.f_spike_max_pool:
                p_conv1_1 = lib_snn.spike_max_pool(
                    a_conv1_1,
                    self.n_conv1_1.get_spike_count(),
                    self.dict_shape['conv1_p']
                )
            else:
                p_conv1_1 = self.pool2d(a_conv1_1)

            s_conv2 = self.conv2(p_conv1_1)
            a_conv2 = self.n_conv2(s_conv2,t)
            s_conv2_1 = self.conv2_1(a_conv2)
            a_conv2_1 = self.n_conv2_1(s_conv2_1,t)

            if self.conf.f_spike_max_pool:
                p_conv2_1 = lib_snn.spike_max_pool(
                    a_conv2_1,
                    self.n_conv2_1.get_spike_count(),
                    self.dict_shape['conv2_p']
                )
            else:
                p_conv2_1 = self.pool2d(a_conv2_1)

            s_conv3 = self.conv3(p_conv2_1)
            a_conv3 = self.n_conv3(s_conv3,t)
            s_conv3_1 = self.conv3_1(a_conv3)
            a_conv3_1 = self.n_conv3_1(s_conv3_1,t)
            s_conv3_2 = self.conv3_2(a_conv3_1)
            a_conv3_2 = self.n_conv3_2(s_conv3_2,t)

            if self.conf.f_spike_max_pool:
                p_conv3_2 = lib_snn.spike_max_pool(
                    a_conv3_2,
                    self.n_conv3_2.get_spike_count(),
                    self.dict_shape['conv3_p']
                )
            else:
                p_conv3_2 = self.pool2d(a_conv3_2)


            s_conv4 = self.conv4(p_conv3_2)
            a_conv4 = self.n_conv4(s_conv4,t)
            s_conv4_1 = self.conv4_1(a_conv4)
            a_conv4_1 = self.n_conv4_1(s_conv4_1,t)
            s_conv4_2 = self.conv4_2(a_conv4_1)
            a_conv4_2 = self.n_conv4_2(s_conv4_2,t)

            if self.conf.f_spike_max_pool:
                p_conv4_2 = lib_snn.spike_max_pool(
                    a_conv4_2,
                    self.n_conv4_2.get_spike_count(),
                    self.dict_shape['conv4_p']
                )
            else:
                p_conv4_2 = self.pool2d(a_conv4_2)

            s_conv5 = self.conv5(p_conv4_2)
            a_conv5 = self.n_conv5(s_conv5,t)
            s_conv5_1 = self.conv5_1(a_conv5)
            a_conv5_1 = self.n_conv5_1(s_conv5_1,t)
            s_conv5_2 = self.conv5_2(a_conv5_1)
            a_conv5_2 = self.n_conv5_2(s_conv5_2,t)

            if self.conf.f_spike_max_pool:
                p_conv5_2 = lib_snn.spike_max_pool(
                    a_conv5_2,
                    self.n_conv5_2.get_spike_count(),
                    self.dict_shape['conv5_p']
                )
            else:
                p_conv5_2 = self.pool2d(a_conv5_2)

            flat = tf.compat.v1.layers.flatten(p_conv5_2)

            s_fc1 = self.fc1(flat)
            #s_fc1_bn = self.fc1_bn(s_fc1,training=f_training)
            #a_fc1 = self.n_fc1(s_fc1_bn,t)
            a_fc1 = self.n_fc1(s_fc1,t)

            s_fc2 = self.fc2(a_fc1)
            #s_fc2_bn = self.fc2_bn(s_fc2,training=f_training)
            #a_fc2 = self.n_fc2(s_fc2_bn,t)
            a_fc2 = self.n_fc2(s_fc2,t)

            s_fc3 = self.fc3(a_fc2)
            #print('a_fc3')
            a_fc3 = self.n_fc3(s_fc3,t)


            #print(str(t)+" : "+str(self.n_fc3.vmem.numpy()))



            ###
#            idx_print=0,10,10,0
#            print("time: {time:} - input: {input:0.5f}, vmem: {vmem:0.5f}, spk: {spike:f} {spike_bool:}\n"
#                  .format(
#                        time=t,
#                        input=inputs.numpy()[idx_print],
#                        vmem=self.n_in.vmem.numpy()[idx_print],
#                        spike=self.n_in.out.numpy()[idx_print],
#                        spike_bool= (self.n_in.out.numpy()[idx_print]>0.0))
#                  )




            #print('synapse')
            #print(s_conv1[0,15,15,0])
            #print('vmem')
            #print(self.n_conv1.vmem[0,15,15,0])
            #print('vth')
            #print(self.n_conv1.vth[0,15,15,0])
            #print('')


            #
            #if self.f_1st_iter == False:
            #    plt_idx=0
            #    subplot_x=4
            #    subplot_y=4
            #    for layer_name, layer in self.list_layer.items():
            #        if not ('bn' in layer_name):
            #            print('subplot {}, {}'.format(int(plt_idx/subplot_x+1),int(plt_idx%subplot_x+1)))
            #            #plt.subplot(subplot_x*subplot_y,int(plt_idx/subplot_x+1),int(plt_idx%subplot_x+1))
            #            #plt.subplot(int(plt_idx/subplot_x+1),int(plt_idx%subplot_x+1),plt_idx+1)
            #            plt.subplot(subplot_y,subplot_y,plt_idx+1)
            #            plt.hist(np.abs(layer.kernel.numpy().flatten()),bins=100)
            #            plt_idx+=1
            #
            #            print("{} - max: {}, mean: {}".format(layer_name,np.max(np.abs(layer.kernel.numpy())),np.mean(np.abs(layer.kernel.numpy()))))
            #
            #    plt.show()

            if self.f_1st_iter == False and self.f_debug_visual == True:
                #self.visual(t)


                synapse=s_conv1
                neuron=self.n_in

                #synapse=s_fc2
                #neuron=self.n_fc2

                synapse_1=s_conv1
                neuron_1=self.n_conv1

                synapse_2 = s_conv1_1
                neuron_2 = self.n_conv1_1


                #self.debug_visual(synapse, neuron, synapse_1, neuron_1, synapse_2, neuron_2, t)
                self.debug_visual_raster(t)

                #if t==self.ts-1:
                #    plt.figure()
                #    #plt.show()

            ##########
            #
            ##########

            if self.f_1st_iter == False:
                if self.conf.f_comp_act:
                    if self.conf.neural_coding=='RATE':
                        self.comp_act_rate(t)
                    elif self.conf.neural_coding=='WEIGHTED_SPIKE':
                        self.comp_act_ws(t)
                    elif self.conf.neural_coding=='BURST':
                        self.comp_act_pro(t)

                if self.conf.f_isi:
                    self.total_isi += self.get_total_isi()
                    self.total_spike_amp += self.get_total_spike_amp()

                    self.f_out_isi(t)


                if self.conf.f_entropy:
                    #print(a_conv1.numpy().shape)
                    #print(self.n_conv1.out.numpy().shape)
                    #print(self.dict_stat_w['conv1'].shape)
                    #self.dict_stat_w['conv4'][t]=self.n_conv4.out.numpy()
                    #self.dict_stat_w['fc2'][t]=self.n_fc2.out.numpy()

                    for idx_l, l in enumerate(self.layer_name):
                        if l !='fc3':
                            self.dict_stat_w[l][t] = self.list_neuron[l].out.numpy()


                    #print(self.dict_stat_w['conv1'])

                if t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
                    output=self.n_fc3.vmem
                    #self.recoding_ret_val(output)
                    self.recoding_ret_val()


                    #num_spike_count = tf.cast(tf.reduce_sum(self.spike_count,axis=[2]),tf.int32)
                    #num_spike_count = tf.reduce_sum(self.spike_count,axis=[2])

            #print(t, self.n_fc3.last_spike_time.numpy())
            #print(t, self.n_fc3.isi.numpy())

        if self.conf.f_entropy and (not self.f_1st_iter):
            self.cal_entropy()


        if self.f_1st_iter:
            self.f_1st_iter = False

            self.conv1_bn(s_conv1,training=f_training)
            self.conv1_1_bn(s_conv1_1,training=f_training)

            self.conv2_bn(s_conv2,training=f_training)
            self.conv2_1_bn(s_conv2_1,training=f_training)

            self.conv3_bn(s_conv3,training=f_training)
            self.conv3_1_bn(s_conv3_1,training=f_training)
            self.conv3_2_bn(s_conv3_2,training=f_training)

            self.conv4_bn(s_conv4,training=f_training)
            self.conv4_1_bn(s_conv4_1,training=f_training)
            self.conv4_2_bn(s_conv4_2,training=f_training)

            self.conv5_bn(s_conv5,training=f_training)
            self.conv5_1_bn(s_conv5_1,training=f_training)
            self.conv5_2_bn(s_conv5_2,training=f_training)

            self.fc1_bn(s_fc1,training=f_training)
            self.fc2_bn(s_fc2,training=f_training)
            self.fc3_bn(s_fc3,training=f_training)

            return 0


        else:

            if not self.conf.f_pruning_channel:
                self.get_total_residual_vmem()

                spike_zero = tf.reduce_sum(self.spike_count,axis=[0,2])
                #spike_zero = tf.count_nonzero(self.spike_count,axis=[0,2])

                if np.any(spike_zero.numpy() == 0.0):
                #if num_spike_count==0.0:
                    print('spike count 0')
                    #print(num_spike_count.numpy())
                    #a = input("press any key to exit")
                    #os.system("Pause")
                    #raw_input("Press any key to exit")
                    #sys.exit(0)

            #plt.hist(self.n_conv1.vmem.numpy().flatten())
            #plt.show()

            if self.conf.f_pruning_channel:
                for idx_l, l in enumerate(self.layer_name):
                    if 'conv' in l:
                        neuron = self.list_neuron[l]
                        kernel_name = self.layer_name[idx_l+1]

                        if 'conv' in kernel_name:
                            self.idx_pruning_channel[kernel_name] =\
                                self.get_pruning_channel_idx(neuron.get_spike_count())


                            n_remain_channel = tf.shape(self.idx_pruning_channel[kernel_name])[-1]
                            n_original_channel = tf.shape(neuron.get_spike_count())[-1]
                            remain_ratio = n_remain_channel/n_original_channel

                            print('%3d / %3d : %.4f'%(n_remain_channel,n_original_channel,remain_ratio))

                            self.f_idx_pruning_channel[kernel_name]=remain_ratio < self.th_idx_pruning_channel
                            #print(self.f_idx_pruning_channel)

            # first_spike_time visualization
            if self.conf.f_record_first_spike_time and self.conf.f_visual_record_first_spike_time:
                print('first spike time')
                _, axes = plt.subplots(4,4)
                idx_plot=0
                for n_name, n in self.list_neuron.items():
                    if not ('fc3' in n_name):
                        #positive = n.first_spike_time > 0
                        #print(n_name+'] min: '+str(tf.reduce_min(n.first_spike_time[positive]))+', mean: '+str(tf.reduce_mean(n.first_spike_time[positive])))
                        #print(tf.reduce_min(n.first_spike_time[positive]))

                        #positive=n.first_spike_time.numpy().flatten() > 0
                        positive=tf.boolean_mask(n.first_spike_time,n.first_spike_time>0)

                        if not tf.equal(tf.size(positive),0):

                            #min=np.min(n.first_spike_time.numpy().flatten()[positive,])
                            #print(positive.shape)
                            #min=np.min(n.first_spike_time.numpy().flatten()[positive])
                            min=tf.reduce_min(positive)
                            #mean=np.mean(n.first_spike_time.numpy().flatten()[positive,])

                            #if self.conf.f_tc_based:
                            #    fire_s=idx_plot*self.conf.time_fire_start
                            #    fire_e=idx_plot*self.conf.time_fire_start+self.conf.time_fire_duration
                            #else:
                            #    fire_s=idx_plot*self.conf.time_fire_start
                            #    fire_e=idx_plot*self.conf.time_fire_start+self.conf.time_fire_duration

                            #fire_s = n.time_start_fire
                            #fire_e = n.time_end_fire

                            fire_s = idx_plot * self.conf.time_fire_start
                            fire_e = idx_plot * self.conf.time_fire_start + self.conf.time_fire_duration

                            axe=axes.flatten()[idx_plot]
                            #axe.hist(n.first_spike_time.numpy().flatten()[positive],bins=range(fire_s,fire_e,1))
                            axe.hist(positive.numpy().flatten(),bins=range(fire_s,fire_e,1))

                            axe.axvline(x=min.numpy(),color='b', linestyle='dashed')

                            axe.axvline(x=fire_s)
                            axe.axvline(x=fire_e)


                        idx_plot+=1



                # file write raw data
                for n_name, n in self.list_neuron.items():
                    if not ('fc3' in n_name):
                        positive=tf.boolean_mask(n.first_spike_time,n.first_spike_time>0).numpy()

                        fname = './spike_time/spike-time'
                        if self.conf.f_load_time_const:
                            fname += '_train-'+str(self.conf.time_const_num_trained_data)+'_tc-'+str(self.conf.tc)+'_tw-'+str(self.conf.time_window)

                        fname += '_'+n_name+'.csv'
                        f = open(fname,'w')
                        wr = csv.writer(f)
                        wr.writerow(positive)
                        f.close()


                plt.show()


        return self.spike_count



    ###########################################################
    ## SNN output
    ###########################################################

    #
    def recoding_ret_val(self):
        output=self.snn_output_func()
        self.snn_output.scatter_nd_update([self.count_accuracy_time_point],tf.expand_dims(output,0))

        tc_int, tc = self.get_total_spike_count()
        self.total_spike_count_int[self.count_accuracy_time_point]+=tc_int
        self.total_spike_count[self.count_accuracy_time_point]+=tc

        self.count_accuracy_time_point+=1

        #num_spike_count = tf.cast(tf.reduce_sum(self.snn_output,axis=[2]),tf.int32)

    def snn_output_func(self):
        snn_output_func_sel = {
            "SPIKE": self.snn_output_layer.spike_counter,
            "VMEM": self.snn_output_layer.vmem,
            "FIRST_SPIKE_TIME": self.snn_output_layer.first_spike_time
        }
        return snn_output_func_sel[self.conf.snn_output_type]




#    def recoding_ret_val(self, output):
#        self.spike_count.scatter_nd_update([self.count_accuracy_time_point],tf.expand_dims(output,0))
#
#        tc_int, tc = self.get_total_spike_count()
#        #self.total_spike_count_int[self.count_accuracy_time_point]+=tc_int
#        #self.total_spike_count[self.count_accuracy_time_point]+=tc
#
#        self.total_spike_count_int[self.count_accuracy_time_point] = self.total_spike_count_int[self.count_accuracy_time_point]+tc_int
#        self.total_spike_count[self.count_accuracy_time_point] = self.total_spike_count[self.count_accuracy_time_point]+tc
#
#        #self.total_spike_count_int[self.count_accuracy_time_point]=tc_int
#        #self.total_spike_count[self.count_accuracy_time_point]=tc
#
#        #print(self.total_spike_count_int[self.count_accuracy_time_point][0])
#
#        self.count_accuracy_time_point+=1


#####
    def call_snn_pruning(self,inputs,f_training,tw):
        self.count_accuracy_time_point=0

        # reset for sample
        self.reset_neuron()

        spike_count = np.zeros((self.num_accuracy_time_point,)+self.n_fc3.get_spike_count().numpy().shape)

        if self.conf.f_comp_act:
            self.save_ann_act(inputs,f_training)
        plt.clf()

        #
        #for t in range(10):
        #    a_in = self.n_in(inputs,t)
        #    s_conv1 = self.conv1(a_in)
        #    a_conv1 = self.n_conv1(s_conv1,t)
#
#
#        idx_a_in = np.where(tf.reduce_sum(a_in,[0,1,2]).numpy()>0)[0]
#        idx_a_conv1 = np.where(tf.reduce_sum(a_conv1,[0,1,2]).numpy()>0)[0]
#        #print(idx)

#        print(idx_a_in)

        #print(self.idx_pruning_channel[0])
        #print(tf.shape(self.idx_pruning_channel[0]))

        #for idx_l, l in enumerate(self.layer_name):
        for k, v in self.idx_pruning_channel.items():
            if self.f_idx_pruning_channel[k]:
                print(k)
                print(tf.shape(self.list_layer[k].kernel))
                print(tf.shape(tuple(v.flatten())))
                self.kernel_pruning_channel[k] = self.list_layer[k].kernel.numpy()[:,:,tuple(v.flatten()),:]
                self.conv_pruning_channel[k] = partial(self.conv_channel_wise_pruning,name=k)
            else:
                self.conv_pruning_channel[k] = self.list_layer[k]



        #
        for t in range(tw):
            if self.verbose == True:
                print('time: '+str(t))
            #x = tf.reshape(inputs,self._input_shape)

            a_in = self.n_in(inputs,t)

            if self.conf.input_spike_mode == 'WEIGHTED_SPIKE' or self.conf.neural_coding == 'WEIGHTED_SPIKE':
            #if self.conf.neural_coding == 'WEIGHTED_SPIKE':
                #if tf.equal(tf.reduce_max(a_in),0.0):
                if (int)(t%self.conf.p_ws) == 0:
                    self.bias_enable()
                else:
                    self.bias_disable()
            else:
                if self.conf.input_spike_mode == 'BURST':
                    if t==0:
                        self.bias_enable()
                    else:
                        if tf.equal(tf.reduce_max(a_in),0.0):
                            self.bias_enable()
                        else:
                            self.bias_disable()

            s_conv1 = self.conv1(a_in)
            a_conv1 = self.n_conv1(s_conv1,t)

            s_conv1_1 = self.conv_pruning_channel['conv1_1'](a_conv1)
            a_conv1_1 = self.n_conv1_1(s_conv1_1,t)

            if self.conf.f_spike_max_pool:
                p_conv1_1 = lib_snn.spike_max_pool(
                    a_conv1_1,
                    self.n_conv1_1.get_spike_count(),
                    self.dict_shape['conv1_p']
                )
            else:
                p_conv1_1 = self.pool2d(a_conv1_1)

            #s_conv2 = self.conv2(p_conv1_1)
            s_conv2 = self.conv_pruning_channel['conv2'](p_conv1_1)
            a_conv2 = self.n_conv2(s_conv2,t)
            #s_conv2_1 = self.conv2_1(a_conv2)
            s_conv2_1 = self.conv_pruning_channel['conv2_1'](a_conv2)
            a_conv2_1 = self.n_conv2_1(s_conv2_1,t)

            if self.conf.f_spike_max_pool:
                p_conv2_1 = lib_snn.spike_max_pool(
                    a_conv2_1,
                    self.n_conv2_1.get_spike_count(),
                    self.dict_shape['conv2_p']
                )
            else:
                p_conv2_1 = self.pool2d(a_conv2_1)

            #s_conv3 = self.conv3(p_conv2_1)
            s_conv3 = self.conv_pruning_channel['conv3'](p_conv2_1)
            a_conv3 = self.n_conv3(s_conv3,t)
            #s_conv3_1 = self.conv3_1(a_conv3)
            s_conv3_1 = self.conv_pruning_channel['conv3_1'](a_conv3)
            a_conv3_1 = self.n_conv3_1(s_conv3_1,t)
            #s_conv3_2 = self.conv3_2(a_conv3_1)
            s_conv3_2 = self.conv_pruning_channel['conv3_2'](a_conv3_1)
            a_conv3_2 = self.n_conv3_2(s_conv3_2,t)

            if self.conf.f_spike_max_pool:
                p_conv3_2 = lib_snn.spike_max_pool(
                    a_conv3_2,
                    self.n_conv3_2.get_spike_count(),
                    self.dict_shape['conv3_p']
                )
            else:
                p_conv3_2 = self.pool2d(a_conv3_2)

            #s_conv4 = self.conv4(p_conv3_2)
            s_conv4 = self.conv_pruning_channel['conv4'](p_conv3_2)
            a_conv4 = self.n_conv4(s_conv4,t)
            #s_conv4_1 = self.conv4_1(a_conv4)
            s_conv4_1 = self.conv_pruning_channel['conv4_1'](a_conv4)
            a_conv4_1 = self.n_conv4_1(s_conv4_1,t)
            #s_conv4_2 = self.conv4_2(a_conv4_1)
            s_conv4_2 = self.conv_pruning_channel['conv4_2'](a_conv4_1)
            a_conv4_2 = self.n_conv4_2(s_conv4_2,t)

            if self.conf.f_spike_max_pool:
                p_conv4_2 = lib_snn.spike_max_pool(
                    a_conv4_2,
                    self.n_conv4_2.get_spike_count(),
                    self.dict_shape['conv4_p']
                )
            else:
                p_conv4_2 = self.pool2d(a_conv4_2)

            #s_conv5 = self.conv5(p_conv3_2)
            s_conv5 = self.conv_pruning_channel['conv5'](p_conv4_2)
            a_conv5 = self.n_conv5(s_conv5,t)
            #s_conv5_1 = self.conv5_1(a_conv5)
            s_conv5_1 = self.conv_pruning_channel['conv5_1'](a_conv5)
            a_conv5_1 = self.n_conv5_1(s_conv5_1,t)
            #s_conv5_2 = self.conv5_2(a_conv5_1)
            s_conv5_2 = self.conv_pruning_channel['conv5_2'](a_conv5_1)
            a_conv5_2 = self.n_conv5_2(s_conv5_2,t)

            if self.conf.f_spike_max_pool:
                p_conv5_2 = lib_snn.spike_max_pool(
                    a_conv5_2,
                    self.n_conv5_2.get_spike_count(),
                    self.dict_shape['conv5_p']
                )
            else:
                p_conv5_2 = self.pool2d(a_conv5_2)

            flat = tf.layers.flatten(p_conv5_2)

            s_fc1 = self.fc1(flat)
            #s_fc1_bn = self.fc1_bn(s_fc1,training=f_training)
            #a_fc1 = self.n_fc1(s_fc1_bn,t)
            a_fc1 = self.n_fc1(s_fc1,t)

            s_fc2 = self.fc2(a_fc1)
            #s_fc2_bn = self.fc2_bn(s_fc2,training=f_training)
            #a_fc2 = self.n_fc2(s_fc2_bn,t)
            a_fc2 = self.n_fc2(s_fc2,t)

            #print(tf.reduce_mean(a_fc2.numpy()))
            #print(tf.reduce_sum(a_fc2.numpy()))

            s_fc3 = self.fc3(a_fc2)
            #print('a_fc3')
            a_fc3 = self.n_fc3(s_fc3,t)

            if self.conf.f_comp_act:
                if self.conf.neural_coding=='RATE':
                    self.comp_act_rate(t)
                elif self.conf.neural_coding=='WEIGHTED_SPIKE':
                    self.comp_act_ws(t)
                elif self.conf.neural_coding=='BURST':
                    self.comp_act_pro(t)

            if self.conf.f_isi:
                self.total_isi += self.get_total_isi()
                self.total_spike_amp += self.get_total_spike_amp()

                self.f_out_isi(t)


            if self.conf.f_entropy:
                for idx_l, l in enumerate(self.layer_name):
                    if l !='fc3':
                        self.dict_stat_w[l][t] = self.list_neuron[l].out.numpy()


            if t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
                # spike count
                #spike_count[self.count_accuracy_time_point,:,:]=(self.n_fc3.get_spike_count().numpy())
                # vmem
                spike_count[self.count_accuracy_time_point,:,:]=(self.n_fc3.vmem.numpy())

                tc_int, tc = self.get_total_spike_count()

                self.total_spike_count_int[self.count_accuracy_time_point]+=tc_int
                self.total_spike_count[self.count_accuracy_time_point]+=tc

                self.count_accuracy_time_point+=1

                num_spike_count = tf.cast(tf.reduce_sum(spike_count,axis=[2]),tf.int32)

        if self.conf.f_entropy:
            self.cal_entropy()
        else:
            self.get_total_residual_vmem()

            if np.any(num_spike_count.numpy() == 0):
                print('spike count 0')
                #print(num_spike_count.numpy())
                #a = input("press any key to exit")
                #os.system("Pause")
                #raw_input("Press any key to exit")
                #sys.exit(0)

        #plt.hist(self.n_conv1.vmem.numpy().flatten())
        #plt.show()
        return spike_count


    def reset_neuron(self):
        for idx, l in self.list_neuron.items():
            l.reset()

    def debug_visual(self, synapse, neuron, synapse_1, neuron_1, synapse_2, neuron_2, t):

        idx_print_s, idx_print_e = 0, 200

        ax=plt.subplot(3,4,1)
        plt.title('w_sum_in (max)')
        self.plot(t,tf.reduce_max(tf.reshape(synapse,[-1])).numpy(), 'bo')
        #plt.subplot2grid((2,4),(0,1),sharex=ax)
        plt.subplot(3,4,2,sharex=ax)
        plt.title('vmem (max)')
        #self.plot(t,tf.reduce_max(neuron.vmem).numpy(), 'bo')
        self.plot(t,tf.reduce_max(tf.reshape(neuron.vmem,[-1])).numpy(), 'bo')
        self.plot(t,tf.reduce_max(tf.reshape(neuron.vth,[-1])).numpy(), 'ro')
        #self.plot(t,neuron.out.numpy()[neuron.out.numpy()>0].sum(), 'bo')
        #plt.subplot2grid((2,4),(0,2))
        plt.subplot(3,4,3,sharex=ax)
        plt.title('# spikes (total)')
        #spike_rate=neuron.get_spike_count()/t
        self.plot(t,tf.reduce_sum(tf.reshape(neuron.spike_counter_int,[-1])).numpy(), 'bo')
        #self.plot(t,neuron.vmem.numpy()[neuron.vmem.numpy()>0].sum(), 'bo')
        #self.plot(t,tf.reduce_max(spike_rate), 'bo')
        #plt.subplot2grid((2,4),(0,3))
        plt.subplot(3,4,4,sharex=ax)
        plt.title('spike neuron idx')
        plt.grid(True)
        #plt.ylim([0,512])
        plt.ylim([0,neuron.vmem.numpy().flatten()[idx_print_s:idx_print_e].size])
        #plt.ylim([0,int(self.n_fc2.dim[1])])
        plt.xlim([0,self.ts])
        #self.plot(t,np.where(self.n_fc2.out.numpy()==1),'bo')
        #if np.where(self.n_fc2.out.numpy()==1).size == 0:
        idx_fire=np.where(neuron.out.numpy().flatten()[idx_print_s:idx_print_e]!=0.0)
        if not len(idx_fire)==0:
            #print(np.shape(idx_fire))
            #print(idx_fire)
            #print(np.full(np.shape(idx_fire),t))
            self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')
            #self.scatter(t,np.argmax(neuron.get_spike_count().numpy().flatten()),'b')

        addr=0,0,0,6
        ax=plt.subplot(3,4,5)
        #self.plot(t,tf.reduce_max(tf.reshape(synapse_1,[-1])).numpy(), 'bo')
        self.plot(t,synapse_1[addr].numpy(), 'bo')
        plt.subplot(3,4,6,sharex=ax)
        #self.plot(t,tf.reduce_max(tf.reshape(neuron_1.vmem,[-1])).numpy(), 'bo')
        self.plot(t,neuron_1.vmem.numpy()[addr], 'bo')
        self.plot(t,neuron_1.vth.numpy()[addr], 'ro')
        plt.subplot(3,4,7,sharex=ax)
        #self.plot(t,neuron_1.vmem.numpy()[neuron_1.vmem.numpy()>0].sum(), 'bo')
        self.plot(t,neuron_1.spike_counter_int.numpy()[addr], 'bo')
        plt.subplot(3,4,8,sharex=ax)
        plt.grid(True)
        plt.ylim([0,neuron_1.vmem.numpy().flatten()[idx_print_s:idx_print_e].size])
        plt.xlim([0,self.ts])
        idx_fire=np.where(neuron_1.out.numpy().flatten()[idx_print_s:idx_print_e]!=0.0)
        #idx_fire=neuron_1.f_fire.numpy().flatten()[idx_print_s:idx_print_e]
        #idx_fire=neuron_1.f_fire.numpy()[0,0,0,0:10]
        #print(neuron_1.vmem.numpy()[0,0,0,1])
        #print(neuron_1.f_fire.numpy()[0,0,0,1])
        #print(idx_fire)
        if not len(idx_fire)==0:
            self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')

        addr=0,0,0,6
        ax=plt.subplot(3,4,9)
        #self.plot(t,tf.reduce_max(tf.reshape(synapse_2,[-1])).numpy(), 'bo')
        self.plot(t,synapse_2[addr].numpy(), 'bo')
        plt.subplot(3,4,10,sharex=ax)
        #self.plot(t,tf.reduce_max(tf.reshape(neuron_2.vmem,[-1])).numpy(), 'bo')
        self.plot(t,neuron_2.vmem.numpy()[addr], 'bo')
        self.plot(t,neuron_2.vth.numpy()[addr], 'ro')
        plt.subplot(3,4,11,sharex=ax)
        #self.plot(t,neuron_2.vmem.numpy()[neuron_2.vmem.numpy()>0].sum(), 'bo')
        self.plot(t,neuron_2.spike_counter_int.numpy()[addr], 'bo')
        plt.subplot(3,4,12,sharex=ax)
        plt.grid(True)
        plt.ylim([0,neuron_2.vmem.numpy().flatten()[idx_print_s:idx_print_e].size])
        plt.xlim([0,self.ts])
        idx_fire=np.where(neuron_2.out.numpy().flatten()[idx_print_s:idx_print_e]!=0.0)
        #idx_fire=neuron_2.f_fire.numpy().flatten()[idx_print_s:idx_print_e]
        #idx_fire=neuron_2.f_fire.numpy()[0,0,0,0:10]
        #print(neuron_2.vmem.numpy()[0,0,0,1])
        #print(neuron_2.f_fire.numpy()[0,0,0,1])
        #print(idx_fire)
        if not len(idx_fire)==0:
            self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')



#        #plt.subplot2grid((2,4),(1,0))
#        plt.subplot(3,4,9,sharex=ax)
#        self.plot(t,tf.reduce_max(s_fc3).numpy(), 'bo')
#        #plt.subplot2grid((2,4),(1,1))
#        plt.subplot(3,4,10,sharex=ax)
#        self.plot(t,tf.reduce_max(self.n_fc3.vmem).numpy(), 'bo')
#        #plt.subplot2grid((2,4),(1,2))
#        plt.subplot(3,4,11,sharex=ax)
#        self.plot(t,tf.reduce_max(self.n_fc3.get_spike_count()).numpy(), 'bo')
#        plt.subplot(3,4,12)
#        plt.grid(True)
#        #plt.ylim([0,self.n_fc3.dim[1]])
#        plt.ylim([0,self.num_class])
#        plt.xlim([0,self.ts])
#        idx_fire=np.where(self.n_fc3.out.numpy()!=0)[1]


        #self.colors = [self.cmap(self.normalize(value)) for value in self.n_fc3.out.numpy()]

        #if not len(idx_fire)==0:
        #    self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')
        #    self.scatter(t,np.argmax(self.n_fc3.get_spike_count().numpy()),'b')
            #self.scatter(t,np.argmax(self.n_fc3.get_spike_count().numpy()),'g','^')
        #else:
        #    #self.scatter(np.broadcast_to(t,self.n_fc3.vmem.numpy().size),self.n_fc3.vmem.numpy(),self.n_fc3.vmem.numpy())
        #    self.scatter(np.broadcast_to(t,self.n_fc3.vmem.numpy().size),self.n_fc3.vmem.numpy(),np.arange(0,1,0.1))
        #    self.scatter(t,np.argmax(self.n_fc3.vmem.numpy()),'b')





    #@tfplot.autowrap
    def debug_visual_raster(self,t):

        subplot_x, subplot_y = 4, 4

        num_subplot = subplot_x*subplot_y
        idx_print_s, idx_print_e = 0, 100

        if t==0:
            plt_idx=0
            #plt.figure()
            plt.close("dummy")
            _, self.debug_visual_axes = plt.subplots(subplot_y,subplot_x)

            for neuron_name, neuron in self.list_neuron.items():
                if not ('fc3' in neuron_name):
                    self.debug_visual_list_neuron[neuron_name]=neuron

                    axe = self.debug_visual_axes.flatten()[plt_idx]

                    axe.set_ylim([0,neuron.out.numpy().flatten()[idx_print_s:idx_print_e].size])
                    axe.set_xlim([0,self.ts])

                    axe.grid(True)

                    if(plt_idx>0):
                        axe.axvline(x=(plt_idx-1)*self.conf.time_fire_start, color='b')                 # integration
                    axe.axvline(x=plt_idx*self.conf.time_fire_start, color='b')                         # fire start
                    axe.axvline(x=plt_idx*self.conf.time_fire_start+self.conf.time_fire_duration, color='b') # fire end

                    plt_idx+=1

        else:
            plt_idx=0
            for neuron_name, neuron in self.debug_visual_list_neuron.items():

                t_fire_s = plt_idx*self.conf.time_fire_start
                t_fire_e = t_fire_s + self.conf.time_fire_duration

                if t >= t_fire_s and t < t_fire_e :
                    if tf.reduce_sum(neuron.out) != 0.0:
                        idx_fire=tf.where(tf.not_equal(tf.reshape(neuron.out,[-1])[idx_print_s:idx_print_e],tf.constant(0,dtype=tf.float32)))

                    #if tf.size(idx_fire) != 0:
                        axe = self.debug_visual_axes.flatten()[plt_idx]
                        self.scatter(tf.fill(idx_fire.shape,t),idx_fire,'r', axe=axe)

                plt_idx+=1



        if t==self.ts-1:
            plt.figure("dummy")




    def get_pruning_channel_idx(self, act):
        #act = tf.reshape(act,-1)
        _ret_val = np.where(tf.reduce_sum(act,[1,2]).numpy()>0)
        #_ret_val = np.where(tf.reduce_sum(act,[1,2]).numpy()>5)

        ret_val_v = []
        ret_val_length = []
        s_idx = 0
        batch_idx = 0
        for idx, val in enumerate(_ret_val[0]):
            if val > batch_idx:
                if idx == s_idx:
                    ret_val_v.append(list([]))
                    ret_val_length.append(0)
                else:
                    ret_val_v.append(list(_ret_val[1][s_idx:idx]))
                    ret_val_length.append(idx-s_idx)
                s_idx = idx
                batch_idx += 1
        ret_val_v.append(list(_ret_val[1][s_idx:]))
        ret_val_length.append(len(_ret_val[1])-s_idx)

        #print(_ret_val[0])
        #print(_ret_val[1])
        #print(ret_val)

        #print(ret_val)
        #print(tf.shape(ret_val))

        #ret_val = zip(ret_val[0], ret_val[1])

        #for (b, c) in ret_val:
        #    print('%d, %d'%(b,c))

        max = tf.reduce_max(ret_val_length)
        #pad = max-ret_val_length

        #for batch_idx in range(len(ret_val)):
        #    pad = max-ret_val_length[batch_idx]
#
#            ret_val[batch_idx].append(np.zeros(pad))

#        print(ret_val)
        #print(pad)

        ret_val = np.zeros([len(ret_val_v),max],dtype=np.int32)

        for idx in range(len(ret_val_v)):
            ret_val[idx,:ret_val_length[idx]] = ret_val_v[idx]
            #ret_val[idx,:len(ret_val_v[idx])] = ret_val_v[idx]

        return ret_val


    def conv_channel_wise_pruning_new(self, act, kernel_p, bias, idx_channel):
        act_p = act

        [b, h ,w, c] = act_p.shape
        #b = a_conv1.shape[0]
        #h = a_conv1.shape[1]
        #w = a_conv1.shape[2]
        #c = a_conv1.shape[3]
        c_o = kernel_p.shape[3]

        act_p = tf.transpose(act_p, [1,2,0,3])
        act_p = tf.reshape(act_p, [1,h,w,b*c])

        s_p = tf.nn.depthwise_conv2d(act_p, kernel_p,strides=[1,1,1,1],padding='SAME')
        s_p = tf.reshape(s_p, [h,w,b,c,c_o])
        s_p = tf.transpose(s_p, [2,0,1,3,4])
        s_p = tf.reduce_sum(s_p, axis=3)
        s_p = tf.nn.bias_add(s_p, bias)

        return s_p

    def conv_channel_wise_pruning(self, act, name):
        kernel_p = self.kernel_pruning_channel[name]
        bias = self.list_layer[name].bias
        idx_channel = self.idx_pruning_channel[name]


        act_p = np.zeros(act.numpy().shape[:3]+idx_channel[0].shape,dtype=np.float32)

        for b_idx in range(self.conf.batch_size):
            tmp = act.numpy()[:,:,:,tuple(idx_channel[b_idx])]
            tmp = tmp[b_idx]
            #tmp = act.numpy()[b_idx,:,:,tuple(idx_channel[b_idx])]
            #tmp = tf.transpose(tmp,[1,2,0])
            #print(tf.shape(tmp))

            act_p[b_idx,:,:,:len(idx_channel[b_idx])] = tmp

        [b, h ,w, c] = act_p.shape
        #b = a_conv1.shape[0]
        #h = a_conv1.shape[1]
        #w = a_conv1.shape[2]
        #c = a_conv1.shape[3]
        c_o = kernel_p.shape[3]

        act_p = tf.transpose(act_p, [1,2,0,3])
        act_p = tf.reshape(act_p, [1,h,w,b*c])

        s_p = tf.nn.depthwise_conv2d(act_p, kernel_p,strides=[1,1,1,1],padding='SAME')
        s_p = tf.reshape(s_p, [h,w,b,c,c_o])
        s_p = tf.transpose(s_p, [2,0,1,3,4])
        s_p = tf.reduce_sum(s_p, axis=3)
        s_p = tf.nn.bias_add(s_p, bias)

        return s_p


    # training time constant for temporal coding
    def train_time_const(self):

        print("models: train_time_const")

        # train_time_const
        name_layer_prev=''
        for name_layer, layer in self.list_neuron.items():
            if not ('fc3' in name_layer):
                dnn_act = self.dnn_act_list[name_layer]
                self.list_neuron[name_layer].train_time_const_fire(dnn_act)

            if not ('in' in name_layer):
                self.list_neuron[name_layer].set_time_const_integ(self.list_neuron[name_layer_prev].time_const_fire)

            name_layer_prev = name_layer


        # train_time_delay
        name_layer_prev=''
        for name_layer, layer in self.list_neuron.items():
            if not ('fc3' in name_layer or 'in' in name_layer):
                dnn_act = self.dnn_act_list[name_layer]
                self.list_neuron[name_layer].train_time_delay_fire(dnn_act)

            if not ('in' in name_layer or 'conv1' in name_layer):
                self.list_neuron[name_layer].set_time_delay_integ(self.list_neuron[name_layer_prev].time_delay_fire)

            name_layer_prev = name_layer


#        if self.conf.f_tc_based:
#            # update integ and fire time
#            name_layer_prev=''
#            for name_layer, layer in self.list_neuron.items():
#                if not ('fc3' in name_layer):
#                    self.list_neuron[name_layer].set_time_fire(self.list_neuron[name_layer].time_const_fire*self.conf.n_tau_fire_start)
#
#                if not ('in' in name_layer):
#                    self.list_neuron[name_layer].set_time_integ(self.list_neuron[name_layer_prev].time_const_integ*self.conf.n_tau_fire_start)
#
#                name_layer_prev = name_layer


    def get_time_const_train_loss(self):

        loss_prec=0
        loss_min=0
        loss_max=0

        for name_layer, layer in self.list_neuron.items():
            if not ('fc3' in name_layer):
                loss_prec += self.list_neuron[name_layer].loss_prec
                loss_min += self.list_neuron[name_layer].loss_min
                loss_max += self.list_neuron[name_layer].loss_max

        return [loss_prec, loss_min, loss_max]


    # TODO: save activation
    ##############################################################
    # save activation for data-based normalization
    ##############################################################
    # distribution of activation - neuron-wise or channel-wise?
    #def save_dist_activation_neuron_vgg16(model):
    def save_activation(self):

        #path_stat='/home/sspark/Projects/05_SNN/stat/'
        #path_stat='./stat/'
        path_stat=self.conf.path_stat
        #f_name_stat='act_n_train_after_w_norm_max_999'
        #f_name_stat='act_n_train'
        f_name_stat_pre=self.conf.prefix_stat
        stat_conf=['max_999']
        #stat_conf=['max','mean','max_999','max_99','max_98']
        #stat_conf=['max_95','max_90']
        #stat_conf=['max','mean','min','max_75','max_25']
        f_stat=collections.OrderedDict()
        #wr_stat=collections.OrderedDict()

        #
        threads=[]

        for idx_l, l in enumerate(self.layer_name_write_stat):
            for idx_c, c in enumerate(stat_conf):
                key=l+'_'+c

                f_name_stat = f_name_stat_pre+'_'+key
                f_name=os.path.join(path_stat,f_name_stat)
                #f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'w')
                #f_stat[key]=open(path_stat'/'f_name_stat)
                #print(f_name)

                f_stat[key]=open(f_name,'w')
                #wr_stat[key]=csv.writer(f_stat[key])


                #for idx_l, l in enumerate(self.layer_name_write_stat):
                threads.append(threading.Thread(target=self.write_stat, args=(f_stat[key], l, c)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        #f = open('./stat/dist_act_neuron_trainset_test_'+stat_conf+conf.model_name,'wb')

        #wr = csv.writer(f,quoting=csv.QUOTE_NONE,escapechar='\n')
        #wr = csv.writer(f)
        #wr.writerow(['max','min','mean','std','99.9','99','98'])
        #wr.writerow(['max','min','mean','99.9','99','98'])

#        for idx_l, l in enumerate(self.layer_name_write_stat):
#            s_layer=self.dict_stat_w[l].numpy()
#            #print(l)
#            #print(np.shape(s_layer))
#            #print(np.shape(s_layer)[1:])
#            #shape_n=np.shape(s_layer)[1:]
#
#            # before
#            #max=np.max(s_layer,axis=0).flatten()
#            #mean=np.mean(s_layer,axis=0).flatten()
#            #max_999=np.nanpercentile(s_layer,99.9,axis=0).flatten()
#            max_999=np.nanpercentile(s_layer,99.9,axis=0).flatten()
#            #max_99=np.nanpercentile(s_layer,99,axis=0).flatten()
#            #max_98=np.nanpercentile(s_layer,98,axis=0).flatten()
#            #max_95=np.nanpercentile(s_layer,95,axis=0).flatten()
#            #max_90=np.nanpercentile(s_layer,90,axis=0).flatten()
#            #wr_stat[l+'_max'].writerow(max)
#            #wr_stat[l+'_mean'].writerow(mean)
#            wr_stat[l+'_max_999'].writerow(max_999)
#            #wr_stat[l+'_max_99'].writerow(max_99)
#            #wr_stat[l+'_max_98'].writerow(max_98)
#            #wr_stat[l+'_max_95'].writerow(max_95)
#            #wr_stat[l+'_max_90'].writerow(max_90)
#
#
#            #min=np.min(s_layer,axis=0).flatten()
#            #max=np.max(s_layer).flatten()
#            #print(max)
#            #print(np.nanpercentile(s_layer,99.9))
#            #print(np.nanpercentile(s_layer,99))
#            #print(np.nanpercentile(s_layer,98))
#            #print(np.nanpercentile(s_layer,95))
#            #print(np.nanpercentile(s_layer,90))
#            #print(np.mean(s_layer))
#            #plt.hist(s_layer.flatten(), log=True, bins=int(max*2))
#            #plt.show()
#
#            # for after w norm stat
#            #max=np.max(s_layer,axis=0).flatten()
#            #mean=np.mean(s_layer,axis=0).flatten()
#            #min=np.mean(s_layer,axis=0).flatten()
#            #max_25=np.nanpercentile(s_layer,25,axis=0).flatten()
#            #max_75=np.nanpercentile(s_layer,75,axis=0).flatten()
#            #hist=np.histogram(s_layer)
#
#            #wr_stat[l+'_max'].writerow(max)
#            #wr_stat[l+'_mean'].writerow(mean)
#            #wr_stat[l+'_min'].writerow(min)
#            #wr_stat[l+'_max_75'].writerow(max_25)
#            #wr_stat[l+'_max_25'].writerow(max_75)
#            #wr_stat[l+'_hist'].writerow()
#
#
#            #print(np.shape(max))
#
#            #np.savetxt('a',max)
#
#            #wr.writerow([np.max(s_layer,axis=0),np.min(s_layer,axis=0),np.mean(s_layer,axis=0),np.std(s_layer,axis=0),np.nanpercentile(s_layer,99.9,axis=0),np.nanpercentile(s_layer,99,axis=0),np.nanpercentile(s_layer,98,axis=0)])
#            #wr.writerow([np.max(s_layer,axis=0),np.min(s_layer,axis=0),np.mean(s_layer,axis=0),np.nanpercentile(s_layer,99.9,axis=0),np.nanpercentile(s_layer,99,axis=0),np.nanpercentile(s_layer,98,axis=0)])
#            #wr.writerow([max,min,mean,max_999,max_99,max_98])
#
#        for idx_l, l in enumerate(self.layer_name_write_stat):
#            for idx_c, c in enumerate(stat_conf):
#                key=l+'_'+c
#                f_stat[key].close()




    def write_stat(self, f_stat, layer_name, stat_conf_name):
        print('write_stat func')

        l = layer_name
        c = stat_conf_name
        s_layer=self.dict_stat_w[l].numpy()

        self._write_stat(f_stat,s_layer,c)


        #f_stat.close()


    def _write_stat(self, f_stat, s_layer, conf_name):
        print('stat cal: '+conf_name)

        if conf_name=='max':
            stat=np.max(s_layer,axis=0).flatten()
            #stat=tf.reshape(tf.reduce_max(s_layer,axis=0),[-1])
        elif conf_name=='max_999':
            stat=np.nanpercentile(s_layer,99.9,axis=0).flatten()
        elif conf_name=='max_99':
            stat=np.nanpercentile(s_layer,99,axis=0).flatten()
        elif conf_name=='max_98':
            stat=np.nanpercentile(s_layer,98,axis=0).flatten()
        else:
            print('stat confiugration not supported')

        print('stat write')
        wr_stat=csv.writer(f_stat)
        wr_stat.writerow(stat)
        f_stat.close()






    ##############################################################
    def plot_dist_activation_vgg16(self):
    #        plt.subplot2grid((6,3),(0,0))
    #        plt.hist(model.stat_a_conv1)
    #        plt.subplot2grid((6,3),(0,1))
    #        plt.hist(model.stat_a_conv1_1)
    #        plt.subplot2grid((6,3),(1,0))
    #        plt.hist(model.stat_a_conv2)
    #        plt.subplot2grid((6,3),(1,1))
    #        plt.hist(model.stat_a_conv2_1)
    #        plt.subplot2grid((6,3),(2,0))
    #        plt.hist(model.stat_a_conv3)
    #        plt.subplot2grid((6,3),(2,1))
    #        plt.hist(model.stat_a_conv3_1)
    #        plt.subplot2grid((6,3),(2,2))
    #        plt.hist(model.stat_a_conv3_1)
    #        plt.subplot2grid((6,3),(3,0))
    #        plt.hist(model.stat_a_conv4)
    #        plt.subplot2grid((6,3),(3,1))
    #        plt.hist(model.stat_a_conv4_1)
    #        plt.subplot2grid((6,3),(3,2))
    #        plt.hist(model.stat_a_conv4_2)
    #        plt.subplot2grid((6,3),(4,0))
    #        plt.hist(model.stat_a_conv5)
    #        plt.subplot2grid((6,3),(4,1))
    #        plt.hist(model.stat_a_conv5_1)
    #        plt.subplot2grid((6,3),(4,2))
    #        plt.hist(model.stat_a_conv5_2)
    #        plt.subplot2grid((6,3),(5,0))
    #        plt.hist(model.stat_a_fc1)
    #        plt.subplot2grid((6,3),(5,1))
    #        plt.hist(model.stat_a_fc2)
    #        plt.subplot2grid((6,3),(5,2))
    #        plt.hist(model.stat_a_fc3)

        print(np.shape(self.dict_stat_w['fc1']))

        plt.hist(self.dict_stat_w["fc1"][:,120],bins=100)
        plt.show()


# distribution of activation - layer-wise
def save_dist_activation_vgg16(model):
    layer_name=[
        #model.stat_a_conv1,
        #model.stat_a_conv1_1
        #model.stat_a_conv2,
        #model.stat_a_conv2_1,
        #model.stat_a_conv3,
        #model.stat_a_conv3_1,
        #model.stat_a_conv3_2,
        #model.stat_a_conv4,
        #model.stat_a_conv4_1,
        #model.stat_a_conv4_2,
        #model.stat_a_conv5,
        #model.stat_a_conv5_1,
        #model.stat_a_conv5_2,
        #model.stat_a_fc1,
        #model.stat_a_fc2,
        model.stat_a_fc3
    ]

    f = open('./stat/dist_act_neuron_trainset_fc_'+conf.model_name,'wb')
    wr = csv.writer(f)
    wr.writerow(['max','min','mean','std','99.9','99','98'])

    for _, s_layer in enumerate(layer_name):
        wr.writerow([np.max(s_layer),np.min(s_layer),np.mean(s_layer),np.std(s_layer),np.nanpercentile(s_layer,99.9),np.nanpercentile(s_layer,99),np.nanpercentile(s_layer,98)])
    f.close()


# distribution of activation - neuron-wise
def save_dist_activation_neuron_vgg16(model):
    # delete later
    layer_name=[
    #    'conv1',
    #    'conv1_1',
    #    'conv2',
    #    'conv2_1'
    #    'conv3',
    #    'conv3_1',
    #    'conv3_2',
    #    'conv4',
    #    'conv4_1',
    #    'conv4_2'
        'conv5',
        'conv5_1',
        'conv5_2',
        'fc1',
        'fc2',
        'fc3'
    ]

    path_stat='/home/sspark/Projects/05_SNN/stat/'
    #f_name_stat='act_n_train_after_w_norm_max_999'
    f_name_stat='act_n_train'
    #stat_conf=['max_999']
    #stat_conf=['max','mean','max_999','max_99','max_98']
    stat_conf=['max_95','max_90']
    #stat_conf=['max','mean','min','max_75','max_25']
    f_stat=collections.OrderedDict()
    wr_stat=collections.OrderedDict()

    for idx_l, l in enumerate(layer_name):
        for idx_c, c in enumerate(stat_conf):
            key=l+'_'+c
            f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+conf.model_name,'w')
            wr_stat[key]=csv.writer(f_stat[key])

    #f = open('./stat/dist_act_neuron_trainset_test_'+stat_conf+conf.model_name,'wb')

    #wr = csv.writer(f,quoting=csv.QUOTE_NONE,escapechar='\n')
    #wr = csv.writer(f)
    #wr.writerow(['max','min','mean','std','99.9','99','98'])
    #wr.writerow(['max','min','mean','99.9','99','98'])

    for idx_l, l in enumerate(layer_name):
        s_layer=model.dict_stat_w[l]
        #print(np.shape(s_layer))
        #print(np.shape(s_layer)[1:])
        #shape_n=np.shape(s_layer)[1:]

        # before
        #max=np.max(s_layer,axis=0).flatten()
        #mean=np.mean(s_layer,axis=0).flatten()
        #max_999=np.nanpercentile(s_layer,99.9,axis=0).flatten()
        #max_99=np.nanpercentile(s_layer,99,axis=0).flatten()
        #max_98=np.nanpercentile(s_layer,98,axis=0).flatten()
        max_95=np.nanpercentile(s_layer,95,axis=0).flatten()
        max_90=np.nanpercentile(s_layer,90,axis=0).flatten()
        #wr_stat[l+'_max'].writerow(max)
        #wr_stat[l+'_mean'].writerow(mean)
        #wr_stat[l+'_max_999'].writerow(max_999)
        #wr_stat[l+'_max_99'].writerow(max_99)
        #wr_stat[l+'_max_98'].writerow(max_98)
        wr_stat[l+'_max_95'].writerow(max_95)
        wr_stat[l+'_max_90'].writerow(max_90)


        #min=np.min(s_layer,axis=0).flatten()
        #max=np.max(s_layer).flatten()
        #print(max)
        #print(np.nanpercentile(s_layer,99.9))
        #print(np.nanpercentile(s_layer,99))
        #print(np.nanpercentile(s_layer,98))
        #print(np.nanpercentile(s_layer,95))
        #print(np.nanpercentile(s_layer,90))
        #print(np.mean(s_layer))
        #plt.hist(s_layer.flatten(), log=True, bins=int(max*2))
        #plt.show()

        # for after w norm stat
        #max=np.max(s_layer,axis=0).flatten()
        #mean=np.mean(s_layer,axis=0).flatten()
        #min=np.mean(s_layer,axis=0).flatten()
        #max_25=np.nanpercentile(s_layer,25,axis=0).flatten()
        #max_75=np.nanpercentile(s_layer,75,axis=0).flatten()
        #hist=np.histogram(s_layer)

        #wr_stat[l+'_max'].writerow(max)
        #wr_stat[l+'_mean'].writerow(mean)
        #wr_stat[l+'_min'].writerow(min)
        #wr_stat[l+'_max_75'].writerow(max_25)
        #wr_stat[l+'_max_25'].writerow(max_75)
        #wr_stat[l+'_hist'].writerow()


        #print(np.shape(max))

        #np.savetxt('a',max)

        #wr.writerow([np.max(s_layer,axis=0),np.min(s_layer,axis=0),np.mean(s_layer,axis=0),np.std(s_layer,axis=0),np.nanpercentile(s_layer,99.9,axis=0),np.nanpercentile(s_layer,99,axis=0),np.nanpercentile(s_layer,98,axis=0)])
        #wr.writerow([np.max(s_layer,axis=0),np.min(s_layer,axis=0),np.mean(s_layer,axis=0),np.nanpercentile(s_layer,99.9,axis=0),np.nanpercentile(s_layer,99,axis=0),np.nanpercentile(s_layer,98,axis=0)])
        #wr.writerow([max,min,mean,max_999,max_99,max_98])

    for idx_l, l in enumerate(layer_name):
        for idx_c, c in enumerate(stat_conf):
            key=l+'_'+c
            f_stat[key].close()










