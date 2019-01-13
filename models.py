import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import regularizers

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops

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

#
# noinspection PyUnboundLocalVariable
class CIFARModel_CNN(tfe.Network):
    def __init__(self, data_format, conf):
        super(CIFARModel_CNN, self).__init__(name='')

        self.data_format = data_format
        self.conf = conf
        self.num_class = self.conf.num_class

        self.f_1st_iter = True
        self.verbose = conf.verbose
        self.f_debug_visual = conf.verbose_visual
        self.f_done_preproc = False

        self.kernel_size = 3
        self.fanin_conv = self.kernel_size*self.kernel_size
        #self.fanin_conv = self.kernel_size*self.kernel_size/9

        self.tw=conf.time_step

        self.accuracy_time_point = range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval)
        self.accuracy_time_point.append(conf.time_step)
        #self.num_accuracy_time_point = int(math.ceil(float(conf.time_step)/float(conf.time_step_save_interval))
        self.num_accuracy_time_point = len(self.accuracy_time_point)


        #
        self.f_skip_bn = self.conf.f_fused_bn

        self.layer_name=[
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
                'PROPOSED': int(5-np.log2(self.conf.n_init_vth))
            }

            self.spike_amp_kind = type_spike_amp_kind[self.conf.neural_coding]+1
            self.total_spike_amp=np.zeros(self.spike_amp_kind)

            type_spike_amp_bin = {
                'RATE': np.power(0.5,range(0,self.spike_amp_kind+1)),
                'WEIGHTED_SPIKE': np.power(0.5,range(0,self.spike_amp_kind+1)),
                'PROPOSED': 32*np.power(0.5,range(0,self.spike_amp_kind+1))
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

        regularizer_type = {
            'L1': regularizers.l1_regularizer(conf.lamb),
            'L2': regularizers.l2_regularizer(conf.lamb)
        }
        kernel_regularizer = regularizer_type[self.conf.regularizer]
        kernel_initializer = initializers.xavier_initializer(True)
        #kernel_initializer = initializers.variance_scaling_initializer(factor=2.0,mode='FAN_IN')    # MSRA init. = He init

        self.layer_list=collections.OrderedDict()
        self.layer_list['conv1'] = self.track_layer(tf.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))


        self.layer_list['conv1_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['conv1_1'] = self.track_layer(tf.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv1_1_bn'] = self.track_layer(tf.layers.BatchNormalization())

        self.layer_list['conv2'] = self.track_layer(tf.layers.Conv2D(128,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv2_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['conv2_1'] = self.track_layer(tf.layers.Conv2D(128,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv2_1_bn'] = self.track_layer(tf.layers.BatchNormalization())

        self.layer_list['conv3'] = self.track_layer(tf.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv3_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['conv3_1'] = self.track_layer(tf.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv3_1_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['conv3_2'] = self.track_layer(tf.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv3_2_bn'] = self.track_layer(tf.layers.BatchNormalization())

        self.layer_list['conv4'] = self.track_layer(tf.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv4_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['conv4_1'] = self.track_layer(tf.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv4_1_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['conv4_2'] = self.track_layer(tf.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv4_2_bn'] = self.track_layer(tf.layers.BatchNormalization())

        self.layer_list['conv5'] = self.track_layer(tf.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv5_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['conv5_1'] = self.track_layer(tf.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv5_1_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['conv5_2'] = self.track_layer(tf.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.layer_list['conv5_2_bn'] = self.track_layer(tf.layers.BatchNormalization())


        self.layer_list['fc1'] = self.track_layer(tf.layers.Dense(512,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer))

        self.layer_list['fc1_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['fc2'] = self.track_layer(tf.layers.Dense(512,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer))
        self.layer_list['fc2_bn'] = self.track_layer(tf.layers.BatchNormalization())

        #self.layer_list['fc3'] = self.track_layer(tf.layers.Dense(self.num_class,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer))
        self.layer_list['fc3'] = self.track_layer(tf.layers.Dense(self.num_class,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer))
        self.layer_list['fc3_bn'] = self.track_layer(tf.layers.BatchNormalization())

        self.dropout_conv = self.track_layer(tf.layers.Dropout(0.3))
        self.dropout_conv2 = self.track_layer(tf.layers.Dropout(0.4))
        self.dropout = self.track_layer(tf.layers.Dropout(0.5))


        # remove later
        self.conv1=self.layer_list['conv1']
        self.conv1_bn=self.layer_list['conv1_bn']
        self.conv1_1=self.layer_list['conv1_1']
        self.conv1_1_bn=self.layer_list['conv1_1_bn']
        self.conv2=self.layer_list['conv2']
        self.conv2_bn=self.layer_list['conv2_bn']
        self.conv2_1=self.layer_list['conv2_1']
        self.conv2_1_bn=self.layer_list['conv2_1_bn']
        self.conv3=self.layer_list['conv3']
        self.conv3_bn=self.layer_list['conv3_bn']
        self.conv3_1=self.layer_list['conv3_1']
        self.conv3_1_bn=self.layer_list['conv3_1_bn']
        self.conv3_2=self.layer_list['conv3_2']
        self.conv3_2_bn=self.layer_list['conv3_2_bn']
        self.conv4=self.layer_list['conv4']
        self.conv4_bn=self.layer_list['conv4_bn']
        self.conv4_1=self.layer_list['conv4_1']
        self.conv4_1_bn=self.layer_list['conv4_1_bn']
        self.conv4_2=self.layer_list['conv4_2']
        self.conv4_2_bn=self.layer_list['conv4_2_bn']
        self.conv5=self.layer_list['conv5']
        self.conv5_bn=self.layer_list['conv5_bn']
        self.conv5_1=self.layer_list['conv5_1']
        self.conv5_1_bn=self.layer_list['conv5_1_bn']
        self.conv5_2=self.layer_list['conv5_2']
        self.conv5_2_bn=self.layer_list['conv5_2_bn']
        self.fc1=self.layer_list['fc1']
        self.fc1_bn=self.layer_list['fc1_bn']
        self.fc2=self.layer_list['fc2']
        self.fc2_bn=self.layer_list['fc2_bn']
        self.fc3=self.layer_list['fc3']
        self.fc3_bn=self.layer_list['fc3_bn']

        pooling_type= {
            'max': self.track_layer(tf.layers.MaxPooling2D((2,2),(2,2),padding='SAME',data_format=data_format)),
            'avg': self.track_layer(tf.layers.AveragePooling2D((2,2),(2,2),padding='SAME',data_format=data_format))
        }

        self.pool2d = pooling_type[self.conf.pooling]

        #self.pool2d_arg ='max': self.track_layer(tf.nn.max_poolwith_argmax((2,2),(2,2),padding='SAME',data_format=data_format)),

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

        self.shape_out_fc1 = tensor_shape.TensorShape([self.conf.batch_size,512])
        self.shape_out_fc2 = tensor_shape.TensorShape([self.conf.batch_size,512])
        self.shape_out_fc3 = tensor_shape.TensorShape([self.conf.batch_size,self.num_class])


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
        self.dict_shape_one_batch['conv1']=[1,]+self.shape_out_conv1.as_list()[1:]
        self.dict_shape_one_batch['conv1_1']=[1,]+self.shape_out_conv1_1.as_list()[1:]
        self.dict_shape_one_batch['conv1_p']=[1,]+self.shape_out_conv1_p.as_list()[1:]
        self.dict_shape_one_batch['conv2']=[1,]+self.shape_out_conv2.as_list()[1:]
        self.dict_shape_one_batch['conv2_1']=[1,]+self.shape_out_conv2_1.as_list()[1:]
        self.dict_shape_one_batch['conv2_p']=[1,]+self.shape_out_conv2_p.as_list()[1:]
        self.dict_shape_one_batch['conv3']=[1,]+self.shape_out_conv3.as_list()[1:]
        self.dict_shape_one_batch['conv3_1']=[1,]+self.shape_out_conv3_1.as_list()[1:]
        self.dict_shape_one_batch['conv3_2']=[1,]+self.shape_out_conv3_2.as_list()[1:]
        self.dict_shape_one_batch['conv3_p']=[1,]+self.shape_out_conv3_p.as_list()[1:]
        self.dict_shape_one_batch['conv4']=[1,]+self.shape_out_conv4.as_list()[1:]
        self.dict_shape_one_batch['conv4_1']=[1,]+self.shape_out_conv4_1.as_list()[1:]
        self.dict_shape_one_batch['conv4_2']=[1,]+self.shape_out_conv4_2.as_list()[1:]
        self.dict_shape_one_batch['conv4_p']=[1,]+self.shape_out_conv4_p.as_list()[1:]
        self.dict_shape_one_batch['conv5']=[1,]+self.shape_out_conv5.as_list()[1:]
        self.dict_shape_one_batch['conv5_1']=[1,]+self.shape_out_conv5_1.as_list()[1:]
        self.dict_shape_one_batch['conv5_2']=[1,]+self.shape_out_conv5_2.as_list()[1:]
        self.dict_shape_one_batch['conv5_p']=[1,]+self.shape_out_conv5_p.as_list()[1:]
        self.dict_shape_one_batch['fc1']=[1,]+self.shape_out_fc1.as_list()[1:]
        self.dict_shape_one_batch['fc2']=[1,]+self.shape_out_fc2.as_list()[1:]
        self.dict_shape_one_batch['fc3']=[1,]+self.shape_out_fc3.as_list()[1:]

        #
        self.dict_stat_r=collections.OrderedDict()  # read
        self.dict_stat_w=collections.OrderedDict()  # write


        if self.conf.f_entropy:
            self.dict_stat_w['conv1']=np.zeros([self.conf.time_step,]+self.shape_out_conv1.as_list()[1:])
            self.dict_stat_w['conv1_1']=np.zeros([self.conf.time_step,]+self.shape_out_conv1_1.as_list()[1:])
            self.dict_stat_w['conv2']=np.zeros([self.conf.time_step,]+self.shape_out_conv2.as_list()[1:])
            self.dict_stat_w['conv2_1']=np.zeros([self.conf.time_step,]+self.shape_out_conv2_1.as_list()[1:])
            self.dict_stat_w['conv3']=np.zeros([self.conf.time_step,]+self.shape_out_conv3.as_list()[1:])
            self.dict_stat_w['conv3_1']=np.zeros([self.conf.time_step,]+self.shape_out_conv3_1.as_list()[1:])
            self.dict_stat_w['conv3_2']=np.zeros([self.conf.time_step,]+self.shape_out_conv3_2.as_list()[1:])
            self.dict_stat_w['conv4']=np.zeros([self.conf.time_step,]+self.shape_out_conv4.as_list()[1:])
            self.dict_stat_w['conv4_1']=np.zeros([self.conf.time_step,]+self.shape_out_conv4_1.as_list()[1:])
            self.dict_stat_w['conv4_2']=np.zeros([self.conf.time_step,]+self.shape_out_conv4_2.as_list()[1:])
            self.dict_stat_w['conv5']=np.zeros([self.conf.time_step,]+self.shape_out_conv5.as_list()[1:])
            self.dict_stat_w['conv5_1']=np.zeros([self.conf.time_step,]+self.shape_out_conv5_1.as_list()[1:])
            self.dict_stat_w['conv5_2']=np.zeros([self.conf.time_step,]+self.shape_out_conv5_2.as_list()[1:])
            self.dict_stat_w['fc1']=np.zeros([self.conf.time_step,]+self.shape_out_fc1.as_list()[1:])
            self.dict_stat_w['fc2']=np.zeros([self.conf.time_step,]+self.shape_out_fc2.as_list()[1:])
            self.dict_stat_w['fc3']=np.zeros([self.conf.time_step,]+self.shape_out_fc3.as_list()[1:])


            self.arr_length = [2,3,4,5,8,10]
            self.total_entropy=np.zeros([len(self.arr_length),len(self.layer_name)+1])


        if self.conf.f_write_stat or self.conf.f_comp_act:
            self.dict_stat_w['conv1']=np.zeros([1,]+self.shape_out_conv1.as_list()[1:])
            self.dict_stat_w['conv1_1']=np.zeros([1,]+self.shape_out_conv1_1.as_list()[1:])
            self.dict_stat_w['conv2']=np.zeros([1,]+self.shape_out_conv2.as_list()[1:])
            self.dict_stat_w['conv2_1']=np.zeros([1,]+self.shape_out_conv2_1.as_list()[1:])
            self.dict_stat_w['conv3']=np.zeros([1,]+self.shape_out_conv3.as_list()[1:])
            self.dict_stat_w['conv3_1']=np.zeros([1,]+self.shape_out_conv3_1.as_list()[1:])
            self.dict_stat_w['conv3_2']=np.zeros([1,]+self.shape_out_conv3_2.as_list()[1:])
            self.dict_stat_w['conv4']=np.zeros([1,]+self.shape_out_conv4.as_list()[1:])
            self.dict_stat_w['conv4_1']=np.zeros([1,]+self.shape_out_conv4_1.as_list()[1:])
            self.dict_stat_w['conv4_2']=np.zeros([1,]+self.shape_out_conv4_2.as_list()[1:])
            self.dict_stat_w['conv5']=np.zeros([1,]+self.shape_out_conv5.as_list()[1:])
            self.dict_stat_w['conv5_1']=np.zeros([1,]+self.shape_out_conv5_1.as_list()[1:])
            self.dict_stat_w['conv5_2']=np.zeros([1,]+self.shape_out_conv5_2.as_list()[1:])
            self.dict_stat_w['fc1']=np.zeros([1,]+self.shape_out_fc1.as_list()[1:])
            self.dict_stat_w['fc2']=np.zeros([1,]+self.shape_out_fc2.as_list()[1:])
            self.dict_stat_w['fc3']=np.zeros([1,]+self.shape_out_fc3.as_list()[1:])



        self.conv_p=collections.OrderedDict()
        self.conv_p['conv1_p']=np.empty(self.dict_shape['conv1_p'],dtype=np.float32)
        self.conv_p['conv2_p']=np.empty(self.dict_shape['conv2_p'],dtype=np.float32)
        self.conv_p['conv3_p']=np.empty(self.dict_shape['conv3_p'],dtype=np.float32)
        self.conv_p['conv4_p']=np.empty(self.dict_shape['conv4_p'],dtype=np.float32)
        self.conv_p['conv5_p']=np.empty(self.dict_shape['conv5_p'],dtype=np.float32)

        # neurons
        if self.conf.nn_mode == 'SNN':
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

            self.neuron_list=collections.OrderedDict()

            #self.neuron_list['in'] = self.track_layer(lib_snn.Neuron(self.input_shape_snn,'IN',1,self.conf))
            self.neuron_list['in'] = self.track_layer(lib_snn.Neuron(self.input_shape_snn,'IN',1,self.conf,nc))


            self.neuron_list['conv1'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv1,n_type,self.fanin_conv,self.conf,nc))
            self.neuron_list['conv1_1'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv1_1,n_type,self.fanin_conv,self.conf,nc))


            self.neuron_list['conv2'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv2,n_type,self.fanin_conv,self.conf,nc))
            self.neuron_list['conv2_1'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv2_1,n_type,self.fanin_conv,self.conf,nc))

            self.neuron_list['conv3'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv3,n_type,self.fanin_conv,self.conf,nc))
            self.neuron_list['conv3_1'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv3_1,n_type,self.fanin_conv,self.conf,nc))
            self.neuron_list['conv3_2'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv3_2,n_type,self.fanin_conv,self.conf,nc))

            #nc = 'PROPOSED'
            #self.conf.n_init_vth=0.125

            self.neuron_list['conv4'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv4,n_type,self.fanin_conv,self.conf,nc))
            self.neuron_list['conv4_1'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv4_1,n_type,self.fanin_conv,self.conf,nc))
            self.neuron_list['conv4_2'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv4_2,n_type,self.fanin_conv,self.conf,nc))

            #self.conf.n_init_vth=0.125/2.0

            self.neuron_list['conv5'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv5,n_type,self.fanin_conv,self.conf,nc))
            self.neuron_list['conv5_1'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv5_1,n_type,self.fanin_conv,self.conf,nc))
            self.neuron_list['conv5_2'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv5_2,n_type,self.fanin_conv,self.conf,nc))




            self.neuron_list['fc1'] = self.track_layer(lib_snn.Neuron(self.shape_out_fc1,n_type,512,self.conf,nc))
            self.neuron_list['fc2'] = self.track_layer(lib_snn.Neuron(self.shape_out_fc2,n_type,512,self.conf,nc))
            #self.neuron_list['fc3'] = self.track_layer(lib_snn.Neuron(self.shape_out_fc3,n_type,512,self.conf))
            self.neuron_list['fc3'] = self.track_layer(lib_snn.Neuron(self.shape_out_fc3,'OUT',512,self.conf,nc))


            # modify later
            self.n_in = self.neuron_list['in'];

            self.n_conv1 = self.neuron_list['conv1']
            self.n_conv1_1 = self.neuron_list['conv1_1']
            #self.n_conv1_1 = tf.contrib.eager.defun(self.neuron_list['conv1_1'])

            self.n_conv2 = self.neuron_list['conv2']
            self.n_conv2_1 = self.neuron_list['conv2_1']

            self.n_conv3 = self.neuron_list['conv3']
            self.n_conv3_1 = self.neuron_list['conv3_1']
            self.n_conv3_2 = self.neuron_list['conv3_2']

            self.n_conv4 = self.neuron_list['conv4']
            self.n_conv4_1 = self.neuron_list['conv4_1']
            self.n_conv4_2 = self.neuron_list['conv4_2']

            self.n_conv5 = self.neuron_list['conv5']
            self.n_conv5_1 = self.neuron_list['conv5_1']
            self.n_conv5_2 = self.neuron_list['conv5_2']

            self.n_fc1 = self.neuron_list['fc1']
            self.n_fc2 = self.neuron_list['fc2']
            self.n_fc3 = self.neuron_list['fc3']

        #
        self.cmap=matplotlib.cm.get_cmap('viridis')
        #self.normalize=matplotlib.colors.Normalize(vmin=min(self.n_fc3.vmem),vmax=max(self.n_fc3.vmem))


    def call(self, inputs, f_training):

        nn_mode = {
            'ANN': self.call_ann,
            'SNN': self.call_snn
        }

        ret_val = nn_mode[self.conf.nn_mode](inputs,f_training)

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

    #
    def w_norm_layer_wise(self):

        f_norm=np.max
        #f_norm=np.mean


        for idx_l, l in enumerate(self.layer_name):
            if idx_l==0:
                self.norm[l]=f_norm(self.dict_stat_r[l])
            else:
                self.norm[l]=f_norm(self.dict_stat_r.values()[idx_l])/f_norm(self.dict_stat_r.values()[idx_l-1])

            self.norm_b[l]=f_norm(self.dict_stat_r[l])

        if self.conf.f_vth_conp:
            for idx_l, l in enumerate(self.layer_name):
                #self.neuron_list[l].set_vth(np.broadcast_to(self.conf.n_init_vth*1.0 + 0.1*self.dict_stat_r[l]/f_norm(self.dict_stat_r[l]),self.dict_shape[l]))
                self.neuron_list[l].set_vth(np.broadcast_to(self.dict_stat_r[l]/f_norm(self.dict_stat_r[l]),self.dict_shape[l]))
                #self.neuron_list[l].set_vth(np.broadcast_to(self.dict_stat_r[l]/np.broadcast_to(f_norm(self.dict_stat_r[l]),self.dict_stat_r[l].shape)   ,self.dict_shape[l]))

        #self.print_act_d()
        # print
        for k, v in self.norm.iteritems():
            print(k +': '+str(v))

        for k, v in self.norm_b.iteritems():
            print(k +': '+str(v))


        deep_layer_const = 1.0

        self.conv1.kernel = self.conv1.kernel/self.norm['conv1']*deep_layer_const
        self.conv1.bias = self.conv1.bias/self.norm_b['conv1']
        self.conv1_1.kernel = self.conv1_1.kernel/self.norm['conv1_1']*deep_layer_const
        self.conv1_1.bias = self.conv1_1.bias/self.norm_b['conv1_1']

        self.conv2.kernel = self.conv2.kernel/self.norm['conv2']*deep_layer_const
        self.conv2.bias = self.conv2.bias/self.norm_b['conv2']
        self.conv2_1.kernel = self.conv2_1.kernel/self.norm['conv2_1']*deep_layer_const
        self.conv2_1.bias = self.conv2_1.bias/self.norm_b['conv2_1']

        self.conv3.kernel = self.conv3.kernel/self.norm['conv3']*deep_layer_const
        self.conv3.bias = self.conv3.bias/self.norm_b['conv3']
        self.conv3_1.kernel = self.conv3_1.kernel/self.norm['conv3_1']*deep_layer_const
        self.conv3_1.bias = self.conv3_1.bias/self.norm_b['conv3_1']
        self.conv3_2.kernel = self.conv3_2.kernel/self.norm['conv3_2']*deep_layer_const
        self.conv3_2.bias = self.conv3_2.bias/self.norm_b['conv3_2']

        self.conv4.kernel = self.conv4.kernel/self.norm['conv4']*deep_layer_const
        self.conv4.bias = self.conv4.bias/self.norm_b['conv4']
        self.conv4_1.kernel = self.conv4_1.kernel/self.norm['conv4_1']*deep_layer_const
        self.conv4_1.bias = self.conv4_1.bias/self.norm_b['conv4_1']
        self.conv4_2.kernel = self.conv4_2.kernel/self.norm['conv4_2']*deep_layer_const
        self.conv4_2.bias = self.conv4_2.bias/self.norm_b['conv4_2']

        self.conv5.kernel = self.conv5.kernel/self.norm['conv5']*deep_layer_const
        self.conv5.bias = self.conv5.bias/self.norm_b['conv5']
        self.conv5_1.kernel = self.conv5_1.kernel/self.norm['conv5_1']*deep_layer_const
        self.conv5_1.bias = self.conv5_1.bias/self.norm_b['conv5_1']
        self.conv5_2.kernel = self.conv5_2.kernel/self.norm['conv5_2']*deep_layer_const
        self.conv5_2.bias = self.conv5_2.bias/self.norm_b['conv5_2']

        self.fc1.kernel = self.fc1.kernel/self.norm['fc1']*deep_layer_const
        self.fc1.bias = self.fc1.bias/self.norm_b['fc1']
        self.fc2.kernel = self.fc2.kernel/self.norm['fc2']*deep_layer_const
        self.fc2.bias = self.fc2.bias/self.norm_b['fc2']
        self.fc3.kernel = self.fc3.kernel/self.norm['fc3']*deep_layer_const
        self.fc3.bias = self.fc3.bias/self.norm_b['fc3']

    #
    def data_based_w_norm(self):
        #stat_file='./stat/dist_act_trainset_'+self.conf.model_name
        #stat_file='./stat/dist_act_neuron_trainset_'+self.conf.model_name
        #stat_file='./stat/act_n_trainset_test_'+self.conf.model_name

        path_stat='./stat/'
        f_name_stat='act_n_train'
        stat_conf=['max','mean','max_999','max_99','max_98']
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

            f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'r')
            r_stat[key]=csv.reader(f_stat[key])

            for row in r_stat[key]:
                self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.dict_shape[l][1:])

            #if self.conf.f_ws:
                #self.dict_stat_r['conv1']=self.dict_stat_r['conv1']/(1-1/np.power(2,8))

            #print(np.shape(self.dict_stat_r[l]))


        self.w_norm_layer_wise()





    #
    def load_act_after_w_norm(self):
        path_stat='./stat/'
        f_name_stat='act_n_train_after_w_norm_max_999'
        f_stat=collections.OrderedDict()
        r_stat=collections.OrderedDict()

        # choose one
        #stat='max'
        stat='mean'
        #stat='min'
        #stat='max_75'
        #stat='max_25'

        for idx_l, l in enumerate(self.layer_name):
            key=l+'_'+stat

            f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'r')
            r_stat[key]=csv.reader(f_stat[key])

            for row in r_stat[key]:
                self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.dict_shape[l][1:])

                #print(self.dict_stat_r[l])


    def print_act_after_w_norm(self):
        self.load_act_after_w_norm()

        self.print_act_d()


    def temporal_norm(self):
        print('Temporal normalization')
        for key, value in self.layer_list.iteritems():
            if self.conf.f_fused_bn:
                if not ('bn' in key):
                    value.kernel=value.kernel/self.tw
                    value.bias=value.bias/self.tw
            else:
                value.kernel=value.kernel/self.tw
                value.bias=value.bias/self.tw


    #
    def preproc_ann_norm(self):
        if self.conf.f_fused_bn:
            self.fused_bn()

        #self.print_model()

        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()

        #self.print_model()


    def call_ann(self,inputs,f_training):
        #print(type(inputs))
        if self.f_1st_iter == False and self.conf.nn_mode=='ANN':
            if self.f_done_preproc == False:
                self.f_done_preproc=True
                self.print_model_conf()
                self.preproc_ann_norm()

            self.f_skip_bn=self.conf.f_fused_bn
        else:
            self.f_skip_bn=False

        x = tf.reshape(inputs,self._input_shape)
        s_conv1 = self.conv1(x)
        if self.f_skip_bn:
            s_conv1_bn = s_conv1
        else:
            s_conv1_bn = self.conv1_bn(s_conv1,training=f_training)

        a_conv1 = tf.nn.relu(s_conv1_bn)
        if f_training:
            a_conv1 = self.dropout_conv(a_conv1,training=f_training)
        s_conv1_1 = self.conv1_1(a_conv1)
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

        s_flat = tf.layers.flatten(p_conv5_2)

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
            if ('bn' in self.conf.model_name) or ('ro' in self.conf.model_name) :
                s_fc3_bn = self.fc3_bn(s_fc3,training=f_training)
            else:
                s_fc3_bn = s_fc3
        #a_fc3 = s_fc3_bn
        if 'ro' in self.conf.model_name:
            a_fc3 = tf.nn.relu(s_fc3_bn)
        else:
            a_fc3 = s_fc3_bn

        #if f_training:
        #   x = self.dropout(x,training=f_training)


        # write stat
        if (self.conf.f_write_stat) and (not self.f_1st_iter):
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
            self.dict_stat_w['conv5']=np.append(self.dict_stat_w['conv5'],a_conv5.numpy(),axis=0)
            self.dict_stat_w['conv5_1']=np.append(self.dict_stat_w['conv5_1'],a_conv5_1.numpy(),axis=0)
            self.dict_stat_w['conv5_2']=np.append(self.dict_stat_w['conv5_2'],a_conv5_2.numpy(),axis=0)
            self.dict_stat_w['fc1']=np.append(self.dict_stat_w['fc1'],a_fc1.numpy(),axis=0)
            self.dict_stat_w['fc2']=np.append(self.dict_stat_w['fc2'],a_fc2.numpy(),axis=0)
            self.dict_stat_w['fc3']=np.append(self.dict_stat_w['fc3'],a_fc3.numpy(),axis=0)


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



        if self.f_1st_iter and self.conf.nn_mode=='ANN':
            print('1st iter')
            self.f_1st_iter = False
            self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)



        return a_out


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
        for idx, (key, value) in enumerate(self.dict_stat_r.iteritems()):
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

        #print(gamma)
        #print(beta)

        #return kernel, bias


    def preproc_ann_to_snn(self):
        print('preprocessing: ANN to SNN')
        if self.conf.f_fused_bn:
            self.fused_bn()

        #self.print_model()
        #print(np.max(self.layer_list.values()[0].kernel))
        #self.temporal_norm()
        #print(np.max(self.layer_list.values()[0].kernel))


        #self.print_model()
        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()

        #if self.conf.f_comp_act:
        #    self.load_act_after_w_norm()

        #self.print_act_after_w_norm()

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

#        plt.ion()
#        plt.figure()
#        plt.hist(self.fc3.kernel.numpy())
#        plt.show()
#        plt.draw()
#        plt.pause(0.001)
#
#        w_in_sum_fc3 = tf.reduce_sum(tf.maximum(self.fc3.kernel,0),axis=[0])
#        plt.subplot(212)
#        plt.hist(w_in_sum_fc3)
#        plt.draw()
#        plt.pause(0.001)
#
#        w_in_max_fc3 = tf.reduce_max(w_in_sum_fc3)
#        self.fc3.kernel = self.fc3.kernel/w_in_max_fc3
#
#        plt.ioff()
#        #plt.show()


    def plot(self, x, y, mark):
        #plt.ion()
        #plt.hist(self.n_fc3.vmem)
        plt.plot(x, y, mark)
        plt.draw()
        plt.pause(0.00000001)
        #plt.ioff()


    def scatter(self, x, y, color, marker='o'):
        #plt.ion()
        plt.scatter(x, y, c=color, s=1, marker=marker)
        plt.draw()
        plt.pause(0.00000001)
        #plt.ioff()

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
        for idx_n, (nn, n) in enumerate(self.neuron_list.iteritems()):
            idx=idx_n-1
            if nn!='in' or nn!='fc3':
                self.total_residual_vmem[idx]+=tf.reduce_sum(tf.abs(n.vmem))
                self.total_residual_vmem[len-1]+=self.total_residual_vmem[idx]

    def get_total_isi(self):
        isi_count=np.zeros(self.conf.time_step)

        for idx_n, (nn, n) in enumerate(self.neuron_list.iteritems()):
            if nn!='in' or nn!='fc3':
                isi_count_n = np.bincount(np.int32(n.isi.numpy().flatten()))
                isi_count_n.resize(self.conf.time_step)
                isi_count = isi_count + isi_count_n

        return isi_count


    def f_out_isi(self,t):
        for idx_n, (nn, n) in enumerate(self.neuron_list.iteritems()):
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

        for idx_n, (nn, n) in enumerate(self.neuron_list.iteritems()):
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

        for idx_n, (nn, n) in enumerate(self.neuron_list.iteritems()):
            idx=idx_n-1
            if nn!='in':
                spike_count_int[idx]=tf.reduce_sum(n.get_spike_count_int())
                spike_count_int[len-1]+=spike_count_int[idx]
                spike_count[idx]=tf.reduce_sum(n.get_spike_count())
                spike_count[len-1]+=spike_count[idx]

                #print(spike_count_int[idx])

        return [spike_count_int, spike_count]

    #def bias_norm_weighted_spike(self):
        #for k, l in self.layer_list.iteritems():
        #    if not 'bn' in k:
        #        l.bias = l.bias/(1-1/np.power(2,8))
        #        #l.bias = l.bias/8.0
        #self.layer_list['conv1'].bias=self.layer_list['conv1'].bias/8.0

    def bias_norm_proposed_method(self):
        for k, l in self.layer_list.iteritems():
            if not 'bn' in k:
                l.bias = l.bias*self.conf.n_init_vth
                #l.bias = l.bias/200
                #l.bias = l.bias*0.0

    def bias_enable(self):
        for k, l in self.layer_list.iteritems():
            if not 'bn' in k:
                l.use_bias = True

    def bias_disable(self):
        for k, l in self.layer_list.iteritems():
            if not 'bn' in k:
                l.use_bias = False


    def comp_act_rate(self,t):
        self.total_comp_act[t,-1]=0.0
        for idx_l, l in enumerate(self.layer_name):
            if l !='fc3':
                self.total_comp_act[t,idx_l]=np.mean(np.abs(self.neuron_list[l].spike_counter.numpy().flatten()/(float)(t+1)-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,-1]+=self.total_comp_act[t,idx_l]


        #l='conv1'
        #print(self.neuron_list[l].spike_counter.numpy().flatten())
        #print(self.dict_stat_w[l].flatten())

    def comp_act_ws(self,t):
        self.total_comp_act[t,-1]=0.0
        for idx_l, l in enumerate(self.layer_name):
            if l !='fc3':
                #self.total_comp_act[t,idx_l]=np.mean(np.abs(self.neuron_list[l].spike_counter.numpy().flatten()/((float)(t+1)/(float)(self.conf.p_ws))-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,idx_l]=np.mean(np.abs(self.neuron_list[l].spike_counter.numpy().flatten()/((float)(t+1)/(float)(self.conf.p_ws))-self.dict_stat_w[l].flatten()))
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



    def call_snn(self,inputs,f_training):

        count_accuracy_time_point=0

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

            spike_count = np.zeros((self.num_accuracy_time_point,)+self.n_fc3.get_spike_count().numpy().shape)

            if self.conf.f_comp_act:
                self.save_ann_act(inputs,f_training)
            #print((self.num_accuracy_time_point,)+self.n_fc3.get_spike_count().numpy().shape)
            #print('spike count')
            #print(spike_count.shape)
            #print(np.shape(spike_count))
            #print(spike_count)

            #spike_count = np.zeros(self.n_fc3.get_spike_count())
            #spike_count = np.zeros(2+np.shape(self.n_fc3.get_spike_count().numpy()))

            #print(self.n_fc3.get_spike_count().numpy().shape)
            #spike_count = np.zeros((2,)+self.n_fc3.get_spike_count().numpy().shape)

            #print(np.shape(spike_count))
        plt.clf()

        #
        for t in range(self.tw):
            if self.verbose == True:
                print('time: '+str(t))
            #x = tf.reshape(inputs,self._input_shape)

            a_in = self.n_in(inputs,t)


            #if self.conf.f_real_value_input_snn:
            #    a_in = inputs
            #else:
            #    a_in = self.n_in(inputs,t)


            if self.conf.input_spike_mode == 'WEIGHTED_SPIKE' or self.conf.neural_coding == 'WEIGHTED_SPIKE':
            #if self.conf.neural_coding == 'WEIGHTED_SPIKE':
                #if tf.equal(tf.reduce_max(a_in),0.0):
                if (int)(t%self.conf.p_ws) == 0:
                    self.bias_enable()
                else:
                    self.bias_disable()
            else:
                if self.conf.input_spike_mode == 'PROPOSED':
                    if t==0:
                        self.bias_enable()
                    else:
                        if tf.equal(tf.reduce_max(a_in),0.0):
                            self.bias_enable()
                        else:
                            self.bias_disable()

            #print(a_in)

            s_conv1 = self.conv1(a_in)
            a_conv1 = self.n_conv1(s_conv1,t)

            s_conv1_1 = self.conv1_1(a_conv1)
            a_conv1_1 = self.n_conv1_1(s_conv1_1,t)

            if self.conf.f_spike_max_pool:
                tmp = tf.reshape(self.n_conv1_1.get_spike_count(),[1,-1,]+self.dict_shape['conv1_1'].as_list()[2:])
                _, arg = tf.nn.max_pool_with_argmax(tmp,(1,2,2,1),(1,2,2,1),padding='SAME')
                arg = tf.reshape(arg,self.dict_shape['conv1_p'])
                a_conv1_1_f = tf.reshape(a_conv1_1,[-1])
                p_conv1_1 = tf.convert_to_tensor(a_conv1_1_f.numpy()[arg],dtype=tf.float32)
            else:
                p_conv1_1 = self.pool2d(a_conv1_1)

            s_conv2 = self.conv2(p_conv1_1)
            a_conv2 = self.n_conv2(s_conv2,t)
            s_conv2_1 = self.conv2_1(a_conv2)
            a_conv2_1 = self.n_conv2_1(s_conv2_1,t)

            if self.conf.f_spike_max_pool:
                tmp = tf.reshape(self.n_conv2_1.get_spike_count(),[1,-1,]+self.dict_shape['conv2_1'].as_list()[2:])
                _, arg = tf.nn.max_pool_with_argmax(tmp,(1,2,2,1),(1,2,2,1),padding='SAME')
                arg = tf.reshape(arg,self.dict_shape['conv2_p'])
                a_conv2_1_f = tf.reshape(a_conv2_1,[-1])
                p_conv2_1 = tf.convert_to_tensor(a_conv2_1_f.numpy()[arg],dtype=tf.float32)
            else:
                p_conv2_1 = self.pool2d(a_conv2_1)

            s_conv3 = self.conv3(p_conv2_1)
            a_conv3 = self.n_conv3(s_conv3,t)
            s_conv3_1 = self.conv3_1(a_conv3)
            a_conv3_1 = self.n_conv3_1(s_conv3_1,t)
            s_conv3_2 = self.conv3_2(a_conv3_1)
            a_conv3_2 = self.n_conv3_2(s_conv3_2,t)

            if self.conf.f_spike_max_pool:
                tmp = tf.reshape(self.n_conv3_2.get_spike_count(),[1,-1,]+self.dict_shape['conv3_2'].as_list()[2:])
                _, arg = tf.nn.max_pool_with_argmax(tmp,(1,2,2,1),(1,2,2,1),padding='SAME')
                arg = tf.reshape(arg,self.dict_shape['conv3_p'])
                a_conv3_2_f = tf.reshape(a_conv3_2,[-1])
                p_conv3_2 = tf.convert_to_tensor(a_conv3_2_f.numpy()[arg],dtype=tf.float32)
            else:
                p_conv3_2 = self.pool2d(a_conv3_2)


            s_conv4 = self.conv4(p_conv3_2)
            a_conv4 = self.n_conv4(s_conv4,t)
            s_conv4_1 = self.conv4_1(a_conv4)
            a_conv4_1 = self.n_conv4_1(s_conv4_1,t)
            s_conv4_2 = self.conv4_2(a_conv4_1)
            a_conv4_2 = self.n_conv4_2(s_conv4_2,t)

            if self.conf.f_spike_max_pool:
                tmp = tf.reshape(self.n_conv4_2.get_spike_count(),[1,-1,]+self.dict_shape['conv4_2'].as_list()[2:])
                _, arg = tf.nn.max_pool_with_argmax(tmp,(1,2,2,1),(1,2,2,1),padding='SAME')
                arg = tf.reshape(arg,self.dict_shape['conv4_p'])
                a_conv4_2_f = tf.reshape(a_conv4_2,[-1])
                p_conv4_2 = tf.convert_to_tensor(a_conv4_2_f.numpy()[arg],dtype=tf.float32)
            else:
                p_conv4_2 = self.pool2d(a_conv4_2)

            s_conv5 = self.conv5(p_conv4_2)
            a_conv5 = self.n_conv5(s_conv5,t)
            s_conv5_1 = self.conv5_1(a_conv5)
            a_conv5_1 = self.n_conv5_1(s_conv5_1,t)
            s_conv5_2 = self.conv5_2(a_conv5_1)
            a_conv5_2 = self.n_conv5_2(s_conv5_2,t)

            if self.conf.f_spike_max_pool:
                tmp = tf.reshape(self.n_conv5_2.get_spike_count(),[1,-1,]+self.dict_shape['conv5_2'].as_list()[2:])
                _, arg = tf.nn.max_pool_with_argmax(tmp,(1,2,2,1),(1,2,2,1),padding='SAME')
                arg = tf.reshape(arg,self.dict_shape['conv5_p'])
                a_conv5_2_f = tf.reshape(a_conv5_2,[-1])
                p_conv5_2 = tf.convert_to_tensor(a_conv5_2_f.numpy()[arg],dtype=tf.float32)
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

            s_fc3 = self.fc3(a_fc2)
            #print('a_fc3')
            a_fc3 = self.n_fc3(s_fc3,t)



            if self.f_1st_iter == False and self.f_debug_visual == True:
                #self.visual(t)

                # ipnut neuron
                #synapse=s_conv1
                #neuron=self.n_in

                # conv1
                synapse=s_conv1
                neuron=self.n_conv1

                # conv2
                #synapse=s_conv2
                #neuron=self.n_conv2

                # conv3
                #synapse=s_conv3
                #neuron=self.n_conv3

                # conv4
                #synapse=s_conv4
                #neuron=self.n_conv4

                # conv5
                #synapse=s_conv5
                #neuron=self.n_conv5

                # conv5
                #synapse=s_fc1
                #neuron=self.n_fc1

                #fc2
                #synapse=s_fc2
                #neuron=self.n_fc2

                ax=plt.subplot(3,4,1)
#                plt.title('w_sum_in (max)')
#                self.plot(t,tf.reduce_max(tf.reshape(synapse,[-1])).numpy(), 'bo')
#                #plt.subplot2grid((2,4),(0,1),sharex=ax)
#                plt.subplot(3,4,2,sharex=ax)
#                plt.title('vmem (max)')
#                #self.plot(t,tf.reduce_max(neuron.vmem).numpy(), 'bo')
#                self.plot(t,tf.reduce_max(tf.reshape(neuron.vmem,[-1])).numpy(), 'bo')
#                #plt.subplot2grid((2,4),(0,2))
#                plt.subplot(3,4,3,sharex=ax)
#                plt.title('# spikes (max)')
#                #spike_rate=neuron.get_spike_count()/t
#                self.plot(t,tf.reduce_max(tf.reshape(neuron.get_spike_count(),[-1])).numpy(), 'bo')
#                #self.plot(t,tf.reduce_max(spike_rate), 'bo')
#                #plt.subplot2grid((2,4),(0,3))
#                plt.subplot(3,4,4,sharex=ax)
#                plt.title('spike neuron idx')
#                plt.grid(True)
#                #plt.ylim([0,512])
#                plt.ylim([0,neuron.vmem.numpy().flatten().size])
#                #plt.ylim([0,int(self.n_fc2.dim[1])])
#                plt.xlim([0,self.tw])
#                #self.plot(t,np.where(self.n_fc2.out.numpy()==1),'bo')
#                #if np.where(self.n_fc2.out.numpy()==1).size == 0:
#                idx_fire=np.where(neuron.out.numpy().flatten()!=0)
#                if not len(idx_fire)==0:
#                    #print(np.shape(idx_fire))
#                    #print(idx_fire)
#                    #print(np.full(np.shape(idx_fire),t))
#                    self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')
#                    #self.scatter(t,np.argmax(neuron.get_spike_count().numpy().flatten()),'b')
#
#

                # conv1
                #synapse=s_conv1
                #neuron=self.n_conv1

                # conv1_1
                #synapse=s_conv1_1
                #neuron=self.n_conv1_1

                synapse=s_fc2
                neuron=self.n_fc2

                ax=plt.subplot(3,4,5)
                plt.title('w_sum_in (max)')
                self.plot(t,tf.reduce_max(tf.reshape(synapse,[-1])).numpy(), 'bo')
                #plt.subplot2grid((2,4),(0,1),sharex=ax)
                plt.subplot(3,4,6,sharex=ax)
                plt.title('vmem (max)')
                #self.plot(t,tf.reduce_max(neuron.vmem).numpy(), 'bo')
                self.plot(t,tf.reduce_max(tf.reshape(neuron.vmem,[-1])).numpy(), 'bo')
                #self.plot(t,neuron.out.numpy()[neuron.out.numpy()>0].sum(), 'bo')
                #plt.subplot2grid((2,4),(0,2))
                plt.subplot(3,4,7,sharex=ax)
                plt.title('# spikes (max)')
                #spike_rate=neuron.get_spike_count()/t
                #self.plot(t,tf.reduce_max(tf.reshape(neuron.get_spike_count(),[-1])).numpy(), 'bo')
                self.plot(t,neuron.vmem.numpy()[neuron.vmem.numpy()>0].sum(), 'bo')
                #self.plot(t,tf.reduce_max(spike_rate), 'bo')
                #plt.subplot2grid((2,4),(0,3))
                plt.subplot(3,4,8,sharex=ax)
                plt.title('spike neuron idx')
                plt.grid(True)
                #plt.ylim([0,512])
                plt.ylim([0,neuron.vmem.numpy().flatten().size])
                #plt.ylim([0,int(self.n_fc2.dim[1])])
                plt.xlim([0,self.tw])
                #self.plot(t,np.where(self.n_fc2.out.numpy()==1),'bo')
                #if np.where(self.n_fc2.out.numpy()==1).size == 0:
                idx_fire=np.where(neuron.out.numpy().flatten()!=0)
                if not len(idx_fire)==0:
                    #print(np.shape(idx_fire))
                    #print(idx_fire)
                    #print(np.full(np.shape(idx_fire),t))
                    self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')
                    #self.scatter(t,np.argmax(neuron.get_spike_count().numpy().flatten()),'b')



                #plt.subplot2grid((2,4),(1,0))
                plt.subplot(3,4,9,sharex=ax)
                self.plot(t,tf.reduce_max(s_fc3).numpy(), 'bo')
                #plt.subplot2grid((2,4),(1,1))
                plt.subplot(3,4,10,sharex=ax)
                self.plot(t,tf.reduce_max(self.n_fc3.vmem).numpy(), 'bo')
                #plt.subplot2grid((2,4),(1,2))
                plt.subplot(3,4,11,sharex=ax)
                self.plot(t,tf.reduce_max(self.n_fc3.get_spike_count()).numpy(), 'bo')
                plt.subplot(3,4,12)
                plt.grid(True)
                #plt.ylim([0,self.n_fc3.dim[1]])
                plt.ylim([0,self.num_class])
                plt.xlim([0,self.tw])
                idx_fire=np.where(self.n_fc3.out.numpy()!=0)[1]


                #self.colors = [self.cmap(self.normalize(value)) for value in self.n_fc3.out.numpy()]

                if not len(idx_fire)==0:
                    self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')
                    self.scatter(t,np.argmax(self.n_fc3.get_spike_count().numpy()),'b')
                    #self.scatter(t,np.argmax(self.n_fc3.get_spike_count().numpy()),'g','^')
                else:
                    #self.scatter(np.broadcast_to(t,self.n_fc3.vmem.numpy().size),self.n_fc3.vmem.numpy(),self.n_fc3.vmem.numpy())
                    self.scatter(np.broadcast_to(t,self.n_fc3.vmem.numpy().size),self.n_fc3.vmem.numpy(),np.arange(0,1,0.1))
                    self.scatter(t,np.argmax(self.n_fc3.vmem.numpy()),'b')


            if self.f_1st_iter == False:
                if self.conf.f_comp_act:
                    if self.conf.neural_coding=='RATE':
                        self.comp_act_rate(t)
                    elif self.conf.neural_coding=='WEIGHTED_SPIKE':
                        self.comp_act_ws(t)
                    elif self.conf.neural_coding=='PROPOSED':
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
                            self.dict_stat_w[l][t] = self.neuron_list[l].out.numpy()


                    #print(self.dict_stat_w['conv1'])

                if t==self.accuracy_time_point[count_accuracy_time_point]-1:
                    # spike count
                    #spike_count[count_accuracy_time_point,:,:]=(self.n_fc3.get_spike_count().numpy())
                    # vmem
                    spike_count[count_accuracy_time_point,:,:]=(self.n_fc3.vmem.numpy())

                    tc_int, tc = self.get_total_spike_count()

                    self.total_spike_count_int[count_accuracy_time_point]+=tc_int
                    self.total_spike_count[count_accuracy_time_point]+=tc

                    #print('time '+str(t))
                    #print(spike_count.shape)
                    #print(self.n_fc3.get_spike_count().numpy())
                    #print(spike_count)
                    count_accuracy_time_point+=1

               #spike_count = self.n_fc3.get_spike_count()
                #print(spike_count)


#                num_spike_count = tf.cast(tf.reduce_sum(spike_count,axis=[1]),tf.int32)
                #spike_count=self.n_fc3.get_spike_count()
                    num_spike_count = tf.cast(tf.reduce_sum(spike_count,axis=[2]),tf.int32)

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
        self.n_in.reset()

        self.n_conv1.reset()
        self.n_conv1_1.reset()

        self.n_conv2.reset()
        self.n_conv2_1.reset()

        self.n_conv3.reset()
        self.n_conv3_1.reset()
        self.n_conv3_2.reset()

        self.n_conv4.reset()
        self.n_conv4_1.reset()
        self.n_conv4_2.reset()

        self.n_conv5.reset()
        self.n_conv5_1.reset()
        self.n_conv5_2.reset()

        self.n_fc1.reset()
        self.n_fc2.reset()
        self.n_fc3.reset()


#
class MNISTModel_CNN(tfe.Network):
    def __init__(self, data_format, conf):
        super(MNISTModel_CNN, self).__init__(name='')

        self.data_format = data_format
        self.conf = conf

        self.f_1st_iter = True

        if self.data_format == 'channels_first':
            self._input_shape = [-1,1,28,28]   # MNIST
            #self._input_shape = [-1,3,32,32]    # CIFAR-10
        else:
            assert self.data_format == 'channels_last'
            self._input_shape = [-1,28,28,1]
            #self._input_shape = [-1,32,32,3]

        use_bias = conf.use_bias

        activation_sel = {
            'ANN': tf.nn.relu,
            'SNN': None
        }

        activation = activation_sel[self.conf.nn_mode]
        #ret_val = nn_mode[self.conf.nn_mode](inputs,f_training)

        #activation = tf.nn.relu
        #kernel_regularizer = regularizers.l2_regularizer(conf.lamb)
        #kernel_regularizer = regularizers.l1_regularizer(conf.lamb)

        regularizer_type = {
            'L1': regularizers.l1_regularizer(conf.lamb),
            'L2': regularizers.l2_regularizer(conf.lamb)
        }

        kernel_regularizer = regularizer_type[self.conf.regularizer]
        kernel_initializer = initializers.xavier_initializer(True)

        #self.conv1 = self.track_layer(tf.layers.Conv2D(32,5,data_format=data_format,activation=tf.nn.relu,use_bias=False,kernel_regularizer=kernel_regularizer,kernel_initializer=initializers.xavier_initializer(True)))
        #self.conv2 = self.track_layer(tf.layers.Conv2D(64,5,data_format=data_format,activation=tf.nn.relu,use_bias=False,kernel_regularizer=regularizers.l2_regularizer(conf.lamb),kernel_initializer=initializers.xavier_initializer(True)))
        #self.fc1 = self.track_layer(tf.layers.Dense(1024,activation=tf.nn.relu,use_bias=False,kernel_regularizer=regularizers.l2_regularizer(conf.lamb),kernel_initializer=initializers.xavier_initializer(True)))
        #self.fc2 = self.track_layer(tf.layers.Dense(10,use_bias=False,kernel_regularizer=regularizers.l2_regularizer(conf.lamb),kernel_initializer=initializers.xavier_initializer(True)))
        #self.dropout = self.track_layer(tf.layers.Dropout(0.5))

        self.conv1 = self.track_layer(tf.layers.Conv2D(32,3,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.conv2 = self.track_layer(tf.layers.Conv2D(64,3,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='same'))
        self.fc1 = self.track_layer(tf.layers.Dense(1024,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer))
        self.fc2 = self.track_layer(tf.layers.Dense(10,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer))
        self.dropout = self.track_layer(tf.layers.Dropout(0.5))

        #print(tf.contrib.framework.current_arg_scope())

        #with arg_scope(
        #    [tf.layers.Conv2D, tf.layers.Dense],
        #    use_bias=True
        #):

        ### cnn_model_1 - max pooling
        ### accu.: 99.23 %
        ### test time: 44.244 s
        #use_bias = False
        #self.conv1 = self.track_layer(tf.layers.Conv2D(32,5,data_format=data_format,activation=tf.nn.relu,use_bias=use_bias))
        #self.conv2 = self.track_layer(tf.layers.Conv2D(64,5,data_format=data_format,activation=tf.nn.relu,use_bias=use_bias))
        #self.fc1 = self.track_layer(tf.layers.Dense(1024,activation=tf.nn.relu,use_bias=use_bias))
        #self.fc2 = self.track_layer(tf.layers.Dense(10,use_bias=use_bias))
        #self.dropout = self.track_layer(tf.layers.Dropout(0.5))

        pooling_type= {
            'max': self.track_layer(tf.layers.MaxPooling2D((2,2),(2,2),padding='SAME',data_format=data_format)),
            'avg': self.track_layer(tf.layers.AveragePooling2D((2,2),(2,2),padding='SAME',data_format=data_format))
        }

        self.pool2d = pooling_type[self.conf.pooling]



        input_shape_one_sample = tensor_shape.TensorShape([1,self._input_shape[1],self._input_shape[2],self._input_shape[3]])
        print(input_shape_one_sample)

        self.shape_out_conv1 = util.cal_output_shape_Conv2D(self.data_format,input_shape_one_sample,32,3,1)
        self.shape_out_conv1_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv1,2,2)

        self.shape_out_conv2 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv1_p,64,3,1)
        self.shape_out_conv2_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv2,2,2)

        self.shape_out_fc1 = tensor_shape.TensorShape([1,1024])
        self.shape_out_fc2 = tensor_shape.TensorShape([1,10])


        print('Conv: '+str(self.shape_out_conv1))
        print('Pool:  '+str(self.shape_out_conv1_p))

        print('Conv: '+str(self.shape_out_conv2))
        print('Pool:  '+str(self.shape_out_conv2_p))

        print('Fc:   '+str(self.shape_out_fc1))
        print('Fc:   '+str(self.shape_out_fc2))



        # neurons
        if self.conf.nn_mode == 'SNN':
            print('SNN mode')


            
            self.input_shape_snn = [1] + self._input_shape[1:]
#
#            self.shape_out_conv1 = util.cal_output_shape_Conv2D('channels_first',self.input_shape_snn,32,3,1)
#            self.shape_out_conv1_p = util.cal_output_shape_Pooling2D('channels_first',self.shape_out_conv1,2,2)
#            self.shape_out_conv2 = util.cal_output_shape_Conv2D('channels_first',self.shape_out_conv1_p,64,3,1)
#            self.shape_out_conv2_p = util.cal_output_shape_Pooling2D('channels_first',self.shape_out_conv2,2,2)
#
#            self.shape_out_fc1 = tensor_shape.TensorShape([1,1024])
#            self.shape_out_fc2 = tensor_shape.TensorShape([1,10])

            print(self.shape_out_conv1)
            print(self.shape_out_conv1_p)
            print(self.shape_out_conv2)
            print(self.shape_out_conv2_p)
            print(self.shape_out_fc1)
            print(self.shape_out_fc2)

            vth = self.conf.n_init_vth
            vinit = self.conf.n_init_vinit
            vrest = self.conf.n_init_vrest
            time_step = self.conf.time_step
            n_type = self.conf.n_type

            #print('neuron input add')
            self.n_in = self.track_layer(lib_snn.Neuron(self.input_shape_snn,'IN',self.conf))
            self.n_conv1 = self.track_layer(lib_snn.Neuron(self.shape_out_conv1,'LIF',self.conf))
            self.n_conv2 = self.track_layer(lib_snn.Neuron(self.shape_out_conv2,'LIF',self.conf))
            self.n_fc1 = self.track_layer(lib_snn.Neuron(self.shape_out_fc1,'LIF',self.conf))
            self.n_out = self.track_layer(lib_snn.Neuron(self.shape_out_fc2,'LIF',self.conf))

    def call(self, inputs, f_training):
        nn_mode = {
            'ANN': self.call_ann,
            'SNN': self.call_snn
        }

        ret_val = nn_mode[self.conf.nn_mode](inputs,f_training)

        return ret_val

    def call_ann(self,inputs,f_training):
        x = tf.reshape(inputs,self._input_shape)
        x = self.conv1(x)
        x = self.pool2d(x)
        x = self.conv2(x)
        x = self.pool2d(x)
        x = tf.layers.flatten(x)
        x = self.fc1(x)
        if f_training:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

    def call_snn(self,inputs,f_training):

        if self.f_1st_iter == False:
            self.reset_neuron()
        else:
            self.f_1st_iter = False

        for t in range(self.conf.time_step):
            act_in = self.n_in(inputs,t)

            _act_conv1 = self.conv1(act_in)
            act_conv1 = self.n_conv1(_act_conv1,t)
            act_conv1_p = self.pool2d(act_conv1)

            _act_conv2 = self.conv2(act_conv1_p)
            act_conv2 = self.n_conv2(_act_conv2,t)
            act_conv2_p = self.pool2d(act_conv2)
            act_conv2_p_f = tf.layers.flatten(act_conv2_p)

            _act_fc1 = self.fc1(act_conv2_p_f)
            act_fc1 = self.n_fc1(_act_fc1,t)

            _act_fc2 = self.fc2(act_fc1)
            act_out = self.n_out(_act_fc2,t)

            #print(self.n_out.get_spike_count())

            if not self.conf.output_dir == '':

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.histogram('spike_in', act_in, step=t)
                    tf.contrib.summary.histogram('spike_conv1', act_conv1, step=t)
                    tf.contrib.summary.histogram('spike_conv2', act_conv2, step=t)
                    tf.contrib.summary.histogram('spike_fc1', act_fc1, step=t)
                    tf.contrib.summary.histogram('spike_out', act_out, step=t)

                    tf.contrib.summary.histogram('spike_count_in', self.n_in.get_spike_count(), step=t)
                    tf.contrib.summary.histogram('spike_count_conv1', self.n_conv1.get_spike_count(), step=t)
                    tf.contrib.summary.histogram('spike_count_conv2', self.n_conv2.get_spike_count(), step=t)
                    tf.contrib.summary.histogram('spike_count_fc1', self.n_fc1.get_spike_count(), step=t)
                    tf.contrib.summary.histogram('spike_count_out', self.n_out.get_spike_count(), step=t)

                    tf.contrib.summary.scalar('spike_count_in_total', tf.reduce_sum(self.n_in.get_spike_count()),step=t)
                    tf.contrib.summary.scalar('spike_count_conv1_total', tf.reduce_sum(self.n_conv1.get_spike_count()),step=t)
                    tf.contrib.summary.scalar('spike_count_conv2_total', tf.reduce_sum(self.n_conv2.get_spike_count()),step=t)
                    tf.contrib.summary.scalar('spike_count_fc1_total', tf.reduce_sum(self.n_fc1.get_spike_count()),step=t)
                    tf.contrib.summary.scalar('spike_count_out_total', tf.reduce_sum(self.n_out.get_spike_count()),step=t)

                    tf.contrib.summary.histogram('act_conv1_pooling', act_conv1_p, step=t)
                    tf.contrib.summary.histogram('act_conv2_pooling', act_conv2_p, step=t)

                    tf.contrib.summary.histogram('vmem_in', self.n_in.vmem, step=t)
                    tf.contrib.summary.histogram('vmem_conv1', self.n_conv1.vmem, step=t)
                    tf.contrib.summary.histogram('vmem_conv2', self.n_conv2.vmem, step=t)
                    tf.contrib.summary.histogram('vmem_fc1', self.n_fc1.vmem, step=t)
                    tf.contrib.summary.histogram('vmem_out', self.n_out.vmem, step=t)

        spike_count = self.n_out.get_spike_count()

        return spike_count

    def reset_neuron(self):
        self.n_in.reset()
        self.n_conv1.reset()
        self.n_conv2.reset()
        self.n_fc1.reset()
        self.n_out.reset()


#
# noinspection PyUnboundLocalVariable
class MNISTModel_MLP(tfe.Network):
    def __init__(self, data_format, conf):
        super(MNISTModel_MLP, self).__init__(name='')

        # configuration
        self.conf = conf

        # internal
        self.f_1st_iter = True

        # model
        # synapses
        self._input_shape = [-1,784]
        self._hidden_shape = [-1,500]
        self._output_shape = [-1,10]

        self.fc1 = self.track_layer(tf.layers.Dense(500,activation=tf.nn.relu,use_bias=False,kernel_regularizer=regularizers.l2_regularizer(conf.lamb),kernel_initializer=initializers.xavier_initializer(True)))
        self.fc2 = self.track_layer(tf.layers.Dense(500,activation=tf.nn.relu,use_bias=False,kernel_regularizer=regularizers.l2_regularizer(conf.lamb),kernel_initializer=initializers.xavier_initializer(True)))
        #self.dropout = self.track_layer(tf.layers.Dropout(0.5))
        self.fc3 = self.track_layer(tf.layers.Dense(10,use_bias=False,kernel_regularizer=regularizers.l2_regularizer(conf.lamb),kernel_initializer=initializers.xavier_initializer(True)))

        # neurons
        if self.conf.nn_mode == 'SNN':
            print('SNN mode')
            print('neuron input add')
            self.n_in = self.track_layer(lib_snn.Neuron(self._input_shape,'IN',vth=self.conf.n_init_vth,vinit=self.conf.n_init_vinit,vrest=self.conf.n_init_vrest))
            self.n_1 = self.track_layer(lib_snn.Neuron(self._hidden_shape,'LIF',vth=self.conf.n_init_vth,vinit=self.conf.n_init_vinit,vrest=self.conf.n_init_vrest))
            self.n_2 = self.track_layer(lib_snn.Neuron(self._hidden_shape,'LIF',vth=self.conf.n_init_vth,vinit=self.conf.n_init_vinit,vrest=self.conf.n_init_vrest))
            self.n_out = self.track_layer(lib_snn.Neuron(self._output_shape,'LIF',vth=self.conf.n_init_vth,vinit=self.conf.n_init_vinit,vrest=self.conf.n_init_vrest))

    #def build(self, _) :
        #print('BUILD')

    # noinspection PyUnboundLocalVariable
    def call(self, inputs, f_training):
        if self.conf.nn_mode == 'ANN':
            #print('ANN mode')
            x = tf.reshape(inputs,self._input_shape)
            x = self.fc1(x)
            x = self.fc2(x)
            #if f_training:
            #    x = self.dropout(x)
            x = self.fc3(x)
            return x
        elif self.conf.nn_mode == 'SNN':
            if self.f_1st_iter == False:
                self.reset_neuron()
            else:
                self.f_1st_iter = False

            #print('SNN mode')

            #print(inputs.shape)
            #x = tf.reshape(inputs,self._input_shape)
            #print(x.shape)

            for t in range(self.conf.time_step):
                #print("time step: %d"%(t))
                act_in = self.n_in(inputs,t)
                _act_fc1 = self.fc1(act_in)
                act_fc1 = self.n_1(_act_fc1,t)
                _act_fc2 = self.fc2(act_fc1)
                act_fc2 = self.n_2(_act_fc2,t)
                _act_fc3 = self.fc3(act_fc2)
                act_out = self.n_out(_act_fc3,t)

                spike_count = self.n_out.get_spike_count()

            # noinspection PyUnboundLocalVariable
            return spike_count
        else :
            print('not supported nn_mode %s' % self.conf.nn_mode)
            sys.exit(1)

    # for neuron
    def reset_neuron(self):
        self.n_in.reset()
        self.n_1.reset()
        self.n_2.reset()
        self.n_out.reset()

