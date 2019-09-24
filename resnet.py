# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet50 model definition compatible with TensorFlow's eager execution.

Reference [Deep Residual Learning for Image
Recognition](https://arxiv.org/abs/1512.03385)

Adapted from tf.keras.applications.ResNet50. A notable difference is that the
model here outputs logits while the Keras model outputs probability.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.framework import tensor_shape


from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import regularizers

from tensorflow.python.keras import layers
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework import add_arg_scope


from tensorflow.python.ops import math_ops

import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import util
import lib_snn


import collections

import csv
import threading

#@add_arg_scope
#def tensorflow.python.keras.layers.Conv2D

class _res_block(tf.keras.Model):
    """_res_block is the block that has no conv layer at shortcut.

    Args:
    kernel_size: the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    data_format: data_format for the input ('channels_first' or
      'channels_last').
    """

    def __init__(
            self,
            kernel_size,
            filters,
            stage,
            block,
            data_format,
            conf,
            input_shape,
            stride_shortcut=(2,2),
            f_shortcut=False,
            ):
        super(_res_block, self).__init__(name='conv'+str(stage)+'_block' + str(block))
        filters1, filters2, filters3 = filters

        conv_name_base = ''
        bn_name_base = ''
        bn_axis = 1 if data_format == 'channels_first' else 3
        #bn_axis=-1
        bn_epsilon=1e-5
        bn_momentum=0.999

        self.conf = conf

        self._input_shape = input_shape

        self.f_shortcut = f_shortcut
        if f_shortcut == True:
            self.stride_shortcut = stride_shortcut
        else:
            self.stride_shortcut = (1,1)

        regularizer_type = {
            'L1': regularizers.l1_regularizer(conf.lamb),
            'L2': regularizers.l2_regularizer(conf.lamb)
        }

        kernel_regularizer = regularizer_type[conf.regularizer]
        kernel_initializer = initializers.xavier_initializer(True)

        #arg_scope = tf.contrib.framework.arg_scope
        #
        #with arg_scope(
        #    [layers.Conv2D],
        #    data_format = data_format,
        #    kernel_initializer = kernel_initializer
        #):

        self.f_1st_iter = True
        self.f_model_load_done = False


        self.type_call = {
            'ANN': self.call_ann,
            'SNN': self.call_snn
        }

        self.type_call_fused_bn = {
            'ANN': self.call_ann_fused_bn,
            'SNN': self.call_snn_fused_bn
        }

        self.type_preproc = {
            'ANN': self.preproc_ann,
            'SNN': self.preproc_snn
        }

        self.conv1 = layers.Conv2D(
                filters1,
                (1, 1),
                padding='valid',
                strides=self.stride_shortcut,
                name=conv_name_base+'1/conv',
                data_format=data_format,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )
        self.conv1_bn = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                name=bn_name_base+'1/bn',
                epsilon=bn_epsilon,
                momentum=bn_momentum
            )

        self.conv2 = tf.keras.layers.Conv2D(
                filters2,
                kernel_size,
                padding='same',
                strides=(1,1),
                data_format=data_format,
                name=conv_name_base+'2/conv',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )
        self.conv2_bn = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                name=bn_name_base+'2/bn',
                epsilon=bn_epsilon,
                momentum=bn_momentum
            )

        self.conv3 = tf.keras.layers.Conv2D(
                filters3,
                (1, 1),
                padding='valid',
                strides=(1,1),
                data_format=data_format,
                name=conv_name_base+'3/conv',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )
        self.conv3_bn = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                name=bn_name_base+'3/bn',
                epsilon=bn_epsilon,
                momentum=bn_momentum
            )

        self.dropout = tf.keras.layers.Dropout(0.5)

        if self.f_shortcut==True:
            self.shortcut = tf.keras.layers.Conv2D(
                    filters3,
                    (1, 1),
                    padding='valid',
                    strides=self.stride_shortcut,
                    name=conv_name_base+'0/conv',
                    data_format=data_format,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer
                )
            self.shortcut_bn = tf.keras.layers.BatchNormalization(
                    axis=bn_axis,
                    name=bn_name_base+'0/bn',
                    epsilon=bn_epsilon,
                    momentum=bn_momentum
                )

        # def neurons
        if self.conf.nn_mode=='SNN':
            self.in_shape = self._input_shape

            self.shape_out_conv1 = util.cal_output_shape_Conv2D(data_format,self.in_shape,filters1,kernel_size,self.stride_shortcut)
            self.shape_out_conv2 = util.cal_output_shape_Conv2D(data_format,self.shape_out_conv1,filters2,kernel_size,1)
            self.shape_out_conv3 = util.cal_output_shape_Conv2D(data_format,self.shape_out_conv2,filters3,kernel_size,1)
            #self.shape_out_out = util.cal_output_shape_Conv2D(data_format,self.shape_out_conv3,filters3,kernel_size,1)
            self.shape_out_out=self.shape_out_conv3

            n_type = self.conf.n_type
            nc = self.conf.neural_coding
            conf = self.conf

            self.neuron_list=collections.OrderedDict()

            self.n_conv1 = lib_snn.Neuron(self.shape_out_conv1,n_type,1,conf,nc)
            self.n_conv2= lib_snn.Neuron(self.shape_out_conv2,n_type,1,conf,nc)
            #self.n_conv3 = lib_snn.Neuron(self.shape_out_conv3,n_type,1,conf,nc)
            self.n_out = lib_snn.Neuron(self.shape_out_out,n_type,1,conf,nc)

            self.neuron_list['conv1']=self.n_conv1
            self.neuron_list['conv2']=self.n_conv2
            #self.neuron_list['conv3']=self.n_conv3
            self.neuron_list['out']=self.n_out


    def fused_bn(self):
        #print('fused_bn')

        conv_bn_fused(self.conv1,self.conv1_bn)
        conv_bn_fused(self.conv2,self.conv2_bn)
        conv_bn_fused(self.conv3,self.conv3_bn)

        if self.f_shortcut==True:
            conv_bn_fused(self.shortcut,self.shortcut_bn)

    def bias_norm(self):
        if self.conf.input_spike_mode=='REAL':
            self.bias_norm_input_real()
        else:
            print('not defined bias norm mode: %', self.conf.input_spike_mode)
            os._exit(0)


    def bias_norm_input_real(self):
        self.conv1.bias = self.conv1.bias/self.conf.time_step
        self.conv2.bias = self.conv2.bias/self.conf.time_step
        self.conv3.bias = self.conv3.bias/self.conf.time_step

        if self.f_shortcut==True:
            self.shortcut.bias = self.shortcut.bias/self.conf.time_step

    def bias_enable(self):
        self.conv1.use_bias = True
        self.conv2.use_bias = True
        self.conv3.use_bias = True

        if self.f_shortcut==True:
            self.shortcut.use_bias = True

    def bias_disable(self):
        self.conv1.use_bias = False
        self.conv2.use_bias = False
        self.conv3.use_bias = False

        if self.f_shortcut==True:
            self.shortcut.use_bias = False



    def preproc(self):
        self.type_preproc[self.conf.nn_mode]()
        self.f_1st_iter = False

    def preproc_ann(self):
        #print('preprocessing')
        if self.conf.f_fused_bn == True:
            if self.conf.en_train == False:
                self.fused_bn()
            else :
                print('fused bn mode is only available in inference')
                os.exit(0)

    def preproc_snn(self):
        self.preproc_ann()

    #@tf.contrib.eager.defun
    def call(self, input_tensor, f_training=False, time_step=0):
        if self.f_model_load_done == True:
            if self.f_1st_iter == True:
                self.preproc()

            if self.conf.f_fused_bn == True:
                ret_val = self.type_call_fused_bn[self.conf.nn_mode](input_tensor, f_training=False, time_step=0)
            else:
                ret_val = self.type_call[self.conf.nn_mode](input_tensor, f_training=False)
        else:
            ret_val = self.type_call['ANN'](input_tensor, f_training)
            if self.conf.nn_mode=='SNN':
                ret_val = self.call_snn_load(input_tensor)
            self.f_model_load_done = True
        return ret_val


    def call_ann(self, input_tensor, f_training=False):
        x = self.conv1(input_tensor)
        x = self.conv1_bn(x, training=f_training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.conv2_bn(x, training=f_training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.conv3_bn(x, training=f_training)

        if f_training:
            x = self.dropout(x,training=f_training)

        if self.f_shortcut==True:
            res = self.shortcut(input_tensor)
            res = self.shortcut_bn(res, training=f_training)
        else:
            res = input_tensor

        x += res
        x = tf.nn.relu(x)

        return x

    def call_ann_fused_bn(self, input_tensor, f_training=False, time_step=0):
        s_conv1 = self.conv1(input_tensor)
        a_conv1 = tf.nn.relu(s_conv1)

        s_conv2 = self.conv2(a_conv1)
        a_conv2 = tf.nn.relu(s_conv2)

        s_conv3 = self.conv3(a_conv2)

        if self.f_shortcut==True:
            res = self.shortcut(input_tensor)
        else:
            res = input_tensor

        s_res = s_conv3 + res
        a_out = tf.nn.relu(s_res)

        return a_out

    def call_snn_load(self, input_tensor, f_training=False, time_step=0):
        s_conv1 = self.conv1(input_tensor)
        a_conv1 = self.n_conv1(s_conv1,time_step)

        s_conv2 = self.conv2(a_conv1)
        a_conv2 = self.n_conv2(s_conv2,time_step)

        s_conv3 = self.conv3(a_conv2)

        if self.f_shortcut==True:
            res = self.shortcut(input_tensor)
        else:
            res = input_tensor

        s_res = s_conv3 + res
        a_out = self.n_out(s_res,time_step)

        return a_out

    def call_snn(self, input_tensor, f_training=False):
        print('*E: call_snn: SNN mode is available only in fused_bn mode')
        os._exit(0)

    def call_snn_fused_bn(self, input_tensor, f_training=False, time_step=0):
        s_conv1 = self.conv1(input_tensor)
        a_conv1 = self.n_conv1(s_conv1,time_step)

        s_conv2 = self.conv2(a_conv1)
        a_conv2 = self.n_conv2(s_conv2,time_step)

        s_conv3 = self.conv3(a_conv2)

        if self.f_shortcut==True:
            res = self.shortcut(input_tensor)
        else:
            res = input_tensor

        s_res = s_conv3 + res
        a_out = self.n_out(s_res,time_step)

        return a_out


    def snn_neuron_reset(self):
        for _, n in self.neuron_list.items():
            n.reset()

        #self.n_conv1.reset()
        #self.n_conv2.reset()
        ##self.n_conv3.reset()
        #self.n_out.reset()


    def get_spike_count(self):
        tc_int=0
        tc=0

        for _, n in self.neuron_list.items():
            tc_int += tf.reduce_sum(n.get_spike_count_int())
            tc += tf.reduce_sum(n.get_spike_count())

        return tc_int, tc



class Resnet50(tf.keras.Model):
    """Instantiates the ResNet50 architecture.

    Args:
    data_format: format for the image. Either 'channels_first' or
      'channels_last'.  'channels_first' is typically faster on GPUs while
      'channels_last' is typically faster on CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats
    name: Prefix applied to names of variables created in the model.
    trainable: Is the model trainable? If true, performs backward
        and optimization after call() method.
    include_top: whether to include the fully-connected layer at the top of the
      network.
    pooling: Optional pooling mode for feature extraction when `include_top`
      is `False`.
      - `None` means that the output of the model will be the 4D tensor
          output of the last convolutional layer.
      - `avg` means that global average pooling will be applied to the output of
          the last convolutional layer, and thus the output of the model will be
          a 2D tensor.
      - `max` means that global max pooling will be applied.
    classes: optional number of classes to classify images into, only to be
      specified if `include_top` is True.

    Raises:
      ValueError: in case of invalid argument for data_format.
    """

    def __init__(
        self,
        data_format,
        conf,
        name=None,
        trainable=True,
        include_top=True,
        pooling=None,
        classes=1000
    ):
        super(Resnet50, self).__init__(name='resnet50')

        valid_channel_values = ('channels_first', 'channels_last')
        if data_format not in valid_channel_values:
          raise ValueError('Unknown data_format: %s. Valid values: %s' %
                           (data_format, valid_channel_values))
        self.include_top = include_top
        self.conf = conf

        regularizer_type = {
            'L1': regularizers.l1_regularizer(conf.lamb),
            'L2': regularizers.l2_regularizer(conf.lamb)
        }

        self.kernel_regularizer = regularizer_type[conf.regularizer]
        self.kernel_initializer = initializers.xavier_initializer(True)


        self.f_1st_iter = True
        self.f_model_load_done = False

        if data_format == 'channels_first':
            self._input_shape = [-1,3,conf.input_size,conf.input_size]
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1,conf.input_size,conf.input_size,3]


        self.type_call = {
            'ANN': self.call_ann,
            'SNN': self.call_snn
        }

        self.type_call_fused_bn = {
            'ANN': self.call_ann_fused_bn,
            'SNN': self.call_snn_fused_bn
        }

        self.type_preproc = {
            'ANN': self.preproc_ann,
            'SNN': self.preproc_snn
        }

        def res_block(filters, stage, block, stride_shortcut=(2,2), f_shortcut=False, input_shape=[1,1,1,1]):
            l = _res_block(
                3,
                filters,
                stage=stage,
                block=block,
                data_format=data_format,
                conf=self.conf,
                stride_shortcut=stride_shortcut,
                f_shortcut=f_shortcut,
                input_shape=input_shape
            )
            return l

        #
        self.accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
        self.accuracy_time_point.append(conf.time_step)
        self.num_accuracy_time_point = len(self.accuracy_time_point)
        self.count_accuracy_time_point = 0

        #self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])
        #self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])
        #self.total_residual_vmem=np.zeros(len(self.layer_name)+1)

        self.total_spike_count=np.zeros([self.num_accuracy_time_point])
        self.total_spike_count_int=np.zeros([self.num_accuracy_time_point])

        #
        filters_conv1 = 64
        kernel_size_conv1 = (7,7)

        #print('output_shape')
        #print(self.conv1.output_shape)

        self.in_shape = [self.conf.batch_size]+self._input_shape[1:]

        # should modify later
        #self.shape_out_conv1 = util.cal_output_shape_Conv2D(data_format,self.in_shape,filters_conv1,kernel_size_conv1,2)
        #self.shape_out_conv1_p = util.cal_output_shape_Pooling2D(data_format,self.shape_out_conv1,3,2)

        self.shape_out_conv1 = tensor_shape.TensorShape([self.in_shape[0]]+[109,109,64])
        self.shape_out_conv1_p = tensor_shape.TensorShape([self.in_shape[0]]+[54,54,64])

        #self.shape_out_blk2a = self.shape_out_blk1[:2]+self.shape_out_blk1[3]/2
        #self.shape_out_blk2 = tensor_shape.TensorShape(self.shape_out_blk1[:3].dims+[256])

        self.shape_out_blk1 = self.shape_out_conv1_p
        self.shape_out_blk2 = tensor_shape.TensorShape([self.in_shape[0]]+[54,54,256])
        self.shape_out_blk3 = tensor_shape.TensorShape([self.in_shape[0]]+[27,27,512])
        self.shape_out_blk4 = tensor_shape.TensorShape([self.in_shape[0]]+[14,14,1024])
        self.shape_out_blk5 = tensor_shape.TensorShape([self.in_shape[0]]+[7,7,2048])

        self.shape_out_out = tensor_shape.TensorShape([self.conf.batch_size,self.conf.num_class])

        # def neurons
        if self.conf.nn_mode=='SNN':
            n_type = self.conf.n_type
            nc = self.conf.neural_coding
            conf = self.conf

            self.neuron_list=collections.OrderedDict()

            self.neuron_list['in'] = lib_snn.Neuron(self.in_shape,'IN',1,conf,nc)
            self.neuron_list['conv1'] = lib_snn.Neuron(self.shape_out_conv1,n_type,1,conf,nc)
            self.neuron_list['out'] = lib_snn.Neuron(self.shape_out_out,'OUT',1,conf,nc)

            self.n_in = self.neuron_list['in']
            self.n_conv1 = self.neuron_list['conv1']
            self.n_out = self.neuron_list['out']

            #
            self.spike_count = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.n_out.dim)),dtype=tf.float32,trainable=False)

        #
        # layer definition
        #
        self.conv1 = tf.keras.layers.Conv2D(
            64, (7, 7),
            strides=(2,2),
            data_format=data_format,
            name='conv1/conv',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer
        )

        #print('output_shape')
        #print(self.conv1.output_shape)

        bn_axis = 1 if data_format == 'channels_first' else 3
        bn_epsilon=1e-5
        bn_momentum=0.999

        self.conv1_bn= tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                name='conv1/bn',
                epsilon=bn_epsilon,
                momentum=bn_momentum
        )

        self.conv1_act = tf.keras.layers.ReLU()

        self.max_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format)

        self.blk2a = res_block([64, 64, 256], stage=2, block='1', stride_shortcut=(1, 1), f_shortcut=True,
                               input_shape=self.shape_out_blk1)
        self.blk2b = res_block([64, 64, 256], stage=2, block='2',input_shape=self.shape_out_blk2)
        self.blk2c = res_block([64, 64, 256], stage=2, block='3',input_shape=self.shape_out_blk2)

        self.blk3a = res_block([128, 128, 512], stage=3, block='1', f_shortcut=True,input_shape=self.shape_out_blk2)
        self.blk3b = res_block([128, 128, 512], stage=3, block='2',input_shape=self.shape_out_blk3)
        self.blk3c = res_block([128, 128, 512], stage=3, block='3',input_shape=self.shape_out_blk3)
        self.blk3d = res_block([128, 128, 512], stage=3, block='4',input_shape=self.shape_out_blk3)

        self.blk4a = res_block([256, 256, 1024], stage=4, block='1', f_shortcut=True,input_shape=self.shape_out_blk3)
        self.blk4b = res_block([256, 256, 1024], stage=4, block='2',input_shape=self.shape_out_blk4)
        self.blk4c = res_block([256, 256, 1024], stage=4, block='3',input_shape=self.shape_out_blk4)
        self.blk4d = res_block([256, 256, 1024], stage=4, block='4',input_shape=self.shape_out_blk4)
        self.blk4e = res_block([256, 256, 1024], stage=4, block='5',input_shape=self.shape_out_blk4)
        self.blk4f = res_block([256, 256, 1024], stage=4, block='6',input_shape=self.shape_out_blk4)

        self.blk5a = res_block([512, 512, 2048], stage=5, block='1', f_shortcut=True,input_shape=self.shape_out_blk4)
        self.blk5b = res_block([512, 512, 2048], stage=5, block='2',input_shape=self.shape_out_blk5)
        self.blk5c = res_block([512, 512, 2048], stage=5, block='3',input_shape=self.shape_out_blk5)

        #self.avg_pool =
        #    tf.keras.layers.AveragePooling2D(
        #        (7, 7), strides=(7, 7), data_format=data_format))

        reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]
        reduction_indices = tf.constant(reduction_indices)
        self.avg_pool = functools.partial(
            tf.reduce_mean,
            reduction_indices=reduction_indices,
            keepdims=False
        )

        self.fc=tf.keras.layers.Dense(classes, name='logits')

        self.dropout = tf.keras.layers.Dropout(0.5)



        #
        # write stat - activation
        #
        if self.conf.f_write_stat:
            self.num_train_dataset = self.conf.num_test_dataset      # tentative
            self.count_stat_w = 0

            self.layer_name_stat_w =[
                'conv1'
            ]


            self.dict_act = collections.OrderedDict()  # activation

            # write stat - activation
            self.stat_conf = ['max', 'max_999']

            self.dict_stat_r = collections.OrderedDict()  # read
            self.dict_stat_w = collections.OrderedDict()  # write

            print('write stat mode: ' + self.conf.act_save_mode + ', ' + str(self.stat_conf))

            for layer_name in self.layer_name_stat_w:
                # self.dict_stat_w[layer_name]=np.zeros([1,]+self.dict_shape[layer_name][1:])
                #self.dict_stat_w[layer_name] = np.zeros([self.conf.num_train_dataset, ] + self.dict_shape[layer_name][1:],

                if self.conf.act_save_mode == 'channel' :
                    self.dict_stat_w[layer_name] = np.zeros([len(self.stat_conf),]
                                                            + [self.num_train_dataset, ]
                                                            + [self.shape_out_conv1.as_list()[-1], ],
                                                            np.float32)
                elif self.conf.act_save_mode == 'neuron' :
                     self.dict_stat_w[layer_name] = np.zeros([len(self.stat_conf),]
                                                            + [self.num_train_dataset, ]
                                                            + self.shape_out_conv1.as_list()[1:],
                                                            np.float32)
                else :
                    assert(False)

                print(layer_name + str(self.dict_stat_w[layer_name].shape))

            self.dict_act['conv1'] = self.conv1_act

        #
        self.blk_list=collections.OrderedDict()
        self.blk_list['blk2a'] = self.blk2a
        self.blk_list['blk2b'] = self.blk2b
        self.blk_list['blk2c'] = self.blk2c

        self.blk_list['blk3a'] = self.blk3a
        self.blk_list['blk3b'] = self.blk3b
        self.blk_list['blk3c'] = self.blk3c
        self.blk_list['blk3d'] = self.blk3d

        self.blk_list['blk4a'] = self.blk4a
        self.blk_list['blk4b'] = self.blk4b
        self.blk_list['blk4c'] = self.blk4c
        self.blk_list['blk4d'] = self.blk4d
        self.blk_list['blk4e'] = self.blk4e
        self.blk_list['blk4f'] = self.blk4f

        self.blk_list['blk5a'] = self.blk5a
        self.blk_list['blk5b'] = self.blk5b
        self.blk_list['blk5c'] = self.blk5c




    def fused_bn(self):
        conv_bn_fused(self.conv1,self.conv1_bn)

    def bias_norm(self):
        #if self.conf.input_spike_mode == 'WEIGHTED_SPIKE' or self.conf.neural_coding == 'WEIGHTED_SPIKE':

        if self.conf.input_spike_mode=='REAL':
            self.bias_norm_input_real()
        else:
            print('not defined bias norm mode: %', self.conf.input_spike_mode)
            os._exit(0)

        #
        for _, blk in self.blk_list.items():
            blk.bias_norm()


    def bias_norm_input_real(self):
        self.conv1.bias = self.conv1.bias/self.conf.time_step

        self.fc.bias = self.fc.bias/self.conf.time_step

    def bias_enable(self):
        self.conv1.use_bias = True
        self.fc.use_bias = True

        for _, blk in self.blk_list.items():
            blk.bias_enable()


    def bias_disable(self):
        self.conv1.use_bias = False
        self.fc.use_bias = False

        for _, blk in self.blk_list.items():
            blk.bias_disable()

    def preproc(self):
        self.type_preproc[self.conf.nn_mode]()
        self.f_1st_iter = False

    def preproc_ann(self):
        #print('preprocessing')
        if self.conf.f_fused_bn == True:
            self.fused_bn()

    def preproc_snn(self):
        if self.conf.f_fused_bn == True:
            self.fused_bn()

        #self.bias_norm()


    #@tf.contrib.eager.defun
    def call(self, input_tensor, labels=None, f_training=False):
        if self.f_model_load_done == True:
            if self.f_1st_iter == True:
                self.preproc()

            if self.conf.f_fused_bn == True:
                ret_val = self.type_call_fused_bn[self.conf.nn_mode](input_tensor, labels, f_training)
            else:
                ret_val = self.type_call[self.conf.nn_mode](input_tensor, f_training)
        else:
            ret_val = self.type_call['ANN'](input_tensor, f_training)
            if self.conf.nn_mode=='SNN':
                ret_val = self.call_snn_load(input_tensor)         # for load neurons
            self.f_model_load_done = True
        return ret_val


    def call_ann(self, input_tensor, f_training=False):
        #print('stage 1')
        s_conv1 = self.conv1(input_tensor)

        s_conv1_bn = self.conv1_bn(s_conv1, training=f_training)
        #a_conv1 = tf.nn.relu(s_conv1_bn)
        a_conv1 = self.conv1_act(s_conv1_bn)
        p_conv1 = self.max_pool(a_conv1)

        #print('stage 2')
        a_blk2a = self.blk2a(p_conv1, f_training=f_training)
        a_blk2b = self.blk2b(a_blk2a, f_training=f_training)
        a_blk2c = self.blk2c(a_blk2b, f_training=f_training)

        #print('stage 3')
        a_blk3a = self.blk3a(a_blk2c, f_training=f_training)
        a_blk3b = self.blk3b(a_blk3a, f_training=f_training)
        a_blk3c = self.blk3c(a_blk3b, f_training=f_training)
        a_blk3d = self.blk3d(a_blk3c, f_training=f_training)

        #print('stage 4')
        a_blk4a = self.blk4a(a_blk3d, f_training=f_training)
        a_blk4b = self.blk4b(a_blk4a, f_training=f_training)
        a_blk4c = self.blk4c(a_blk4b, f_training=f_training)
        a_blk4d = self.blk4d(a_blk4c, f_training=f_training)
        a_blk4e = self.blk4e(a_blk4d, f_training=f_training)
        a_blk4f = self.blk4f(a_blk4e, f_training=f_training)

        #print('stage 5')
        a_blk5a = self.blk5a(a_blk4f, f_training=f_training)
        a_blk5b = self.blk5b(a_blk5a, f_training=f_training)
        a_blk5c = self.blk5c(a_blk5b, f_training=f_training)

        #
        p_blk5c = self.avg_pool(a_blk5c)
        s_out = self.fc(p_blk5c)
        a_out = s_out

        return a_out

    def call_ann_fused_bn(self, input_tensor, labels=None, f_training=False):

        #input_tensor = tf.add(input_tensor, [103.939, 116.779, 123.68])
        #input_tensor = tf.div(input_tensor, 255)
        #input_tensor = tf.multiply(input_tensor, 255)
        #input_tensor = tf.subtract(input_tensor, [103.939, 116.779, 123.68])

        #print('stage 1')
        s_conv1 = self.conv1(input_tensor)
        a_conv1 = self.conv1_act(s_conv1)
        p_conv1 = self.max_pool(a_conv1)

        #print('stage 2')
        a_blk2a = self.blk2a(p_conv1, f_training=f_training)
        a_blk2b = self.blk2b(a_blk2a, f_training=f_training)
        a_blk2c = self.blk2c(a_blk2b, f_training=f_training)

        #print('stage 3')
        a_blk3a = self.blk3a(a_blk2c, f_training=f_training)
        a_blk3b = self.blk3b(a_blk3a, f_training=f_training)
        a_blk3c = self.blk3c(a_blk3b, f_training=f_training)
        a_blk3d = self.blk3d(a_blk3c, f_training=f_training)

        #print('stage 4')
        a_blk4a = self.blk4a(a_blk3d, f_training=f_training)
        a_blk4b = self.blk4b(a_blk4a, f_training=f_training)
        a_blk4c = self.blk4c(a_blk4b, f_training=f_training)
        a_blk4d = self.blk4d(a_blk4c, f_training=f_training)
        a_blk4e = self.blk4e(a_blk4d, f_training=f_training)
        a_blk4f = self.blk4f(a_blk4e, f_training=f_training)

        #print('stage 5')
        a_blk5a = self.blk5a(a_blk4f, f_training=f_training)
        a_blk5b = self.blk5b(a_blk5a, f_training=f_training)
        a_blk5c = self.blk5c(a_blk5b, f_training=f_training)

        #
        p_blk5c = self.avg_pool(a_blk5c)
        s_out = self.fc(p_blk5c)
        a_out = s_out



        if self.conf.f_write_stat:
            self.dict_act['conv1'] = a_conv1
#
#            #print(tf.reduce_max(a_conv1.numpy()))
            #print(tf.shape(self.dict_act['conv1']))
            #print(tf.shape(tf.reduce_max(self.dict_act['conv1'],axis=[1,2])))

            for layer_name in self.layer_name_stat_w:
                act = self.dict_act[layer_name]

                if self.conf.act_save_mode=='channel':
                    self.dict_stat_w[layer_name][0][self.count_stat_w:self.count_stat_w+self.conf.batch_size] = tf.reduce_max(act,axis=[1,2])
                    #self.dict_stat_w[layer_name][0][self.count_stat_w:self.count_stat_w+self.conf.batch_size] = tf.reduce_max(act,axis=[1,2])
#                    #print(np.max(act.numpy(),axis=[0,1]))
#                    #print(shape(np.max(act.numpy(),axis=[0,1])))
#                elif self.conf.act_save_mode=='neuron':
#                    self.dict_stat_w[layer_name][self.count_stat_w]=act.numpy()
#
            self.count_stat_w += self.conf.batch_size



        return a_out

    def call_snn_load(self, input_tensor):
        t = 0
        a_in = self.n_in(input_tensor, t)

        #print('stage 1')
        s_conv1 = self.conv1(a_in)
        a_conv1 = self.n_conv1(s_conv1,t)
        p_conv1 = self.max_pool(a_conv1)

        #print('stage 2')
        a_blk2a = self.blk2a(p_conv1, time_step=t)
        a_blk2b = self.blk2b(a_blk2a, time_step=t)
        a_blk2c = self.blk2c(a_blk2b, time_step=t)

        #print('stage 3')
        a_blk3a = self.blk3a(a_blk2c, time_step=t)
        a_blk3b = self.blk3b(a_blk3a, time_step=t)
        a_blk3c = self.blk3c(a_blk3b, time_step=t)
        a_blk3d = self.blk3d(a_blk3c, time_step=t)

        #print('stage 4')
        a_blk4a = self.blk4a(a_blk3d, time_step=t)
        a_blk4b = self.blk4b(a_blk4a, time_step=t)
        a_blk4c = self.blk4c(a_blk4b, time_step=t)
        a_blk4d = self.blk4d(a_blk4c, time_step=t)
        a_blk4e = self.blk4e(a_blk4d, time_step=t)
        a_blk4f = self.blk4f(a_blk4e, time_step=t)

        #print('stage 5')
        a_blk5a = self.blk5a(a_blk4f, time_step=t)
        a_blk5b = self.blk5b(a_blk5a, time_step=t)
        a_blk5c = self.blk5c(a_blk5b, time_step=t)

        #
        p_blk5c = self.avg_pool(a_blk5c)
        s_out = self.fc(p_blk5c)
        a_out = self.n_out(s_out,t)

        a_out = self.n_out.vmem

        return a_out




    def call_snn(self, input_tensor, f_training=False):
        print('*E: call_snn: SNN mode is available only in fused_bn mode')
        os._exit(0)


    def call_snn_fused_bn(self, input_tensor, labels, f_training=False):

        self.snn_init_sample()

        input_tensor = tf.add(input_tensor, [103.939, 116.779, 123.68])
        input_tensor = tf.div(input_tensor, 255.0)

        #input_tensor = tf.add(input_tensor, [])

        for t in range(self.conf.time_step):
            # input
            a_in = self.n_in(input_tensor, t)


            a_in = tf.multiply(a_in, 255)

            if self.conf.input_spike_mode=='POISSON':
                a_in = tf.subtract(a_in, [103.939/self.conf.time_step, 116.779/self.conf.time_step, 123.68/self.conf.time_step])
            elif self.conf.input_spike_mode=='WEIGHTED_SPIKE':
                a_in = tf.subtract(a_in, [103.939/self.conf.p_ws, 116.779/self.conf.p_ws, 123.68/self.conf.p_ws])
            elif self.conf.input_spike_mode=='PROPOSED':
                print('not implemented yet')
                os._exit(0)
            else:
                a_in = tf.subtract(a_in, [103.939, 116.779, 123.68])


            #if t % 80 == 0 :
            if t == 0 :
                self.bias_enable()
            else:
                a_in = tf.multiply(a_in, 0.0)
                self.bias_disable()

            #print('stage 1')
            #s_conv1 = self.conv1(input_tensor)
            s_conv1 = self.conv1(a_in)
            a_conv1 = self.n_conv1(s_conv1,t)
            if self.conf.f_spike_max_pool:
                tmp = tf.reshape(self.n_conv1.get_spike_count(),[1,-1,]+self.shape_out_conv1.as_list()[2:])
                _, arg = tf.nn.max_pool_with_argmax(tmp,(1,3,3,1),(1,2,2,1),padding='VALID')
                arg = tf.reshape(arg,self.shape_out_conv1_p)
                a_conv1_f = tf.reshape(a_conv1,[-1])
                p_conv1 = tf.convert_to_tensor(a_conv1_f.numpy()[arg],dtype=tf.float32)
            else:
                p_conv1 = self.max_pool(a_conv1)


            #print('stage 2')
            a_blk2a = self.blk2a(p_conv1, time_step=t)
            a_blk2b = self.blk2b(a_blk2a, time_step=t)
            a_blk2c = self.blk2c(a_blk2b, time_step=t)

            #print('stage 3')
            a_blk3a = self.blk3a(a_blk2c, time_step=t)
            a_blk3b = self.blk3b(a_blk3a, time_step=t)
            a_blk3c = self.blk3c(a_blk3b, time_step=t)
            a_blk3d = self.blk3d(a_blk3c, time_step=t)

            #print('stage 4')
            a_blk4a = self.blk4a(a_blk3d, time_step=t)
            a_blk4b = self.blk4b(a_blk4a, time_step=t)
            a_blk4c = self.blk4c(a_blk4b, time_step=t)
            a_blk4d = self.blk4d(a_blk4c, time_step=t)
            a_blk4e = self.blk4e(a_blk4d, time_step=t)
            a_blk4f = self.blk4f(a_blk4e, time_step=t)

            #print('stage 5')
            a_blk5a = self.blk5a(a_blk4f, time_step=t)
            a_blk5b = self.blk5b(a_blk5a, time_step=t)
            a_blk5c = self.blk5c(a_blk5b, time_step=t)

            #
            p_blk5c = self.avg_pool(a_blk5c)
            s_out = self.fc(p_blk5c)

            #if t > 100:
            #    a_out = self.n_out(s_out,t)

            a_out = self.n_out(s_out,t)

            ret_val = self.n_out.vmem


            if self.conf.verbose_visual == True:
                #self.verbose_visual(input_tensor,self.n_in,t)
                #self.verbose_visual(s_conv1,self.n_conv1,t)
                self.verbose_visual(p_blk5c,self.blk5c.n_out,s_out,self.n_out,t,labels)



            if t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
                #output = self.n_out.get_spike_count().numpy()
                #output = self.n_out.vmem.numpy()
                #self.snn_out_recode(output)

                output = self.n_out.vmem
                self.recoding_ret_val(output)


        #return ret_val
        return self.spike_count

    def snn_neuron_reset(self):
        self.n_conv1.reset()

        for _, blk in self.blk_list.items():
            blk.snn_neuron_reset()

        self.n_out.reset()

    def snn_init_sample(self):
        self.snn_neuron_reset()
        self.count_accuracy_time_point=0
        #self.ret_snn = np.zeros((self.num_accuracy_time_point,)+self.conf.num_class)
        self.ret_snn = np.zeros((self.num_accuracy_time_point,)+self.n_out.vmem.numpy().shape)

        self.spike_count.assign(tf.zeros((self.num_accuracy_time_point,)+tuple(self.n_out.dim)))

#
#    def snn_out_recode(self, output):
#        self.ret_snn[self.count_accuracy_time_point,:,:]=output
#
#        tc_int, tc = self.get_total_spike_count()
#
#        self.total_spike_count_int[self.count_accuracy_time_point]+=tc_int
#        self.total_spike_count[self.count_accuracy_time_point]+=tc
#
#        #print('time '+str(t))
#        #print(spike_count.shape)
#        #print(self.n_fc3.get_spike_count().numpy())
#        #print(spike_count)
#        self.count_accuracy_time_point+=1


    def recoding_ret_val(self, output):
        self.spike_count.scatter_nd_update([self.count_accuracy_time_point],tf.expand_dims(output,0))

        tc_int, tc = self.get_total_spike_count()
        self.total_spike_count_int[self.count_accuracy_time_point]+=tc_int
        self.total_spike_count[self.count_accuracy_time_point]+=tc

        self.count_accuracy_time_point+=1


    def get_total_spike_count(self):
        tc_int = 0
        tc = 0

        tc_int += tf.reduce_sum(self.n_conv1.get_spike_count_int())
        tc += tf.reduce_sum(self.n_conv1.get_spike_count())

        for _, blk in self.blk_list.items():
            tc_int_tmp, tc_tmp = blk.get_spike_count()
            tc_int += tc_int_tmp
            tc += tc_tmp

        tc_int += tf.reduce_sum(self.n_out.get_spike_count_int())
        tc += tf.reduce_sum(self.n_out.get_spike_count())

        return tc_int, tc

    def verbose_visual(self, synapse1, neuron1, synapse2, neuron2, t, label):

        num_neuron1 = len(neuron1.out.numpy().flatten())
        num_neuron2 = len(neuron2.out.numpy().flatten())

        ax = plt.subplot(3, 4, 5)
        plt.title('w_sum_in (max)')
        plot(t, tf.reduce_max(synapse1).numpy(), 'bo')

        plt.subplot(3, 4, 6, sharex=ax)
        plt.title('vmem (max)')
        plot(t, tf.reduce_max(neuron1.vmem).numpy(), 'bo')
        #scatter(np.broadcast_to(t, neuron1.vmem.numpy().flatten().size), neuron1.vmem.numpy().flatten(),
        #             np.arange(0, 1.0, 1.0/(neuron1.vmem.numpy().flatten().size)))

        plt.subplot(3, 4, 7, sharex=ax)
        #plt.title('# spikes (max)')
        plt.title('spike count (effective) (max)')
        plot(t, tf.reduce_max(neuron1.get_spike_count()).numpy(), 'bo')

        plt.subplot(3, 4, 8, sharex=ax)
        plt.title('spike neuron idx')
        plt.grid(True)
        plt.ylim([0, num_neuron1])
        plt.xlim([0, self.conf.time_step])
        #idx_fire = np.where(neuron1.out.numpy() != 0)[1]
        idx_fire = np.where(neuron1.out.numpy().flatten() != 0)

        if neuron1.n_type == 'OUT':
            #scatter(np.full(np.shape(idx_fire), t, dtype=int), idx_fire, 'r')
            #scatter(t, np.argmax(neuron1.get_spike_count().numpy()), 'b')
            scatter(t, np.argmax(neuron1.vmem.numpy()), 'b')
            #plt.axhline(y=label, xmin=0.0, xmax=1.0, color='k')
            plt.axhline(y=int(label), color='k')
        else:
            scatter(np.full(np.shape(idx_fire), t, dtype=int), idx_fire, 'r')

        ##
        # plt.subplot2grid((2,4),(1,0))
        plt.subplot(3, 4, 9, sharex=ax)
        plot(t, tf.reduce_max(synapse2).numpy(), 'bo')
        # plt.subplot2grid((2,4),(1,1))
        plt.subplot(3, 4, 10, sharex=ax)
        plot(t, tf.reduce_max(neuron2.vmem).numpy(), 'bo')
        #if neuron2.n_type=='OUT':
        #    scatter(np.broadcast_to(t, neuron2.vmem.numpy().size), neuron2.vmem.numpy(),
        #                 np.arange(0, 1, 1.0/self.conf.num_class))
        plt.subplot(3, 4, 11, sharex=ax)
        plot(t, tf.reduce_max(neuron2.get_spike_count()).numpy(), 'bo')
        plt.subplot(3, 4, 12)
        plt.grid(True)
        plt.ylim([0, num_neuron2])
        plt.xlim([0, self.conf.time_step])
        #idx_fire = np.where(neuron2.out.numpy() != 0)[1]
        idx_fire = np.where(neuron2.out.numpy().flatten() != 0)

        if neuron2.n_type == 'OUT':
            pred = np.argmax(neuron2.vmem.numpy())
            if label==pred:
                scatter(t, pred, 'b')
            else:
                scatter(t, pred, 'r')
                scatter(t, label, 'g')

            #scatter(t, np.argmax(neuron2.vmem.numpy()), 'b')
            #plt.axhline(y=label, xmin=0.0, xmax=1.0, color='k')
            #plt.axhline(y=int(label), color='k', linewidth=0.2)
        else:
            scatter(np.full(np.shape(idx_fire), t, dtype=int), idx_fire, 'r')

#############################
#
#############################

    # start here - 181211
    # path modifi
    # save activation value
    def save_activation(self):
        layer_name = self.layer_name_stat_w

        path_stat=self.conf.path_stat
        #path_stat=os.path.join(path_stat,self.conf.ann_model)
        #path_stat=os.path.join(path_stat,self.conf.dataset)
        path_stat=os.path.join(path_stat,self.conf.model_name)
        #prefix_stat=self.conf.prefix_stat

        if not os.path.isdir(path_stat):
            tf.gfile.MakeDirs(path_stat)

        #path = path_stat+prefix_stat
        stat_conf = self.stat_conf

        threads=[]
        for idx_l, l in enumerate(layer_name):
            threads.append(threading.Thread(target=self.write_stat, args=(path_stat, l, stat_conf)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def write_stat(self, path_stat, layer_name, stat_conf):
        print('write_stat func')
        f_stat=collections.OrderedDict()
        wr_stat=collections.OrderedDict()

        l = layer_name
        for idx_c, c in enumerate(stat_conf):
            key=l+'_'+c
            filename = self.conf.prefix_stat+'_'+key
            path = os.path.join(path_stat,filename)
            f_stat[key]=open(path,'w')
            wr_stat[key]=csv.writer(f_stat[key])

        s_layer=self.dict_stat_w[l]

        threads=[]
        for idx_c, c in enumerate(stat_conf):
            key=l+'_'+c
            threads.append(threading.Thread(target=self._write_stat, args=(s_layer, f_stat[key], wr_stat[key], c)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()


    def _write_stat(self, s_layer, f_stat, wr_stat, conf_name):
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
        wr_stat.writerow(stat)
        f_stat.close()







def conv_bn_fused(conv, bn):
    gamma=bn.gamma
    beta=bn.beta
    mean=bn.moving_mean
    var=bn.moving_variance
    ep=bn.epsilon
    inv=math_ops.rsqrt(var+ep)
    inv*=gamma

    conv.kernel = conv.kernel*math_ops.cast(inv,conv.kernel.dtype)
    conv.bias = ((conv.bias-mean)*inv+beta)



def plot(x, y, mark):
    #plt.ion()
    #plt.hist(self.n_fc3.vmem)
    plt.plot(x, y, mark)
    plt.draw()
    plt.pause(0.00000001)
    #plt.ioff()


def scatter(x, y, color, marker='o'):
    #plt.ion()
    plt.scatter(x, y, c=color, s=1, marker=marker)
    plt.draw()
    plt.pause(0.00000001)
    #plt.ioff()

