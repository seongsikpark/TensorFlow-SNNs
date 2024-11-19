import tensorflow as tf
import tensorflow.contrib.eager as tfe

import tensorflow.keras.layers as layers

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
#class cnn_mnist(tfe.Network):
class cnn_mnist(tf.keras.Model):
    def __init__(self, data_format, conf):
        super(cnn_mnist, self).__init__(name='')


        self.data_format = data_format
        self.conf = conf

        if self.data_format == 'channels_first':
            self._input_shape = [-1,1,28,28]   # MNIST
        else:
            assert self.data_format == 'channels_last'
            self._input_shape = [-1,28,28,1]


        #
        self.dim_i = 784
        self.dim_c_1 = 15
        self.dim_k = 5
        self.dim_c_2 = 40
        self.dim_h = 300
        self.dim_o = conf.num_class

        #self._input_shape = [-1,self.dim_i]
        self._hidden_shape = [-1,self.dim_h]
        self._output_shape = [-1,self.dim_o]

        self.i_shape = (self.conf.batch_size,) + tuple(self._input_shape[1:])
        self.h_shape = (self.conf.batch_size,) + tuple(self._hidden_shape[1:])
        self.o_shape = (self.conf.batch_size,) + tuple(self._output_shape[1:])

        # internal
        self.f_model_load_done = False
        self.f_1st_iter = True

        self.type_call = {
            'ANN': self.call_ann,
            'SNN': self.call_snn
        }

        #
        regularizer_type = {
            'L1': regularizers.l1_regularizer(conf.lamb),
            'L2': regularizers.l2_regularizer(conf.lamb)
        }

        kernel_regularizer = regularizer_type[conf.regularizer]
        kernel_initializer = initializers.xavier_initializer(True)

        #
        self.list_l = []        # list - layers
        self.list_s = []        # list - PSP
        self.list_a = []        # list - activation values
        self.list_n = []        # list - neurons
        self.list_pl = []       # list - pooling layer
        self.list_p = []        # list - pooling values


        self.list_a.append([])      # for input

        self.list_l.append(
            layers.Conv2D(
                self.dim_c_1,
                self.dim_k,
                data_format=data_format,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
                padding='valid'
            )
        )

        self.list_s.append([])
        self.list_n.append(tf.nn.relu)
        self.list_a.append([])
        self.list_pl.append(
            layers.MaxPooling2D((2,2),(2,2),padding='SAME',data_format=data_format)
        )
        self.list_p.append([])



        self.list_l.append(
            layers.Conv2D(
                self.dim_c_2,
                self.dim_k,
                data_format=data_format,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
                padding='valid'
            )
        )

        self.list_s.append([])
        self.list_n.append(tf.nn.relu)
        self.list_a.append([])
        self.list_pl.append(
            layers.MaxPooling2D((2,2),(2,2),padding='SAME',data_format=data_format)
        )
        self.list_p.append([])



        self.list_l.append(layers.Dense(
            self.dim_h,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer
        ))

        self.list_s.append([])
        self.list_n.append(tf.nn.relu)
        self.list_a.append([])

        self.list_l.append(layers.Dense(
            self.dim_o,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer
        ))

        self.list_s.append([])
        self.list_n.append(tf.nn.relu)
        self.list_a.append([])

        # neurons
        if self.conf.nn_mode == 'SNN':
            print('SNN mode')
            self.list_n=[]


            nc = self.conf.neural_coding


            self.list_n.append(lib_snn.Neuron(self.i_shape,'IN',1,self.conf,nc,0,'in'))
            self.list_n.append(lib_snn.Neuron([self.conf.batch_size,24,24,15],'IF',1,self.conf,nc,1,'conv1'))
            self.list_n.append(lib_snn.Neuron([self.conf.batch_size,8,8,40],'IF',1,self.conf,nc,2,'conv2'))
            self.list_n.append(lib_snn.Neuron(self.h_shape,'IF',1,self.conf,nc,1,'fc1'))
            self.list_n.append(lib_snn.Neuron(self.o_shape,'IF',1,self.conf,nc,2,'fc2'))

            self.accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
            self.accuracy_time_point.append(conf.time_step)
            self.num_accuracy_time_point = len(self.accuracy_time_point)
            self.count_accuracy_time_point = 0

            self.spike_count = np.zeros((self.num_accuracy_time_point,)+self.o_shape)
            #self.spike_count = np.zeros(self.o_shape)
            #self.spike_count = np.zeros([self.num_accuracy_time_point,].extend(self.o_shape))


            #self.shape_out_conv1 = util.cal_output_shape_Conv2D(self.data_format,self.in_shape,64,self.kernel_size,1)



    def preproc(self):
        print('not define yet')
        os._exit(0)

    def batch_padding(self,input_tensor):
        d = input_tensor.numpy().shape[0]
        t = self.i_shape[0]
        g = t-d

        return tf.concat([input_tensor,tf.zeros((g,)+self.i_shape[1:])],axis=0)

    def total_spike_count_int(self):
        print('total_spike_count_int: not implemented yet')

    def total_spike_count(self):
        print('total_spike_count: not implemented yet')


    def call(self, input_tensor, f_training):
        #print(tf.shape(input_tensor))
        input_tensor = tf.reshape(input_tensor,self._input_shape)

        #print(tf.shape(input_tensor))
        #print(self._input_shape)
        if self.f_model_load_done == True:
            #if self.f_1st_iter == True:
            #    self.preproc()

            # modify later
            #if self.conf.f_fused_bn == True:
            #    ret_val = self.type_call_fused_bn[self.conf.nn_mode](input_tensor, f_training)
            #else:
            #    ret_val = self.type_call[self.conf.nn_mode](input_tensor, f_training)

            if self.conf.nn_mode=='SNN':
                if input_tensor.numpy().shape != self.i_shape:
                    input_tensor = self.batch_padding(input_tensor)

            ret_val = self.type_call[self.conf.nn_mode](input_tensor, f_training)
        else:
            #ret_val = self.type_call['ANN'](input_tensor, f_training)
            ret_val = self.type_call[self.conf.nn_mode](input_tensor, f_training)

            #if self.conf.nn_mode=='SNN':
            #    ret_val = self.type_call['SNN'](input_tensor, f_training)
            self.f_model_load_done = True
        return ret_val


    def call_ann(self, inputs, f_training):
        self.list_a[0] = inputs
        self.list_s[0] = self.list_l[0](self.list_a[0])
        self.list_a[1] = self.list_n[1](self.list_s[0])
        self.list_p[0] = self.list_pl[0](self.list_a[1])

        #print(tf.shape(self.list_p[0]))

        self.list_s[1] = self.list_l[1](self.list_p[0])
        self.list_a[2] = self.list_n[2](self.list_s[1])
        self.list_p[1] = self.list_pl[1](self.list_a[2])
        #print(tf.shape(self.list_p[0]))

        self.p_flat = tf.layers.flatten(self.list_p[1])

        #self.list_s[2] = self.list_l[2](self.list_p[1])
        self.list_s[2] = self.list_l[2](self.p_flat)
        self.list_a[3] = self.list_n[3](self.list_s[2])

        #print(tf.shape(self.list_a[3]))

        self.list_s[3] = self.list_l[3](self.list_a[3])
        self.list_a[4] = self.list_s[3]
        #print(tf.shape(self.list_a[4]))

        ret_val = self.list_a[4]

        #print(tf.shape(ret_val))

        return ret_val

    def recoding_ret_val(self, output_neuron):
        output_neuron = output_neuron
        # spike count
        #print(self.o_shape)
        #print(tf.shape(self.spike_count))
        #print(self.count_accuracy_time_point)
        #print(tf.shape(output_neuron.get_spike_count()))
        self.spike_count[self.count_accuracy_time_point,:,:]=(output_neuron.get_spike_count().numpy())
        # vmem
        #spike_count[count_accuracy_time_point,:,:]=(output_neuron.vmem.numpy())

        # get total spike count
        #tc_int, tc = self.get_total_spike_count()
        #
        #self.total_spike_count_int[count_accuracy_time_point]+=tc_int
        #self.total_spike_count[count_accuracy_time_point]+=tc

        self.count_accuracy_time_point+=1


        #num_spike_count = tf.cast(tf.reduce_sum(spike_count,axis=[2]),tf.int32)



    def call_snn(self, inputs, f_training):
        if self.f_model_load_done:
            self.reset_neuron()

        for t in range(self.conf.time_step):

            self.list_a[0] = self.list_n[0](inputs,t)
            self.list_s[0] = self.list_l[0](self.list_a[0])

            print(self.list_a[0].shape)
            print(self.list_s[0].shape)




            self.list_a[1] = self.list_n[1](self.list_s[0],t)
            self.list_s[1] = self.list_l[1](self.list_a[1])


            print(self.list_a[1].shape)


            self.list_a[2] = self.list_n[2](self.list_s[1],t)

            if t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
               self.recoding_ret_val(self.list_n[2])


        #ret_val = self.list_n[1].get_spike_count()/self.conf.time_step    # rate-coding
        ret_val = self.spike_count/self.conf.time_step

        return ret_val


    # for neuron
    def reset_neuron(self):

        self.count_accuracy_time_point = 0
        self.spike_count = np.zeros((self.num_accuracy_time_point,)+self.o_shape)

        for idx_n, n in enumerate(self.list_n):
            n.reset()

