import tensorflow as tf


from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops

import tensorflow_probability as tfp
tfd = tfp.distributions

#
#import util
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
#from lib_snn import layer as lib_snn.layer
#from lib_snn import anal as lib_snn.anal
#from lib_snn import util as lib_snn.util
import lib_snn


#
# noinspection PyUnboundLocalVariable
#class CIFARModel_CNN(tfe.Network):
#class CIFARModel_CNN(tf.keras.layers):
class CIFARModel_CNN(lib_snn.model.Model,tf.keras.layers.Layer):
    def __init__(self, input_shape, data_format, conf):
        #super(CIFARModel_CNN, self).__init__(name='')
        lib_snn.model.Model.__init__(self, input_shape, data_format, conf)
        tf.keras.layers.Layer.__init__(self,name='')

        #
        self.kernel_size = 3
        data_format = self.data_format
        use_bias = self.use_bias
        kernel_regularizer = self.kernel_regularizer
        kernel_initializer = self.kernel_initializer
        padding = 'SAME'

        #
        self.list_layer['conv1'] = tf.keras.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv1_1'] = tf.keras.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv1_1_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv2'] = tf.keras.layers.Conv2D(128,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv2_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv2_1'] = tf.keras.layers.Conv2D(128,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv2_1_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv3'] = tf.keras.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv3_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv3_1'] = tf.keras.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv3_1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv3_2'] = tf.keras.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv3_2_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv4'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv4_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv4_1'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv4_1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv4_2'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv4_2_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv5'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv5_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv5_1'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv5_1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv5_2'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding=padding)
        self.list_layer['conv5_2_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['fc1'] = tf.keras.layers.Dense(512,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer)
        self.list_layer['fc1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['fc2'] = tf.keras.layers.Dense(512,activation=None,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer)
        self.list_layer['fc2_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['fc3'] = tf.keras.layers.Dense(self.num_class,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer)
        self.list_layer['fc3_bn'] = tf.keras.layers.BatchNormalization()

        self.dropout_conv = tf.keras.layers.Dropout(0.3)
        self.dropout_conv2 = tf.keras.layers.Dropout(0.4)
        self.dropout = tf.keras.layers.Dropout(0.5)

        #
        self.list_shape['conv1'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.in_shape,64,self.kernel_size,1)
        self.list_shape['conv1_1'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv1'],64,self.kernel_size,1)
        self.list_shape['conv1_p'] = lib_snn.util.cal_output_shape_Pooling2D(self.data_format,self.list_shape['conv1_1'],2,2)

        self.list_shape['conv2'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv1_p'],128,self.kernel_size,1)
        self.list_shape['conv2_1'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv2'],128,self.kernel_size,1)
        self.list_shape['conv2_p'] = lib_snn.util.cal_output_shape_Pooling2D(self.data_format,self.list_shape['conv2_1'],2,2)

        self.list_shape['conv3'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv2_p'],256,self.kernel_size,1)
        self.list_shape['conv3_1'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv3'],256,self.kernel_size,1)
        self.list_shape['conv3_2'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv3_1'],256,self.kernel_size,1)
        self.list_shape['conv3_p'] = lib_snn.util.cal_output_shape_Pooling2D(self.data_format,self.list_shape['conv3_2'],2,2)

        self.list_shape['conv4'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv3_p'],512,self.kernel_size,1)
        self.list_shape['conv4_1'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv4'],512,self.kernel_size,1)
        self.list_shape['conv4_2'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv4_1'],512,self.kernel_size,1)
        self.list_shape['conv4_p'] = lib_snn.util.cal_output_shape_Pooling2D(self.data_format,self.list_shape['conv4_2'],2,2)

        self.list_shape['conv5'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv4_p'],512,self.kernel_size,1)
        self.list_shape['conv5_1'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv5'],512,self.kernel_size,1)
        self.list_shape['conv5_2'] = lib_snn.util.cal_output_shape_Conv2D(self.data_format,self.list_shape['conv5_1'],512,self.kernel_size,1)
        self.list_shape['conv5_p'] = lib_snn.util.cal_output_shape_Pooling2D(self.data_format,self.list_shape['conv5_2'],2,2)

        self.list_shape['fc1'] = tensor_shape.TensorShape([self.conf.batch_size,512]).as_list()
        self.list_shape['fc2'] = tensor_shape.TensorShape([self.conf.batch_size,512]).as_list()
        self.list_shape['fc3'] = tensor_shape.TensorShape([self.conf.batch_size,self.num_class]).as_list()

    #
    #
    def build(self, _):
        lib_snn.model.Model.build(self, _)

        # TODO: move to snn libary?
        # SNN setup
        if self.en_snn:
            print('---- SNN Mode ----')
            print('Neuron setup')


            ########
            # Neuron setup
            ########
            self.list_neuron['in'] = lib_snn.layer.Neuron(self.in_shape_snn,self.conf,'IN',self.conf.neural_coding,0,'in')

            for idx, l_name in enumerate(self.list_layer_name):
                depth=idx+1

                if (depth < len(self.list_layer_name)):
                    n_type = self.conf.n_type
                else:
                    n_type = 'OUT'

                self.list_neuron[l_name] = lib_snn.layer.Neuron(self.list_shape[l_name],self.conf, \
                                                          n_type,self.conf.neural_coding,depth,l_name)


            ########
            # snn output declare - should be after neuron setup
            ########
            self.snn_output_layer = self.list_neuron[next(reversed(self.list_neuron))]
            self.snn_output = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_layer.dim)),dtype=tf.float32,trainable=False)
            self.spike_count = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_layer.dim)),dtype=tf.float32,trainable=False)



            # T2FSNN + GO
            if self.conf.neural_coding=='TEMPORAL' and self.conf.f_load_time_const:
                file_name = self.conf.tk_file_name+'_itr-{:d}'.format(self.conf.time_const_num_trained_data)

                if self.conf.f_train_tk_outlier:
                    file_name+="_outlier"

                print('load trained time constant: file_name: {:s}'.format(file_name))

                file = open(file_name,'r')
                lines = csv.reader(file)

                # load tk
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

            # SNN setup check
            self.snn_setup_check()



        # surrogate DNN model for SNN training w/ TTFS coding
        if self.conf.f_surrogate_training_model:

            self.list_tk=collections.OrderedDict()

            for l_name, l in self.list_layer.items():
                if (not 'bn' in l_name):
                    if (not 'fc3' in l_name):
                        self.list_tk[l_name] = lib_snn.layer.Temporal_kernel([], [], self.conf)

            #
            self.enc_st_n_tw = self.conf.enc_st_n_tw

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
                conv1=self.list_layer['conv1'],
                conv1_bn=self.list_layer['conv1_bn'],
                conv1_1=self.list_layer['conv1_1'],
                conv1_1_bn=self.list_layer['conv1_1_bn'],
                conv2=self.list_layer['conv2'],
                conv2_bn=self.list_layer['conv2_bn'],
                conv2_1=self.list_layer['conv2_1'],
                conv2_1_bn=self.list_layer['conv2_1_bn'],
                conv3=self.list_layer['conv3'],
                conv3_bn=self.list_layer['conv3_bn'],
                conv3_1=self.list_layer['conv3_1'],
                conv3_1_bn=self.list_layer['conv3_1_bn'],
                conv3_2=self.list_layer['conv3_2'],
                conv3_2_bn=self.list_layer['conv3_2_bn'],
                conv4=self.list_layer['conv4'],
                conv4_bn=self.list_layer['conv4_bn'],
                conv4_1=self.list_layer['conv4_1'],
                conv4_1_bn=self.list_layer['conv4_1_bn'],
                conv4_2=self.list_layer['conv4_2'],
                conv4_2_bn=self.list_layer['conv4_2_bn'],
                conv5=self.list_layer['conv5'],
                conv5_bn=self.list_layer['conv5_bn'],
                conv5_1=self.list_layer['conv5_1'],
                conv5_1_bn=self.list_layer['conv5_1_bn'],
                conv5_2=self.list_layer['conv5_2'],
                conv5_2_bn=self.list_layer['conv5_2_bn'],
                fc1=self.list_layer['fc1'],
                fc1_bn=self.list_layer['fc1_bn'],
                fc2=self.list_layer['fc2'],
                fc2_bn=self.list_layer['fc2_bn'],
                fc3=self.list_layer['fc3'],
                fc3_bn=self.list_layer['fc3_bn'],
                list_tk=self.list_tk
            )
        else:
            load_layer_ann_checkpoint = tf.train.Checkpoint(
                conv1=self.list_layer['conv1'],
                conv1_bn=self.list_layer['conv1_bn'],
                conv1_1=self.list_layer['conv1_1'],
                conv1_1_bn=self.list_layer['conv1_1_bn'],
                conv2=self.list_layer['conv2'],
                conv2_bn=self.list_layer['conv2_bn'],
                conv2_1=self.list_layer['conv2_1'],
                conv2_1_bn=self.list_layer['conv2_1_bn'],
                conv3=self.list_layer['conv3'],
                conv3_bn=self.list_layer['conv3_bn'],
                conv3_1=self.list_layer['conv3_1'],
                conv3_1_bn=self.list_layer['conv3_1_bn'],
                conv3_2=self.list_layer['conv3_2'],
                conv3_2_bn=self.list_layer['conv3_2_bn'],
                conv4=self.list_layer['conv4'],
                conv4_bn=self.list_layer['conv4_bn'],
                conv4_1=self.list_layer['conv4_1'],
                conv4_1_bn=self.list_layer['conv4_1_bn'],
                conv4_2=self.list_layer['conv4_2'],
                conv4_2_bn=self.list_layer['conv4_2_bn'],
                conv5=self.list_layer['conv5'],
                conv5_bn=self.list_layer['conv5_bn'],
                conv5_1=self.list_layer['conv5_1'],
                conv5_1_bn=self.list_layer['conv5_1_bn'],
                conv5_2=self.list_layer['conv5_2'],
                conv5_2_bn=self.list_layer['conv5_2_bn'],
                fc1=self.list_layer['fc1'],
                fc1_bn=self.list_layer['fc1_bn'],
                fc2=self.list_layer['fc2'],
                fc2_bn=self.list_layer['fc2_bn'],
                fc3=self.list_layer['fc3'],
                fc3_bn=self.list_layer['fc3_bn']
            )

        return load_layer_ann_checkpoint



    ###########################################################################
    ## processing
    ###########################################################################

    def reset_per_run_snn(self):
        self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.list_layer_name)+1])
        self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.list_layer_name)+1])


    def reset_per_sample_snn(self):
        self.reset_neuron()
        #self.snn_output = np.zeros((self.num_accuracy_time_point,)+self.list_neuron['fc1'].get_spike_count().numpy().shape)
        self.snn_output.assign(tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_layer.dim)))
        self.count_accuracy_time_point=0



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
            lib_snn.anal.save_ann_act(self,inputs,f_training)

        # gradient-based optimization of TC and td in temporal coding (TTFS)
        if self.en_opt_time_const_T2FSNN:
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


        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()

        #self.print_act_after_w_norm()

    def preproc_surrogate_training_model(self):
        if self.f_loss_enc_spike_dist:
            self.dist_beta_sample_func()



    #
    # TODO: input neuron ?
    def load_temporal_kernel_para(self):
        if self.conf.verbose:
            print('preprocessing: load_temporal_kernel_para')

        for l_name in self.list_layer_name:

            if l_name != self.list_layer_name[-1]:
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
    def fused_bn(self):
        print('---- BN Fusion ----')
        for l_name, l in self.list_layer.items():
            if (not 'in' in l_name) and (not 'bn' in l_name):
                l_name_bn = l_name+'_bn'
                if l_name_bn in self.list_layer:
                    self.bn_fusion(l,self.list_layer[l_name_bn])



    def defused_bn(self):
        print('---- BN Defusion ----')
        for l_name, l in self.list_layer.items():
            if (not 'in' in l_name) and (not 'bn' in l_name):
                l_name_bn = l_name+'_bn'
                if l_name_bn in self.list_layer:
                    self.bn_defusion(l,self.list_layer[l_name_bn])



    #
    def w_norm_layer_wise(self):
        print('layer-wise normalization')
        f_norm=np.max

        for idx_l, l in enumerate(self.list_layer_name):
            if idx_l==0:
                self.norm[l]=f_norm(self.dict_stat_r[l])
            else:
                self.norm[l]=f_norm(list(self.dict_stat_r.values())[idx_l])/f_norm(list(self.dict_stat_r.values())[idx_l-1])

            self.norm_b[l]=f_norm(self.dict_stat_r[l])

        # print
        for k, v in self.norm.items():
            print(k +': '+str(v))

        for k, v in self.norm_b.items():
            print(k +': '+str(v))


        #
        for l_name, l in self.list_layer.items():
            if (not 'in' in l_name) and (not 'bn' in l_name):
                #l.kernel = tf.math.divide(l.kernel,self.norm[l_name])
                #l.bias = tf.math.divide(l.bias,self.norm_b[l_name])
                l.kernel = l.kernel/self.norm[l_name]
                l.bias = l.bias/self.norm_b[l_name]


        # TODO: move
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

            for l_name, l in self.list_layer.items():
                if (not 'in' in l_name) and (not 'bn' in l_name):
                    if not(self.conf.input_spike_mode=='REAL' and l_name=='conv1'):
                        l.kernel = l.kernel*layer_const
                        l.bias = l.bias*bias_const

    #
    def data_based_w_norm(self):
        print('---- Data-based weight normalization ----')

        path_stat=self.conf.path_stat
        f_name_stat_pre=self.conf.prefix_stat

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

        for idx_l, l in enumerate(self.list_layer_name):
            key=l+'_'+stat

            f_name_stat = f_name_stat_pre+'_'+key
            f_name=os.path.join(path_stat,f_name_stat)
            f_stat[key]=open(f_name,'r')
            r_stat[key]=csv.reader(f_stat[key])

            for row in r_stat[key]:
                self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])

        self.w_norm_layer_wise()



    def print_act_after_w_norm(self):
        self.print_act_stat_r()


    #
    #def preproc_ann_norm(self):
        #if self.conf.f_fused_bn:
            #self.fused_bn()
#
        ##self.print_model()
#
        ## weight normalization - data based
        #if self.conf.f_w_norm_data:
            #self.data_based_w_norm()
#
        ##self.print_model()
#
#
    def call_ann(self,inputs,f_training, tw=0, epoch=0):
#
        x = tf.reshape(inputs,self._input_shape)

        a_in = x

        s_conv1 = self.list_layer['conv1'](a_in)

        if self.f_skip_bn:
            s_conv1_bn = s_conv1
        else:
            s_conv1_bn = self.list_layer['conv1_bn'](s_conv1,training=f_training)

        a_conv1 = tf.nn.relu(s_conv1_bn)
        if f_training:
            a_conv1 = self.dropout_conv(a_conv1,training=f_training)
        s_conv1_1 = self.list_layer['conv1_1'](a_conv1)

        if self.f_skip_bn:
            s_conv1_1_bn = s_conv1_1
        else:
            s_conv1_1_bn = self.list_layer['conv1_1_bn'](s_conv1_1,training=f_training)
        a_conv1_1 = tf.nn.relu(s_conv1_1_bn)
        p_conv1_1 = self.pool2d(a_conv1_1)
        #if f_training:
        #    x = self.dropout_conv(x,training=f_training)

        s_conv2 = self.list_layer['conv2'](p_conv1_1)
        if self.f_skip_bn:
            s_conv2_bn = s_conv2
        else:
            s_conv2_bn = self.list_layer['conv2_bn'](s_conv2,training=f_training)
        a_conv2 = tf.nn.relu(s_conv2_bn)
        if f_training:
           a_conv2 = self.dropout_conv2(a_conv2,training=f_training)
        s_conv2_1 = self.list_layer['conv2_1'](a_conv2)
        if self.f_skip_bn:
            s_conv2_1_bn = s_conv2_1
        else:
            s_conv2_1_bn = self.list_layer['conv2_1_bn'](s_conv2_1,training=f_training)
        a_conv2_1 = tf.nn.relu(s_conv2_1_bn)
        p_conv2_1 = self.pool2d(a_conv2_1)
        #if f_training:
        #   x = self.dropout_conv2(x,training=f_training)

        s_conv3 = self.list_layer['conv3'](p_conv2_1)
        if self.f_skip_bn:
            s_conv3_bn = s_conv3
        else:
            s_conv3_bn = self.list_layer['conv3_bn'](s_conv3,training=f_training)
        a_conv3 = tf.nn.relu(s_conv3_bn)
        if f_training:
           a_conv3 = self.dropout_conv2(a_conv3,training=f_training)
        s_conv3_1 = self.list_layer['conv3_1'](a_conv3)
        if self.f_skip_bn:
            s_conv3_1_bn = s_conv3_1
        else:
            s_conv3_1_bn = self.list_layer['conv3_1_bn'](s_conv3_1,training=f_training)
        a_conv3_1 = tf.nn.relu(s_conv3_1_bn)
        if f_training:
           a_conv3_1 = self.dropout_conv2(a_conv3_1,training=f_training)
        s_conv3_2 = self.list_layer['conv3_2'](a_conv3_1)
        if self.f_skip_bn:
            s_conv3_2_bn = s_conv3_2
        else:
            s_conv3_2_bn = self.list_layer['conv3_2_bn'](s_conv3_2,training=f_training)
        a_conv3_2 = tf.nn.relu(s_conv3_2_bn)
        p_conv3_2 = self.pool2d(a_conv3_2)
        #if f_training:
        #   x = self.dropout_conv2(x,training=f_training)

        s_conv4 = self.list_layer['conv4'](p_conv3_2)
        if self.f_skip_bn:
            s_conv4_bn = s_conv4
        else:
            s_conv4_bn = self.list_layer['conv4_bn'](s_conv4,training=f_training)
        a_conv4 = tf.nn.relu(s_conv4_bn)
        if f_training:
           a_conv4 = self.dropout_conv2(a_conv4,training=f_training)
        s_conv4_1 = self.list_layer['conv4_1'](a_conv4)
        if self.f_skip_bn:
            s_conv4_1_bn = s_conv4_1
        else:
            s_conv4_1_bn = self.list_layer['conv4_1_bn'](s_conv4_1,training=f_training)
        a_conv4_1 = tf.nn.relu(s_conv4_1_bn)
        if f_training:
           a_conv4_1 = self.dropout_conv2(a_conv4_1,training=f_training)
        s_conv4_2 = self.list_layer['conv4_2'](a_conv4_1)
        if self.f_skip_bn:
            s_conv4_2_bn = s_conv4_2
        else:
            s_conv4_2_bn = self.list_layer['conv4_2_bn'](s_conv4_2,training=f_training)
        a_conv4_2 = tf.nn.relu(s_conv4_2_bn)
        p_conv4_2 = self.pool2d(a_conv4_2)
        #if f_training:
        #   x = self.dropout_conv2(x,training=f_training)

        s_conv5 = self.list_layer['conv5'](p_conv4_2)
        if self.f_skip_bn:
            s_conv5_bn = s_conv5
        else:
            s_conv5_bn = self.list_layer['conv5_bn'](s_conv5,training=f_training)
        a_conv5 = tf.nn.relu(s_conv5_bn)
        if f_training:
           a_conv5 = self.dropout_conv2(a_conv5,training=f_training)
        s_conv5_1 = self.list_layer['conv5_1'](a_conv5)
        if self.f_skip_bn:
            s_conv5_1_bn = s_conv5_1
        else:
            s_conv5_1_bn = self.list_layer['conv5_1_bn'](s_conv5_1,training=f_training)
        a_conv5_1 = tf.nn.relu(s_conv5_1_bn)
        if f_training:
           a_conv5_1 = self.dropout_conv2(a_conv5_1,training=f_training)
        s_conv5_2 = self.list_layer['conv5_2'](a_conv5_1)
        if self.f_skip_bn:
            s_conv5_2_bn = s_conv5_2
        else:
            s_conv5_2_bn = self.list_layer['conv5_2_bn'](s_conv5_2,training=f_training)
        a_conv5_2 = tf.nn.relu(s_conv5_2_bn)
        p_conv5_2 = self.pool2d(a_conv5_2)

        s_flat = tf.compat.v1.layers.flatten(p_conv5_2)

        if f_training:
           s_flat = self.dropout(s_flat,training=f_training)

        s_fc1 = self.list_layer['fc1'](s_flat)
        if self.f_skip_bn:
            s_fc1_bn = s_fc1
        else:
            s_fc1_bn = self.list_layer['fc1_bn'](s_fc1,training=f_training)
        a_fc1 = tf.nn.relu(s_fc1_bn)
        if f_training:
           a_fc1 = self.dropout(a_fc1,training=f_training)

        s_fc2 = self.list_layer['fc2'](a_fc1)
        if self.f_skip_bn:
            s_fc2_bn = s_fc2
        else:
            s_fc2_bn = self.list_layer['fc2_bn'](s_fc2,training=f_training)
        a_fc2 = tf.nn.relu(s_fc2_bn)
        if f_training:
           a_fc2 = self.dropout(a_fc2,training=f_training)

        s_fc3 = self.list_layer['fc3'](a_fc2)
        if self.f_skip_bn:
            s_fc3_bn = s_fc3
        else:
            s_fc3_bn = self.list_layer['fc3_bn'](s_fc3,training=f_training)
        a_fc3 = s_fc3_bn

        a_out = a_fc3


        #
        if self.f_1st_iter and self.conf.nn_mode=='ANN':
            print('1st iter')
            self.f_1st_iter = False
            self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)

        #
        if not self.f_1st_iter and (self.en_opt_time_const_T2FSNN or self.conf.f_write_stat):
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
        if (self.conf.f_write_stat) and (not self.f_1st_iter):
            self.write_act()


        return a_out


    # TODO: move other spot
    #
    def write_act(self):

        for l_name in self.list_layer_write_stat:
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
#        if not self.f_load_model_done:
#        #if f_training==False or tf.random.uniform(shape=(),minval=0,maxval=1)<pr_layer:
#            v_in = x
#            t_in = self.list_tk['in'](v_in,'enc', self.epoch, f_training)
#            v_in_dec= self.list_tk['in'](t_in, 'dec', self.epoch, f_training)
#            a_in = v_in_dec
#        else:
#            a_in = x
        a_in = x

        s_conv1 = self.list_layer['conv1'](a_in)
        if self.f_skip_bn:
            s_conv1_bn = s_conv1
        else:
            s_conv1_bn = self.list_layer['conv1_bn'](s_conv1,training=f_training)


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
        s_conv1_1 = self.list_layer['conv1_1'](a_conv1)

        #pred = tf.reduce_mean(self.list_layer['conv1_1'].kernel,[0,1])

        if self.f_skip_bn:
            s_conv1_1_bn = s_conv1_1
        else:
            s_conv1_1_bn = self.list_layer['conv1_1_bn'](s_conv1_1,training=f_training)

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

        s_conv2 = self.list_layer['conv2'](p_conv1_1)
        if self.f_skip_bn:
            s_conv2_bn = s_conv2
        else:
            s_conv2_bn = self.list_layer['conv2_bn'](s_conv2,training=f_training)

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
        s_conv2_1 = self.list_layer['conv2_1'](a_conv2)
        if self.f_skip_bn:
            s_conv2_1_bn = s_conv2_1
        else:
            s_conv2_1_bn = self.list_layer['conv2_1_bn'](s_conv2_1,training=f_training)

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

        s_conv3 = self.list_layer['conv3'](p_conv2_1)
        if self.f_skip_bn:
            s_conv3_bn = s_conv3
        else:
            s_conv3_bn = self.list_layer['conv3_bn'](s_conv3,training=f_training)

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
        s_conv3_1 = self.list_layer['conv3_1'](a_conv3)
        if self.f_skip_bn:
            s_conv3_1_bn = s_conv3_1
        else:
            s_conv3_1_bn = self.list_layer['conv3_1_bn'](s_conv3_1,training=f_training)

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
        s_conv3_2 = self.list_layer['conv3_2'](a_conv3_1)
        if self.f_skip_bn:
            s_conv3_2_bn = s_conv3_2
        else:
            s_conv3_2_bn = self.list_layer['conv3_2_bn'](s_conv3_2,training=f_training)

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

        s_conv4 = self.list_layer['conv4'](p_conv3_2)
        if self.f_skip_bn:
            s_conv4_bn = s_conv4
        else:
            s_conv4_bn = self.list_layer['conv4_bn'](s_conv4,training=f_training)

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
        s_conv4_1 = self.list_layer['conv4_1'](a_conv4)
        if self.f_skip_bn:
            s_conv4_1_bn = s_conv4_1
        else:
            s_conv4_1_bn = self.list_layer['conv4_1_bn'](s_conv4_1,training=f_training)

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
        s_conv4_2 = self.list_layer['conv4_2'](a_conv4_1)
        if self.f_skip_bn:
            s_conv4_2_bn = s_conv4_2
        else:
            s_conv4_2_bn = self.list_layer['conv4_2_bn'](s_conv4_2,training=f_training)

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

        s_conv5 = self.list_layer['conv5'](p_conv4_2)
        if self.f_skip_bn:
            s_conv5_bn = s_conv5
        else:
            s_conv5_bn = self.list_layer['conv5_bn'](s_conv5,training=f_training)

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
        s_conv5_1 = self.list_layer['conv5_1'](a_conv5)
        if self.f_skip_bn:
            s_conv5_1_bn = s_conv5_1
        else:
            s_conv5_1_bn = self.list_layer['conv5_1_bn'](s_conv5_1,training=f_training)

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
        s_conv5_2 = self.list_layer['conv5_2'](a_conv5_1)
        if self.f_skip_bn:
            s_conv5_2_bn = s_conv5_2
        else:
            s_conv5_2_bn = self.list_layer['conv5_2_bn'](s_conv5_2,training=f_training)

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

        s_fc1 = self.list_layer['fc1'](s_flat)
        if self.f_skip_bn:
            s_fc1_bn = s_fc1
        else:
            s_fc1_bn = self.list_layer['fc1_bn'](s_fc1,training=f_training)

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

        s_fc2 = self.list_layer['fc2'](a_fc1)
        if self.f_skip_bn:
            s_fc2_bn = s_fc2
        else:
            s_fc2_bn = self.list_layer['fc2_bn'](s_fc2,training=f_training)

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


        s_fc3 = self.list_layer['fc3'](a_fc2)
        if self.f_skip_bn:
            s_fc3_bn = s_fc3
        else:
            if ('bn' in self.conf.model_name) or ('ro' in self.conf.model_name) :
                s_fc3_bn = self.list_layer['fc3_bn'](s_fc3,training=f_training)
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



        a_out = a_fc3

        if self.f_1st_iter and self.conf.nn_mode=='ANN':
            print('1st iter')
            self.f_1st_iter = False
            self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)


        if not self.f_1st_iter and self.en_opt_time_const_T2FSNN:
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

        print('Conv1   S: '+str(self.list_layer['conv1'].kernel.get_shape()))
        print('Conv1   N: '+str(self.list_shape['conv1']))
        print('Conv1_1 S: '+str(self.list_layer['conv1_1'].kernel.get_shape()))
        print('Conv1_1 N: '+str(self.list_shape['conv1_1']))
        print('Pool1   N: '+str(self.list_shape['conv1_p']))

        print('Conv2   S: '+str(self.list_layer['conv2'].kernel.get_shape()))
        print('Conv2   N: '+str(self.list_shape['conv2']))
        print('Conv2_1 S: '+str(self.list_layer['conv2_1'].kernel.get_shape()))
        print('Conv2_1 N: '+str(self.list_shape['conv2_1']))
        print('Pool2   N: '+str(self.list_shape['conv2_p']))

        print('Conv3   S: '+str(self.list_layer['conv3'].kernel.get_shape()))
        print('Conv3   N: '+str(self.list_shape['conv3']))
        print('Conv3_1 S: '+str(self.list_layer['conv3_1'].kernel.get_shape()))
        print('Conv3_1 N: '+str(self.list_shape['conv3_1']))
        print('Conv3_2 S: '+str(self.list_layer['conv3_2'].kernel.get_shape()))
        print('Conv3_2 N: '+str(self.list_shape['conv3_2']))
        print('Pool3   N: '+str(self.list_shape['conv3_p']))

        print('Conv4   S: '+str(self.list_layer['conv4'].kernel.get_shape()))
        print('Conv4   N: '+str(self.list_shape['conv4']))
        print('Conv4_1 S: '+str(self.list_layer['conv4_1'].kernel.get_shape()))
        print('Conv4_1 N: '+str(self.list_shape['conv4_1']))
        print('Conv4_1 S: '+str(self.list_layer['conv4_2'].kernel.get_shape()))
        print('Conv4_2 N: '+str(self.list_shape['conv4_2']))
        print('Pool4   N: '+str(self.list_shape['conv4_p']))

        print('Conv5   S: '+str(self.list_layer['conv5'].kernel.get_shape()))
        print('Conv5   N: '+str(self.list_shape['conv5']))
        print('Conv5_1 S: '+str(self.list_layer['conv5_1'].kernel.get_shape()))
        print('Conv5_1 N: '+str(self.list_shape['conv5_1']))
        print('Conv5_1 S: '+str(self.list_layer['conv5_1'].kernel.get_shape()))
        print('Conv5_2 N: '+str(self.list_shape['conv5_2']))
        print('Pool5   N: '+str(self.list_shape['conv5_p']))

        print('Fc1     S: '+str(self.list_layer['fc1'].kernel.get_shape()))
        print('Fc1     N: '+str(self.list_shape['fc1']))
        print('Fc2     S: '+str(self.list_layer['fc2'].kernel.get_shape()))
        print('Fc2     N: '+str(self.list_shape['fc2']))
        print('Fc3     S: '+str(self.list_layer['fc3'].kernel.get_shape()))
        print('Fc3     N: '+str(self.list_shape['fc3']))


    # TODO: move to util
    def print_act_stat_r(self):
        print('print activation stat')

        fig, axs = plt.subplots(6,3)

        axs=axs.ravel()

        #for idx_l, (name_l,stat_l) in enumerate(self.dict_stat_r):
        for idx, (key, value) in enumerate(self.dict_stat_r.items()):
            axs[idx].hist(value.flatten())

        plt.show()

    # TODO: move to util
    # batch normalization fusion - conv, fc
    def bn_fusion(self, layer, bn):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        layer.kernel = layer.kernel*math_ops.cast(inv,layer.kernel.dtype)
        layer.bias = ((layer.bias-mean)*inv+beta)

    # TODO: move to util
    # batch normalization defusion - conv, fc
    def fc_bn_defused(self, layer, bn):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        layer.kernel = layer.kernel/inv
        layer.bias = (layer.bias-beta)/inv+mean


    # TODO: move to visual
    def plot(self, x, y, mark):
        #plt.ion()
        #plt.hist(self.list_neuron['fc3'].vmem)
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


    def cal_entropy(self):
        #total_pattern=np.empty(len(self.list_layer_name))

        for il, length in enumerate(self.arr_length_entropy):
            #total_pattern=0
            #total_pattern=np.zeros(1)
            for idx_l, l in enumerate(self.list_layer_name):
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
        if self.conf.neural_coding=="RATE":
            if t==0:
                self.bias_enable()
            else:
                self.bias_disable()
        elif self.conf.input_spike_mode == 'WEIGHTED_SPIKE' or self.conf.neural_coding == 'WEIGHTED_SPIKE':
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
        #plt.clf()

        #
        for t in range(tw):
            if self.verbose == True:
                print('time: '+str(t))

            self.bias_control(t)

            a_in = self.list_neuron['in'](inputs,t)


            ####################
            #
            ####################
            s_conv1 = self.list_layer['conv1'](a_in)
            a_conv1 = self.list_neuron['conv1'](s_conv1,t)

            s_conv1_1 = self.list_layer['conv1_1'](a_conv1)
            a_conv1_1 = self.list_neuron['conv1_1'](s_conv1_1,t)

            if self.conf.f_spike_max_pool:
                p_conv1_1 = lib_snn.layer.spike_max_pool(
                    a_conv1_1,
                    self.list_neuron['conv1_1'].get_spike_count(),
                    self.list_shape['conv1_p']
                )
            else:
                p_conv1_1 = self.pool2d(a_conv1_1)

            s_conv2 = self.list_layer['conv2'](p_conv1_1)
            a_conv2 = self.list_neuron['conv2'](s_conv2,t)
            s_conv2_1 = self.list_layer['conv2_1'](a_conv2)
            a_conv2_1 = self.list_neuron['conv2_1'](s_conv2_1,t)

            if self.conf.f_spike_max_pool:
                p_conv2_1 = lib_snn.layer.spike_max_pool(
                    a_conv2_1,
                    self.list_neuron['conv2_1'].get_spike_count(),
                    self.list_shape['conv2_p']
                )
            else:
                p_conv2_1 = self.pool2d(a_conv2_1)

            s_conv3 = self.list_layer['conv3'](p_conv2_1)
            a_conv3 = self.list_neuron['conv3'](s_conv3,t)
            s_conv3_1 = self.list_layer['conv3_1'](a_conv3)
            a_conv3_1 = self.list_neuron['conv3_1'](s_conv3_1,t)
            s_conv3_2 = self.list_layer['conv3_2'](a_conv3_1)
            a_conv3_2 = self.list_neuron['conv3_2'](s_conv3_2,t)

            if self.conf.f_spike_max_pool:
                p_conv3_2 = lib_snn.layer.spike_max_pool(
                    a_conv3_2,
                    self.list_neuron['conv3_2'].get_spike_count(),
                    self.list_shape['conv3_p']
                )
            else:
                p_conv3_2 = self.pool2d(a_conv3_2)

            s_conv4 = self.list_layer['conv4'](p_conv3_2)
            a_conv4 = self.list_neuron['conv4'](s_conv4,t)
            s_conv4_1 = self.list_layer['conv4_1'](a_conv4)
            a_conv4_1 = self.list_neuron['conv4_1'](s_conv4_1,t)
            s_conv4_2 = self.list_layer['conv4_2'](a_conv4_1)
            a_conv4_2 = self.list_neuron['conv4_2'](s_conv4_2,t)

            if self.conf.f_spike_max_pool:
                p_conv4_2 = lib_snn.layer.spike_max_pool(
                    a_conv4_2,
                    self.list_neuron['conv4_2'].get_spike_count(),
                    self.list_shape['conv4_p']
                )
            else:
                p_conv4_2 = self.pool2d(a_conv4_2)

            s_conv5 = self.list_layer['conv5'](p_conv4_2)
            a_conv5 = self.list_neuron['conv5'](s_conv5,t)
            s_conv5_1 = self.list_layer['conv5_1'](a_conv5)
            a_conv5_1 = self.list_neuron['conv5_1'](s_conv5_1,t)
            s_conv5_2 = self.list_layer['conv5_2'](a_conv5_1)
            a_conv5_2 = self.list_neuron['conv5_2'](s_conv5_2,t)

            if self.conf.f_spike_max_pool:
                p_conv5_2 = lib_snn.layer.spike_max_pool(
                    a_conv5_2,
                    self.list_neuron['conv5_2'].get_spike_count(),
                    self.list_shape['conv5_p']
                )
            else:
                p_conv5_2 = self.pool2d(a_conv5_2)

            flat = tf.compat.v1.layers.flatten(p_conv5_2)

            s_fc1 = self.list_layer['fc1'](flat)
            #s_fc1_bn = self.list_layer['fc1_bn'](s_fc1,training=f_training)
            #a_fc1 = self.list_neuron['fc1'](s_fc1_bn,t)
            a_fc1 = self.list_neuron['fc1'](s_fc1,t)

            s_fc2 = self.list_layer['fc2'](a_fc1)
            #s_fc2_bn = self.list_layer['fc2_bn'](s_fc2,training=f_training)
            #a_fc2 = self.list_neuron['fc2'](s_fc2_bn,t)
            a_fc2 = self.list_neuron['fc2'](s_fc2,t)

            s_fc3 = self.list_layer['fc3'](a_fc2)
            #print('a_fc3')
            a_fc3 = self.list_neuron['fc3'](s_fc3,t)


            #print(str(t)+" : "+str(self.list_neuron['fc3'].vmem.numpy()))



            if self.f_1st_iter == False and self.f_debug_visual == True:
                #synapse=s_conv1
                #neuron=self.list_neuron['in']
                #synapse_1=s_conv1
                #neuron_1=self.list_neuron['conv1']
                #synapse_2 = s_conv1_1
                #neuron_2 = self.list_neuron['conv1_1']
                #lib_snn.util.debug_visual(synapse, neuron, synapse_1, neuron_1, synapse_2, neuron_2, t)
                lib_snn.util.debug_visual_raster(t)



            ##########
            #
            ##########

            if self.f_1st_iter == False:
                if self.conf.f_comp_act:
                    lib_snn.anal.comp_act(self)

                if self.conf.f_isi:
                    self.total_isi += self.get_total_isi()
                    self.total_spike_amp += self.get_total_spike_amp()

                    self.f_out_isi(t)


                if self.conf.f_entropy:
                    for idx_l, l in enumerate(self.list_layer_name):
                        if l !='fc3':
                            self.dict_stat_w[l][t] = self.list_neuron[l].out.numpy()


                    #print(self.dict_stat_w['conv1'])

                if t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
                    output=self.list_neuron['fc3'].vmem
                    self.recoding_ret_val()


                    #num_spike_count = tf.cast(tf.reduce_sum(self.spike_count,axis=[2]),tf.int32)
                    #num_spike_count = tf.reduce_sum(self.spike_count,axis=[2])

            #print(t, self.list_neuron['fc3'].last_spike_time.numpy())
            #print(t, self.list_neuron['fc3'].isi.numpy())

        if self.conf.f_entropy and (not self.f_1st_iter):
            self.cal_entropy()

        # dummy run
        if self.f_1st_iter:
        #if False:
            self.f_1st_iter = False

            self.list_layer['conv1_bn'](s_conv1,training=f_training)
            self.list_layer['conv1_1_bn'](s_conv1_1,training=f_training)

            self.list_layer['conv2_bn'](s_conv2,training=f_training)
            self.list_layer['conv2_1_bn'](s_conv2_1,training=f_training)

            self.list_layer['conv3_bn'](s_conv3,training=f_training)
            self.list_layer['conv3_1_bn'](s_conv3_1,training=f_training)
            self.list_layer['conv3_2_bn'](s_conv3_2,training=f_training)

            self.list_layer['conv4_bn'](s_conv4,training=f_training)
            self.list_layer['conv4_1_bn'](s_conv4_1,training=f_training)
            self.list_layer['conv4_2_bn'](s_conv4_2,training=f_training)

            self.list_layer['conv5_bn'](s_conv5,training=f_training)
            self.list_layer['conv5_1_bn'](s_conv5_1,training=f_training)
            self.list_layer['conv5_2_bn'](s_conv5_2,training=f_training)

            self.list_layer['fc1_bn'](s_fc1,training=f_training)
            self.list_layer['fc2_bn'](s_fc2,training=f_training)
            self.list_layer['fc3_bn'](s_fc3,training=f_training)

            return 0


        else:

            self.get_total_residual_vmem()

            spike_zero = tf.reduce_sum(self.snn_output,axis=[0,2])

            if np.any(spike_zero.numpy() == 0.0):
                print('spike count 0')

            #plt.hist(self.list_neuron['conv1'].vmem.numpy().flatten())
            #plt.show()


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





    # TODO: move to process
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


    def reset_neuron(self):
        for idx, l in self.list_neuron.items():
            l.reset()




    ###########################################################
    ## training time constant for temporal coding
    ###########################################################

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


    # TODO: save activation, parameterize
    ##############################################################
    # save activation for data-based normalization
    ##############################################################
    # distribution of activation - neuron-wise or channel-wise?
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

        for idx_l, l in enumerate(self.list_layer_write_stat):
            for idx_c, c in enumerate(stat_conf):
                key=l+'_'+c

                f_name_stat = f_name_stat_pre+'_'+key
                f_name=os.path.join(path_stat,f_name_stat)
                #f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'w')
                #f_stat[key]=open(path_stat'/'f_name_stat)
                #print(f_name)

                f_stat[key]=open(f_name,'w')
                #wr_stat[key]=csv.writer(f_stat[key])


                #for idx_l, l in enumerate(self.list_layer_write_stat):
                threads.append(threading.Thread(target=self.write_stat, args=(f_stat[key], l, c)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()


    def write_stat(self, f_stat, layer_name, stat_conf_name):
        print('---- write_stat ----')

        l = layer_name
        c = stat_conf_name
        s_layer=self.dict_stat_w[l].numpy()

        self._write_stat(f_stat,s_layer,c)


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

        print('stat write - {}'.format(f_stat.name))
        wr_stat=csv.writer(f_stat)
        wr_stat.writerow(stat)
        f_stat.close()



















