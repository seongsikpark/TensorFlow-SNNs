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

#
class MNISTModel_CNN(tfe.Network):
    def __init__(self, data_format, conf):
        super(MNISTModel_CNN, self).__init__(name='')

        self.data_format = data_format
        self.conf = conf
        self.num_class = self.conf.num_class

        self.f_1st_iter = True
        self.verbose = conf.verbose
        self.f_debug_visual = conf.verbose_visual
        self.f_done_preproc = False

        self.kernel_size = 5
        self.fanin_conv = self.kernel_size*self.kernel_size
        #self.fanin_conv = self.kernel_size*self.kernel_size/9

        self.tw=conf.time_step

        self.accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
        self.accuracy_time_point.append(conf.time_step)
        #self.num_accuracy_time_point = int(math.ceil(float(conf.time_step)/float(conf.time_step_save_interval))
        self.num_accuracy_time_point = len(self.accuracy_time_point)


        #
        self.f_skip_bn = self.conf.f_fused_bn

        self.layer_name=[
            'conv1',
            'conv2',
            'fc1',
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

            self.spike_amp_bin=type_spike_amp_bin[self.conf.neural_coding]
            self.spike_amp_bin=self.spike_amp_bin[::-1]
            self.spike_amp_bin[0]=0.0

        if self.conf.f_comp_act:
            self.total_comp_act=np.zeros([self.conf.time_step,len(self.layer_name)+1])


        self.output_layer_isi=np.zeros(self.num_class)
        self.output_layer_last_spike_time=np.zeros(self.num_class)

        # nomarlization factor
        self.norm=collections.OrderedDict()
        self.norm_b=collections.OrderedDict()


        if self.data_format == 'channels_first':
            self._input_shape = [-1,1,28,28]   # MNIST
        else:
            assert self.data_format == 'channels_last'
            self._input_shape = [-1,28,28,1]


        if conf.nn_mode == 'ANN':
            use_bias = conf.use_bias
        else :
            use_bias = conf.use_bias

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
        self.layer_list['conv1'] = self.track_layer(tf.layers.Conv2D(12,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='valid'))
        self.layer_list['conv1_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['conv2'] = self.track_layer(tf.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='valid'))
        self.layer_list['conv2_bn'] = self.track_layer(tf.layers.BatchNormalization())
        self.layer_list['fc1'] = self.track_layer(tf.layers.Dense(self.num_class,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer))

        self.dropout = self.track_layer(tf.layers.Dropout(0.5))

        # remove later
        self.conv1=self.layer_list['conv1']
        self.conv1_bn=self.layer_list['conv1_bn']
        self.conv2=self.layer_list['conv2']
        self.conv2_bn=self.layer_list['conv2_bn']
        self.fc1=self.layer_list['fc1']





        pooling_type= {
            'max': self.track_layer(tf.layers.MaxPooling2D((2,2),(2,2),padding='SAME',data_format=data_format)),
            'avg': self.track_layer(tf.layers.AveragePooling2D((2,2),(2,2),padding='SAME',data_format=data_format))
        }

        self.pool2d = pooling_type[self.conf.pooling]
        self.act_relu = tf.nn.relu

        input_shape_one_sample = tensor_shape.TensorShape([1,self._input_shape[1],self._input_shape[2],self._input_shape[3]])
        self.in_shape = [self.conf.batch_size]+self._input_shape[1:]

        self.shape_out_conv1 = util.cal_output_shape_Conv2D_pad_val(self.data_format,self.in_shape,12,self.kernel_size,1)
        self.shape_out_conv1_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv1,2,2)

        self.shape_out_conv2 = util.cal_output_shape_Conv2D_pad_val(self.data_format,self.shape_out_conv1_p,64,self.kernel_size,1)
        self.shape_out_conv2_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv2,2,2)

        self.shape_out_fc1 = tensor_shape.TensorShape([self.conf.batch_size,self.num_class])


        self.dict_shape=collections.OrderedDict()
        self.dict_shape['conv1']=self.shape_out_conv1
        self.dict_shape['conv1_p']=self.shape_out_conv1_p
        self.dict_shape['conv2']=self.shape_out_conv2
        self.dict_shape['conv2_p']=self.shape_out_conv2_p
        self.dict_shape['fc1']=self.shape_out_fc1


        self.dict_shape_one_batch=collections.OrderedDict()
        self.dict_shape_one_batch['conv1']=[1,]+self.shape_out_conv1.as_list()[1:]
        self.dict_shape_one_batch['conv1_p']=[1,]+self.shape_out_conv1_p.as_list()[1:]
        self.dict_shape_one_batch['conv2']=[1,]+self.shape_out_conv2.as_list()[1:]
        self.dict_shape_one_batch['conv2_p']=[1,]+self.shape_out_conv2_p.as_list()[1:]
        self.dict_shape_one_batch['fc1']=[1,]+self.shape_out_fc1.as_list()[1:]

        #
        self.dict_stat_r=collections.OrderedDict()  # read
        self.dict_stat_w=collections.OrderedDict()  # write


        if self.conf.f_entropy:
            self.dict_stat_w['conv1']=np.zeros([self.conf.time_step,]+self.shape_out_conv1.as_list()[1:])
            self.dict_stat_w['conv2']=np.zeros([self.conf.time_step,]+self.shape_out_conv2.as_list()[1:])
            self.dict_stat_w['fc1']=np.zeros([self.conf.time_step,]+self.shape_out_fc1.as_list()[1:])


        if self.conf.f_write_stat or self.conf.f_comp_act:
            self.dict_stat_w['conv1']=np.zeros([1,]+self.shape_out_conv1.as_list()[1:])
            self.dict_stat_w['conv2']=np.zeros([1,]+self.shape_out_conv2.as_list()[1:])
            self.dict_stat_w['fc1']=np.zeros([1,]+self.shape_out_fc1.as_list()[1:])


        self.conv_p=collections.OrderedDict()
        self.conv_p['conv1_p']=np.empty(self.dict_shape['conv1_p'],dtype=np.float32)
        self.conv_p['conv2_p']=np.empty(self.dict_shape['conv2_p'],dtype=np.float32)

        # neurons
        if self.conf.nn_mode == 'SNN':
            print('Neuron setup')

            self.input_shape_snn = [self.conf.batch_size] + self._input_shape[1:]

            print('Input shape snn: '+str(self.input_shape_snn))

            n_type = self.conf.n_type
            nc = self.conf.neural_coding

            self.neuron_list=collections.OrderedDict()

            self.neuron_list['in'] = self.track_layer(lib_snn.Neuron(self.input_shape_snn,'IN',1,self.conf,nc))
            self.neuron_list['conv1'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv1,n_type,self.fanin_conv,self.conf,nc))
            self.neuron_list['conv2'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv2,n_type,self.fanin_conv,self.conf,nc))
            self.neuron_list['fc1'] = self.track_layer(lib_snn.Neuron(self.shape_out_fc1,'OUT',512,self.conf,nc))


            # modify later
            self.n_in = self.neuron_list['in'];
            self.n_conv1 = self.neuron_list['conv1']
            self.n_conv2 = self.neuron_list['conv2']
            self.n_fc1 = self.neuron_list['fc1']

        #
        self.cmap=matplotlib.cm.get_cmap('viridis')
        #self.normalize=matplotlib.colors.Normalize(vmin=min(self.n_fc1.vmem),vmax=max(self.n_fc1.vmem))


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
        self.conv_bn_fused(self.conv2, self.conv2_bn, 1.0)

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

        self.conv2.kernel = self.conv2.kernel/self.norm['conv2']*deep_layer_const
        self.conv2.bias = self.conv2.bias/self.norm_b['conv2']


        self.fc1.kernel = self.fc1.kernel/self.norm['fc1']*deep_layer_const
        self.fc1.bias = self.fc1.bias/self.norm_b['fc1']

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


        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()



    def call_ann(self,inputs,f_training):
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
        p_conv1 = self.pool2d(a_conv1)

        s_conv2 = self.conv2(p_conv1)
        if self.f_skip_bn:
            s_conv2_bn = s_conv2
        else:
            s_conv2_bn = self.conv2_bn(s_conv2,training=f_training)
        a_conv2 = tf.nn.relu(s_conv2_bn)
        p_conv2 = self.pool2d(a_conv2)

        s_flat = tf.layers.flatten(p_conv2)

        if f_training:
           s_flat = self.dropout(s_flat,training=f_training)

        a_fc1 = self.fc1(s_flat)

        if self.conf.f_comp_act and (not self.f_1st_iter):
            self.dict_stat_w['conv1']=a_conv1.numpy()
            self.dict_stat_w['conv2']=a_conv2.numpy()
            self.dict_stat_w['fc1']=a_fc1.numpy()

        a_out = a_fc1

        if self.f_1st_iter and self.conf.nn_mode=='ANN':
            print('1st iter')
            self.f_1st_iter = False
            self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)


        #print(s_conv1.numpy().shape)
        #print(s_conv2.numpy().shape)
        #print(a_fc1.numpy().shape)


        # write stat
        if (self.conf.f_write_stat) and (not self.f_1st_iter):
            self.dict_stat_w['conv1']=np.append(self.dict_stat_w['conv1'],a_conv1.numpy(),axis=0)
            self.dict_stat_w['conv2']=np.append(self.dict_stat_w['conv2'],a_conv2.numpy(),axis=0)
            self.dict_stat_w['fc1']=np.append(self.dict_stat_w['fc1'],a_fc1.numpy(),axis=0)

        return a_out


    def print_model_conf(self):
        # print model configuration
        print('Input   N: '+str(self.in_shape))

        print('Conv1   S: '+str(self.conv1.kernel.get_shape()))
        print('Conv1   N: '+str(self.shape_out_conv1))
        print('Pool1   N: '+str(self.shape_out_conv1_p))

        print('Conv2   S: '+str(self.conv2.kernel.get_shape()))
        print('Conv2   N: '+str(self.shape_out_conv2))
        print('Pool2   N: '+str(self.shape_out_conv2_p))

        print('fc1     S: '+str(self.fc1.kernel.get_shape()))
        print('fc1     N: '+str(self.shape_out_fc1))


    def print_act_d(self):
        print('print activation')

        fig, axs = plt.subplots(6,3)

        axs=axs.ravel()

        #for idx_l, (name_l,stat_l) in enumerate(self.dict_stat_r):
        for idx, (key, value) in enumerate(self.dict_stat_r.iteritems()):
            axs[idx].hist(value.flatten())

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

        #print(np.max(self.layer_list.values()[0].kernel))
        #self.temporal_norm()
        #print(np.max(self.layer_list.values()[0].kernel))


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

        w_in_sum_conv2 = tf.reduce_sum(tf.maximum(self.conv2.kernel,0),axis=[0,1,2])
        #w_in_max_conv2 = tf.reduce_max(w_in_sum_conv2+self.conv2.bias)
        w_in_max_conv2 = tf.reduce_max(w_in_sum_conv2)
        self.conv2.kernel = self.conv2.kernel/w_in_max_conv2

        w_in_sum_fc1 = tf.reduce_sum(tf.maximum(self.fc1.kernel,0),axis=[0])
        #w_in_max_fc1 = tf.reduce_max(w_in_sum_fc1+self.fc1.bias)
        w_in_max_fc1 = tf.reduce_max(w_in_sum_fc1)
        # w_in_min_fc1 = tf.reduce_min(w_in_sum_fc1)
        #self.fc1.kernel = self.fc1.kernel-kernel_min/2
        self.fc1.kernel = self.fc1.kernel/w_in_max_fc1


        if self.verbose == True:
            print('w_in_max_conv1:' +str(w_in_max_conv1))

            print('w_in_max_conv2:' +str(w_in_max_conv2))
            print('w_in_max_conv2_1:' +str(w_in_max_conv2_1))

            print('w_in_max_fc1:' +str(w_in_max_fc1))

#        plt.ion()
#        plt.figure()
#        plt.hist(self.fc1.kernel.numpy())
#        plt.show()
#        plt.draw()
#        plt.pause(0.001)
#
#        w_in_sum_fc1 = tf.reduce_sum(tf.maximum(self.fc1.kernel,0),axis=[0])
#        plt.subplot(212)
#        plt.hist(w_in_sum_fc1)
#        plt.draw()
#        plt.pause(0.001)
#
#        w_in_max_fc1 = tf.reduce_max(w_in_sum_fc1)
#        self.fc1.kernel = self.fc1.kernel/w_in_max_fc1
#
#        plt.ioff()
#        #plt.show()


    def plot(self, x, y, mark):
        #plt.ion()
        #plt.hist(self.n_fc1.vmem)
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
    def get_total_residual_vmem(self):
        len=self.total_residual_vmem.shape[0]
        for idx_n, (nn, n) in enumerate(self.neuron_list.iteritems()):
            idx=idx_n-1
            if nn!='in' or nn!='fc1':
                self.total_residual_vmem[idx]+=tf.reduce_sum(tf.abs(n.vmem))
                self.total_residual_vmem[len-1]+=self.total_residual_vmem[idx]

    def get_total_isi(self):
        isi_count=np.zeros(self.conf.time_step)

        for idx_n, (nn, n) in enumerate(self.neuron_list.iteritems()):
            if nn!='in' or nn!='fc1':
                isi_count_n = np.bincount(np.int32(n.isi.numpy().flatten()))
                isi_count_n.resize(self.conf.time_step)
                isi_count = isi_count + isi_count_n

        return isi_count

    def get_total_spike_amp(self):
        spike_amp=np.zeros(self.spike_amp_kind)
        #print(range(0,self.spike_amp_kind)[::-1])
        #print(np.power(0.5,range(0,self.spike_amp_kind)))

        for idx_n, (nn, n) in enumerate(self.neuron_list.iteritems()):
            if nn!='in' or nn!='fc1':
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

    def bias_norm_weighted_spike(self):
        for k, l in self.layer_list.iteritems():
            if not 'bn' in k:
            #if (not 'bn' in k) and (not 'fc1' in k) :
                #l.bias = l.bias/(1-1/np.power(2,8))
                l.bias = l.bias/8.0

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
            if l !='fc1':
                self.total_comp_act[t,idx_l]=np.mean(np.abs(self.neuron_list[l].spike_counter.numpy().flatten()/(float)(t+1)-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,-1]+=self.total_comp_act[t,idx_l]


        #l='conv1'
        #print(self.neuron_list[l].spike_counter.numpy().flatten())
        #print(self.dict_stat_w[l].flatten())

    def comp_act_ws(self,t):
        self.total_comp_act[t,-1]=0.0
        for idx_l, l in enumerate(self.layer_name):
            if l !='fc1':
                #self.total_comp_act[t,idx_l]=np.mean(np.abs(self.neuron_list[l].spike_counter.numpy().flatten()/((float)(t+1)/(float)(self.conf.p_ws))-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,idx_l]=np.mean(np.abs(self.neuron_list[l].spike_counter.numpy().flatten()/((float)(t+1)/(float)(self.conf.p_ws))-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,-1]+=self.total_comp_act[t,idx_l]


    def comp_act_pro(self,t):
        self.comp_act_ws(t)

    def save_ann_act(self,inputs,f_training):
        self.call_ann(inputs,f_training)


    def cal_entropy(self):
        length = 4

        for idx_l, l in enumerate(self.layer_name):
            if l !='fc1':
                #print(self.dict_stat_w[l].shape)
                self.dict_stat_w[l][np.nonzero(self.dict_stat_w[l])]=1.0

                #print(self.dict_stat_w[l])

                #print(np.array2string(str(self.dict_stat_w[l]),max_line_width=4))
                print(self.dict_stat_w[l].shape)

                num_words = self.dict_stat_w[l].shape[0]/length
                tmp = np.zeros((num_words,)+(self.dict_stat_w[l].shape[1:]))

                for idx in range(num_words):
                    for idx_length in range(length):
                        tmp[idx] += self.dict_stat_w[l][idx*length+idx_length]*np.power(2,idx_length)

                #print(tmp)

                #plt.hist(tmp.flatten())
                #plt.show()
                #print(tmp.shape)

                print(stats.entropy(np.histogram(tmp.flatten())))




    def call_snn(self,inputs,f_training):

        count_accuracy_time_point=0
        inputs = tf.reshape(inputs,self._input_shape)

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


                #if self.conf.input_spike_mode == 'WEIGHTED_SPIKE' or self.conf.neural_coding== 'WEIGHTED_SPIKE':
                #    self.bias_norm_weighted_spike()


            spike_count = np.zeros((self.num_accuracy_time_point,)+self.n_fc1.get_spike_count().numpy().shape)

            if self.conf.f_comp_act:
                self.save_ann_act(inputs,f_training)
        plt.clf()

        #
        for t in range(self.tw):
            if self.verbose == True:
                print('time: '+str(t))
            #x = tf.reshape(inputs,self._input_shape)

            a_in = self.n_in(inputs,t)

            if self.conf.f_real_value_input_snn:
                if self.conf.neural_coding=='WEIGHTED_SPIKE':
                    a_in = inputs/self.conf.p_ws
                else:
                    a_in = inputs
            else:
                a_in = self.n_in(inputs*2.0,t)

            if self.conf.input_spike_mode == 'WEIGHTED_SPIKE' or self.conf.neural_coding== 'WEIGHTED_SPIKE':
                if (int)(t%self.conf.p_ws) == 0:
                    self.bias_enable()
                else:
                    #self.bias_enable()
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

            if self.conf.f_spike_max_pool:
                tmp = tf.reshape(self.n_conv1.get_spike_count(),[1,-1,]+self.dict_shape['conv1'].as_list()[2:])
                _, arg = tf.nn.max_pool_with_argmax(tmp,(1,2,2,1),(1,2,2,1),padding='SAME')
                arg = tf.reshape(arg,self.dict_shape['conv1_p'])
                a_conv1_f = tf.reshape(a_conv1,[-1])
                p_conv1 = tf.convert_to_tensor(a_conv1_f.numpy()[arg],dtype=tf.float32)
            else:
                p_conv1 = self.pool2d(a_conv1)

            s_conv2 = self.conv2(p_conv1)
            a_conv2 = self.n_conv2(s_conv2,t)

            if self.conf.f_spike_max_pool:
                tmp = tf.reshape(self.n_conv2.get_spike_count(),[1,-1,]+self.dict_shape['conv2'].as_list()[2:])
                _, arg = tf.nn.max_pool_with_argmax(tmp,(1,2,2,1),(1,2,2,1),padding='SAME')
                arg = tf.reshape(arg,self.dict_shape['conv2_p'])
                a_conv2_f = tf.reshape(a_conv2,[-1])
                p_conv2 = tf.convert_to_tensor(a_conv2_f.numpy()[arg],dtype=tf.float32)
            else:
                p_conv2 = self.pool2d(a_conv2)

            flat = tf.layers.flatten(p_conv2)

            s_fc1 = self.fc1(flat)
            #print('a_fc1')
            a_fc1 = self.n_fc1(s_fc1,t)



            if self.f_1st_iter == False:
                if t==self.accuracy_time_point[count_accuracy_time_point]-1:
                    # spike count
                    #spike_count[count_accuracy_time_point,:,:]=(self.n_fc3.get_spike_count().numpy())
                    # vmem
                    spike_count[count_accuracy_time_point,:,:]=(self.n_fc1.vmem.numpy())

                    tc_int, tc = self.get_total_spike_count()

                    self.total_spike_count_int[count_accuracy_time_point]+=tc_int
                    self.total_spike_count[count_accuracy_time_point]+=tc

                    #print('time '+str(t))
                    #print(spike_count.shape)
                    #print(self.n_fc3.get_spike_count().numpy())
                    #print(spike_count)
                    count_accuracy_time_point+=1

                    num_spike_count = tf.cast(tf.reduce_sum(spike_count,axis=[2]),tf.int32)


        if self.conf.f_entropy and (not self.f_1st_iter):
            self.cal_entropy()


        if self.f_1st_iter:
            self.f_1st_iter = False

            self.conv1_bn(s_conv1,training=f_training)
            self.conv2_bn(s_conv2,training=f_training)

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
        self.n_conv2.reset()
        self.n_fc1.reset()

