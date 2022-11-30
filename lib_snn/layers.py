import tensorflow as tf
# import tensorflow.contrib.eager as tfe

# import tensorflow_probability as tfp

import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers

from keras import backend

import sys

from tensorflow.python.ops import math_ops

# custom gradient
import lib_snn.ops.nn_grad


#

import numpy as np

import matplotlib.pyplot as plt

import collections

#
import lib_snn

#
from lib_snn.model import Model
from lib_snn.sim import glb_t
from lib_snn.sim import glb
from lib_snn.sim import glb_plot

#from main_hp_tune import conf

from config import conf

#
# ~class Layer(tf.keras.layers.Layer):
# def __init__(self, input_shape, data_format, conf):


# abstract class
# Layer
class Layer():
    index = None    # layer index count starts from InputGenLayer

    def __init__(self, use_bn, activation, last_layer=False, kwargs=None):
        #
        self.depth = -1

        #
        self.conf = conf
        # self.use_bn = Model.use_bn
        self.use_bn = use_bn
        #self.en_snn = Model.en_snn
        self.en_snn = (self.conf.nn_mode == 'SNN' or self.conf.f_validation_snn)
        #self.en_snn = None

        #self.use_bias = True
        #self.use_bias = conf.use_bias

        self.lmb = conf.lmb
        regularizer_type = {
            #'L1': regularizers.l1(conf.lmb),
            #'L2': regularizers.l2(conf.lmb)
            'L1': regularizers.l1(self.lmb),
            'L2': regularizers.l2(self.lmb),
        }

        self.kernel_regularizer = regularizer_type[self.conf.regularizer]

        #
        #last_layer = kwargs.pop('last_layer', None)
        #if last_layer:
            #self.last_layer = True
        #else:
            #self.last_layer = False

        self.synapse=False
        self.last_layer = last_layer

        #
        self.prev_layer = None

        #
        self.out_s = None  # output - synapse
        self.out_b = None  # output - batch norm.
        self.out_n = None  # output - neuron

        # name
        #name = kwargs.pop('name', None)
        #if name is not None:
        #    name_bn = name + '_bn'
        #    name_act = name + '_act'
        #else:
        #    name_bn = None
        #    name_act = None

        # ReLU-6
        relu_max_value = kwargs.pop('relu_max_value',None)

        # tdbn
        tdbn_arg = kwargs.pop('tdbn',None)


        # batch norm.
        if self.use_bn:
            if tdbn_arg is None:
                tdbn = conf.mode=='train' and conf.nn_mode=='SNN' and conf.tdbn
            else:
                tdbn = tdbn_arg

            #self.bn = tf.keras.layers.BatchNormalization(name=name_bn)
            #self.bn = tf.keras.layers.BatchNormalization()
            #self.bn = tf.keras.layers.BatchNormalization(epsilon=1.001e-5,name=name_bn)
            #self.bn = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)

            fused = self.conf.tf_fused_bn

            self.bn = lib_snn.layers_new.BatchNormalization(epsilon=1.001e-5,en_tdbn=tdbn,fused=fused)

            #self.bn = lib_snn.layers_new.BatchNormalization(epsilon=1.0)
        else:
            self.bn = None

        self.f_skip_bn = False
        #self.f_skip_bn = (self.conf.nn_mode == 'ANN' and self.conf.f_fused_bn) or (self.conf.nn_mode == 'SNN')
        #self.f_skip_bn = self.conf.f_fused_bn

        # activation, neuron
        #self.activation = activation
        #self.act = activation
        self.act = None

        # DNN mode
        if activation == 'relu':
            #self.act_dnn = tf.keras.layers.ReLU(max_value=relu_max_value, name=name_act)
            self.act_dnn = tf.keras.layers.ReLU(max_value=relu_max_value)
            #self.act_dnn = tf.keras.layers.ReLU(max_value=6.0)
        elif activation == 'softmax':
            #self.act_dnn = tf.keras.layers.Softmax(name=name_act)
            self.act_dnn = tf.keras.layers.Softmax()
        else:
            self.act_dnn = None

        # self.act_dnn = activation
        self.act_snn = None

        #
        self.shape_n = None

        #
        self.output_shape_fixed_batch = None

        #
        self.en_record_output = False
        self.record_output = None
        self.record_logit = None

        # neuron setup
        if self.en_snn:
            if self.last_layer:
                self.n_type = 'OUT'
            else:
                self.n_type = self.conf.n_type

        # bias control
        self.bias_en_time = 0
        #self.f_bias_ctrl = False
        self.bias_ctrl_sub = None

        # bias control - SNN inference
        if self.en_snn:
            self.bias_control = self.conf.bias_control
        else:
            self.bias_control = False

    #
    def build(self, input_shapes):
        # super(Conv2D,self).build(input_shapes)
        # print(Conv2D.__mro__[2])
        # tf.keras.layers.Conv2D.build(self,input_shapes)
        # self.__mro__[2].build(self,input_shapes)

        # build ann model
        #print('build layer - {}'.format(self.name))
        super().build(input_shapes)
        # print(super())
        # print(super().build)

        #print(super().build)

        #assert False

        #
        if isinstance(self,lib_snn.layers.InputGenLayer):
            self.output_shape_fixed_batch = input_shapes
        else:
            self.output_shape_fixed_batch = super().compute_output_shape(input_shapes)

        # self.act_snn = lib_snn.layers.Neuron(self.output_shape_fixed_batch,self.conf,\
        # n_type,self.conf.neural_coding,depth,self.name)

        # self.act_dnn = tf.keras.layers.ReLU()

        # if not self.en_snn:
        # self.act = self.act_dnn

        if self.en_snn:
            #print('---- SNN Mode ----')
            #print('Neuron setup')

            # self.shape_n = lib_snn.util.cal_output_shape_Conv2D(self.data_format,input_shapes,self.filters,self.kernel_size,self.strides[0])
            # self.output_shape_fixed_batch = super().compute_output_shape(input_shapes)

            # print('output')
            # print(self.output_shape_fixed_batch)

            # TODO: InputGenLayer
            #if isinstance(self,lib_snn.layers.Identity):
            #if hasattr(self,'activation'):
            #if hasattr(self, 'activation'):
            #if self.activation is None:
            if self.act_dnn is None:
                self.act_snn = lambda x,y,z : tf.identity(x)
                #print(self.name)
                #assert False
            else:
                n_type = self.n_type

                if isinstance(self,lib_snn.layers.Add):
                #if isinstance(self, lib_snn.layers.Conv2D):
                    n_type = 'IF'
                    #n_type = 'LIF'

                self.act_snn = lib_snn.neurons.Neuron(self.output_shape_fixed_batch, self.conf, \
                                                      n_type, self.conf.neural_coding, self.depth, 'n_'+self.name)


            #self.act_snn = lib_snn.neurons.Neuron(self.output_shape_fixed_batch, self.conf, \
            #                                      self.n_type, self.conf.neural_coding, self.depth, 'n_'+self.name)

            #
            self.bias_ctrl_sub = tf.zeros(self.output_shape_fixed_batch)
            #self.f_bias_ctrl = tf.constant(False,dtype=tf.bool,shape=self.output_shape_fixed_batch[0])
            self.f_bias_ctrl = tf.constant(False,dtype=tf.bool,shape=[self.output_shape_fixed_batch[0],self.output_shape_fixed_batch[-1]])

        #else:
        #    if isinstance(self,lib_snn.layers.Identity):
        #        self.act_dnn = lambda x,y : tf.identity(x)

        # setup activation
        if self.en_snn:
            self.act = self.act_snn
        else:
            self.act = self.act_dnn

        #
        if self.conf.fine_tune_quant:
            #self.quant_max = tf.Variable(tf.zeros([]),trainable=False,name=name+'/quant_max')
            #self.quant_max = tf.constant(0.0,shape=[],name='quant_max')

            if hasattr(self, 'act'):
                #if not isinstance(layer,lib_snn.layers.InputGenLayer):
                #if not layer.act_dnn is None:
                #if not (layer.activation is None):
                if not (self.act_dnn is None):
                    #print(self)
                    #print(self.name)
                    stat = lib_snn.calibration.read_stat(None, self, 'max_999')

                    if self.conf.f_w_norm_data:
                        stat_max = tf.ones(shape=[])
                    else:
                        stat_max = tf.reduce_max(stat)
                    #self.quant_max = tf.constant(stat_max,shape=[],name=self.name+'/quant_max')
                    #layer.quant_max = tf.Variable(stat_max, trainable=False, name='quant_max')

                    if self.depth==100:
                        #self.vth_l = tf.Variable(initial_value=stat_max,shape=[],name=self.name+'/vth_l',trainable=True)
                        self.vth_l = tf.Variable(initial_value=stat_max,shape=[],name=self.name+'/vth_l',trainable=False)
                    else:
                        self.vth_l = tf.constant(stat_max,shape=[],name=self.name+'/vth_l')

        #
        self.built = True
        #print('build layer - done')

    def init_record_output(self):
        self.record_output = tf.Variable(tf.zeros(self.output_shape_fixed_batch),trainable=False,name='record_output')
        #
        f_hold_temporal_tensor=True
        if self.conf.f_hold_temporal_tensor:
            output_shape_list = self.output_shape_fixed_batch.as_list()
            output_shape_list.insert(0,self.conf.time_step)
            output_shape = tf.TensorShape(output_shape_list)

            #[time, batch, width, height, channel]
            self.record_output = tf.Variable(tf.zeros(output_shape),trainable=False,name='record_output')

        if self.last_layer:
            self.record_logit= tf.Variable(tf.zeros(self.output_shape_fixed_batch),trainable=False,name='record_logit')


    #
    # def call(self,input,training):
    # s = super().call(input)
    #
    # s = tf.nn.relu(s)
    #
    # return s

    #
    # def call_set_aside_for_future(self,input,training):
    def call(self, input, training):
        #print('layer - {:}, training - {:}'.format(self.name,training))
        #print('layer call - {}'.format(self.name))

        if training is None:
            training = backend.learning_phase()


        s = super().call(input)


        if False:
            if (self.name == 'predictions') and (not self.conf.full_test):
                print(input)
                print(s)
                print(self.bias)

        # print('depth: {}, name: {}'.format(self.depth, self.name))
        # if self.depth==1:
        # print(super().call(input))
        #    print(super().call)
        # print(s)
        # print(self.kernel)

        # bias control test
        #if (glb.model_compiled) and (self.depth*2 > glb_t.t) and (self.conf.nn_mode=='SNN'):
            #self.use_bias = False
        #else:
            #self.use_bias = self.conf.use_bias

        #self.use_bias = False

        # bias control
        if self.en_snn and self.bias_control:
            s = self.bias_control_run(s)


        if (self.use_bn) and (not self.f_skip_bn):
            b = self.bn(s, training=training)
            #if glb.model_compiled:
            #    assert False

            #if False:
            #if True:
            if self.conf.debug_mode and self.conf.verbose_snn_train:
                if glb.model_compiled:
                    #print('{:.3e}'.format(tf.reduce_max(b)))
                    #print('after bn> {} - max: {}, mean: {}'.format(self.name,tf.reduce_max(b),tf.reduce_mean(b)))
                    print('before bn> {} - max: {:.3g}, mean: {:.3g}, var: {:.3g}'
                          .format(self.name,tf.reduce_max(s),tf.reduce_mean(s),tf.math.reduce_variance(s)))
                    print('after bn> {} - max: {:.3g}, mean: {:.3g}, var: {:.3g}, moving_mean: {:.3g}, moving_var: {:.3g}'
                          .format(self.name,tf.reduce_max(b),tf.reduce_mean(b),tf.math.reduce_variance(b), tf.reduce_mean(self.bn.moving_mean),tf.reduce_mean(self.bn.moving_variance)))
                    #print('after bn> {} - max: {:.3e}, mean: {:.3e}'.format(self.name))

                    if self.name=='conv1':
                        print(self.bn.moving_variance)
        else:
            b = s
            #print('here')
            #print(self.name)

        if self.act is None:
            n = b
        else:
            if self.en_snn:
                n = self.act(b, glb_t.t, training)
                if self.last_layer and (not (self.act_dnn is None)):
                    # softmax
                    n = self.act_dnn(n)
            else:
                if self.conf.fine_tune_quant and not self.last_layer:
                    n = lib_snn.calibration.clip_floor_act(b, self.vth_l, 64.0)
                else:
                    n = self.act(b)

                #
                #num_bits=np.log2(self.conf.time_step)
                #n=tf.quantize_and_dequantize_v4(n, 0, 1, signed_input=False, num_bits=num_bits, range_given=True)
                #n = tf.clip_by_value(n,0,1)

        #self.quant_act=True
        #if self.quant_act:
        ##    #n = tf.quantization.fake_quant_with_min_max_vars(n,0,1,num_bits=8)
        #    #n = tf.quantization.fake_quant_with_min_max_vars(n,0,1,num_bits=8)
            #if self.conf.fine_tune_quant and (not self.last_layer):
            #if self.conf.fine_tune_quant:
                #n=tf.quantize_and_dequantize_v4(n, 0, 1, signed_input=False, num_bits=8, range_given=True)
                #n = tf.quantize_and_dequantize_v4(n, 0, self.quant_max, signed_input=False, num_bits=8, range_given=True)
                #n = tf.quantize_and_dequantize_v4(n, 0, self.quant_max, signed_input=False, num_bits=6, range_given=True)
                #n = tf.quantize_and_dequantize_v4(n, 0, self.quant_max, signed_input=False, num_bits=4, range_given=True)
                #n = tf.clip_by_value(tf.math.floor(b*64/self.quant_max)/64,0,1)*self.quant_max
                #n=lib_snn.calibration.clip_floor_act(b, self.quant_max, 64.0)

                #print('layer name')
                #print(self.name)

                #n=lib_snn.calibration.clip_floor_act(b)
                #n=lib_snn.calibration.clip_floor_shift_act(b, self.vth_l, 64.0)
                #n = tf.quantize_and_dequantize_v4(n, 0, self.quant_max, signed_input=False, num_bits=12, range_given=True)
        #    n=tf.quantize_and_dequantize_v4(n, 0, 1, signed_input=False, num_bits=16, range_given=True)
        #    #n = tf.quantization.quantize(n,0,1,T=tf.qint8)
        #if self.name=='conv1' and glb.model_compiled:
        #    print('layer - {}'.format(self.name))
        #    self.n =n
        #    print
        #    assert False
        ret = n

        #if self.en_snn:
        #    if self.last_layer:
        #        #if conf.snn_output_type ==
        #        time = conf.time_step - self.bias_en_time
        #        ret = ret / time
        #        #ret = ret

        #

        #if self.name=='fc1' or self.name=='fc2':
            #print(s)
            #print(b)
            #print()

        #print(self.name)
        #print(n[0])

        #if (self.name == 'predictions') and (glb.model_compiled) and (not conf.full_test):
        #    print(n)

        if False:
            if (self.name == 'predictions') and (glb.model_compiled) and (not conf.full_test):
                #print('time: {}'.format(t))
                #print(ret)

                if self.conf.num_test_data == 1:
                    print('curr')
                    print(tf.argmax(b,axis=1))
                    print(b)
                    print('acum')
                    print(tf.argmax(n,axis=1))
                    print(n)
                else:
                    print('curr')
                    print(tf.argmax(b,axis=1)[conf.verbose_visual_idx])
                    print(b[conf.verbose_visual_idx])
                    print('acum')
                    print(tf.argmax(n,axis=1)[conf.verbose_visual_idx])
                    print(n[conf.verbose_visual_idx])


        if self.en_record_output:
            #self.record_output = ret
            if self.conf.f_hold_temporal_tensor:
                t=glb_t.t
                indices = [[t]]
                ret_t = tf.expand_dims(ret,axis=0)
                tf.tensor_scatter_nd_update(self.record_output,indices,ret_t)
                #indices=[[:,t]]
                #tf.tensor_scatter_nd_update(self.record_output,[:,t,])
            else:
                self.record_output.assign(ret)
            #if self.name == 'predictions':
            if self.last_layer:
                #self.record_logit = b
                self.record_logit.assign(b)

        # debug
        # TODO: debug mode set - glb.model_compiled and self.conf.debug_mode and ~~
        #if (glb.model_compiled) and (self.conf.debug_mode and self.conf.nn_mode=='SNN'):
            #self.plot()

        #
        #if (not self.conf.full_test) and (glb.model_compiled) and (self.conf.debug_mode and self.conf.nn_mode=='SNN'):
        #    if self.name=='conv1':
        #        print('{:4d}'.format(glb_t.t))
        #        print('{:4d}: {}'.format(glb_t.t, b.numpy().flatten()[0:10]))
        #        print('{:4d}: {}'.format(glb_t.t, self.act.vmem.numpy().flatten()[0:10]))
                #print('{:4d}: {}'.format(glb_t.t, self.act.spike_count_int.numpy().flatten()[0:10]))
        #        print(': {}'.format(self.act.spike_count_int.numpy().flatten()[0:10]))

        #if (glb.model_compiled) and (self.conf.debug_mode and self.conf.nn_mode == 'SNN'):
        #    self.plot_neuron()

        if False:
        #if self.conf.debug_mode and self.conf.nn_mode=='SNN':
            #print('neuron input')
            #print('{}: {} - in {}, out {}'.format(glb_t.t,self.depth,tf.reduce_mean(b),tf.reduce_mean(n)))

            #print('neuron output')
            #print('{}: {} - {}'.format(glb_t.t,self.depth,tf.reduce_mean(n)))
            #print('{}: {}'.format(self.depth,tf.reduce_mean(self.act_snn.vmem)))
            #print('{}: {}'.format(self.depth,tf.reduce_mean(self.act_snn.vmem)))

            #if self.depth==0:
                #print('input')
                #print(tf.reduce_mean(input))

            #if (self.depth==1):
                #print(self.name)
                #print('{}: {} - {}'.format(glb_t.t, self.depth, self.act.out))
                #print('{}: {} - {}'.format(glb_t.t, self.depth, n[0,0,0]))

            #if (self.depth == 15) and (glb_t.t % 10 == 0):
            #    print(tf.reduce_mean(self.act.get_spike_count()))

            #if self.depth==1:
                #print(input)
                #print(tf.reduce_mean(input))
                #print(tf.reduce_mean(s))
                #print(tf.reduce_mean(b))
                #print(tf.reduce_mean(n))

            #print(self.name)
            #print(self.built)

            #print_period=10
            print_period=self.conf.time_step_save_interval
            #if (self.depth==4 or self.depth==8 or self.depth==12) and (glb_t.t%print_period==(print_period-1)):
            #if glb_t.t % print_period == (print_period - 1):
            #if (glb_t.t % print_period == 0) and (self.depth>1):
            if (glb_t.t % print_period == 0) and (glb.model_compiled):
                #print(self.name)
                #print('{}: {} - spike count {}'.format(glb_t.t, self.depth, self.act.get_spike_count()))
                #print('{}: {} - spike count mean {}'.format(glb_t.t, self.depth, tf.reduce_mean(self.act.get_spike_count())))
                print('{:4d}: {:3d} {:>8} - spike count max {:8.2f}, mean_s {:8.4f}, mean_s_t {:8.4f}'.format(glb_t.t, self.depth,
                                                                    self.name,
                                                                    tf.reduce_max(self.act.get_spike_count()),
                                                                    tf.reduce_mean(self.act.get_spike_count()),
                                                                    tf.reduce_mean(self.act.get_spike_count())/glb_t.t))

                #if (self.depth==4 or self.depth==8 or self.depth==12 or self.depth==16) and (glb_t.t%print_period==(print_period-1)):
            #if (self.depth == 1) and (glb_t.t % 1 == 0):
                #print('{}: {} - {}'.format(self.depth,glb_t.t,tf.reduce_mean(self.act_snn.spike_counter)))
                #print('{}: {} - in {}, out {}'.format(glb_t.t,self.depth,tf.reduce_mean(self.act_snn.input),tf.reduce_mean(self.act_snn.out)))
                #print('{}: {} - {}'.format(glb_t.t,self.depth,self.act_snn.out))

            if False:
                if (self.depth == 16) and (glb_t.t % print_period == 0):

                    if self.en_snn:
                        #print('{}: {} - input {}'.format(glb_t.t, self.depth, input))
                        #print('{}: {} - s {}'.format(glb_t.t, self.depth, s))
                        print('{}: {} - out {}'.format(glb_t.t, self.depth, self.act.out))
                        #print('{}: {} - {}'.format(glb_t.t, self.depth, self.act.vth))
                    else:
                        print('{}: {} - {}'.format(glb_t.t, self.depth, b))

            #print('{}: {} - {}'.format(glb_t.t,self.depth,tf.reduce_mean(self.act_snn.out)))

        return ret

    #
    def bias_control_run(self, synaptic_output):
        if hasattr(self, 'bias') and self.en_snn:
            if self.conf.use_bias and tf.reduce_any(self.f_bias_ctrl):
                ret = tf.subtract(synaptic_output, self.bias_ctrl_sub)
                return ret
            else:
                return synaptic_output
        else:
            return synaptic_output


    # TODO: move
    def plot(self):

        #if self.name=='conv1':

        axe=glb_plot.axes.flatten()[self.depth]

        #idx=0,0,0,0
        #idx=7
        idx=glb_plot.idx

        if self.name=='predictions':
            idx=7
            out = self.act.vmem.numpy().flatten()[idx]                      # vmem
        else:
            out = self.act.get_spike_count_int().numpy().flatten()[idx]     # spike

        #print('{} - {}, {}'.format(glb_t.t,spike,spike/glb_t.t))
        #lib_snn.util.plot(glb_t.t,spike/glb_t.t,axe=axe)
        lib_snn.util.plot(glb_t.t,out/glb_t.t,axe=axe)

    def plot_neuron(self):
        assert False
        #for idx, layer_name in glb_plot.layers:

        for idx in range(0, 16):
            axe = glb_plot.axes.flatten()[idx]
            out = self.act.get_spike_count_int().numpy().flatten()[idx]  # spike
            lib_snn.util.plot(glb_t.t, out / glb_t.t, axe=axe)

    #
    def reset(self):
        if self.en_snn:
            if self.act_snn is not None:
                self.act_snn.reset()


    #
    def bn_fusion(self):
        self._bn_fusion(self.bn)
        self.f_skip_bn = True


    def bn_defusion(self):

        assert False, 'not implemented and verified yet'
        self._bn_defusion(self, self.bn)

    # batch normalization fusion - conv, fc
    def _bn_fusion(self, bn):
        gamma = bn.gamma
        beta = bn.beta
        mean = bn.moving_mean
        var = bn.moving_variance
        ep = bn.epsilon
        inv = math_ops.rsqrt(var + ep)
        inv *= gamma

        #if self.name == 'expanded_conv_depthwise':
        #    print(self.depthwise_kernel.shape)

        if isinstance(self,lib_snn.layers.DepthwiseConv2D):
            inv_e = tf.expand_dims(inv,0)
            inv_e = tf.expand_dims(inv_e,0)
            inv_e = tf.expand_dims(inv_e,-1)
            self.depthwise_kernel = self.depthwise_kernel* math_ops.cast(inv_e, self.depthwise_kernel.dtype)
            #if self.name=='expanded_conv_depthwise':
            #    print(self.depthwise_kernel.shape)
        else:
            self.kernel = self.kernel * math_ops.cast(inv, self.kernel.dtype)

        self.bias = ((self.bias - mean) * inv + beta)
        #self.bias = self.bias*2
        #assert False

#
def _bn_defusion(layer, bn):
    gamma = bn.gamma
    beta = bn.beta
    mean = bn.moving_mean
    var = bn.moving_variance
    ep = bn.epsilon
    inv = math_ops.rsqrt(var + ep)
    inv *= gamma

    layer.kernel = layer.kernel / inv
    layer.bias = (layer.bias - beta) / inv + mean

# Input
class InputLayer(Layer, tf.keras.layers.InputLayer):
    # class InputLayer(Layer):
    def __init__(self,
                 input_shape=None,
                 batch_size=None,
                 dtype=None,
                 input_tensor=None,
                 sparse=None,
                 name=None,
                 ragged=None,
                 type_spec=None,
                 **kwargs):
        tf.keras.layers.InputLayer.__init__(
            self,
            input_shape,
            batch_size,
            dtype,
            input_tensor,
            sparse,
            name,
            ragged,
            type_spec,
            **kwargs)
        Layer.__init__(self, False, None, kwargs=kwargs)

        print('init')
        # assert False

    def build(self, input_shapes):
        #print('build input')
        # super().build(input_shapes)

        assert False

    # def call(self, inputs):
    # def call(self, inputs, *args, ** kwargs):
    def call(self, inputs, training):
        #print('call input')

        assert False


# custom input layer - for spike input generation
class InputGenLayer(Layer, tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        tf.keras.layers.Layer.__init__(self, **kwargs)
        Layer.__init__(self, False, None, kwargs=kwargs)

        #self.act_dnn = tf.identity()

        #
        Layer.index = 0             # start of models
        self.depth = Layer.index
        self.n_type = 'IN'
        #self.use_bias=conf.use_bias
        #self.kernel=1           # dummy
        #self.bias=0

    #def call(self, inputs, training):
        # print('input gen layer - call')
        #print(inputs)
    #    return inputs




# Conv2D
class Conv2D(Layer, tf.keras.layers.Conv2D):
    # class Conv2D(tf.keras.layers.Conv2D,Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 activation=None,
                 activity_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 kernel_constraint=None,
                 bias_constraint=None,
                 bias_regularizer=None,
                 use_bn=False,  # use batch norm.
                 **kwargs):

        Layer.__init__(self, use_bn, activation, kwargs=kwargs)

        tf.keras.layers.Conv2D.__init__(
            self,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=conf.data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=conf.use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer='zeros',
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=bias_regularizer,
            #bias_regularizer=self.kernel_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            # dynamic=True,
            **kwargs)

        #
        Layer.index += 1
        self.depth = Layer.index
        self.synapse=True


# Conv2D
class DepthwiseConv2D(Layer, tf.keras.layers.DepthwiseConv2D):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 use_bn=False,  # use batch norm.
                 **kwargs):
        Layer.__init__(self, use_bn, activation, kwargs=kwargs)

        tf.keras.layers.DepthwiseConv2D.__init__(
            self,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        #
        Layer.index += 1
        self.depth = Layer.index
        self.synapse=True



# Dense
class Dense(Layer, tf.keras.layers.Dense):
    def __init__(self,
                 units,
                 activation=None,
                 # use_bias=True
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 #bias_initializer='glorot_uniform',
                 # kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_bn=False,  # use batch norm.
                 last_layer=False,
                 **kwargs):

        Layer.__init__(self, use_bn, activation, last_layer, kwargs=kwargs)

        tf.keras.layers.Dense.__init__(
            self,
            units,
            activation=None,
            use_bias=conf.use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            #bias_initializer='zeros',
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=bias_regularizer,
            #bias_regularizer=self.kernel_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            # dynamic=True,
            **kwargs)

        #
        Layer.index += 1
        self.depth = Layer.index
        self.synapse=True


#
class Add(Layer, tf.keras.layers.Add):
    def __init__(self, use_bn=False, epsilon=0.001, activation=None, **kwargs):
        tf.keras.layers.Add.__init__(self, **kwargs)

        #Layer.__init__(self, use_bn=use_bn, epsilon=epsilon, activation=activation, kwargs=kwargs)
        Layer.__init__(self, use_bn=use_bn, activation=activation, kwargs=kwargs)

        # self.act = activation

        #self.use_bias = False
        #self.kernel = tf.constant(1.0, shape=[], name='kernel')
        #self.bias = tf.zeros(shape=[],name='bias')

        #print(self.output)
        #assert False

        #
        Layer.index += 1
        self.depth = Layer.index

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_bn': self.use_bn,
            'act': self.act,
        })
        return config


    def call_tmp(self, input, training):

        x = Layer.call(self,input,training)

        if self.use_bias:
            x = x + self.bias

        return x

#
class Identity(Layer, tf.keras.layers.Layer):
    def __init__(self, use_bn=False, epsilon=0.001, activation=None, **kwargs):
        tf.keras.layers.Layer.__init__(self, **kwargs)
        #Layer.__init__(self, use_bn=use_bn, epsilon=epsilon, activation=activation, kwargs=kwargs)
        Layer.__init__(self, use_bn=use_bn, activation=activation, kwargs=kwargs)

        self.kernel = tf.constant(1.0, shape=[], name='kernel')
        self.bias = tf.zeros(shape=[],name='bias')

    #def build(self,input_shape):
        ##self.kernel = self.add_weight("kernel",shape=[],initializer='ones',trainable=False)
        #self.kernel = tf.Variable("kernel",shape=[],initializer='ones',trainable=False)

    def call(self, inputs, training):
        ret = tf.multiply(inputs,self.kernel)
        ret = tf.add(ret,self.bias)

        if self.en_record_output:
            #self.record_output = ret
            self.record_output.assign(ret)

        return ret

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'use_bn': self.use_bn,
            'act': self.act,
            #'kernel': self.kernel,
        })
        return config


# MaxPolling2D
class MaxPool2D(Layer, tf.keras.layers.MaxPool2D):
    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding='valid',
                 # data_format=None,
                 **kwargs):

        if strides!=(2,2):
            assert False, 'only support stride (2x2) in maxpooling2d'

        tf.keras.layers.MaxPool2D.__init__(
            self,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=conf.data_format,
            **kwargs)

        Layer.__init__(self, False, None, kwargs=kwargs)

        self.prev_layer_set_done=False

    def build(self, input_shapes):
        tf.keras.layers.MaxPool2D.build(self, input_shapes)

        self.output_shape_fixed_batch = super().compute_output_shape(input_shapes)
        #print('maxpool')
        #print(self.output_shape_fixed_batch)

    def call_tmp(self, inputs):
        # s = super().call(self,inputs)
        # return s
        return tf.keras.layers.MaxPool2D.call(self, inputs)

    def call(self, inputs):

        #assert False
        #if not Model.f_load_model_done:
            #return tf.keras.layers.MaxPool2D.call(self, inputs)

        if self.en_snn and self.prev_layer_set_done:

            #print('current_layer - {}'.format(self.name))
            #print('prev layer - {}'.format(self.prev_layer.name))


            # spike_count = Model.model.get_layer(name=self.prev_layer_name).act.get_spike_count()
            spike_count = self.prev_layer.act.get_spike_count()
            #print(spike_count)
            output_shape = self.output_shape_fixed_batch

            # print('spike count - {}'.format(tf.reduce_sum(spike_count)))
            #print(self.name)
            ret = lib_snn.layers.spike_max_pool_2d_22(inputs, spike_count, output_shape)
            #print(inputs.shape)
            #print(ret.shape)
            return ret
        else:
            return tf.keras.layers.MaxPool2D.call(self, inputs)


# GlobalAveragePooling2D
class GlobalAveragePooling2D(Layer, tf.keras.layers.GlobalAveragePooling2D):
    def __init__(self,
                 **kwargs):

        tf.keras.layers.GlobalAveragePooling2D.__init__(self,**kwargs)
        Layer.__init__(self, use_bn=False, activation=None, last_layer=False, kwargs=kwargs)

    #def call(self, inputs):

        #name='avg_pool')(x)

# ZeroPadding2D
class ZeroPadding2D(Layer, tf.keras.layers.ZeroPadding2D):
    def __init__(self,
                 **kwargs):

        tf.keras.layers.ZeroPadding2D.__init__(self,**kwargs)
        Layer.__init__(self, use_bn=False, activation=None, last_layer=False, kwargs=kwargs)



############################################################
## spike max pool (spike count based gating function)
############################################################
#@tf.function
# spike max pool 2d, stride = (2,2)
@tf.custom_gradient
def spike_max_pool_2d_22(feature_map, spike_count, output_shape):
    # tmp = tf.reshape(spike_count,(1,-1,)+spike_count.numpy().shape[2:])

    #assert False
    #tmp = tf.reshape(spike_count, (1, -1,) + tuple(tf.shape(spike_count)[2:]))
    tmp = tf.reshape(spike_count, (1, -1,) + tuple(spike_count.shape[2:]))
    #print(output_shape)
    #print(tf.shape(spike_count)[2:].numpy())

    #print(tmp)
    #assert False
    #tmp = tf.reshape(spike_count, (1, -1,) + tf.shape(spike_count)[2:])

    # old
    _, arg = tf.nn.max_pool_with_argmax(tmp, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')
    conv_f = tf.reshape(feature_map, [-1])
    arg = tf.reshape(arg, [-1])

    p_conv = tf.gather(conv_f, arg)
    p_conv = tf.reshape(p_conv, output_shape)

    #_, arg = tf.nn.max_pool_with_argmax(spike_count, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

    #_, arg = tf.nn.max_pool_with_argmax(tmp, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME', include_batch_in_index=True)
    # arg = tf.reshape(arg,output_shape)

    #print(arg)

    #print(feature_map.shape)

    #print(arg.shape)
    #assert False

    # p_conv = tf.convert_to_tensor(conv_f.numpy()[arg],dtype=tf.float32)

    def grad(upstream):
        #print(p_conv.shape)
        #print(upstream.shape)
        #assert False

        if False:
            dim_in_b, dim_in_y, dim_in_x, dim_in_c = upstream.shape

            dim_out_b=dim_in_b
            dim_out_y=dim_in_y*2
            dim_out_x=dim_in_x*2
            dim_out_c=dim_in_c

            dim_out = [dim_out_b,dim_out_x,dim_out_y,dim_out_c]
            #print(dim_out)

            print(upstream.shape)
            print(dim_out)

            upstream_expand = tf.broadcast_to(upstream,dim_out)

            #dy_dx_z = tf.zeros(dy_dx.shape)

            dim_flat=tf.reduce_prod(tmp.shape)
            zeros = tf.zeros(dim_flat)
            ones = tf.ones(arg.shape)

            #print(arg)
            #print(zeros)
            #print(ones)

            arg_wrap = [[_arg] for _arg in arg]

            grad_mask = tf.tensor_scatter_nd_update(zeros,arg_wrap,ones)
            grad_mask = tf.reshape(grad_mask,upstream_expand.shape)

            #dy_dx = tf.where(grad_mask)
            dy_dx = tf.math.multiply(upstream_expand,grad_mask)

        zeros = tf.zeros(conv_f.shape)
        #ones = tf.ones(arg.shape)
        #print(upstream.shape)
        upstream_flat=tf.reshape(upstream,[-1])
        #arg_wrap = [[_arg] for _arg in arg]
        arg_wrap = tf.reshape(arg,[arg.shape[0],1])

        dy_dx = tf.tensor_scatter_nd_update(zeros,arg_wrap,upstream_flat)
        dy_dx = tf.reshape(dy_dx, feature_map.shape)

        #assert False

        return dy_dx, tf.stop_gradient(spike_count), tf.stop_gradient(output_shape)

    return p_conv, grad

# def spike_max_pool_temporal(feature_map, spike_count, output_shape):


# def spike_max_pool(feature_map, spike_count, output_shape):
#
#    max_pool = {
#        'TEMPORAL': spike_max_pool_rate
#    }.get(self.neural_coding, spike_max_pool_rate)
#
#    p_conv = max_pool(feature_map, spike_count, output_shape)
#
##    return p_conv
