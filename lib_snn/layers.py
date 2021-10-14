import tensorflow as tf
#import tensorflow.contrib.eager as tfe

#import tensorflow_probability as tfp

import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers

import sys


from tensorflow.python.ops import math_ops

import numpy as np

import matplotlib.pyplot as plt

import collections

#
import lib_snn

#
from lib_snn.model import Model


#
#~class Layer(tf.keras.layers.Layer):
    #def __init__(self, input_shape, data_format, conf):


# abstract class
# Layer
class Layer():
    index=-1
    def __init__(self,use_bn,activation,**kwargs):
        #
        self.depth=-1

        #
        self.conf = Model.conf
        #self.use_bn = Model.use_bn
        self.use_bn=use_bn
        self.en_snn = Model.en_snn

        #
        self.prev_layer=None

        #
        self.out_s = None       # output - synapse
        self.out_b = None       # output - batch norm.
        self.out_n = None       # output - neuron

        # name
        name = kwargs.pop('name', None)
        if name is not None:
            name_bn = name+'_bn'
            name_act = name+'_act'
        else:
            name_bn = None
            name_act = None

        # batch norm.
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(name=name_bn)
            #self.bn = tf.keras.layers.BatchNormalization(epsilon=1.001e-5,name=name_bn)
        else:
            self.bn = None

        # activation, neuron
        # DNN mode
        if activation == 'relu':
            self.act_dnn = tf.keras.layers.ReLU(name=name_act)
        elif activation == 'softmax':
            self.act_dnn = tf.keras.layers.Softmax(name=name_act)
        else:
            self.act_dnn = None

        #self.act_dnn = activation
        self.act_snn = None

        #
        self.shape_n = None

        #
        self.output_shape_fixed_batch=None

        # neuron setup
        if self.en_snn:
            self.n_type = self.conf.n_type



    #
    def build(self, input_shapes):
        #super(Conv2D,self).build(input_shapes)
        #print(Conv2D.__mro__[2])
        #tf.keras.layers.Conv2D.build(self,input_shapes)
        #self.__mro__[2].build(self,input_shapes)

        #print('build layer')
        super().build(input_shapes)
        #print(super())
        #print(super().build)


        #
        self.output_shape_fixed_batch = super().compute_output_shape(input_shapes)
        #print(self.output_shape_fixed_batch)

        #self.act_snn = lib_snn.layers.Neuron(self.output_shape_fixed_batch,self.conf,\
                                             #n_type,self.conf.neural_coding,depth,self.name)

        #self.act_dnn = tf.keras.layers.ReLU()

        #if not self.en_snn:
            #self.act = self.act_dnn

        if self.en_snn:
            print('---- SNN Mode ----')
            print('Neuron setup')

            #self.shape_n = lib_snn.util.cal_output_shape_Conv2D(self.data_format,input_shapes,self.filters,self.kernel_size,self.strides[0])
            #self.output_shape_fixed_batch = super().compute_output_shape(input_shapes)

            #print('output')
            #print(self.output_shape_fixed_batch)


            self.act_snn = lib_snn.neurons.Neuron(self.output_shape_fixed_batch,self.conf,\
                                                self.n_type,self.conf.neural_coding,self.depth,self.name)

        # setup activation
        if self.en_snn:
            self.act = self.act_snn
        else:
            self.act = self.act_dnn

        #
        self.built = True

    #
    #def call(self,input,training):
        #s = super().call(input)
#
        #s = tf.nn.relu(s)
#
        #return s

    #
    #def call_set_aside_for_future(self,input,training):
    def call(self,input,training):
        #print('layer call')
        s = super().call(input)

        #print('depth: {}, name: {}'.format(self.depth, self.name))
        #if self.depth==1:
            #print(super().call(input))
        #    print(super().call)
            #print(s)
            #print(self.kernel)


        if (self.use_bn) and (not Model.f_skip_bn):
            b = self.bn(s,training=training)
        else:
            b = s

        if self.act is None:
            n = b
            #assert False
        else:
            if self.en_snn:
                n = self.act(b,Model.t)
            else:
                n = self.act(b)
        ret = n

        return ret

    #
    def bn_fusion(self):
        self._bn_fusion(self,self.bn)


    # batch normalization fusion - conv, fc
    def _bn_fusion(self, layer, bn):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        layer.kernel = layer.kernel*math_ops.cast(inv,layer.kernel.dtype)
        layer.bias = ((layer.bias-mean)*inv+beta)

    #
    def bn_defusion(self):

        assert False, 'not implemented and verified yet'
        self._bn_defusion(self,self.bn)

    #
    def _bn_defusion(self, layer, bn):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        layer.kernel = layer.kernel/inv
        layer.bias = (layer.bias-beta)/inv+mean


# Input
class InputLayer(Layer,tf.keras.layers.InputLayer):
#class InputLayer(Layer):
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
        Layer.__init__(self,False,None,**kwargs)



        print('init')
        #assert False

    def build(self, input_shapes):
        print('build input')
        #super().build(input_shapes)

        assert False

    #def call(self, inputs):
    #def call(self, inputs, *args, ** kwargs):
    def call(self, inputs, training):
        print('call input')

        assert False


# custom input layer - for spike input generation
class InputGenLayer(Layer,tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        Layer.__init__(self,False,None,**kwargs)
        tf.keras.layers.Layer.__init__(self)

        #
        Layer.index+= 1
        self.depth = Layer.index

    def call(self, inputs, training):
        #print('input gen layer - call')
        #print(inputs)
        return inputs



# Conv2D
class Conv2D(Layer,tf.keras.layers.Conv2D):
#class Conv2D(tf.keras.layers.Conv2D,Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 activation=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_bn=False,                          # use batch norm.
                 **kwargs):

        tf.keras.layers.Conv2D.__init__(
            self,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=Model.data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=Model.use_bias,
            #kernel_initializer=Model.kernel_initializer,
            bias_initializer='zeros',
            kernel_regularizer=Model.kernel_regularizer,
            bias_regularizer=None,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            #dynamic=True,
            **kwargs)

        Layer.__init__(self,use_bn,activation,**kwargs)

        #
        Layer.index+= 1
        self.depth = Layer.index



# Dense
class Dense(Layer,tf.keras.layers.Dense):
    def __init__(self,
                 units,
                 activation=None,
                 #use_bias=True
                 #kernel_initializer='glorot_uniform',
                 #bias_initializer='zeros',
                 #kernel_regularizer=None,
                 #bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_bn=False,                          # use batch norm.
                 **kwargs):

        tf.keras.layers.Dense.__init__(
            self,
            units,
            activation=None,
            use_bias=Model.use_bias,
            #kernel_initializer=Model.kernel_initializer,
            bias_initializer='zeros',
            kernel_regularizer=Model.kernel_regularizer,
            bias_regularizer=None,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            #dynamic=True,
            **kwargs)

        Layer.__init__(self,use_bn,activation,**kwargs)

        #
        Layer.index += 1
        self.depth = Layer.index


#
class Add(Layer,tf.keras.layers.Add):
    def __init__(self, use_bn=False, epsilon=0.001, activation=None, **kwargs):
        tf.keras.layers.Add.__init__(self, **kwargs)

        Layer.__init__(self,use_bn=use_bn,epsilon=epsilon,activation=activation,**kwargs)

        #self.act = activation

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


# MaxPolling2D
class MaxPool2D(Layer,tf.keras.layers.MaxPool2D):
    def __init__(self,
                 pool_size=(2,2),
                 strides=None,
                 padding='valid',
                 #data_format=None,
                 **kwargs):

        tf.keras.layers.MaxPool2D.__init__(
            self,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=Model.data_format,
            **kwargs)

        Layer.__init__(self,False,None,**kwargs)


    def build(self, input_shapes):
        tf.keras.layers.MaxPool2D.build(self,input_shapes)

        self.output_shape_fixed_batch = super().compute_output_shape(input_shapes)
        print('maxpool')
        print(self.output_shape_fixed_batch)


    def call(self,inputs):
        #s = super().call(self,inputs)
        #return s
        return tf.keras.layers.MaxPool2D.call(self, inputs)



    def call_set_aside(self,inputs):

        if not Model.f_load_model_done:
            return tf.keras.layers.MaxPool2D.call(self,inputs)

        if self.en_snn:

            #spike_count = Model.model.get_layer(name=self.prev_layer_name).act.get_spike_count()
            spike_count = self.prev_layer.act.get_spike_count()
            output_shape = self.output_shape_fixed_batch

            #print('spike count - {}'.format(tf.reduce_sum(spike_count)))
            return lib_snn.layers.spike_max_pool(inputs,spike_count,output_shape)
        else:
            return tf.keras.layers.MaxPool2D.call(self,inputs)














############################################################
## spike max pool (spike count based gating function)
############################################################
def spike_max_pool(feature_map, spike_count, output_shape):
    #tmp = tf.reshape(spike_count,(1,-1,)+spike_count.numpy().shape[2:])
    tmp = tf.reshape(spike_count,(1,-1,)+tuple(tf.shape(spike_count)[2:]))
    _, arg = tf.nn.max_pool_with_argmax(tmp,(1,2,2,1),(1,2,2,1),padding='SAME')
    #arg = tf.reshape(arg,output_shape)
    conv_f = tf.reshape(feature_map,[-1])
    arg = tf.reshape(arg,[-1])

    p_conv = tf.gather(conv_f, arg)
    p_conv = tf.reshape(p_conv,output_shape)

    #p_conv = tf.convert_to_tensor(conv_f.numpy()[arg],dtype=tf.float32)

    return p_conv


#def spike_max_pool_temporal(feature_map, spike_count, output_shape):


#def spike_max_pool(feature_map, spike_count, output_shape):
#
#    max_pool = {
#        'TEMPORAL': spike_max_pool_rate
#    }.get(self.neural_coding, spike_max_pool_rate)
#
#    p_conv = max_pool(feature_map, spike_count, output_shape)
#
##    return p_conv
