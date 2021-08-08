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
    index=0
    def __init__(self,use_bn,activation):
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

        # batch norm.
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization()
        else:
            self.bn = None

        # activation, neuron
        # DNN mode
        self.act_dnn = activation
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


            self.act_snn = lib_snn.layers.Neuron(self.output_shape_fixed_batch,self.conf,\
                                                self.n_type,self.conf.neural_coding,self.depth,self.name)

        # setup activation
        if self.en_snn:
            self.act = self.act_snn
        else:
            self.act = self.act_dnn

        #
        self.built = True


    #
    def call(self,input,training):
        #print('layer call')
        s = super().call(input)

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
        Layer.__init__(self,False,None)

        #
        Layer.index+= 1
        self.depth = Layer.index

    def build(self):
        print('build input')
        #super().build(input_shapes)

        assert False

    def call(self):
        print('call input')

        assert False





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
            kernel_initializer=Model.kernel_initializer,
            bias_initializer='zeros',
            kernel_regularizer=Model.kernel_regularizer,
            bias_regularizer=None,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            #dynamic=True,
            **kwargs)

        Layer.__init__(self,use_bn,activation)

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
            kernel_initializer=Model.kernel_initializer,
            bias_initializer='zeros',
            kernel_regularizer=Model.kernel_regularizer,
            bias_regularizer=None,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            #dynamic=True,
            **kwargs)

        Layer.__init__(self,use_bn,activation)

        #
        Layer.index += 1
        self.depth = Layer.index


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

        Layer.__init__(self,False,None)


    def build(self, input_shapes):
        tf.keras.layers.MaxPool2D.build(self,input_shapes)

        self.output_shape_fixed_batch = super().compute_output_shape(input_shapes)
        print('maxpool')
        print(self.output_shape_fixed_batch)


    def call(self,inputs):

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







#
#class Neuron(tf.layers.Layer):
class Neuron(tf.keras.layers.Layer):
    def __init__(self,dim,conf,n_type,neural_coding,depth=0,name='',**kwargs):

    #def __init__(self, dim, n_type, fan_in, conf, neural_coding, depth=0, name='', **kwargs):
        #super(Neuron, self).__init__(name="")
        super(Neuron, self).__init__(name=name)

        self.dim = dim
        self.dim_one_batch = [1,]+dim[1:]

        self.n_type = n_type
        #self.fan_in = fan_in

        self.conf = conf

        self.neural_coding=neural_coding

        #
        self.vmem = None


        # stat for weighted spike
        #self.en_stat_ws = True
        self.en_stat_ws = False

        #self.zeros = np.zeros(self.dim,dtype=np.float32)
        #self.zeros = tf.constant(0.0,shape=self.dim,dtype=tf.float32)
        #self.fires = np.full(self.dim, self.conf.n_in_init_vth,dtype=np.float32)

        self.zeros = tf.zeros(self.dim,dtype=tf.float32)
        self.fires = tf.constant(self.conf.n_in_init_vth,shape=self.dim,dtype=tf.float32)

        self.depth = depth

        #if self.conf.f_record_first_spike_time:
            #self.init_first_spike_time = -1.0
        #self.init_first_spike_time = self.conf.time_fire_duration*self.conf.init_first_spike_time_n
        self.init_first_spike_time = 100000



        if self.conf.neural_coding=='TEMPORAL':
            self.time_const_init_fire = self.conf.tc
            self.time_const_init_integ = self.conf.tc

            self.time_delay_init_fire = 0.0
            self.time_delay_init_integ = 0.0


            if self.conf.f_tc_based:
                self.time_start_integ_init = (self.depth-1)*self.conf.time_fire_start
                self.time_start_fire_init = (self.depth)*self.conf.time_fire_start

                self.time_end_integ_init = self.time_start_integ_init + self.conf.time_fire_duration
                self.time_end_fire_init = self.time_start_fire_init + self.conf.time_fire_duration
            else:
                self.time_start_integ_init = (self.depth-1)*self.conf.time_fire_start
                self.time_start_fire_init = (self.depth)*self.conf.time_fire_start

                self.time_end_integ_init = self.time_start_integ_init + self.conf.time_fire_duration
                self.time_end_fire_init = self.time_start_fire_init + self.conf.time_fire_duration


        #self.spike_counter = tf.Variable(name="spike_counter",dtype=tf.float32,initial_value=tf.zeros(self.dim,dtype=tf.float32),trainable=False)
        #self.spike_counter_int = tf.Variable(name="spike_counter_int",dtype=tf.float32,initial_value=tf.zeros(self.dim,dtype=tf.float32),trainable=False)
        #self.f_fire = tf.Variable(name='f_fire', dtype=tf.bool, initial_value=tf.constant(False,dtype=tf.bool,shape=self.dim),trainable=False)

        #
        #self.vmem = tf.Variable(shape=self.dim,dtype=tf.float32,initial_value=tf.constant(self.conf.n_init_vinit,shape=self.dim),trainable=False,name='vmem')

    def build(self, _):
        print('neuron build')
        super().build(_)

        if self.n_type == 'IN':
            init_vth = self.conf.n_in_init_vth
        else:
            init_vth = self.conf.n_init_vth

        self.vth_init = self.add_variable("vth_init",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(init_vth),trainable=False)
        #self.vth_init = tfe.Variable(init_vth)
        self.vth = self.add_variable("vth",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(init_vth),trainable=False)

        self.vmem = self.add_variable("vmem",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.conf.n_init_vinit),trainable=False)
        #self.vmem = tf.Variable("vmem",shape=self.dim,dtype=tf.float32,initial_value=tf.constant(self.conf.n_init_vinit),trainable=False)
        #self.vmem = tf.Variable(shape=self.dim,dtype=tf.float32,initial_value=tf.constant(self.conf.n_init_vinit,shape=self.dim),trainable=False,name='vmem')

        self.out = self.add_variable("out",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
        #self.out = self.add_variable("out",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=True)

        if self.conf.f_isi:
            self.last_spike_time = self.add_variable("last_spike_time",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
            self.isi = self.add_variable("isi",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        self.spike_counter_int = self.add_variable("spike_counter_int",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
        self.spike_counter = self.add_variable("spike_counter",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        self.f_fire = self.add_variable("f_fire",shape=self.dim,dtype=tf.bool,trainable=False)

        if self.conf.f_tot_psp:
            self.tot_psp = self.add_variable("tot_psp",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        if self.conf.f_refractory:
            self.refractory = self.add_variable("refractory",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
            self.t_set_refractory= self.add_variable("t_set_refractory",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(-1.0),trainable=False)

        #self.depth = self.add_variable("depth",shape=self.dim,dtype=tf.int32,initializer=tf.zeros_initializer,trainable=False)

        #if self.conf.neural_coding=='TEMPORAL':
        #if self.conf.f_record_first_spike_time:
        #    self.first_spike_time=self.add_variable("first_spike_time",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.init_first_spike_time),trainable=False)


        # stat for weighted spike
        if self.en_stat_ws:
            self.stat_ws = self.add_variable("stat_ws",shape=self.conf.p_ws,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        # relative spike time of each layer
        self.first_spike_time=self.add_variable("first_spike_time",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.init_first_spike_time),trainable=False)
        if self.conf.neural_coding=='TEMPORAL':
            #if self.conf.f_record_first_spike_time:
            #    self.first_spike_time=self.add_variable("first_spike_time",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.init_first_spike_time),trainable=False)



            #self.time_const=self.add_variable("time_const",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.conf.tc),trainable=False)

            # TODO: old - scalar version
            self.time_const_integ=self.add_variable("time_const_integ",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_const_init_integ),trainable=False)
            self.time_const_fire=self.add_variable("time_const_fire",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_const_init_fire),trainable=False)
            self.time_delay_integ=self.add_variable("time_delay_integ",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_delay_init_integ),trainable=False)
            self.time_delay_fire=self.add_variable("time_delay_fire",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_delay_init_fire),trainable=False)

            self.time_start_integ=self.add_variable("time_start_integ",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_start_integ_init),trainable=False)
            self.time_end_integ=self.add_variable("time_end_integ",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_end_integ_init),trainable=False)
            self.time_start_fire=self.add_variable("time_start_fire",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_start_fire_init),trainable=False)
            self.time_end_fire=self.add_variable("time_end_fire",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_end_fire_init),trainable=False)

#            self.time_const_integ=self.add_variable("time_const_integ",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_const_init_integ),trainable=False)
#            self.time_const_fire=self.add_variable("time_const_fire",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_const_init_fire),trainable=False)
#            self.time_delay_integ=self.add_variable("time_delay_integ",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_delay_init_integ),trainable=False)
#            self.time_delay_fire=self.add_variable("time_delay_fire",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_delay_init_fire),trainable=False)
#
#            self.time_start_integ=self.add_variable("time_start_integ",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_start_integ_init),trainable=False)
#            self.time_end_integ=self.add_variable("time_end_integ",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_end_integ_init),trainable=False)
#            self.time_start_fire=self.add_variable("time_start_fire",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_start_fire_init),trainable=False)
#            self.time_end_fire=self.add_variable("time_end_fire",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_end_fire_init),trainable=False)


            print_loss=True

            if self.conf.f_train_tk and print_loss:
                self.loss_prec=self.add_variable("loss_prec",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
                self.loss_min=self.add_variable("loss_min",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
                self.loss_max=self.add_variable("loss_max",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)


#        if self.conf.noise_en and (self.conf.noise_type=="JIT"):
#            #self.jit_max = tf.floor(4*self.conf.noise_pr)       # 4 sigma
#            self.jit_max = int(tf.floor(4*self.conf.noise_pr))       # 4 sigma
#            shape = tensor_shape.TensorShape(self.vmem.shape)
#            #shape = tensor_shape.TensorShape(self.out.shape+[int(self.jit_max),])
#            #shape = tensor_shape.TensorShape([int(self.jit_max),]+self.out.shape)
#
#            #self.jit_q = self.add_variable("jit_q",shape=shape,dtype=tf.bool,initializer=tf.constant_initializer(False),trainable=False)
#            self.jit_q = self.add_variable("jit_q",shape=shape,dtype=tf.int32,initializer=tf.constant_initializer(False),trainable=False)

        self.built = True

    def call(self,inputs,t):
        
        #print('neuron call')

        #self.reset_each_time()

        # reshape
        #vth = tf.reshape(self.vth,self.dim)
        #vmem = tf.reshape(self.vmem,self.dim)
        #out = tf.reshape(self.out,self.dim)
        #inputs = tf.reshape(inputs,self.dim)

        #if inputs.shape[0] != 1:
        #    print('not supported batch mode in SNN test mode')
        #    sys.exit(1)
        #else:
        #    inputs = tf.reshape(inputs,self.vmem.shape)
        #inputs = tf.reshape(inputs,[-1]+self.vmem.shape[1:])

        # run_fwd
        run_type = {
            'IN': self.run_type_in,
            'IF': self.run_type_if,
            'LIF': self.run_type_lif,
            'OUT': self.run_type_out
        }[self.n_type](inputs,t)

        #out_ret = tf.reshape(self.out,self.dim)
        out_ret = self.out

        return out_ret

    # reset - time step
    def reset_each_time(self):
        self.reset_out()

    # reset - sample
    def reset(self):
        #print('reset neuron')
        self.reset_vmem()
        #self.reset_out()
        self.reset_spike_count()
        self.reset_vth()

        if self.conf.f_tot_psp:
            self.reset_tot_psp()
        if self.conf.f_isi:
            self.last_spike_time = tf.zeros(self.last_spike_time.shape)
            self.isi = tf.zeros(self.isi.shape)
        if self.conf.f_refractory:
            self.refractory = tf.zeros(self.refractory.shape)
            self.t_set_refractory = tf.constant(-1.0,dtype=tf.float32,shape=self.refractory.shape)

        if self.conf.f_record_first_spike_time:
            self.reset_first_spike_time()

    def reset_spike_count(self):
        #self.spike_counter = tf.zeros(self.dim)
        #self.spike_counter_int = tf.zeros(self.dim)

        self.spike_counter.assign(tf.zeros(self.dim,dtype=tf.float32))
        self.spike_counter_int.assign(tf.zeros(self.dim,dtype=tf.float32))

        #self.spike_counter.assign(self.zeros)
        #self.spike_counter.assign(tf.zeros(self.dim))
        #self.spike_counter_int = tf.zeros(self.out.shape)
        #self.spike_counter_int.assign(tf.zeros(self.dim))

    #
    def reset_vmem(self):
        #assert False
        #self.vmem = tf.constant(self.conf.n_init_vinit,tf.float32,self.vmem.shape)
        self.vmem.assign(tf.constant(self.conf.n_init_vinit,tf.float32,self.vmem.shape))

        #print(type(self.vmem))
        #assert False

    #
    def reset_tot_psp(self):
        self.tot_psp = tf.zeros(tf.shape(self.tot_psp))

    #
    def reset_out(self):
        self.out = tf.zeros(self.out.shape)

    def reset_vth(self):
        self.vth = self.vth_init

    #
    def reset_first_spike_time(self):
        self.first_spike_time=tf.constant(self.init_first_spike_time,shape=self.first_spike_time.shape,dtype=tf.float32)

    ##
    def set_vth_temporal_kernel(self,t):
        # exponential decay
        #self.vth = tf.constant(tf.exp(-float(t)/self.conf.tc),tf.float32,self.out.shape)
        #self.vth = tf.constant(tf.exp(-t/self.conf.tc),tf.float32,self.out.shape)
        time = tf.subtract(t,self.time_delay_fire)

        if self.conf.f_qvth:
            time = tf.add(time,0.5)

        #print(self.vth.shape)
        #print(self.time_const_fire.shape)
        #self.vth = tf.constant(tf.exp(tf.divide(-time,self.time_const_fire)),tf.float32,self.out.shape)

        # TODO: check
        self.vth = tf.exp(tf.divide(-time,self.time_const_fire))
        #
        #self.vth = tf.multiply(self.conf.n_init_vth,tf.exp(tf.divide(-time,self.time_const_fire)))

        # polynomial
        #self.vth = tf.constant(tf.add(-tf.pow(t/self.conf.tc,2),1.0),tf.float32,self.out.shape)


    ##
    def input_spike_real(self,inputs,t):
        # TODO: check it
        if self.conf.neural_coding=="WEIGHTED_SPIKE":
            self.out=tf.truediv(inputs,self.conf.p_ws)
        elif self.conf.neural_coding=="TEMPORAL":
            if t==0:
                self.out=inputs
            else:
                self.out = tf.zeros(self.out.shape)
        else:
            self.out=inputs

#        else:
#            self.out=inputs
#            #assert False
        #self.out=inputs

    def input_spike_poission(self,inputs,t):
        # Poission input
        vrand = tf.random_uniform(self.vmem.shape,minval=0.0,maxval=1.0,dtype=tf.float32)

        self.f_fire = inputs>=vrand

        self.out = tf.where(self.f_fire,self.fires,self.zeros)
        #self.out = tf.where(self.f_fire,tf.constant(1.0,tf.float32,self.out.shape),tf.zeros(self.vmem.shape))

    def input_spike_weighted_spike(self,inputs,t):
        assert False
        # weighted synpase input
        t_mod = (int)(t%8)
        if t_mod == 0:
            self.vmem = inputs
            self.vth = tf.constant(0.5,tf.float32,self.vth.shape)
        else:
            self.vth = tf.multiply(self.vth,0.5)

        f_fire = self.vmem >= self.vth

        self.out = tf.where(f_fire,self.vth,tf.zeros(self.out.shape))

        self.vmem = tf.subtract(self.vmem,self.out)

    def input_spike_burst(self,inputs,t):
        assert False
        if t == 0:
            self.vmem = inputs

        f_fire = self.vmem >= self.vth

        self.out = tf.where(f_fire,self.vth,tf.zeros(self.out.shape))

        self.vmem = tf.subtract(self.vmem,self.out)

        self.vth = tf.where(f_fire,self.vth*2.0,self.vth_init)

        # repeat input
        if tf.equal(tf.reduce_max(self.out),0.0):
            self.vmem = inputs

    def input_spike_temporal(self,inputs,t):

        if t == 0:
            self.vmem = inputs

        #kernel = self.temporal_kernel(t)
        #self.vth = tf.constant(kernel,tf.float32,self.out.shape)
        #self.vth = tf.constant(tf.exp(-t/self.conf.tc),tf.float32,self.out.shape)
        #self.vth = self.temporal_kernel(t)
        self.set_vth_temporal_kernel(t)
        #print(self.vth[0,1,1,1])

        #print("input_spike_temporal: depth: "+str(self.depth)+", t_glb: "+str(t)+", t_loc: "+str(t)+", kernel: "+str(self.vth[0,0,0,0].numpy()))


        if self.conf.f_refractory:
            self.f_fire = (self.vmem >= self.vth) & \
                          tf.equal(self.refractory,tf.constant(0.0,tf.float32,self.refractory.shape))
                        #(self.vth >= tf.constant(10**(-5),tf.float32,self.vth.shape) ) & \

        else:
            self.f_fire = (self.vmem >= self.vth) & (self.vth >= 10**(-5))

        #print(f_fire)
        #print(self.vth >= 10**(-1))

        # reset by subtraction
        #self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        #self.vmem = tf.subtract(self.vmem,self.out)

        # reset by zero
        self.out = tf.where(self.f_fire,tf.ones(self.out.shape),tf.zeros(self.out.shape))
        self.vmem = tf.where(self.f_fire,tf.zeros(self.out.shape),self.vmem)

        if self.conf.f_refractory:
            #self.cal_refractory_temporal(self.f_fire)
            self.cal_refractory_temporal(t)


        #self.vth = tf.where(f_fire,self.vth*2.0,self.vth_init)

        # repeat input
        #print(tf.reduce_max(self.out))
        #if tf.equal(tf.reduce_max(self.out),0.0):
        #    self.vmem = inputs

    def spike_dummy_input(self,inputs,t):
        self.out = tf.zeros(self.out.shape)
        self.f_fire = tf.constant(False,tf.bool,self.f_fire.shape)

    def spike_dummy_fire(self,t):
        self.out = tf.zeros(self.out.shape)
        self.f_fire = tf.constant(False,tf.bool,self.f_fire.shape)

    # (min,max) = (0.0,1.0)
    def input_spike_gen(self,inputs,t):

        if self.conf.f_tc_based:
            f_run_temporal = (t < self.conf.n_tau_fire_duration*self.time_const_fire)
        else:
            f_run_temporal = (t < self.conf.time_fire_duration)


        input_spike_mode ={
            'REAL': self.input_spike_real,
            'POISSON': self.input_spike_poission,
            'WEIGHTED_SPIKE': self.input_spike_weighted_spike,
            'BURST': self.input_spike_burst,
            'TEMPORAL': self.input_spike_temporal if f_run_temporal else self.spike_dummy_input
        }

        input_spike_mode[self.conf.input_spike_mode](inputs,t)


    #
    def leak(self):
        assert False
        self.vmem = tf.multiply(self.vmem,0.7)

    #
    def cal_isi(self, f_fire, t):
        f_1st_spike = self.last_spike_time == 0.0
        f_isi_update = np.logical_and(f_fire, np.logical_not(f_1st_spike))

        self.isi = tf.where(f_isi_update,tf.constant(t,tf.float32,self.isi.shape)-self.last_spike_time,np.zeros(self.isi.shape))

        self.last_spike_time = tf.where(f_fire,tf.constant(t,tf.float32,self.last_spike_time.shape),self.last_spike_time)


    #
    def integration(self,inputs,t):

        if self.conf.neural_coding=="TEMPORAL":
            t_int_s = self.time_start_integ_init
            t_int_e = self.time_end_integ_init

            f_run_int_temporal = (t >= t_int_s and t < t_int_e) or (t==0)   # t==0 : for bias integration
        else:
            f_run_int_temporal = False

        #print(type(self.vmem))
        #assert False

        # intergation
        {
            'TEMPORAL': self.integration_temporal if f_run_int_temporal else lambda inputs, t : None,
            'NON_LINEAR': self.integration_non_lin
        }.get(self.neural_coding, self.integration_default) (inputs,t)


        ########################################
        # common
        ########################################

        #
        if self.conf.f_positive_vmem:
            self.vmem = tf.maximum(self.vmem,tf.constant(0.0,tf.float32,self.vmem.shape))

        #
        if self.conf.f_tot_psp:
            self.tot_psp = tf.add(self.tot_psp, inputs)

        # debug
        #if self.depth==3:
        #    print(self.vmem.numpy())


    @tf.function
    def integration_default(self,inputs,t):
        self.vmem.assign(tf.add(self.vmem,inputs))

    #
    def integration_temporal(self,inputs,t):
        if(t==0) :
            time = 0
        else :
            time = self.relative_time_integ(t)

            time = time - self.time_delay_integ
        #time = tf.zeros(self.vmem.shape)

        if self.conf.noise_en:
            if self.conf.noise_type=="JIT" or self.conf.noise_type=="JIT-A":
                rand = tf.random.normal(shape=inputs.shape,mean=0.0,stddev=self.conf.noise_pr)

                if self.conf.noise_type=="JIT-A":
                    rand = tf.abs(rand)
                    rand = tf.floor(rand)
                else:
                    f_positive = tf.greater_equal(rand,tf.zeros(shape=rand.shape))
                    rand = tf.where(f_positive, tf.floor(rand), tf.math.ceil(rand))

                time = tf.add(time,rand)


        #receptive_kernel = tf.constant(tf.exp(-time/self.conf.tc),tf.float32,self.vmem.shape)

        #receptive_kernel = tf.constant(tf.exp(-time/self.time_const_integ),tf.float32,self.vmem.shape)
        receptive_kernel = tf.exp(tf.divide(-time,self.time_const_integ))

        #print("integration_temporal: depth: "+str(self.depth)+", t_glb: "+str(t)+", t_loc: "+str(time)+", kernel: "+str(receptive_kernel[0,0,0,0].numpy()))

        #
        #print('test')
        #print(inputs.shape)
        #print(receptive_kernel.shape)

        if self.depth==1:
            if self.conf.input_spike_mode=='REAL':
                psp = inputs
            else:
                psp = tf.multiply(inputs,receptive_kernel)
        else:
            psp = tf.multiply(inputs,receptive_kernel)
        #print(inputs[0,0,0,1])
        #print(psp[0,0,0,1])


        #addr=0,10,10,0
        #print("int: glb {}: loc {} - in {:0.3f}, kernel {:.03f}, psp {:0.3f}".format(t, time,inputs[addr],receptive_kernel[addr],psp[addr]))

        #
        #self.vmem = tf.add(self.vmem,inputs)
        self.vmem = tf.add(self.vmem,psp)


    def integration_non_lin(self,inputs,t):

        #
        #alpha=tf.constant(1.0,tf.float32,self.vmem.shape)
        #beta=tf.constant(0.1,tf.float32,self.vmem.shape)
        #gamma=tf.constant(1.0,tf.float32,self.vmem.shape)
        #eps=tf.constant(0.01,tf.float32,self.vmem.shape)
        #non_lin=tf.multiply(alpha,tf.log(tf.add(beta,tf.divide(gamma,tf.abs(tf.add(self.vmem,eps))))))

        #
        #alpha=tf.constant(0.0459,tf.float32,self.vmem.shape)
        #beta=tf.constant(-7.002,tf.float32,self.vmem.shape)

        #
        alpha=tf.constant(0.3,tf.float32,self.vmem.shape)
        beta=tf.constant(-2.30259,tf.float32,self.vmem.shape)


        #alpha=tf.constant(0.459,tf.float32,self.vmem.shape)
        #alpha=tf.constant(1.5,tf.float32,self.vmem.shape)

        non_lin=tf.multiply(alpha,tf.exp(tf.multiply(beta,tf.abs(self.vmem))))

        psp = tf.multiply(inputs,non_lin)


        dim = tf.size(tf.shape(inputs))
        if tf.equal(dim, tf.constant(4)):
            idx=0,0,0,0
        elif tf.equal(dim, tf.constant(3)):
            idx=0,0,0
        elif tf.equal(dim, tf.constant(2)):
            idx=0,0

        #print("vmem: {:g}, non_lin: {:g}, inputs: {:g}, psp: {:g}".format(self.vmem[idx],non_lin[idx],inputs[idx],psp[idx]))

        self.vmem = tf.add(self.vmem,psp)

    # TODO: move
    ############################################################
    ## noise function
    ############################################################
    def noise_jit(self,t):
        rand = tf.random.normal(self.out.shape,mean=0.0,stddev=self.conf.noise_pr)
        time_jit= tf.cast(tf.floor(tf.abs(rand)),dtype=tf.int32)

        f_jit_del = tf.not_equal(time_jit,tf.zeros(self.out.shape,dtype=tf.int32))
        t_mod = t%self.jit_max
        pow_t_mod = tf.pow(2,t_mod)
        one_or_zero = tf.cast(tf.math.mod(tf.truediv(self.jit_q,pow_t_mod,2),2),tf.float32)
        f_jit_ins = tf.equal(one_or_zero,tf.ones(self.out.shape))
        self.jit_q = tf.where(f_jit_ins,tf.subtract(self.jit_q,pow_t_mod),self.jit_q)


        f_fire_and_jit_del = tf.math.logical_and(self.f_fire,f_jit_del)

        jit_t = tf.math.floormod(tf.add(tf.constant(t,shape=time_jit.shape,dtype=tf.int32),time_jit),self.jit_max)

        jit_q_t = tf.add(self.jit_q,tf.pow(2,jit_t))

        self.jit_q = tf.where(f_fire_and_jit_del,jit_q_t,self.jit_q)

        # jitter - del
        out_tmp = tf.where(f_fire_and_jit_del,tf.zeros(self.out.shape),self.out)

        # jitter - insert
        # reset by subtraction
        self.out = tf.where(f_jit_ins,self.vth,out_tmp)


    def noise_del(self):
        #print(self.out)
        #print(tf.reduce_sum(self.out))

        rand = tf.random.uniform(self.out.shape,minval=0.0,maxval=1.0)

        f_noise = tf.less(rand,tf.constant(self.conf.noise_pr,shape=self.out.shape))

        f_fire_and_noise = tf.math.logical_and(self.f_fire,f_noise)

        #out_noise = tf.where(f_fire_and_noise,tf.zeros(self.out.shape),self.out)

        #if self.neural_coding=="TEMPORAL":
            #self.out = tf.where(f_fire_and_noise,tf.zeros(self.out.shape),self.out)
            #self.f_fire = tf.where(f_fire_and_noise,tf.constant(False,tf.bool,self.f_fire.shape),self.f_fire)
        #else:
        self.out = tf.where(f_fire_and_noise,tf.zeros(self.out.shape),self.out)
        self.f_fire = tf.where(f_fire_and_noise,tf.constant(False,tf.bool,self.f_fire.shape),self.f_fire)

        #print("out_noise: {}".format(tf.reduce_sum(out_noise)))

        #assert False



    ############################################################
    ## fire function
    ############################################################
    def fire(self,t):

        #
        # for TTFS coding
        #
        if self.conf.neural_coding=="TEMPORAL":
            t_fire_s = self.time_start_fire_init
            t_fire_e = self.time_end_fire_init
            f_run_fire = t >= t_fire_s and t < t_fire_e
        else:
            f_run_fire = False


        {
            'RATE': self.fire_rate,
            'WEIGHTED_SPIKE': self.fire_weighted_spike,
            'BURST': self.fire_burst,
            #'TEMPORAL': self.fire_temporal if t >= self.depth*self.conf.time_window and t < (self.depth+1)*self.conf.time_window else self.spike_dummy_fire
            'TEMPORAL': self.fire_temporal if f_run_fire else self.spike_dummy_fire,
            'NON_LINEAR': self.fire_non_lin
        }.get(self.neural_coding, self.fire_rate) (t)

        #
        if self.conf.noise_en:

            ## noise - jitter
            #if self.conf.noise_type=='JIT':
            #    self.noise_jit(t)

            # TODO: modify it to switch style
            # noise - DEL spikes
            if self.conf.noise_type == 'DEL':
                self.noise_del()

    #
    # TODO
    #@tf.function
    def fire_condition_check(self):
        return tf.math.greater_equal(self.vmem,self.vth)


    #
    def fire_rate(self,t):
        #self.f_fire = self.vmem >= self.vth
        self.f_fire = self.fire_condition_check()
        self.out = tf.where(self.f_fire,self.fires,self.zeros)

        # reset
        # vmem -> vrest

        # reset by subtraction
        #self.vmem = tf.where(f_fire,self.vmem-self.vth,self.vmem)
        self.vmem.assign(tf.subtract(self.vmem,self.out))

        # reset to zero
        #self.vmem = tf.where(f_fire,tf.constant(self.conf.n_init_vreset,tf.float32,self.vmem.shape,self.vmem)

        if self.conf.f_isi:
            self.cal_isi(self.f_fire,t)


    def fire_weighted_spike(self,t):
        # weighted synpase input
        t_mod = (int)(t%self.conf.p_ws)
        if t_mod == 0:
            # TODO: check
            #self.vth = tf.constant(0.5,tf.float32,self.vth.shape)
            self.vth = tf.constant(self.conf.n_init_vth,tf.float32,self.vth.shape)
        else:
            self.vth = tf.multiply(self.vth,0.5)

        if self.conf.f_refractory:
            #self.f_fire = np.logical_and(self.vmem >= self.vth,np.equal(self.refractory,0.0))
            # TODO: check
            #self.f_fire = tf.logical_and(self.vmem >= self.vth, tf.equal(self.refractory,0.0))
            assert False, 'modify refractory'
        else:
            self.f_fire = self.vmem >= self.vth


        if self.conf.f_refractory:
            print('fire_weighted_spike, refractory: not implemented yet')

        self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))

        self.vmem = tf.subtract(self.vmem,self.out)

        # noise - jit
        if self.conf.noise_en:
            if self.conf.noise_type=="JIT" or self.conf.noise_type=="JIT-A":
                rand = tf.random.normal(shape=self.out.shape,mean=0.0,stddev=self.conf.noise_pr)

                if self.conf.noise_type=="JIT-A":
                    rand = tf.abs(rand)
                    rand = tf.floor(rand)
                else:
                    f_positive = tf.greater_equal(rand,tf.zeros(shape=rand.shape))
                    rand = tf.where(f_positive, tf.floor(rand), tf.math.ceil(rand))

                pow_rand = tf.pow(2.0,-rand)
                out_jit = tf.multiply(self.out,pow_rand)
                self.out = tf.where(self.f_fire,out_jit,self.out)

        # stat for weighted spike
        if self.en_stat_ws:
            count = tf.cast(tf.math.count_nonzero(self.out),tf.float32)
            #print(count)
           # tf.tensor_scatter_nd_add(self.stat_ws,[[t_mod]],[count])
            #self.stat_ws.scatter_add(tf.IndexedSlices(10.0,1))
            self.stat_ws.scatter_add(tf.IndexedSlices(count,t_mod))
            #tf.tensor_scatter_nd_add(self.stat_ws,[[1]],[10])

            print(self.stat_ws)
            #plt.hist(self.stat_ws.numpy())
            #plt.show()

        if self.conf.f_isi:
            self.cal_isi(self.f_fire,t)


    #
    def fire_burst(self,t):
        if self.conf.f_refractory:
            #self.f_fire = np.logical_and(self.vmem >= self.vth,np.equal(self.refractory,0.0))
            # TODO: check
            #self.f_fire = tf.logical_and(self.vmem >= self.vth, np.equal(self.refractory,0.0))
            assert False, 'modify refractory'

        else:
            self.f_fire = self.vmem >= self.vth


        #print(f_fire)
        #print(np.equal(self.refractory,0.0))
        #print(self.refractory)


        # reset by subtraction
        self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        self.vmem = tf.subtract(self.vmem,self.out)

        if self.conf.f_refractory:
            self.cal_refractory(self.f_fire)


        # exp increasing order
        self.vth = tf.where(self.f_fire,self.vth*2.0,self.vth_init)
        #self.vth = tf.where(f_fire,self.vth*1.5,self.vth_init)
        # exp decreasing order
        #self.vth = tf.where(f_fire,self.vth*0.5,self.vth_init)
        #self.vth = tf.where(f_fire,self.vth*0.9,self.vth_init)


        if self.conf.noise_en:
            if self.conf.noise_type=="JIT" or self.conf.noise_type=="JIT-A":
                rand = tf.random.normal(shape=self.out.shape,mean=0.0,stddev=self.conf.noise_pr)

                if self.conf.noise_type=="JIT-A":
                    rand = tf.abs(rand)
                    rand = tf.floor(rand)
                else:
                    f_positive = tf.greater_equal(rand,tf.zeros(shape=rand.shape))
                    rand = tf.where(f_positive, tf.floor(rand), tf.math.ceil(rand))

                pow_rand = tf.pow(2.0,rand)
                out_jit = tf.multiply(self.out,pow_rand)
                self.out = tf.where(self.f_fire,out_jit,self.out)

        if self.conf.f_isi:
            self.cal_isi(self.f_fire,t)

    #
    def fire_temporal(self,t):

        time = self.relative_time_fire(t)

        #
        # encoding
        # dynamic threshold (vth)
        #
        self.set_vth_temporal_kernel(time)

        if self.conf.f_refractory:

            # TODO: check - case 1 or 2
            # case 1
            #self.f_fire = (self.vmem >= self.vth) & \
            #                tf.equal(self.refractory,tf.zeros(self.refractory.shape))
            self.f_fire = (self.vmem >= self.vth) & \
                              tf.equal(self.refractory,tf.constant(0.0,tf.float32,self.refractory.shape))

            # case 2
            ##self.f_fire = (self.vmem >= self.vth) & \
            ##                tf.equal(self.refractory,tf.zeros(self.refractory.shape))
            #f_fire = tf.greater_equal(self.vmem,self.vth)
            ##f_refractory = tf.greater_equal(tf.constant(t,tf.float32),self.refractory)
            ##f_refractory = tf.greater_equal(tf.constant(t,tf.float32),self.refractory)
            #f_refractory = tf.greater(tf.constant(t,tf.float32),self.refractory)
            ##self.f_fire = (self.vmem >= self.vth) & \
            ##                  tf.equal(self.refractory,tf.constant(0.0,tf.float32,self.refractory.shape))
            #self.f_fire = tf.logical_and(f_fire,f_refractory)

        else:
            self.f_fire = (self.vmem >= self.vth) & (self.vth >= 10**(-5))

        #
        # reset
        #

        # reset by zero
        self.out = tf.where(self.f_fire,tf.ones(self.out.shape),tf.zeros(self.out.shape))
        self.vmem = tf.where(self.f_fire,tf.zeros(self.out.shape),self.vmem)

        # reset by subtraction
        #self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        #self.vmem = tf.subtract(self.vmem,self.out)


        #addr=0,10,10,0
        #print("fire: glb {}: loc {} - vth {:0.3f}, kernel {:.03f}, out {:0.3f}".format(t, time,self.vmem[addr],self.vth[addr],self.out[addr]))

        if self.conf.f_refractory:
            #self.cal_refractory_temporal(self.f_fire)
            self.cal_refractory_temporal(t)


    #
    def fire_non_lin(self,t):

        #
        self.f_fire = self.vmem >= self.vth
        self.out = tf.where(self.f_fire,self.fires,self.zeros)

        #
        #rand = tf.random_uniform(shape=self.vmem.shape,minval=0.90*self.vth,maxval=self.vth)
        #self.f_fire = tf.logical_or(self.vmem >= self.vth,self.vmem >= rand)

        #self.out = tf.where(self.f_fire,self.fires,self.zeros)


        # reset by subtract
        #self.vmem = tf.subtract(self.vmem,self.out)

        # reset to zero
        #self.out = tf.where(self.f_fire,tf.ones(self.out.shape),tf.zeros(self.out.shape))
        self.vmem = tf.where(self.f_fire,tf.zeros(self.out.shape),self.vmem)


        if self.conf.f_isi:
            self.cal_isi(self.f_fire,t)



    def fire_type_out(self, t):
        f_fire = self.vmem >= self.vth

        self.vmem = tf.where(f_fire,self.vmem-self.vth,self.vmem)

        self.out = tf.where(f_fire,tf.constant(1.0,tf.float32,self.out.shape),tf.zeros(self.vmem.shape))


        #self.isi = tf.where(f_fire,tf.constant(t,tf.float32,self.isi.shape)-self.last_spike_time,self.isi)
        #self.last_spike_time = tf.where(f_fire,tf.constant(t,tf.float32,self.last_spike_time.shape),self.last_spike_time)



    ############################################################
    ##
    ############################################################

    def cal_refractory(self,f_fire):
        f_refractory_update = np.logical_and(np.not_equal(self.vth-self.vth_init,0.0),np.logical_not(f_fire))
        refractory_update = 2.0*np.log2(self.vth/self.vth_init)

        self.refractory = tf.maximum(self.refractory-1,tf.constant(0.0,tf.float32,self.refractory.shape))

        self.refractory = tf.where(f_refractory_update,refractory_update,self.refractory)

        #print(tf.reduce_max(self.vth))
        #print(self.vth_init)
        #print(np.not_equal(self.vth,self.vth_init))
        #print(np.logical_not(f_fire))
        #print(f_refractory_update)
        #self.refractory = tf.where(f_fire,tf.constant(0.0,tf.float32,self.refractory.shape),self.refractory)
        #print(self.refractory)

        #print(tf.reduce_max(np.log2(self.vth/self.vth_init)))

    # TODO: refractory
    def cal_refractory_temporal_original(self,f_fire):
        #self.refractory = tf.where(f_fire,tf.constant(self.conf.time_step,tf.float32,self.refractory.shape),self.refractory)
        self.refractory = tf.where(f_fire,tf.constant(10000.0,tf.float32,self.refractory.shape),self.refractory)

    def cal_refractory_temporal(self, t):
        if self.conf.noise_robust_en:

            inf_t = 10000.0

            t_b = self.conf.noise_robust_spike_num
            #t_b = 2

            f_init_refractory = tf.equal(self.t_set_refractory,tf.constant(-1.0,dtype=tf.float32,shape=self.t_set_refractory.shape))



            f_first_spike = tf.logical_and(f_init_refractory, self.f_fire)

            t = tf.constant(t,dtype=tf.float32,shape=self.refractory.shape)
            t_b = tf.constant(t_b,dtype=tf.float32,shape=self.refractory.shape)
            inf_t = tf.constant(inf_t,dtype=tf.float32,shape=self.refractory.shape)

            self.t_set_refractory = tf.where(f_first_spike,tf.add(t,t_b),self.t_set_refractory)
            #self.t_set_refractory = tf.where(f_first_spike,t,self.t_set_refractory)

            #
            f_add_vmem = self.f_fire
            #add_amount = tf.divide(self.vth,2.0)
            add_amount = self.vth
            self.vmem = tf.where(f_add_vmem,tf.add(self.vmem,add_amount),self.vmem)


            #
            f_set_inf_refractory = tf.equal(t,self.t_set_refractory)
            self.refractory = tf.where(f_set_inf_refractory,inf_t,self.refractory)

            #print(self.t_set_refractory)

            #
            #self.refractory = tf.where(self.f_fire,tf.constant(10000.0,tf.float32,self.refractory.shape),self.refractory)
            #f_set_refractory = tf.logical_and(self.f_fire,tf.equal(self.refractory,tf.zeros(shape=self.refractory.shape)))
            #self.t_set_refractory = tf.where(f_set_refractory,t,self.t_set_refractory)

            #self.refractory = tf.where(self.f_fire,inf_t,self.refractory)

            #print(type(t.numpy()[0,0,0,0]))
            #print(type(self.t_set_refractory.numpy()[0,0,0,0]))
            #print(t.shape)
            #print(self.t_set_refractory.shape)
            #f_refractory = tf.equal(t,self.t_set_refractory)
            #print(self.t_set_refractory[0])
            #self.refractory = tf.where(f_refractory,inf_t,self.refractory)

            #print(t_int)
            #print(self.depth)
            #print(self.t_set_refractory.numpy()[0,0,0,0:10])
            #print(f_init_refractory.numpy()[0,0,0,0:10])
            #print(self.refractory.numpy()[0,0,0,0:10])
        else:
            self.refractory = tf.where(self.f_fire,tf.constant(10000.0,tf.float32,self.refractory.shape),self.refractory)



    #
    def count_spike(self, t):
        {
            'TEMPORAL': self.count_spike_temporal
        }.get(self.neural_coding, self.count_spike_default) (t)


    def count_spike_default(self, t):
        #self.spike_counter_int = tf.where(self.f_fire,self.spike_counter_int+1.0,self.spike_counter_int)
        #self.spike_counter = tf.add(self.spike_counter, self.out)

        self.spike_counter_int.assign(tf.where(self.f_fire,self.spike_counter_int+1.0,self.spike_counter_int))
        self.spike_counter.assign(tf.add(self.spike_counter, self.out))

    def count_spike_temporal(self, t):
        self.spike_counter_int = tf.add(self.spike_counter_int, self.out)
        self.spike_counter = tf.where(self.f_fire, tf.add(self.spike_counter,self.vth),self.spike_counter_int)

        if self.conf.f_record_first_spike_time:
            #spike_time = self.relative_time_fire(t)
            spike_time = t

            self.first_spike_time = tf.where(
                                        self.f_fire,\
                                        tf.constant(spike_time,dtype=tf.float32,\
                                        shape=self.first_spike_time.shape),\
                                        self.first_spike_time)



    ############################################################
    ## run fwd pass
    ############################################################

    def run_type_in(self,inputs,t):
        #print('run_type_in')
        self.input_spike_gen(inputs,t)
        self.count_spike(t)

    #
    def run_type_if(self,inputs,t):
        #print('run_type_if')
        self.integration(inputs,t)
        self.fire(t)
        self.count_spike(t)

    #
    def run_type_lif(self,inputs,t):
        #print('run_type_lif')
        self.leak()
        self.integration(inputs,t)
        self.fire(t)
        self.count_spike(t)

    def run_type_out(self,inputs,t):
        #print("output layer")
        #self.integration(inputs,t)
        ##self.fire_type_out(t)

        if self.conf.snn_output_type in ('SPIKE', 'FIRST_SPIKE_TIME'):
            # in current implementation, output layer acts as IF neuron.
            # If the other types of neuron is needed for the output layer,
            # the declarations of neuron layers in other files should be modified.
            self.run_type_if(inputs,t)
        else:
            self.integration(inputs,t)


    ############################################################
    ##
    ############################################################

    def set_vth(self,vth):
        #self.vth = self.vth.assign(vth)
        self.vth.assign(vth)

    def get_spike_count(self):
        spike_count = tf.reshape(self.spike_counter,self.dim)
        return spike_count

    def get_spike_count_int(self):
        #spike_count_int = tf.reshape(self.spike_counter_int,self.dim)
        return self.spike_counter_int

    def get_spike_rate(self):
        #return self.get_spike_count_int()/self.conf.time_step
        return self.get_spike_count_int()/self.conf.time_step

    def get_tot_psp(self):
        return self.tot_psp

    def get_isi(self):
        return self.isi



    def set_time_const_init_integ(self, time_const_init_integ):
        self.time_const_init_integ = time_const_init_integ

    def set_time_const_init_fire(self, time_const_init_fire):
        self.time_const_init_fire = time_const_init_fire

    def set_time_delay_init_integ(self, time_delay_init_integ):
        self.time_delay_init_integ = time_delay_init_integ

    def set_time_delay_init_fire(self, time_delay_init_fire):
        self.time_delay_init_fire = time_delay_init_fire


    #
    def set_time_const_integ(self, time_const_integ):
        self.time_const_integ = time_const_integ

    def set_time_const_fire(self, time_const_fire):
        self.time_const_fire = time_const_fire

    def set_time_delay_integ(self, time_delay_integ):
        self.time_delay_integ = time_delay_integ

    def set_time_delay_fire(self, time_delay_fire):
        self.time_delay_fire = time_delay_fire




    #def set_time_const_integ(self, time_const_integ):
    #    self.time_const_integ = time_const_integ

    #def set_time_delay_integ(self, time_delay_integ):
    #    self.time_delay_integ = time_delay_integ


    def set_time_integ(self, time_start_integ):
        self.time_start_integ = time_start_integ
        self.time_end_integ = self.time_start_integ + self.conf.time_fire_duration

    def set_time_fire(self, time_start_fire):
        self.time_start_fire = time_start_fire
        self.time_end_fire = self.time_start_fire + self.conf.time_fire_duration







    ############################################################
    ## training time constant (tau) for TTFS coding
    ## gradient-based optimization (DAC-20)
    ############################################################
    def train_time_const_fire(self,dnn_act):
        #print("snn_lib: train_time_const")
        #print(dnn_act)
        #self.time_const_integ = tf.zeros([])
        #self.time_const_fire = tf.multiply(self.time_const_fire,0.1)

        # delta - -1/(2tau^2)(x-x_hat)(x_hat)

        #spike_time = self.first_spike_time-self.depth*self.conf.time_fire_start*self.time_const_fire-self.time_delay_fire

        #if self.conf.f_tc_based:
        #    spike_time = self.first_spike_time-self.depth*self.conf.n_tau_fire_start*self.time_const_fire-self.time_delay_fire
        #else:
        #    spike_time = self.first_spike_time-self.depth*self.conf.time_fire_start-self.time_delay_fire


        spike_time = self.relative_time_fire(self.first_spike_time)
        #spike_time = self.first_spike_time
        spike_time_sub_delay = spike_time-self.time_delay_fire


        x = dnn_act

        #x_hat = tf.where(
        #            tf.equal(self.first_spike_time,tf.constant(self.init_first_spike_time,shape=self.first_spike_time.shape,dtype=tf.float32)), \
        #            tf.zeros(self.first_spike_time.shape), \
        #            tf.exp(-(spike_time_sub_delay/self.time_const_fire)))


        x_hat = tf.where(
                    self.flag_fire(),\
                    tf.zeros(self.first_spike_time.shape), \
                    tf.exp(-(spike_time_sub_delay/self.time_const_fire)))


        #x_hat = tf.exp(-self.first_spike_time/self.time_const_fire)

        #loss = tf.reduce_sum(tf.square(x-x_hat))

        #print(x[0])
        #print(x_hat[0])
        #print(tf.reduce_min(x_hat))
        #print(tf.reduce_max(x_hat))
        #print(tf.reduce_min(spike_time_sub_delay))
        #print(tf.reduce_max(spike_time_sub_delay))

        loss_prec = tf.reduce_mean(tf.square(x-x_hat))
        loss_prec = loss_prec/2.0

        # l2
        delta1 = tf.subtract(x,x_hat)
        delta1 = tf.multiply(delta1,x_hat)
        #delta1 = tf.multiply(delta1, tf.subtract(self.first_spike_time,self.time_delay_fire))
        delta1 = tf.multiply(delta1, spike_time_sub_delay)

        if tf.equal(tf.size(tf.boolean_mask(delta1,delta1>0)),0):
            delta1 = tf.zeros([])
        else:
            delta1 = tf.reduce_mean(tf.boolean_mask(delta1,delta1>0))


        dim = tf.size(tf.shape(x))
        if tf.equal(dim, tf.constant(4)):
            reduce_axis = [1,2,3]
        elif tf.equal(dim, tf.constant(3)):
            reduce_axis = [1,2]
        elif tf.equal(dim, tf.constant(2)):
            reduce_axis = [1]


        if self.conf.f_train_tk_outlier:
            #x_min = tf.tfp.stats.percentile(tf.boolean_mask(x,x>0),0.01)
            #x_min = tf.constant(np.percentile(tf.boolean_mask(x,x>0).numpy(),1),dtype=tf.float32,shape=[])

            x_pos = tf.where(x>tf.zeros(x.shape),x,tf.zeros(x.shape))
            x_min = tf.constant(np.percentile(x_pos.numpy(),2,axis=reduce_axis),dtype=tf.float32,shape=x_pos.shape[0])

            #print("min: {:e}, min_0.01: {:e}".format(tf.reduce_min(tf.boolean_mask(x,x>0)),x_min))
        else:
            #~x_min = tf.reduce_min(tf.boolean_mask(x,x>0))
            x_pos = tf.where(x>tf.zeros(x.shape),x,tf.zeros(x.shape))
            x_min = tf.reduce_min(x_pos,axis=reduce_axis)



        if self.conf.f_tc_based:
            fire_duration = self.conf.n_tau_fire_duration*self.time_const_fire
        else:
            fire_duration = self.conf.time_fire_duration

        #x_hat_min = tf.exp(-(self.conf.time_fire_duration/self.time_const_fire))
        x_hat_min = tf.exp(-(fire_duration-self.time_delay_fire)/self.time_const_fire)

        loss_min = tf.reduce_mean(tf.square(x_min-x_hat_min))
        loss_min = loss_min/2.0

        x_min = tf.reduce_mean(x_min)




        delta2 = tf.subtract(x_min,x_hat_min)
        delta2 = tf.multiply(delta2,x_hat_min)

        if self.conf.f_tc_based:
            delta2 = tf.multiply(delta2, tf.subtract(self.conf.n_tau_time_window*self.time_const_fire,self.time_delay_fire))
        else:
            delta2 = tf.multiply(delta2, tf.subtract(self.conf.time_window,self.time_delay_fire))


        #delta2 = tf.reduce_mean(delta2)

        #
        #idx=0,0,0,0
        #print("x: {:e}, x_hat: {:e}".format(x[idx],x_hat[idx]))
        #print("x_min: {:e}, x_hat_min: {:e}".format(x_min,x_hat_min))

        # l1


        #
        delta1 = tf.divide(delta1,tf.square(self.time_const_fire))
        delta2 = tf.divide(delta2,tf.square(self.time_const_fire))


        #rho1 = 10.0
        #rho2 = 100.0


        rho1 = 1.0
        rho2 = 1.0

        #
        delta = tf.add(tf.multiply(delta1,rho1),tf.multiply(delta2,rho2))

        #print("name: {:s}, del: {:e}, del1: {:e}, del2: {:e}".format(self.name,delta,delta1,delta2))


        #self.time_const_fire = tf.subtract(self.time_const_fire, delta)
        self.time_const_fire = tf.add(self.time_const_fire, delta)


        #
        #idx=0,10,10,0
        #print('x: {:e}, vmem: {:e}, x_hat: {:e}, delta: {:e}'.format(x[idx],self.vmem[idx],x_hat[idx],delta))

        print("name: {:s}, loss_prec: {:g}, loss_min: {:g}, tc: {:f}".format(self.name,loss_prec,loss_min,self.time_const_fire))

        self.loss_prec = loss_prec
        self.loss_min = loss_min

        #
        #print("name: {:s}, tc: {:f}".format(self.name,self.time_const_fire))


        #print("\n")

    ############################################################
    ## training time delay (td) for TTFS coding
    ## gradient-based optimization (DAC-20)
    ############################################################
    def train_time_delay_fire(self, dnn_act):

#        if self.conf.f_train_tk_outlier:
#            t_ref = self.depth*self.conf.time_fire_start
#            t_min = np.percentile(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0).numpy(),0.01)
#            t_min = t_min-t_ref
#        else:
#            t_ref = self.depth*self.conf.time_fire_start
#            t_min = tf.reduce_min(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0))
#            t_min = t_min-t_ref


        #t_ref = self.depth*self.conf.time_fire_start*self.time_const_fire

        if self.conf.f_tc_based:
            t_ref = self.depth*self.conf.n_tau_fire_start*self.time_const_fire
        else:
            t_ref = self.depth*self.conf.time_fire_start


        dim = tf.size(tf.shape(self.first_spike_time))
        if tf.equal(dim, tf.constant(4)):
            reduce_axis = [1,2,3]
        elif tf.equal(dim, tf.constant(3)):
            reduce_axis = [1,2]
        elif tf.equal(dim, tf.constant(2)):
            reduce_axis = [1]

        #print(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0,keepdims=True).shape)

        #t_min = tf.reduce_min(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0))
        t_min = tf.where(tf.equal(self.first_spike_time,self.init_first_spike_time),tf.constant(99999.9,shape=self.first_spike_time.shape),self.first_spike_time)
        t_min = tf.reduce_min(t_min,axis=reduce_axis)
        #t_min = t_min-t_ref
        t_min = self.relative_time_fire(t_min)

        x_max = tf.exp(-(t_min-self.time_delay_fire)/self.time_const_fire)

        x_max_hat = tf.exp(self.time_delay_fire/self.time_const_fire)

        loss_max = tf.reduce_mean(tf.square(x_max-x_max_hat))
        loss_max = loss_max/2.0

        delta = tf.subtract(x_max,x_max_hat)
        delta = tf.multiply(delta,x_max_hat)
        delta = tf.divide(delta,self.time_const_fire)
        delta = tf.reduce_mean(delta)

        rho = 1.0

        delta = tf.multiply(delta,rho)

        #print(self.first_spike_time)
        #print(t_min)
        ##print(x_max)
        #print(x_max_hat)

        #print("t_min: {:f}".format(t_min))
        #print("x_max: {:f}".format(x_max))
        #print("x_max_hat: {:f}".format(x_max_hat))
        #print("delta: {:f}".format(delta))

        #self.time_delay_fire = tf.subtract(self.time_delay_fire,delta)
        self.time_delay_fire = tf.add(self.time_delay_fire,delta)

        #print("name: {:s}, del: {:e}, td: {:e}".format(self.name,delta,self.time_delay_fire))
        print("name: {:s}, loss_max: {:e}, td: {:f}".format(self.name,loss_max,self.time_delay_fire))

        self.loss_max = loss_max


    ############################################################
    ## This function is needed for fire phase in the temporal coding (TTFS)
    ## time converter: absolute time (global time) -> relative time in each time window (local time)
    ## t: absolute time (global time)
    ## time: relative time (local time)
    ############################################################
    def relative_time_fire(self, t):
        if self.conf.f_tc_based:
            time = t-self.depth*self.conf.n_tau_fire_start*self.time_const_fire
        else:
            time = t-self.depth*self.conf.time_fire_start
        return time

    ############################################################
    ## This function is needed for integration phase in the temporal coding (TTFS)
    ## time converter: absolute time (global time) -> relative time in each time window (local time)
    ## t: absolute time (global time)
    ## time: relative time (local time)
    ############################################################
    def relative_time_integ(self, t):
        if self.conf.f_tc_based:
            time = t-(self.depth-1)*self.conf.n_tau_fire_start*self.time_const_integ
        else:
            time = t-(self.depth-1)*self.conf.time_fire_start
        return time


    #
    def flag_fire(self):
        ret = tf.not_equal(self.spike_counter,tf.constant(0.0,tf.float32,self.spike_counter.shape))
        return ret


    ###########################################################################
    ## SNN training w/ TTFS coding
    ###########################################################################



###############################################################################
## Temporal kernel for surrogate model training
## enc(t)=ta*exp(-(t-td)/tc)
###############################################################################
class Temporal_kernel(tf.keras.layers.Layer):
    def __init__(self,dim_in,dim_out,conf):
        super(Temporal_kernel, self).__init__()

        #
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dim_in_one_batch = [1,]+dim_in[1:]
        self.dim_out_one_batch = [1,]+dim_out[1:]

        #
        self.init_tc = conf.tc
        self.init_td = self.init_tc*np.log(conf.td)
        self.init_tw = conf.time_window

        self.epoch_start_t_int = conf.epoch_start_train_t_int
        self.epoch_start_clip_tw = conf.epoch_start_train_clip_tw
        self.epoch_start_train_tk = conf.epoch_start_train_tk
        # start epoch training with floor function - quantization
        # before this epoch, training with round founction
        self.epoch_start_train_floor = conf.epoch_start_train_floor

        #
        #self.enc_st_n_tw = conf.enc_st_n_tw
        # encoding maximum spike time
        self.ems_mode = conf.ems_loss_enc_spike
        self.ems_nt_mult_tw = conf.enc_st_n_tw*conf.time_window

        #
        self.f_td_training=conf.f_td_training

        # encoding decoding para couple
        self.f_enc_dec_couple = True
        #self.f_enc_dec_couple = False

        # double tc
        #self.f_double_tc = True
        self.f_double_tc = False


    def build(self, _):

        # TODO: parameterize
        # which one ?
        # a para per layer
        # neuron-wise para
        self.tc = self.add_variable("tc",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_tc),trainable=True)
        #self.td = self.add_variable("td",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_td),trainable=True)
        self.td = self.add_variable("td",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_td),trainable=self.f_td_training)
        self.tw = self.add_variable("tw",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_tw),trainable=False)

        if self.f_double_tc:
            self.tc_1 = self.add_variable("tc_1",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(10.0),trainable=True)
            self.td_1 = self.add_variable("td_1",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=True)


        #
        # decoding para
        #self.tc_dec = self.add_variable("tc_dec",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_tc),trainable=True)
        #self.td_dec = self.add_variable("td_dec",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_td),trainable=True)
        ##self.tw_dec = self.add_variable("tw_dec",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_tw),trainable=False)

        #
        # decoding para

        if not self.f_enc_dec_couple:
            self.tc_dec = self.add_variable("tc_dec",shape=self.dim_out_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_tc),trainable=True)
            self.td_dec = self.add_variable("td_dec",shape=self.dim_out_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_td),trainable=True)
            self.tw_dec = self.add_variable("tw_dec",shape=self.dim_out_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_tw),trainable=False)



        # input - encoding target
        self.in_enc = self.add_variable("in_enc",shape=self.dim_out,dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=False)
        # output of encoding - spike time
        self.out_enc = self.add_variable("out_enc",shape=self.dim_out,dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=False)
        # output of decoding
        self.out_dec = self.add_variable("out_dec",shape=self.dim_in,dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=False)



    def call(self, input, mode, epoch, training):
        mode_sel={
            'enc': self.call_encoding,
            'dec': self.call_decoding
        }

        ret = mode_sel[mode](input, epoch, training)

        return ret

    def call_encoding(self, input, epoch, training):

        #
        self.in_enc = input

        #
        t_float = self.call_encoding_kernel(input)

        #
        infer_mode = (training==False)and(epoch<0)
        #
        #if False:
        #if ((training==False) and (epoch==-1)) or ((training == True) and (epoch > self.epoch_start_t_int)):
        if ((training==True)and(tf.math.greater(epoch,self.epoch_start_t_int))) or (training==False):
            # TODO: parameterize
            #if epoch > self.epoch_start_t_int+100:
            #if epoch > self.epoch_start_t_int:
            #    t = tf.ceil(t_float)
            #else:
            #    t = tf.quantization.fake_quant_with_min_max_vars(t_float,0,tf.pow(2.0,16.0)-1,16)
            #   #` t=tf.round(t_float)


            #if (epoch < self.epoch_start_train_floor) and not infer_mode:
            if ((training==True) and (tf.math.greater(epoch,self.epoch_start_train_floor))) or (training==False):
                #t = tf.quantization.fake_quant_with_min_max_vars(t_float,0,tf.pow(2.0,16.0)-1,16)
                #t = tf.math.add(tf.math.floor(t_float),1)
                t = tf.math.ceil(t_float)
                #t = tf.math.ceil(t)
            else:
                t = tf.quantization.fake_quant_with_min_max_vars(t_float, 0, tf.pow(2.0, 16.0) - 1, 16)


            #tmp = tf.where(tf.equal(t,0),100,t)
            #print(tf.reduce_min(tmp))

        else :
            t = t_float



#        #
#        if (training == False) or ((training==True) and (epoch > self.epoch_start_clip_tw)):
#        #if False:
#        #if True:
#            #print(t)
#            t=tf.math.minimum(t, self.tw)
#            #print(self.tw)


        #print('min: {:}, max:{:}'.format(tf.reduce_min(t),tf.reduce_max(t)))
        #print(t)

        #
        self.out_enc = t

        #print(tf.reduce_mean(self.tc))
        #print(tf.reduce_mean(self.td))
        #print(tf.reduce_mean(self.ta))

        return t


    def call_encoding_kernel(self, input):

        if self.ems_mode== 'f':
            eps = 1.0E-30
        elif self.ems_mode == 'n':
            #eps = tf.math.exp(-float(self.ems))
            eps = tf.math.exp(tf.math.divide(tf.math.subtract(self.td,self.ems_nt_mult_tw),self.tc))
        else:
            assert False, 'not supported encoding maximum spike mode - {}'.format(self.ems_mode)


        #x = tf.nn.relu(input)
        #x = tf.divide(x,self.ta)
        x = tf.nn.relu(input)
        x = tf.add(x,eps)
        x = tf.math.log(x)

        if self.f_double_tc:
            #t = tf.subtract(self.td, tf.multiply(x,self.tc))
            A = tf.math.log(tf.add(tf.exp(tf.divide(self.td,self.tc)),tf.exp(tf.divide(self.td_1,self.tc_1))))
            x = tf.subtract(A,x)
            t = tf.multiply(tf.divide(tf.add(self.tc,self.tc_1),2),x)
        else:
            t = tf.subtract(self.td, tf.multiply(x,self.tc))

        t = tf.nn.relu(t)

        #print(t)

        #print(t)
        #print(self.td)

        return t


    def call_decoding(self, t, epoch, training):


        #x = tf.multiply(self.ta,tf.exp(tf.divide(tf.subtract(self.td,t),self.tc)))

        if self.f_enc_dec_couple:
            tw_target = self.tw
            td = self.td
            tc = self.tc

            #if epoch > 500:
            #    tw_target = 1.5*self.tw - self.tw/1000*epoch

        else:
            #tw_target = self.tw_dec
            #td = self.td_dec
            #tc = self.tc_dec

            tw_target = self.tw
            td = self.td
            tc = self.tc
        #
        if self.f_double_tc:
            x = tf.add(tf.exp(tf.divide(tf.subtract(td,t),tc)),tf.exp(tf.divide(tf.subtract(self.td_1,t),self.tc_1)))
        else:
            x = tf.exp(tf.divide(tf.subtract(td,t),tc))



        #if False:
        #if (training == False) or ((training==True) and (epoch > self.epoch_start_clip_tw)):
        #if (training==True)and(epoch > self.epoch_start_clip_tw) or (training==False)and(epoch<0):
        if ((training == True) and (tf.math.greater(epoch,self.epoch_start_clip_tw))) or (training == False):
            #if epoch > 300:
            #    tw_target = self.tw/2
            #else:
            #    tw_target = self.tw

            #
            tk_min = tf.exp(tf.divide(tf.subtract(td,tw_target),tc))

            #print('min: {:}, max:{:}'.format(tf.reduce_min(x),tf.reduce_max(x)))

            #print('')
            #print(tw_target)
            #print(tk_min)

            x_clipped = tf.where(x>=tf.broadcast_to(tk_min,shape=x.shape),\
                                 x,tf.constant(0.0,shape=x.shape,dtype=tf.float32))

            #x_clipped = x-tk_min
            #x_clipped = tf.nn.relu(x_clipped)
            #x_clipped = x_clipped+tk_min

            x = x_clipped

        self.out_dec = x


        #print('min: {:}, max:{:}'.format(tf.reduce_min(x),tf.reduce_max(x)))

        return x

    #
    def set_init_td_by_target_range(self, act_target_range):

        td=tf.multiply(self.tc,tf.math.log(act_target_range))
        self.td.assign(tf.constant(td,dtype=tf.float32,shape=self.td.shape))






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
