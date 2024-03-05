import tensorflow as tf
# import tensorflow.contrib.eager as tfe

# import tensorflow_probability as tfp

import tensorflow.keras.regularizers as regularizers

from keras import backend

from tensorflow.python.ops import math_ops

import tensorflow_probability as tfp

# custom gradient


#

#
import lib_snn

#
from lib_snn.sim import glb_t
from lib_snn.sim import glb

#
#from lib_snn.sim import glb_plot
#from lib_snn.sim import glb_plot_act
#from lib_snn.sim import glb_plot_syn
#from lib_snn.sim import glb_plot_bn
#from lib_snn.sim import glb_plot_kernel

#from main_hp_tune import conf

#from config import conf
#from config_common import conf
from absl import flags
conf = flags.FLAGS

from keras.engine.input_spec import InputSpec


import layers_new


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

            self.bn = layers_new.batch_normalization.BatchNormalization(epsilon=1.001e-5,en_tdbn=tdbn,fused=fused)

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
        #if self.conf.nn_mode=='SNN' and not self.conf.snn_training_spatial_first:
            #print(self.name)
            #print(self.input_spec)
            #if not self.input_spec is None:
                #if not self.input_spec.ndim is None:
                    #self.input_spec = InputSpec(ndim=self.input_spec.ndim+1)


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

        #
        #self._call_full_argspec
        #self._expects_training_arg = 'training' in self._call_full_argspec.args
        #self._expects_mask_arg = 'mask' in self._call_full_argspec.args

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

        # snn training - temporal first
        #if not self.conf.snn_training_spatial_first:
            #self.out = tf.TensorArray(dtype=tf.float32,
                        #size=self.conf.time_step,element_shape=self.output_shape_fixed_batch,clear_after_read=False)


        # integrated output
        # only for Conv2D
        if hasattr(self, 'f_output_integ'):
            if self.f_output_integ:
                self.output_integ = tf.zeros(shape=self.output_shape_fixed_batch,name=self.name+'_output_integ')


        #
        # integrated output
        # only for Conv2D
        #if conf.debug_syn_output:
        if hasattr(self, 'f_output_t'):
            if self.f_output_t:
                self._outputs = tf.TensorArray(
                    dtype=tf.float32,
                    size=conf.time_step,
                    element_shape=self.output_shape_fixed_batch,
                    clear_after_read=False,
                    tensor_array_name='outputs')


        #
        #self.built = True
        #print('build layer - done')

    def init_record_output(self):
        #self.record_output = tf.Variable(tf.zeros(self.output_shape_fixed_batch),trainable=False,name='record_output')
        self.record_output = tf.TensorArray(dtype=tf.float32,
                                size=self.conf.time_step,element_shape=self.output_shape_fixed_batch,clear_after_read=False)

        #
        f_hold_temporal_tensor=True
        #if self.conf.f_hold_temporal_tensor:
        if False:
            output_shape_list = self.output_shape_fixed_batch.as_list()
            output_shape_list.insert(0,self.conf.time_step)
            output_shape = tf.TensorShape(output_shape_list)

            #[time, batch, width, height, channel]
            self.record_output = tf.Variable(tf.zeros(output_shape),trainable=False,name='record_output')

        #if self.last_layer:
        if False:
            #self.record_logit= tf.Variable(tf.zeros(self.output_shape_fixed_batch),trainable=False,name='record_logit')
            self.record_logit = tf.TensorArray(dtype=tf.float32, size=self.conf.time_step,element_shape=self.output_shape_fixed_batch)


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
    #def call(self, input, **kwargs):
        #print('layer - {:}, training - {:}'.format(self.name,training))
        #print('layer call - {}'.format(self.name))

        #if self.conf.debug_mode:
        #    print('layer name: {:}'.format(self.name))

        if training is None:
            training = backend.learning_phase()

        # test - add noise to input
        #noise_input_layer = keras.layers.GaussianNoise(0.7)
        #noise_input_layer = keras.layers.GaussianNoise(0.1)
        #noise_input_layer = keras.layers.GaussianNoise(1.0)
        #input_noise = noise_input_layer(input,training)
        #input = tf.where(input==0,input_noise,input)

        #n_dim = len(input.shape)
        #reduction_axes = [i for i in range(n_dim-1)]
        #variance_input = tf.math.reduce_variance(input,axis=reduction_axes)
        #input = tf.where(variance_input<0.1,input_noise,input)
        #input = input_noise

        # new
        #if not self.conf.snn_training_spatial_first and self.act is None and self.bn is None:
        #if not self.conf.snn_training_spatial_first:
        #if True:
        if False:

            range_ts = range(1, self.conf.time_step + 1)


            out_arr = tf.TensorArray(dtype=tf.float32,
                                      size=self.conf.time_step,element_shape=self.output_shape_fixed_batch,clear_after_read=False)
            out_arr = []
            for t in range_ts:
                #layer_in = input.read(t-1)
                layer_in = input[t-1]

                layer_out = super().call(layer_in)

                out_arr.append(layer_out)

                #out_arr = out_arr.write(t-1,layer_out)

            out_arr=tf.convert_to_tensor(out_arr)

            return out_arr
        else:
            #self._expects_training_arg = ('training' in method_arg_spec.args or
            #                              method_arg_spec.varkw is not None)
            #self._expects_mask_arg = ('mask' in method_arg_spec.args or
            #                          method_arg_spec.varkw is not None)

            #if self._expects_training_arg:
                #print(self.name)
                ##output = super().call(input,training)
                ##output = super().call(input)
                #output = super
            #else:
                #output = super().call(input)

            #output = super().call(input,**kwargs)

            # TODO: add codes to check parent's class call argument - including training or not
            if isinstance(self, lib_snn.layers.BatchNormalization):
                output = super().call(input,training)
            else:
                output = super().call(input)

            # regularization
            if conf.reg_psp and self.depth > 1:
                #print('reg_psp: {:}'.format(self.name))
                lib_snn.layers.reg_psp(self,output)

                if False:
                    #h_min = -1.0
                    #h_max = 1.0
                    h_min = tf.reduce_min(output)
                    h_max = tf.reduce_max(output)


                    if False: # histogram
                        #hist = tf.histogram_fixed_width(inputs,[tf.reduce_min(inputs),tf.reduce_max(inputs)])
                        hist = tf.histogram_fixed_width(output,[h_min,h_max])
                        num_inputs = tf.reduce_sum(hist)
                        #hist = tf.where(hist==0,tf.constant(1.0e-5,shape=hist.shape),hist)
                        p = tf.cast(hist / num_inputs,dtype=tf.float32)
                    else:   # fitting to normal dist.
                        mean = tf.reduce_mean(output)
                        std = tf.math.reduce_std(output)
                        pdf = tfp.distributions.Normal(mean,std)
                        p = pdf.prob(output)

                        print("before")
                        print(output)
                        print("mean out: {:}, std out: {:}".format(mean,std))
                        print("min p: {:}, max p: {:}".format(tf.reduce_min(p),tf.reduce_max(p)))


                    #e = tf.math.multiply_no_nan(tf.math.log(p)/tf.math.log(tf.cast(2.0,dtype=tf.float64)),p)
                    e = tf.math.multiply_no_nan(tf.math.log(p)/tf.math.log(2.0),p)
                    #e = tf.where(p==0,tf.zeros(e.shape),e)
                    e = -tf.reduce_mean(e)

                    print(self.name)
                    print(e)
                    print("min p: {:}, max p: {:}".format(tf.reduce_min(p),tf.reduce_max(p)))

                    if tf.reduce_any(tf.math.is_nan(e)):
                        print(self.name)
                        print(e)
                        #print(output)
                        #assert False


                    #e = tf.clip_by_value(e, 1,10)
                    #print(e)
                    #self.add_loss(0.01*e)
                    self.add_loss(conf.reg_syn_in_const*e)

            #if conf.debug_syn_output:
            if hasattr(self, 'f_output_t'):
                if self.f_output_t:
                    t = glb_t.t
                    #t=1
                    self._outputs = self._outputs.write(t - 1, output)

                    #output = self._outputs.read(t-1)

            return output

        #
        # temporal first
        if False:
            print(self.name)
            print(input.shape)
            print('self.built : {:}'.format(self.built))
            self.time_axis=0
            if isinstance(self,lib_snn.layers.InputGenLayer):
                if not self.conf.snn_training_spatial_first and self.built:
                    #input_expand = tf.expand_dims(input,axis=time_axis)
                    input_expand_shape = [self.conf.time_step,]+input.shape
                    input = tf.broadcast_to(input,shape=input_expand_shape)
                    print('input expand - time axis: {:}'.format(self.time_axis))

                    self.output_shape_fixed_batch=[self.conf.time_step,]+self.output_shape_fixed_batch

            print('output dim')
            print(self.output_shape_fixed_batch)

            # spatial or temporal_first
            if self.conf.snn_training_spatial_first or (not self.built):
                _input = input
                s = super().call(_input)
            else:

                range_ts = range(1, self.conf.time_step + 1)

                #s = tf.zeros([self.conf.time_step,]+self.output_shape_fixed_batch)
                s = tf.zeros(self.output_shape_fixed_batch)
                for t in range_ts:
                    #_input = tf.gather(input,)
                    #_input = input[t]
                    #_input = input
                    #_input = input[:,t-1,:]
                    _input = input[t-1,:,:,:,:]       # [t,b,h,w,c] -> [b,h,w,c]
                    _s = super().call(_input)
                    _s = tf.expand_dims(_s,axis=self.time_axis)

                    print('s')
                    print(s.shape)

                    print('_s')
                    print(_s.shape)

                    indices = tf.constant([[t-1]])
                    index_depth=1

                    s=tf.tensor_scatter_nd_update(s,indices,_s)
                    #s=tf.tensor_scatter_nd_update(s,[[0]],_s)

        #
        #print(input)
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

            # add noise test - sspark, 221206
            if False:
                n_dim = len(s.shape)
                reduction_axes = [i for i in range(n_dim-1)]
                variance = tf.math.reduce_variance(s,axis=reduction_axes)
                reduced_dim = variance.shape
                #if variance < tf.constant(0.3,shape=reduced_dim):
                #n=tf.cond(variance < tf.constant(0.3,shape=reduced_dim), lambda: add_noise(n), lambda: n)

                #noise_layer = keras.layers.GaussianNoise(1.0-variance)
                #noise_layer = keras.layers.GaussianNoise(variance)
                #noise_layer = keras.layers.GaussianNoise(0.1)
                #noise_layer = keras.layers.GaussianNoise(0.5)
                #noise_layer = keras.layers.GaussianNoise(0.7)
                #noise_layer = keras.layers.GaussianNoise(1.0)
                #s_noise = noise_layer(s,training)

            #print(s)
            if False:
                print('layer - '+self.name)
                print('before - variance')
                print(variance)
                print(tf.reduce_min(variance))

            #target_variance = 0.3
            #s=tf.where(variance<tf.constant(target_variance,shape=reduced_dim),s_noise,s)

            if False:
                print(variance<tf.constant(target_variance,shape=reduced_dim))
                variance_after = tf.math.reduce_variance(s,axis=reduction_axes)
                print('after - variance')
                print(variance_after)
                print(tf.reduce_min(variance_after))


            b = self.bn(s, training=training)
            #if glb.model_compiled:
            #    assert False


        else:
            b = s
            #print('here')
            #print(self.name)

        # for debug
        #if self.last_layer:
        #    print(s)


        if self.act is None:
            n = b
        else:
            if self.en_snn:
                n = self.act(b, glb_t.t, training)

                if self.last_layer and (not (self.act_dnn is None)):
                    # softmax
                    n = self.act_dnn(n)

                else:
                    pass


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

        if self.conf.debug_mode and self.conf.verbose_snn_train:
            if glb.model_compiled:
                if (self.use_bn) and (not self.f_skip_bn):
                    #print('{:.3e}'.format(tf.reduce_max(b)))
                    #print('after bn> {} - max: {}, mean: {}'.format(self.name,tf.reduce_max(b),tf.reduce_mean(b)))
                    print('before bn> {} - max: {:.3g}, mean: {:.3g}, var: {:.3g}'
                          .format(self.name,tf.reduce_max(s),tf.reduce_mean(s),tf.math.reduce_variance(s)))
                    print('after bn> {} - max: {:.3g}, mean: {:.3g}, var: {:.3g}, moving_mean: {:.3g}, moving_var: {:.3g}'
                          .format(self.name,tf.reduce_max(b),tf.reduce_mean(b),tf.math.reduce_variance(b), tf.reduce_mean(self.bn.moving_mean),tf.reduce_mean(self.bn.moving_variance)))
                    #print('after bn> {} - max: {:.3e}, mean: {:.3e}'.format(self.name))

                    #if self.name=='conv1':
                    #    print(self.bn.moving_variance)

                #
                if hasattr(self,'act') and isinstance(self.act,lib_snn.neurons.Neuron):
                    print('\n spike count - layer')
                    spike_count = self.act.spike_count
                    print('{: <10}: - sum {:.3e}, mean {:.3e}, non-zero percent {:.3e}'
                          .format(self.name,tf.reduce_sum(spike_count),tf.reduce_mean(spike_count)
                                  ,tf.math.count_nonzero(spike_count,dtype=tf.float32)/tf.cast(tf.reduce_prod(spike_count.shape),dtype=tf.float32)))


        # plot
        #if True and (glb.model_compiled):
        if self.conf.verbose_visual and (glb.model_compiled):
            from lib_snn.sim import glb_plot_act
            from lib_snn.sim import glb_plot_syn
            from lib_snn.sim import glb_plot_bn
            from lib_snn.sim import glb_plot_kernel

            lib_snn.util.plot_hist(glb_plot_syn,s,1000,norm_fit=True)
            lib_snn.util.plot_hist(glb_plot_bn,b,1000,norm_fit=True)
            lib_snn.util.plot_hist(glb_plot_act,n,100,range=[-1,2])
            if hasattr(self,'kernel'):
                lib_snn.util.plot_hist(glb_plot_kernel,self.kernel,1000)

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


        if False: # old
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

        # new
        if self.en_record_output:
            self.record_output = self.record_output.write(glb_t.t-1,ret)
            #self.record_logit.assign(b)


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

        #
        if hasattr(self, 'f_output_integ'):
            if self.f_output_integ:
                if glb.t == 0:
                    self.output_integ = ret
                else:
                    self.output_integ = self.output_integ + ret


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

        from lib_snn.sim import glb_plot
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

    # v2
    def bn_fusion_v2(self,bn_layer):
        self._bn_fusion(bn_layer)

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



    # set pre and post neuron
    def set_n_pre_post(self,n_pre, n_post):
        self.n_pre = n_pre
        self.n_post = n_post


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

    def call(self, inputs, training):
        # print('input gen layer - call')
        if conf.input_data_time_dim:
            if inputs.shape.__len__() > 4:
                return inputs[:,0,:,:,:]    # for model build

        #print(inputs)
        return inputs

        #
#    def call(self, inputs, training):
#        inputs_e = tf.TensorArray(dtype=tf.float32,
#                             size=self.conf.time_step,element_shape=inputs.shape,clear_after_read=False)
#
#        #for t in range(1,conf.time_step+1):
#        #    inputs_e = inputs_e.write(t-1,inputs)
#
#        #ret = tf.convert_to_tensor(inputs_e)
#
#        #ret = tuple(inputs_e)
#        if self.conf.nn_mode=='SNN' and not self.conf.snn_training_spatial_first:
#            inputs = tf.expand_dims(inputs,axis=0)
#            ret = tf.repeat(inputs,self.conf.time_step,axis=0)
##
#        else:
#            ret = inputs
#
#        return ret






# Conv2D
#@tf.custom_gradient
#class Conv2D(Layer, tf.keras.layers.Conv2D):
class Conv2D(Layer, layers_new.conv.Conv2D):
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
                 #kernel_initializer='glorot_normal',
                 #kernel_initializer='he_normal',
                 kernel_constraint=None,
                 bias_constraint=None,
                 bias_regularizer=None,
                 use_bn=False,  # use batch norm.
                 **kwargs):

        Layer.__init__(self, use_bn, activation, kwargs=kwargs)

        layers_new.conv.Conv2D.__init__(
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

        # integrated output
        self.f_output_integ = False
        #self.f_output_integ = True
        self.f_output_t = conf.debug_syn_output



# DepthwiseConv2D
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
        assert False
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
#class Dense(Layer, tf.keras.layers.Dense):
class Dense(Layer, layers_new.dense.Dense):
    def __init__(self,
                 units,
                 activation=None,
                 # use_bias=True
                 kernel_initializer='glorot_uniform',
                 #kernel_initializer='glorot_normal',
                 #kernel_initializer='he_normal',
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

        #tf.keras.layers.Dense.__init__(
        layers_new.dense.Dense.__init__(
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

        #Layer.__init__(self, use_bn=use_bn, epsilon=epsilon, activation=activation, kwargs=kwargs)
        Layer.__init__(self, use_bn=use_bn, activation=activation, kwargs=kwargs)

        tf.keras.layers.Add.__init__(self, **kwargs)

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
        #Layer.__init__(self, use_bn=use_bn, epsilon=epsilon, activation=activation, kwargs=kwargs)
        Layer.__init__(self, use_bn=use_bn, activation=activation, kwargs=kwargs)

        tf.keras.layers.Layer.__init__(self, **kwargs)

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
            #self.record_output.assign(ret)
            self.record_output = self.record_output.write(glb_t.t,ret)

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

        Layer.__init__(self, False, None, kwargs=kwargs)

        tf.keras.layers.MaxPool2D.__init__(
            self,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=conf.data_format,
            **kwargs)


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


class AveragePooling2D(Layer, tf.keras.layers.AveragePooling2D):

    def __init__(self,
           pool_size=(2, 2),
           strides=None,
           padding='valid',
           data_format=None,
           **kwargs):

        Layer.__init__(self, use_bn=False, activation=None, last_layer=False, kwargs=kwargs)

        tf.keras.layers.AveragePooling2D.__init__(self,
                                            pool_size=pool_size,
                                            strides=strides,
                                            padding=padding,
                                            data_format=data_format,
                                            **kwargs)

        #self.input_spec = InputSpec(ndim=self.input_spec.ndim+1)



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

# Flatten
class Flatten(Layer, tf.keras.layers.Flatten):
    def __init__(self,
                 data_format,
                 **kwargs):

        Layer.__init__(self, use_bn=False, activation=None, last_layer=False, kwargs=kwargs)

        tf.keras.layers.Flatten.__init__(
            self,
            data_format=data_format,
            **kwargs)

#    def call(self, input, training=None):
#
#        if self.conf.nn_mode=='SNN' and not self.conf.snn_training_spatial_first:
#
#            output = []
#
#            range_ts = range(1, self.conf.time_step + 1)
#            for t in range_ts:
#                _input = input[t-1]
#                _output = tf.keras.layers.Flatten.call(self, _input)
#                output.append(_output)
#
#            output = tf.convert_to_tensor(output)
#
#        else:
#            output = tf.keras.layers.Flatten.call(self, input)
#
#        return output



class BatchNormalization(Layer, layers_new.batch_normalization.BatchNormalization):

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 renorm=False,
                 renorm_clipping=None,
                 renorm_momentum=0.99,
                 fused=None,
                 trainable=True,
                 virtual_batch_size=None,
                 adjustment=None,
                 name=None,
                 **kwargs):

        Layer.__init__(self, use_bn=False, activation=None, last_layer=False, kwargs=kwargs)

        layers_new.batch_normalization.BatchNormalization.__init__(self,
                 axis=axis,
                 momentum=momentum,
                 epsilon=epsilon,
                 center=center,
                 scale=scale,
                 beta_initializer=beta_initializer,
                 gamma_initializer=gamma_initializer,
                 moving_mean_initializer=moving_mean_initializer,
                 moving_variance_initializer=moving_variance_initializer,
                 beta_regularizer=beta_regularizer,
                 gamma_regularizer=gamma_regularizer,
                 beta_constraint=beta_constraint,
                 gamma_constraint=gamma_constraint,
                 renorm=renorm,
                 renorm_clipping=renorm_clipping,
                 renorm_momentum=renorm_momentum,
                 fused=fused,
                 trainable=trainable,
                 virtual_batch_size=virtual_batch_size,
                 adjustment=adjustment,
                 name=name,
                 **kwargs)

#    def build(self, input_shape):
#
#        if self.conf.nn_mode=='SNN' and not self.conf.snn_training_spatial_first:
#            input_shape = input_shape[1:]       # [t,b,w,h,c] -> [b,w,h,c]
#
#        layers_new.batch_normalization.BatchNormalization.build(self,input_shape)
#
#        if self.conf.nn_mode=='SNN' and not self.conf.snn_training_spatial_first:
#            self.input_spec = InputSpec(ndim=self.input_spec.ndim+1,axes=self.input_spec.axes)




############################################################
## Temporal wrapper - temporal function
############################################################
def tfn(layer, input):
    if conf.nn_mode=='SNN' and not conf.snn_training_spatial_first:

        range_ts = range(1, conf.time_step + 1)

        f_temporal_reduction = False
        #
        layers_temporal_reduction = []
        layers_temporal_reduction.append(lib_snn.layers.BatchNormalization)
        #layers_temporal_reduction.append(lib_snn.layers.MaxPool2D)
        #layers_temporal_reduction.append(lib_snn.layers.AveragePooling2D)

        if type(layer) in layers_temporal_reduction:
            f_temporal_reduction = True

        #out_arr = []

        #
        if not isinstance(input,tf.TensorArray):
            in_arr = tf.TensorArray(
                dtype=tf.float32,
                size=conf.time_step,
                element_shape=input.shape,
                clear_after_read=False,
                tensor_array_name='in_arr')

        else:
            in_arr = input


        if f_temporal_reduction:
            layer_in = tf.reduce_mean(in_arr.stack(),axis=0)

            layer_out = layer(layer_in)

            #
            out_arr = tf.TensorArray(
                dtype=tf.float32,
                size=conf.time_step,
                element_shape=layer_out.shape,
                clear_after_read=False,
                tensor_array_name='out_arr')

            for t in range_ts:
                #out_arr.append(layer_out)
                out_arr = out_arr.write(t-1,layer_out)

        else:

            for t in range_ts:
                #layer_in = input.read(t-1)
                #layer_in = input[t-1]
                layer_in = in_arr.read(t-1)

                layer_out = layer(layer_in)


                #
                if t-1==0:
                    out_arr = tf.TensorArray(
                        dtype=tf.float32,
                        size=conf.time_step,
                        element_shape=layer_out.shape,
                        clear_after_read=False,
                        tensor_array_name='out_arr')

                #out_arr.append(layer_out)
                out_arr = out_arr.write(t-1,layer_out)

                #out_arr = out_arr.write(t-1,layer_out)

        #out_arr=tf.convert_to_tensor(out_arr)

        return out_arr
    else:
        return layer(input)






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



############################################################
## regularization (psp, entropy min.)
############################################################

def reg_psp(layer,psp):
    output = psp

    h_min = tf.reduce_min(output)
    h_max = tf.reduce_max(output)


    if False:  # histogram
        # hist = tf.histogram_fixed_width(inputs,[tf.reduce_min(inputs),tf.reduce_max(inputs)])
        hist = tf.histogram_fixed_width(output, [h_min, h_max])
        num_inputs = tf.reduce_sum(hist)
        # hist = tf.where(hist==0,tf.constant(1.0e-5,shape=hist.shape),hist)
        p = tf.cast(hist / num_inputs, dtype=tf.float32)
    else:  # fitting to normal dist.

        if False:
            mean = tf.reduce_mean(output)
            std = tf.math.reduce_std(output)
            pdf = tfp.distributions.Normal(mean, std)
            p = pdf.prob(output)

            print("before")
            print(output)
            print("mean out: {:}, std out: {:}".format(mean, std))
            print("min p: {:}, max p: {:}".format(tf.reduce_min(p),
                                                  tf.reduce_max(p)))

        #
        #p = lib_snn.layers.prob_fit_norm_dist(output)
        e = lib_snn.layers.prob_fit_norm_dist(output)


    if False:
        # e = tf.math.multiply_no_nan(tf.math.log(p)/tf.math.log(tf.cast(2.0,dtype=tf.float64)),p)
        #e = tf.math.divide_no_nan(tf.math.log(p),tf.math.log(2.0))
        e = tf.math.log(p)
        e = tf.math.multiply_no_nan(e, p)
        # e = tf.where(p==0,tf.zeros(e.shape),e)
        e = -tf.reduce_mean(e)

        print(layer.name)
        print(e)
        print("min p: {:}, max p: {:}".format(tf.reduce_min(p), tf.reduce_max(p)))

        if tf.reduce_any(tf.math.is_nan(e)):
            print(layer.name)
            print(e)
            # print(output)
            # assert False

    # e = tf.clip_by_value(e, 1,10)
    # print(e)
    # self.add_loss(0.01*e)
    #self.add_loss(conf.reg_syn_in_const * e)
    layer.add_loss(conf.reg_psp_const * e)



@tf.custom_gradient
def prob_fit_norm_dist(x):
    mean = tf.reduce_mean(x)
    std = tf.math.reduce_std(x)
    pdf = tfp.distributions.Normal(mean, std)
    p = pdf.prob(x)

    #
    eps=conf.reg_psp_eps
    log_pi = tf.math.log(p+eps)
    h = tf.math.multiply_no_nan(log_pi, p)
    # e = tf.where(p==0,tf.zeros(e.shape),e)
    if conf.reg_psp_min:
        h = -tf.reduce_mean(h)
    else:
        h = tf.reduce_mean(h)

    #print(h)
    #print("min p: {:}, max p: {:}".format(tf.reduce_min(p), tf.reduce_max(p)))

    def grad(upstream):
        d = -tf.math.divide(tf.math.subtract(x,mean), tf.math.square(std))
        ret = d*p
        #de_dp = tf.math.divide_no_nan(ret,p)
        num_psp = tf.cast(tf.math.reduce_prod(log_pi.shape),dtype=tf.float32)
        if conf.reg_psp_min:
            dh_dp = -(log_pi+tf.ones(shape=log_pi.shape))/num_psp
        else:
            dh_dp = (log_pi+tf.ones(shape=log_pi.shape))/num_psp
        ret = ret*dh_dp
        ret = ret*upstream
        #ret = ret * 0.0
        #ret = tf.zeros(shape=ret.shape)

        if False:
            print('')
            print('upstream: min {:}, max {:}'.format(tf.reduce_min(upstream),tf.reduce_max(upstream)))
            print('std: min {:}, max {:}'.format(tf.reduce_min(std),tf.reduce_max(std)))
            print('d: min {:}, max {:}'.format(tf.reduce_min(d),tf.reduce_max(d)))
            print('dh_dp: min {:}, max {:}'.format(tf.reduce_min(dh_dp),tf.reduce_max(dh_dp)))
            print('grad: min {:}, max {:}'.format(tf.reduce_min(ret),tf.reduce_max(ret)))
            print('')

        return ret


    #return p, grad
    return h, grad


# l2 norm
@tf.custom_gradient
def l2_norm(x,name):
    out = tf.sqrt(tf.reduce_sum(tf.square(x)))

    def grad(upstream):
        dy_dx = tf.multiply(x,tf.math.rsqrt(tf.reduce_sum(tf.square(x))))
        condition = tf.math.count_nonzero(x,dtype=tf.int32)==0
        ret_grad = tf.where(condition, tf.zeros(upstream.shape),upstream*dy_dx)


        #if True:
        #if False:
        if conf.verbose_snn_train:
            print('l2_norm - {:}'.format(name))

            var = x
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}, non_zero {:.3g}'
                  .format('inputs',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var),tf.math.count_nonzero(var,dtype=tf.int32)/tf.math.reduce_prod(var.shape)))

            #print(condition)
            #print(tf.reduce_mean(ret_grad))

            var = upstream
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('y_backprop',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = ret_grad
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('dx',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            print('')

        return ret_grad, None

    return out, grad
