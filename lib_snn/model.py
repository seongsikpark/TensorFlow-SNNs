
import tensorflow as tf

#
import os
import csv

#
import numpy as np
import collections

#
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import compile_utils

#
import keras

#

#
from absl import flags
flags = flags.FLAGS

#
from tqdm import tqdm

#
import lib_snn
from lib_snn.sim import glb
from lib_snn.sim import glb_t

#
#from lib_snn.sim import glb_plot
#from lib_snn.sim import glb_plot_1
#from lib_snn.sim import glb_plot_2
#from lib_snn.sim import glb_plot_3
#from lib_snn.sim import glb_plot_1x2

#from lib_snn.sim import glb_plot_gradient_kernel
#from lib_snn.sim import glb_plot_gradient_gamma
#from lib_snn.sim import glb_plot_gradient_beta


from config import config
conf = config.flags

class Model(tf.keras.Model):
    count=0
    def __init__(self, inputs, outputs, batch_size, input_shape, num_class, conf, **kwargs):
    #def __init__(self, batch_size, input_shape, data_format, num_class, conf, **kwargs):

        #print("lib_SNN - Layer - init")


        lmb = kwargs.pop('lmb', None)
        n_dim_classifier = kwargs.pop('n_dim_classifier', None)

        #
        super(Model, self).__init__(inputs=inputs,outputs=outputs,**kwargs)
        #super(Model, self).__init__(**kwargs)

        #
        Model.count += 1
        #assert Model.count==1, 'We have only one Model instance'

        #
        self.batch_size = batch_size
        self.in_shape = input_shape

        #
        self.verbose = conf.verbose

        #
        self.conf = conf
        Model.data_format = conf.data_format
        self.kernel_size = None                 # for conv layer
        #self.num_class = conf.num_class
        self.num_class = num_class
        Model.use_bias = conf.use_bias

        #
        Model.f_1st_iter = True
        self.f_1st_iter_stat = True
        Model.f_load_model_done = False
        #self.f_debug_visual = conf.verbose_visual
        self.f_done_preproc = False
        #self.f_skip_bn = False      # for the 1st iteration
        Model.f_skip_bn = False      # for the 1st iteration
        Model.f_dummy_run=True

        # keras model
        self.model = None

        # init done
        self.init_done = False

        # multi-gpu strategy
        self.dist_strategy = None

        #
        self.ts=conf.time_step
        self.epoch = -1

        # time step for SNN
        Model.t=0

        # lists
        #self.list_layer_name=None
        self.list_layer=[]
        self.list_layer_name=[]
        #self.list_layer=collections.OrderedDict()
        self.list_neuron=collections.OrderedDict()
        self.list_shape=collections.OrderedDict()
        #self.list_layer_name_write_stat = [k for k in list(self.list_layer.keys()) if not k == 'in']

        #
        self.dict_stat_r=collections.OrderedDict()  # read
        self.dict_stat_w=collections.OrderedDict()  # write
        self.dnn_act_list=collections.OrderedDict()


        #
        self.layers_w_kernel = []
        self.layers_w_neuron = []

        self.total_num_neurons = None

        self.en_record_output = None

        # input
        #self._input_shape = [-1]+input_shape.as_list()
        self._input_shape = [-1]+list(input_shape)
        #self.in_shape = [self.conf.batch_size]+self._input_shape[1:]
        self.in_shape_snn = [self.conf.batch_size] + self._input_shape[1:]


        # output
        #if False:
        self.count_accuracy_time_point=0
        self.accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
        #self.accuracy_time_point = list(tf.range(conf.time_step_save_interval,conf.time_step,delta=conf.time_step_save_interval))
        self.accuracy_time_point.append(conf.time_step)
        self.num_accuracy_time_point = len(self.accuracy_time_point)
        #self.accuracy_metrics = [None] * len(self.accuracy_time_point)
        self.accuracy_results = list(range(self.num_accuracy_time_point))
        self.accuracy_metrics = list(range(self.num_accuracy_time_point))
        self.loss_metrics = list(range(self.num_accuracy_time_point))

        #
        #self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.list_layer_name)+1])
        #self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.list_layer_name)+1])
        #self.total_residual_vmem=np.zeros(len(self.list_layer_name)+1)
        self.total_spike_count=None
        self.total_spike_count_int=None
        self.total_residual_vmem=None

        self.snn_output_neuron = None
        self.snn_output = None
        self.spike_count = None

        #
        self.activation = tf.nn.relu

        #kernel_initializer = initializers.xavier_initializer(True)
        #self.kernel_initializer = initializers.GlorotUniform()
        #Model.kernel_initializer = initializers.Zeros()
        #Model.kernel_initializer = initializers.Zeros()
        #kernel_initializer = initializers.variance_scaling_initializer(factor=2.0,mode='FAN_IN')    # MSRA init. = He init



        #pooling_type= {
        #    'max': tf.keras.layers.MaxPooling2D((2,2),(2,2),padding='SAME',data_format=data_format),
        #    'avg': tf.keras.layers.AveragePooling2D((2,2),(2,2),padding='SAME',data_format=data_format)
        #}

        #self.pool2d = pooling_type[self.conf.pooling]


        #
        self.run_mode = {
            'ANN': self.call_ann if not self.conf.f_surrogate_training_model else self.call_ann_surrogate_training,
            'SNN': self.call_snn
        }

        self.run_mode_load_model = {
            'ANN': self.call_ann if not self.conf.f_surrogate_training_model else self.call_ann_surrogate_training,
            'SNN': self.call_snn
        }

        # training mode
        #self.en_training = self.conf.training_mode
        #self.en_train = self.conf.en_train
        self.en_train = ('train' in self.conf.mode)

        # SNN mode
        #Model.en_snn = (self.conf.nn_mode == 'SNN' or self.conf.f_validation_snn)
        #self.en_snn = (self.conf.nn_mode == 'SNN' or self.conf.f_validation_snn)
        self.nn_mode = conf.nn_mode
        self.en_snn = (self.nn_mode == 'SNN' or self.conf.f_validation_snn)

        # DNN-to-SNN conversion, save dist. act. of DNN
        self.en_write_stat = (self.nn_mode=='ANN' and self.conf.f_write_stat)

        # SNN, temporal coding, time const. training after DNN-to-SNN conversion (T2FSNN + GO)
        self.en_opt_time_const_T2FSNN = (self.nn_mode=='SNN' and not self.en_train \
                                         and self.conf.neural_coding=='TEMPORAL' and self.conf.f_train_tk)

        # comparison activation - ANN vs. SNN
        self.en_comp_act = (self.nn_mode=='SNN' and self.conf.f_comp_act)


        # data-based weight normalization
        if self.conf.f_w_norm_data:
            self.norm=collections.OrderedDict()
            self.norm_b=collections.OrderedDict()

        # bias control - SNN inference
        if self.en_snn:
            self.bias_control = self.conf.bias_control
        else:
            self.bias_control = False

        # debugging
        #if self.f_debug_visual:
        if flags._run_for_visual_debug:
            #self.debug_visual_threads = []
            self.debug_visual_axes = []
            self.debug_visual_list_neuron = collections.OrderedDict()


        # stdp-pathway
        #self.en_stdp_pathway = False
        #self.en_stdp_pathway = True

        #
        #if self.en_stdp_pathway:
        if conf.en_stdp_pathway:
            if conf.model!='VGG16':
                assert False

            '''
            self.get_layer('conv1_1').set_n_pre_post('n_conv1','n_conv1_1')
            self.get_layer('conv2').set_n_pre_post('n_conv1_1','n_conv2')
            self.get_layer('conv2_1').set_n_pre_post('n_conv2','n_conv2_1')
            self.get_layer('conv3').set_n_pre_post('n_conv2_1','n_conv3')
            self.get_layer('conv3_1').set_n_pre_post('n_conv3','n_conv3_1')
            self.get_layer('conv3_2').set_n_pre_post('n_conv3_1','n_conv3_2')
            self.get_layer('conv4').set_n_pre_post('n_conv3_2','n_conv4')
            self.get_layer('conv4_1').set_n_pre_post('n_conv4','n_conv4_1')
            self.get_layer('conv4_2').set_n_pre_post('n_conv4_1','n_conv4_2')
            self.get_layer('conv5').set_n_pre_post('n_conv4_2','n_conv5')
            self.get_layer('conv5_1').set_n_pre_post('n_conv5','n_conv5_1')
            self.get_layer('conv5_2').set_n_pre_post('n_conv5_1','n_conv5_2')
            self.get_layer('fc1').set_n_pre_post('n_conv5_2','n_fc1')
            self.get_layer('fc2').set_n_pre_post('n_fc1','n_fc2')
            '''

            n_conv1 = self.get_layer('n_conv1')
            n_conv1_1 = self.get_layer('n_conv1_1')
            n_conv2 = self.get_layer('n_conv2')
            n_conv2_1 = self.get_layer('n_conv2_1')
            n_conv3 = self.get_layer('n_conv3')
            n_conv3_1 = self.get_layer('n_conv3_1')
            n_conv3_2 = self.get_layer('n_conv3_2')
            n_conv4 = self.get_layer('n_conv4')
            n_conv4_1 = self.get_layer('n_conv4_1')
            n_conv4_2 = self.get_layer('n_conv4_2')
            n_conv5 = self.get_layer('n_conv5')
            n_conv5_1 = self.get_layer('n_conv5_1')
            n_conv5_2 = self.get_layer('n_conv5_2')
            n_fc1 = self.get_layer('n_fc1')
            n_fc2 = self.get_layer('n_fc2')

            self.get_layer('conv1_1').set_n_pre_post(n_conv1,n_conv1_1)
            #self.get_layer('conv2').set_n_pre_post(n_conv1_1,n_conv2)
            self.get_layer('conv2_1').set_n_pre_post(n_conv2,n_conv2_1)
            #self.get_layer('conv3').set_n_pre_post(n_conv2_1,n_conv3)
            self.get_layer('conv3_1').set_n_pre_post(n_conv3,n_conv3_1)
            self.get_layer('conv3_2').set_n_pre_post(n_conv3_1,n_conv3_2)
            #self.get_layer('conv4').set_n_pre_post(n_conv3_2,n_conv4)
            self.get_layer('conv4_1').set_n_pre_post(n_conv4,n_conv4_1)
            self.get_layer('conv4_2').set_n_pre_post(n_conv4_1,n_conv4_2)
            #self.get_layer('conv5').set_n_pre_post(n_conv4_2,n_conv5)
            self.get_layer('conv5_1').set_n_pre_post(n_conv5,n_conv5_1)
            self.get_layer('conv5_2').set_n_pre_post(n_conv5_1,n_conv5_2)
            #self.get_layer('fc1').set_n_pre_post(n_conv5_2,n_fc1)
            self.get_layer('fc2').set_n_pre_post(n_fc1,n_fc2)


    #def init_graph(self, inputs, outputs,**kwargs):
        #super(Model, self).__init__(inputs=inputs,outputs=outputs,**kwargs)


    #def build(self, input_shape):
        #super(Model, self).build(input_shape)
        ## initialize the graph
        #img_input = tf.keras.layers.Input(shape=self.in_shape, batch_size=self.batch_size)
        #out = self.call_ann(img_input,training=False)
        #self._is_graph_network = True
        #self._init_graph_network(inputs=img_input,outputs=out)


    # TODO: move this function
    def spike_max_pool_setup(self):

        nodes_by_depth = self._nodes_by_depth
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        #print(depth_keys)
        prev_layer = None
        for depth in depth_keys:
            nodes = nodes_by_depth[depth]
            for node in nodes:

                #print(node.layer)
                if isinstance(node.layer,lib_snn.layers.MaxPool2D):
                    node.layer.prev_layer = prev_layer
                    node.layer.prev_layer_set_done  = True

                prev_layer = node.layer

        #assert False

    #
    #def set_layers_nn_mode(self):
        #for l in self.layers:
            #if isinstance(l, lib_snn.layers.Layer):
                #l.set_en_snn(self.nn_mode)


    #
    # after init
    def build_set_aside(self, input_shapes):
        #print('build lib snn - Model')

        #
        for idx_layer, layer in enumerate(self.model.layers):
            count_params = layer.count_params()

            if count_params > 0:
                self.list_layer.append(layer)

        #
        for idx_layer, layer in enumerate(self.list_layer):
            self.list_layer_name.append(layer.name)

        #
        print('Layer list')
        print(self.list_layer_name)


        # set prev_layer_name
        prev_layer = None
        for idx_layer, layer in enumerate(self.model.layers):
            layer.prev_layer = prev_layer
            prev_layer = layer


        self.list_layer_name_write_stat = self.list_layer_name



        # data-based weight normalization


        # write stat - acitvation distribution
        if self.en_write_stat:
            lib_snn.anal.init_write_stat(self)

        # train time const after DNN-to-SNN conversion (T2FSNN + GO)
        if self.conf.neural_coding=='TEMPORAL' and self.conf.f_load_time_const:
            lib_snn.ttfs_temporal_kernel.T2FSNN_load_time_const(self)

        # surrogate DNN model for SNN training w/ TTFS coding
        if self.conf.f_surrogate_training_model:
            lib_snn.ttfs_temporal_kernel.surrogate_training_setup()

        # analysis - ANN vs. SNN activation
        if self.en_comp_act:
            assert False, 'f_comp_act mode is not validated yet'
            lib_snn.anal.init_comp_act(self)

        # analysis - ISI
        if self.conf.f_isi:
            assert False, 'f_isi mode is not validated yet'
            lib_snn.anal.init_anal_isi(self)

        # analysis - entropy
        if self.conf.f_entropy:
            assert False, 'f_entropy mode is not validated yet'
            lib_snn.anal.init_anal_entropy(self)

        print(self.list_layer_name[-1])

        ########
        # snn output declare - should be after neuron setup
        ########
        #self.output_layer=self.model.get_layer(name=self.list_layer_name[-1])
        self.output_layer=self.list_layer[-1]


        if Model.en_snn:
            self.snn_output_neuron=self.output_layer.act

            self.snn_output = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_neuron.dim)),
                                          dtype=tf.float32,trainable=False,name='snn_output')
            #self.spike_count = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_neuron.dim)),dtype=tf.float32,trainable=False)


    # TODO: check
    def snn_setup_check(self):
        # output setup
        assert not (self.snn_output_neuron is None), 'snn_output_neuron should be assigned after neuron setup'
        assert not (self.snn_output is None), 'snn_output should be assigned after neuron setup'
        assert not (self.spike_count is None), 'spike_count should be assigned after neuron setup'

    ###########################################################################
    ## call
    ###########################################################################

    def call(self, inputs, training=None, mask=None):

        #ret_val = self.run_mode[self.conf.nn_mode](inputs, training, self.conf.time_step, epoch)
        #ret_val = self.run_mode[self.conf.nn_mode](inputs, training)

        #ret_val = self.run_mode[self.nn_mode](inputs,training,mask)
        #ret_val = self.call_snn(inputs,training,mask)

        ret_val = {
            #'ANN': self.call_ann if not self.conf.f_surrogate_training_model else self.call_ann_surrogate_training,
            'ANN': self.call_ann if not self.conf.f_surrogate_training_model else self.call_ann_surrogate_training,
            #'SNN': self.call_snn if not training else self.call_snn_one_time_step
            'SNN': self.call_snn
        }[self.nn_mode] (inputs,training,mask)


        return ret_val

    #
    #def __call__(self, inputs, training, epoch=-1, f_val_snn=False):
    def call_set_aside(self, inputs, training, epoch=-1, f_val_snn=False):
        #print("lib_SNN - Model - call")

        if Model.f_load_model_done:
            #print('Model - f_load_model_done')

            # pre-processing
            self.preproc(inputs,training,f_val_snn)

            # run
            if (self.en_opt_time_const_T2FSNN):
                # run ANN
                self.run_mode['ANN'](inputs,training,self.conf.time_step,epoch)

                # run SNN
                ret_val = self.run_mode[self.nn_mode](inputs,training,self.conf.time_step,epoch)

                # training time constant
                self.train_time_const()
            else:
                # inference - rate, phase, and burst coding
                if f_val_snn:
                    assert False, 'f_val_snn mode is not validated yet'
                    ret_val = self.call_snn(inputs,training,self.conf.time_step,epoch)
                else:
                    ret_val = self.run_mode[self.nn_mode](inputs,training,self.conf.time_step,epoch)


            # post-processing
            self.postproc(inputs)
        else:
            print('Dummy run')

            if self.nn_mode=='SNN' and self.conf.f_surrogate_training_model:
                ret_val = self.call_ann_surrogate_training(inputs,False,self.conf.time_step,epoch)

            # validation on SNN
            if self.conf.en_train and self.conf.f_validation_snn:
                ret_val = self.call_snn(inputs,False,1,0)

            #ret_val = self.run_mode_load_model[self.conf.nn_mode](inputs,training,self.conf.time_step,epoch)
            #ret_val = self.run_mode_load_model[self.conf.nn_mode](inputs,False,self.conf.time_step,epoch)
            ret_val = self.run_mode_load_model[self.nn_mode](inputs,False,2,epoch)

            Model.f_load_model_done=True

            #
            #if self.f_1st_iter and self.conf.nn_mode=='ANN':
                #print('1st iter - dummy run')
                #self.f_1st_iter = False
                #self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)

            self.f_1st_iter = False

            Model.f_skip_bn = (self.nn_mode=='ANN' and self.conf.f_fused_bn) or (self.nn_mode=='SNN')

        return ret_val

    #
    def call_ann(self,inputs,training=None,mask=None):
        ret = self._run_internal_graph(inputs, training=training, mask=mask)
        return ret

    #
    #@tf.function
    #def call_snn(self,inputs,training, tw, epoch):
    def call_snn(self,inputs,training=None, mask=None):


        # return tensor - [batch, time, output]

        #range_ts = range(1, self.conf.time_step + 1)
        #range_ts = range(1, int(self.conf.time_step/2.0) + 1)

        #range_ts = range(1, self.conf.time_step + 1)
        #range_ts = range(1, int(self.conf.time_step/2) + 1)
        #range_ts = range(1, 1+ 1)


        self.time_axis=0
        if self.conf.snn_training_spatial_first:
            if training:
                if self.conf.sptr:
                    # single time step
                    y_pred = self._run_internal_graph(inputs, training=training, mask=mask)
                else:
                    y_pred = self._run_internal_graph(inputs, training=training, mask=mask)
            else:
                y_pred = self._run_internal_graph_snn_s_first(inputs, training=training, mask=mask)

            ret = y_pred

            if False:
                #range_ts = range(1, 1+ 1)
                #range_ts = range(1, 1+ 1)
                range_ts = range(1, self.conf.time_step + 1)

                glb_t.reset()
                # TODO: modify
                if training:
                    for t in range_ts:
                        layer_in = inputs

                        for layer in self.layers:
                            layer_out = layer(layer_in)
                            layer_in = layer_out

                            # debug
                            if False:
                                #if True:
                                if hasattr(layer,'act') and isinstance(layer.act,lib_snn.neurons.Neuron):
                                    spike_count = layer.act.spike_count
                                    spike = layer.act.out
                                    print('spike count> {}: - sum {:.3e}, mean {:.3e}'.format(layer.name,tf.reduce_sum(spike_count),tf.reduce_mean(spike_count)))
                                    print('spike - sum {:.3e}, mean {:.3e}'.format(tf.reduce_sum(spike),tf.reduce_mean(spike)))
                        glb_t()

                    ret = layer_out

        #else:   # temporal first - new
        #elif False:
        elif True:


            #
            #inputs_e = tf.TensorArray(
            #                dtype=inputs.dtype,
            #                size=self.conf.time_step,
            #                element_shape=inputs.shape,
            #                tensor_array_name='layer_in')
#
#            for t_in_write in range_ts:
#                inputs_e=inputs_e.write(t_in_write-1,inputs)

            #
            y_pred = self._run_internal_graph_snn_t_first(inputs, training=training, mask=mask)

            #ret = y_pred[-1]
            #ret = y_pred.read(self.conf.time_step-1)
            ret = y_pred

            #





        else:   # temporal first - old
            #inputs_expand_shape = [self.conf.time_step,]+inputs.shape
            #inputs_e = tf.broadcast_to(inputs,shape=inputs_expand_shape)
            #print('input expand - time axis: {:}'.format(self.time_axis))

            range_ts = range(1, self.conf.time_step+ 1)


            #
            layer_in = tf.TensorArray(
                            dtype=inputs.dtype,
                            size=self.conf.time_step,
                            element_shape=inputs.shape,
                            tensor_array_name='layer_in')

            for t_in_write in range_ts:
                layer_in=layer_in.write(t_in_write-1,inputs)

            #out = self._run_internal_graph(inputs, training=training, mask=mask)

            #layer_in = inputs_e
            for layer in self.layers:
                #print(layer.name)
                #print(layer)
                if False:
                    if hasattr(layer,'output_shape_fixed_batch'):
                        layer_out = tf.zeros([self.conf.time_step,]+layer.output_shape_fixed_batch)
                    else:
                        #if isinstance(layer,tf.keras.layers.Input):
                        #if not isinstance(layer,keras.engine.input_layer.InputLayer):
                        #    assert False

                        assert isinstance(layer,keras.engine.input_layer.InputLayer) \
                               or isinstance(layer,keras.layers.regularization.dropout.Dropout) \
                               or isinstance(layer,keras.layers.reshaping.flatten.Flatten)

                        layer_out = tf.zeros(layer_in.shape)

                output_shape = layer.output_shape
                if isinstance(layer.output_shape, list):
                    output_shape=output_shape[0]
                #layer_out = tf.zeros((self.conf.time_step,)+output_shape)
                #layer_out = tf.zeros_like(self.conf.time_step)
                #layer_out = []

                # reset
                glb_t.reset()
                f_time_reduction = False    # flag - time reduction


                #
                #print(layer)
                if isinstance(layer,
                              layers_new.batch_normalization.BatchNormalization):
                    #print('aa')
                    f_time_reduction = True


                if f_time_reduction:
                    layer_in_mean_t = tf.reduce_mean(layer_in.stack(),axis=self.time_axis)
                    _layer_in = layer_in_mean_t

                    # run
                    _layer_out = layer(_layer_in)

                    #
                    output_ta_t = tf.TensorArray(
                        dtype=_layer_out.dtype,
                        size=self.conf.time_step,
                        element_shape=_layer_out.shape,
                        clear_after_read=false,
                        tensor_array_name='output_ta_t')

                    for t in range_ts:
                        output_ta_t = output_ta_t.write(t-1,_layer_out)

                else:
                    if False:
                        #
                        # time step 0
                        t=0
                        #_layer_in = layer_in[0]
                        _layer_in = layer_in.read(0)

                        # run
                        _layer_out = layer(_layer_in)

                        output_ta_t = tf.TensorArray(
                                dtype=_layer_out.dtype,
                                size=self.conf.time_step,
                                element_shape=_layer_out.shape,
                                clear_after_read=False,
                                tensor_array_name='output_ta_t')


                        #output_ta_t = tuple(
                        #    ta.write(t, out)
                        #    for ta, out in zip(output_ta_t, _layer_out))
                        output_ta_t = output_ta_t.write(t,_layer_out)

                        glb_t()

                    #
                    for t in range_ts:
                        #_layer_in = layer_in[t-1]
                        _layer_in = layer_in.read(t-1)

                        # run
                        _layer_out = layer(_layer_in)

                        #
                        #print(t)
                        if t-1==0:
                            output_ta_t = tf.TensorArray(
                                dtype=_layer_out.dtype,
                                size=self.conf.time_step,
                                element_shape=_layer_out.shape,
                                clear_after_read=False,
                                tensor_array_name='output_ta_t')

                        #_layer_out = tf.expand_dims(_layer_out,axis=self.time_axis)

                        #layer_out.append(_layer_out)

                        #output_ta_t = tuple(
                        #    ta.write(t, out)
                        #    for ta, out in zip(output_ta_t, _layer_out))

                        output_ta_t = output_ta_t.write(t-1,_layer_out)

                        #indices = tf.constant([[t-1,:]])
                        #updates = _layer_out
                        #layer_out = tf.tensor_scatter_nd_update(layer_out,indices,updates)

                        glb_t()

                    #layer_out = tf.stack([lo for lo in layer_out],axis=0)
                layer_out = output_ta_t
                layer_in = layer_out


            ret = layer_out.read(self.conf.time_step-1)
            #print(layer_out.read(7))

        return ret



        # plot control
        #f_plot = (self.conf.verbose_visual) and (not self.conf.full_test) and (glb.model_compiled) and (self.conf.debug_mode and self.conf.nn_mode == 'SNN')
        f_plot = (flags._run_for_visual_debug) and (not self.conf.full_test) and (glb.model_compiled) and (self.conf.debug_mode and self.nn_mode == 'SNN')
        #f_plot = f_plot and (self.conf.num_test_data==1)

        # tf.expand_dims(self.bias_ctrl_sub,axis=(1,2))
        if self.bias_control:
            self.bias_control_run_pre()

        #
        #for t in range(1,self.conf.time_step+1):
        if self.conf.full_test:
            range_ts = range(1, self.conf.time_step + 1)
        else:
            range_ts = tqdm(range(1, self.conf.time_step + 1),desc="SNN Run")

        #for t in tqdm(range(1, self.conf.time_step + 1),desc="SNN Run"):
        for t in range_ts:
            #self.bias_control(t)

            #self.bias_disable()

            #if not self.conf.full_test:
            #    print('time: {}'.format(t))

            #ret = self._run_internal_graph(inputs, training=training, mask=mask)
            y_pred = self._run_internal_graph(inputs, training=training, mask=mask)

            #
            #print(t)
            #print(self.count_accuracy_time_point)
            # TODO:
            if self.conf.mode=='inference' and \
                self.init_done and (t == self.accuracy_time_point[self.count_accuracy_time_point]):
            #if (t == self.accuracy_time_point[self.count_accuracy_time_point]):
                self.record_acc_spike_time_point(inputs,y_pred)


            # plot output
            #self.plot_output()

            #
            #if (glb.model_compiled) and (self.conf.debug_mode and self.conf.nn_mode == 'SNN'):
            if f_plot:

                from lib_snn.sim import glb_plot
                from lib_snn.sim import glb_plot_1
                from lib_snn.sim import glb_plot_2
                from lib_snn.sim import glb_plot_3
                from lib_snn.sim import glb_plot_1x2

                self.plot_layer_neuron_act(glb_plot)
                self.plot_layer_neuron_vmem(glb_plot_1)
                self.plot_layer_neuron_out(glb_plot_2)
                self.plot_layer_neuron_input(glb_plot_3)

                self.plot_logit_t_and_accum(glb_plot_1x2)

            if self.bias_control:
                #self.bias_control_run()
                self.bias_control_run_dynamic_bn()

            #
            if False:
                if (glb.model_compiled) and (not self.conf.full_test):
                    l = self.get_layer('predictions')
                    #print(l.bias_en_time)
                    #print(self.conf.time_step)
                    time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
                    #print(time)
                    print('')
                    print('logit - {} - {}'.format(t,self.get_layer('predictions').record_output/time))

            if t==self.conf.time_step:
                ret=y_pred

            # end of time step - increase global time
            glb_t()

        return ret


    #
    def call_snn_one_time_step(self, inputs, training=None, mask=None):

        y_pred = self._run_internal_graph(inputs, training=training, mask=mask)

        return y_pred

    #
    def bias_control_run_pre(self):
        #print("bias_control_reset")
        if (glb.model_compiled) and (self.conf.debug_mode and self.nn_mode == 'SNN'):
            #for idx_layer, layer in enumerate(self.layers_w_neuron):
            #for idx_layer, layer in enumerate(self.layers_w_kernel):
            for idx_layer, layer in enumerate(self.layers_bias_control):
                layer.f_bias_ctrl = tf.fill(tf.shape(layer.f_bias_ctrl), True)
                # print(layer.f_bias_ctrl)
                # assert False

                layer.use_bias = self.conf.use_bias

                # if (idx_layer == 0) or (self.conf.input_spike_mode and idx_layer==1) :
                #if (idx_layer == 0) or (idx_layer == 1):
                if (idx_layer == 0):
                    if self.conf.use_bias:
                        layer.bias_en_time = 0
                        # layer.f_bias_ctrl = False
                        layer.f_bias_ctrl = tf.fill(tf.shape(layer.f_bias_ctrl), False)
                else:
                    # layer.use_bias = T
                    layer.f_bias_ctrl = tf.fill(tf.shape(layer.f_bias_ctrl), True)
                    layer.bias_ctrl_sub = tf.broadcast_to(layer.bias, layer.output_shape_fixed_batch)

            #
            #for idx_layer, layer in enumerate(self.layers_bias_control):
            #    if idx_layer == 0:
            #        continue
            #    print('{} - prev: {}'.format(layer.name,self.prev_layer_name[layer.name]))
            #
            #assert False

    #
    def bias_control_run(self):

        bias_control_level = 'layer'
        #bias_control_level = 'channel'

        if (glb.model_compiled) and (self.conf.debug_mode and self.nn_mode == 'SNN'):
            # print('fired neuron')

            if bias_control_level=='layer':
                #for idx_layer, layer in enumerate(self.layers_w_neuron):
                #for idx_layer, layer in enumerate(self.layers_w_kernel):
                for idx_layer, layer in enumerate(self.layers_bias_control):
                    # if layer.use_bias != self.conf.use_bias:
                    # print(layer.use_bias)
                    # print(tf.reduce_any(layer.f_bias_ctrl))
                    if layer.use_bias == tf.reduce_any(layer.f_bias_ctrl):

                        if 'VGG' in self.name:
                            prev_layer = self.layers_bias_control[idx_layer - 1]
                            #prev_layer = self.layers_w_neuron[idx_layer - 1]
                        elif 'ResNet' in self.name:
                            prev_layer_name = self.prev_layer_name_bias_control[layer.name]
                            prev_layer = self.get_layer(prev_layer_name)
                            #prev_layer = self.layers_bias_control[idx_layer - 1]
                        else:
                            assert False

                        #print('test here')
                        #print(layer.name)
                        #print('prev - {}'.format(prev_layer.name))
                        #print('layer - {}'.format(layer.name))
                        #print(prev_layer.name)
                        #print(prev_layer.act.dim)

                        if isinstance(prev_layer, lib_snn.layers.Conv2D):
                            axis = [1, 2, 3]
                        elif isinstance(prev_layer, lib_snn.layers.Dense):
                            axis = [1]
                        else:
                            #print(prev_layer)
                            #print(layer)
                            #print(prev_layer.act)
                            #print(prev_layer.act.dim)
                            #print('prev_layer: {}'.format(prev_layer.name))
                            #print('layer: {}'.format(layer.name))
                            if len(prev_layer.act.dim)==4:
                                axis = [1,2,3]
                            else:
                                assert False

                        n_neurons = prev_layer.act.num_neurons

                        #
                        # spike ratio
                        # spike = tf.reduce_sum(self.layers_w_neuron[idx_layer-1].act.spike_count_int,axis=axis)
                        spike = tf.reduce_sum(prev_layer.act.spike_count_int, axis=axis)
                        #spike = tf.math.count_nonzero(prev_layer.act.spike_count_int, dtype=tf.float32, axis=axis)

                        # num spike neurons
                        #spike = tf.math.count_nonzero(prev_layer.act.spike_count_int, axis=axis)
                        #spike = tf.cast(spike,tf.float32)

                        f_spike = tf.greater(spike / n_neurons, self.bias_control_th[layer.name])

                        # layer.f_bias_ctrl = tf.greater(spike/n_neurons,rate_bias_on)

                        #print(f_spike)
                        # print(f_spike.shape)
                        # print(layer.f_bias_ctrl)
                        # assert False

                        if tf.reduce_any(f_spike):
                            # if layer.f_bias_ctrl
                            #print('{} - {}: bias on - control off'.format(glb_t.t, layer.name))
                            # layer.use_bias = f_spike
                            layer.bias_en_time = glb_t.t
                            layer.f_bias_ctrl = tf.math.logical_not(f_spike)

                            #
                            if self.conf.leak_off_after_bias_en:
                                if isinstance(layer.act,lib_snn.neurons.Neuron):
                                #if isinstance(layer.act, lib_snn.neurons.Neuron) and (layer.name!='predictions'):
                                    layer.act.set_leak_const(tf.ones(layer.act.leak_const.shape))

                                if 'block' in layer.name:
                                    conv_block_name = layer.name.split('_')
                                    conv_name = conv_block_name[2]
                                    conv_block_name = conv_block_name[0] + '_' + conv_block_name[1]

                                    if 'conv2' in conv_name:
                                        conv_block_out = conv_block_name + '_out'
                                        layer_conv_block_out = self.get_layer(conv_block_out)
                                        layer_conv_block_out.act.set_leak_const(
                                            tf.ones(layer_conv_block_out.act.leak_const.shape))


                            if isinstance(layer, lib_snn.layers.Conv2D):
                                ctrl = tf.expand_dims(layer.f_bias_ctrl, axis=1)
                                ctrl = tf.expand_dims(ctrl, axis=2)
                                ctrl = tf.expand_dims(ctrl, axis=3)
                            elif isinstance(layer, lib_snn.layers.Dense):
                                ctrl = tf.expand_dims(layer.f_bias_ctrl, axis=1)
                            elif len(prev_layer.act.dim) == 4:
                                ctrl = tf.expand_dims(layer.f_bias_ctrl, axis=1)
                                ctrl = tf.expand_dims(ctrl, axis=2)
                                ctrl = tf.expand_dims(ctrl, axis=3)
                            else:
                                assert False

                            bias_batch = tf.broadcast_to(layer.bias, layer.bias_ctrl_sub.shape)

                            # layer.bias_ctrl_sub = tf.where(layer.f_bias_ctrl,layer)
                            layer.bias_ctrl_sub = tf.where(ctrl, bias_batch, tf.zeros(layer.bias_ctrl_sub.shape))
            elif bias_control_level == 'channel':
                assert False, 'only vgg implemented'
                for idx_layer, layer in enumerate(self.layers_bias_control):
                    if layer.use_bias == tf.reduce_any(layer.f_bias_ctrl):
                        prev_layer = self.layers_bias_control[idx_layer - 1]

                        if isinstance(prev_layer, lib_snn.layers.Conv2D):
                            axis_reduce_batch = [1, 2]
                            axis = [1, 2]

                            spike = tf.math.count_nonzero(prev_layer.act.spike_count_int, dtype=tf.float32, axis=axis_reduce_batch)

                            n_neurons = tf.gather(prev_layer.act.dim, axis)
                            n_neurons = tf.reduce_prod(n_neurons)
                            n_neurons = tf.cast(n_neurons, dtype=tf.float32)

                        elif isinstance(prev_layer, lib_snn.layers.Dense):
                            axis_reduce_batch = [1]
                            axis = [1]

                            spike = prev_layer.act.spike_count_int

                            n_neurons = prev_layer.act.dim[1]
                        else:
                            assert False

                        # spike = tf.reduce_sum(self.layers_w_neuron[idx_layer-1].act.spike_count_int,axis=axis)
                        #spike = tf.reduce_sum(prev_layer.act.spike_count_int, axis=axis_reduce_batch)
                        #spike = tf.math.count_nonzero(prev_layer.act.spike_count_int, dtype=tf.float32, axis=axis_reduce_batch)

                        #assert False


                        #r_spike = tf.expand_dims(spike/n_neurons,axis=0)
                        r_spike = spike/n_neurons
                        f_spike = tf.greater(r_spike, self.bias_control_th_ch[prev_layer.name])

                        # layer.f_bias_ctrl = tf.greater(spike/n_neurons,rate_bias_on)

                        #print(f_spike)
                        print(layer.name)
                        print(f_spike.shape)
                        # print(layer.f_bias_ctrl)
                        # assert False

                        if tf.reduce_any(f_spike):
                            # if layer.f_bias_ctrl
                            #print('{} - {}: bias on - control off'.format(glb_t.t, layer.name))
                            # layer.use_bias = f_spike
                            layer.bias_en_time = glb_t.t
                            layer.f_bias_ctrl = tf.math.logical_not(f_spike)

                            if isinstance(layer, lib_snn.layers.Conv2D):
                                ctrl = tf.expand_dims(layer.f_bias_ctrl, axis=1)
                                ctrl = tf.expand_dims(ctrl, axis=2)
                                #ctrl = tf.expand_dims(ctrl, axis=3)
                            elif isinstance(layer, lib_snn.layers.Dense):
                                ctrl = layer.f_bias_ctrl
                            #    ctrl = tf.expand_dims(layer.f_bias_ctrl, axis=1)
                            #else:
                            #    assert False

                            #assert False

                            bias_batch = tf.broadcast_to(layer.bias, layer.bias_ctrl_sub.shape)

                            # layer.bias_ctrl_sub = tf.where(layer.f_bias_ctrl,layer)
                            layer.bias_ctrl_sub = tf.where(ctrl, bias_batch, tf.zeros(layer.bias_ctrl_sub.shape))

            else:
                assert False

    #
    def bias_control_run_dynamic_bn(self):

        bias_control_level = 'layer'
        #bias_control_level = 'channel'

        if (glb.model_compiled) and (self.conf.debug_mode and self.nn_mode == 'SNN'):
            # print('fired neuron')

            # channel-wise only
            #if bias_control_level=='layer' :
            if True:
                #for idx_layer, layer in enumerate(self.layers_w_neuron):
                #for idx_layer, layer in enumerate(self.layers_w_kernel):
                for idx_layer, layer in enumerate(self.layers_bias_control):
                    # if layer.use_bias != self.conf.use_bias:
                    # print(layer.use_bias)
                    # print(tf.reduce_any(layer.f_bias_ctrl))
                    if layer.use_bias == tf.reduce_any(layer.f_bias_ctrl):

                        if 'VGG' in self.name:
                            prev_layer = self.layers_bias_control[idx_layer - 1]
                            #prev_layer = self.layers_w_neuron[idx_layer - 1]
                        elif 'ResNet' in self.name:
                            prev_layer_name = self.prev_layer_name_bias_control[layer.name]
                            prev_layer = self.get_layer(prev_layer_name)
                            #prev_layer = self.layers_bias_control[idx_layer - 1]
                        else:
                            assert False

                        #print('test here')
                        #print(layer.name)
                        #print('prev - {}'.format(prev_layer.name))
                        #print('layer - {}'.format(layer.name))
                        #print(prev_layer.name)
                        #print(prev_layer.act.dim)

                        if isinstance(prev_layer, lib_snn.layers.Conv2D):
                            axis = [1, 2, 3]
                            #axis = [1, 2]
                        elif isinstance(prev_layer, lib_snn.layers.Dense):
                            axis = [1]
                            #pass
                        else:
                            #print(prev_layer)
                            #print(layer)
                            #print(prev_layer.act)
                            #print(prev_layer.act.dim)
                            #print('prev_layer: {}'.format(prev_layer.name))
                            #print('layer: {}'.format(layer.name))
                            if len(prev_layer.act.dim)==4:
                                axis = [1,2,3]
                                #axis = [1,2]
                            else:
                                assert False

                        n_neurons = prev_layer.act.num_neurons


                        spike = prev_layer.act.spike_count_int
                        #
                        # spike ratio
                        # spike = tf.reduce_sum(self.layers_w_neuron[idx_layer-1].act.spike_count_int,axis=axis)
                        #spike = tf.reduce_sum(prev_layer.act.spike_count_int, axis=axis)
                        #spike = tf.math.count_nonzero(prev_layer.act.spike_count_int, dtype=tf.float32, axis=axis)

                        # num spike neurons
                        #spike = tf.math.count_nonzero(prev_layer.act.spike_count_int, axis=axis)
                        #spike = tf.cast(spike,tf.float32)


                        #spike = tf.reduce_mean()

                        # new method - dnn activation based
                        #dnn_act_pre = self.model_ann.get_layer(prev_layer.name).record_output

                        # dnn_act stat
                        #dnn_act_pre = lib_snn.calibration.read_stat(None, prev_layer, 'mean')
                        #dnn_act_pre = tf.broadcast_to(tf.expand_dims(dnn_act_pre,axis=0),spike.shape)
                        #dnn_act_pre = dnn_act_pre*0.05

                        # random
                        if glb_t.t==1:
                        #if glb_t.t%10 == 1:
                        #if True:
                            mean=prev_layer.bn.beta
                            std=prev_layer.bn.gamma
                            dnn_act_pre = tf.random.normal(spike.shape,mean=mean,stddev=std)
                            #dnn_act_pre = dnn_act_pre*0.05
                            dnn_act_pre = dnn_act_pre*self.conf.dynamic_bn_dnn_act_scale
                            dnn_act_pre = tf.clip_by_value(dnn_act_pre,0.0,1.0)

                            layer.dnn_act_pre = dnn_act_pre
                        else:
                            dnn_act_pre = layer.dnn_act_pre

                        #dnn_act_pre = tf.reduce_mean(dnn_act_pre,axis=0,keepdims=True)
                        #dnn_act_pre = tf.broadcast_to(dnn_act_pre,spike.shape)

                        # dnn_act
                        #dnn_act_pre = tf.reduce_mean(dnn_act_pre, axis=axis)

                        c_dnn_act_non_zero = tf.cast(tf.math.count_nonzero(dnn_act_pre,axis=axis), tf.float32)

                        #diff = tf.abs(tf.subtract(dnn_act_pre,spike/self.conf.time_step))
                        #diff = tf.abs(tf.subtract(dnn_act_pre,spike/glb_t.t))

                        diff = tf.subtract(dnn_act_pre,spike/glb_t.t)
                        #diff = tf.clip_by_value(diff,0.0,dnn_act_pre)
                        diff = tf.math.pow(diff,2.0)

                        #diff = tf.math.pow(tf.subtract(dnn_act_pre,spike/glb_t.t),2.0)
                        #diff = tf.math.pow(tf.subtr(dnn_act_pre,spike/glb_t.t)/2.0,2.0)
                        #diff = tf.math.pow(tf.subtract(dnn_act_pre,spike/glb_t.t),3.0)
                        #diff = tf.math.pow(tf.subtract(dnn_act_pre,spike/glb_t.t),4.0)
                        #diff = tf.math.pow(tf.subtract(dnn_act_pre,spike/glb_t.t)/2.0,4.0)
                        #diff_norm = tf.where(tf.equal(dnn_act_pre,0.0),tf.zeros(dnn_act_pre.shape),diff/dnn_act_pre)
                        diff_norm = tf.math.divide_no_nan(diff,dnn_act_pre)
                        diff_norm = tf.clip_by_value(diff_norm,0.0,1.0)
                        #diff_mean = tf.reduce_mean(diff_norm, axis=axis)
                        diff_sum = tf.reduce_sum(diff_norm, axis=axis)
                        diff_mean = tf.math.divide(diff_sum,c_dnn_act_non_zero)

                        #self.dnn_act_pre=dnn_act_pre
                        #self.spike=spike
                        #print(layer.name)
                        #assert False

                        #bias_ctrl = (1-tf.abs(tf.subtract(dnn_act_pre,spike/self.conf.time_step))/dnn_act_pre)
                        #bias_ctrl = (1-diff_mean)
                        bias_ctrl = diff_mean
                        #bias_ctrl = tf.clip_by_value(bias_ctrl,0.0,1.0)
                        #assert False

                        #bias_ctrl_avg = tf.reduce_mean(bias_ctrl,axis)


                        #if layer.name=='conv1_1':
                        #if False:
                            #print('time at {}'.format(glb_t.t))
                            #print('spike/t at {}'.format(glb_t.t))
                            #print(spike[0,0,0]/glb_t.t)
                            #print('dnn_act')
                            #print(dnn_act_pre[0,0,0])
                            #print('diff')
                            #print(diff[0,0,0])
                            #print('diff_norm')
                            #print(diff_norm[0,0,0])
                            #print('diff_sum')
                            #print(diff_sum)
                            #print('bias_ctrl')
                            #print(bias_ctrl)
                            #print('bias_ctrl (avg)')
                            #print(bias_ctrl_avg)
                            #print('c_dnn_act_non_zero')
                            #print(c_dnn_act_non_zero)

                        #print(dnn_act_pre)
                        #print(spike/self.conf.time_step)
                        #if

                        bias = layer.bias

                        #bias_batch = tf.expand_dims(bias,axis=0)
                        #if False: # layer-wise
                        if True: # layer-wise
                            bias_ctrl = tf.expand_dims(bias_ctrl,axis=1)
                            bias_ctrl = tf.broadcast_to(bias_ctrl,[bias_ctrl.shape[0],bias.shape[0]])

                        bias_batch = tf.expand_dims(bias,axis=0)
                        bias_batch = tf.broadcast_to(bias_batch,bias_ctrl.shape)


                        bias_fmap = tf.multiply(bias_batch,bias_ctrl)


                        if glb_t.t == 1:
                            #layer.init_bias_ctrl_avg = bias_ctrl_avg
                            layer.init_bias_ctrl = bias_ctrl
                        else:
                            #if bias_ctrl_avg < layer.init_bias_ctrl_avg*0.5:
                            #    if layer.bias_en_time==0:
                            #        layer.bias_en_time = glb_t.t

                            #layer.bias_ctrl_sub = tf.zeros(layer.bias_ctrl_sub.shape)
                            #layer.bias_ctrl_sub = tf.where(bias_ctrl<layer.init_bias_ctrl, \
                            #                               bias_fmap, tf.zeros(layer.bias_ctrl_sub.shape))

                            bias_fmap = tf.where(bias_ctrl<layer.init_bias_ctrl*self.conf.dynamic_bn_test_const, \
                                                           bias_fmap, tf.zeros(bias_fmap.shape))


                        if isinstance(layer, lib_snn.layers.Conv2D):
                            #bias = tf.expand_dims(bias,axis=0)
                            bias_fmap = tf.expand_dims(bias_fmap,axis=1)
                            bias_fmap = tf.expand_dims(bias_fmap,axis=2)
                        elif isinstance(layer, lib_snn.layers.Dense):
                            #bias = tf.expand_dims(bias,axis=0)
                            pass
                        elif len(prev_layer.act.dim) == 4:
                            #bias = tf.expand_dims(bias,axis=0)
                            bias_fmap = tf.expand_dims(bias_fmap,axis=1)
                            bias_fmap = tf.expand_dims(bias_fmap,axis=2)
                        else:
                            assert False


                        #bias_batch = tf.broadcast_to(bias, layer.bias_ctrl_sub.shape)
                        #bias_batch = bias_batch*bias_ctrl

                        bias_fmap = tf.broadcast_to(bias_fmap, layer.bias_ctrl_sub.shape)

                        layer.bias_ctrl_sub = bias_fmap


                        if False:

                            f_spike = tf.greater(spike / n_neurons, self.bias_control_th[layer.name])

                            # layer.f_bias_ctrl = tf.greater(spike/n_neurons,rate_bias_on)

                            #print(f_spike)
                            # print(f_spike.shape)
                            # print(layer.f_bias_ctrl)
                            # assert False

                            if tf.reduce_any(f_spike):
                                # if layer.f_bias_ctrl
                                #print('{} - {}: bias on - control off'.format(glb_t.t, layer.name))
                                # layer.use_bias = f_spike
                                layer.bias_en_time = glb_t.t
                                layer.f_bias_ctrl = tf.math.logical_not(f_spike)

                                #
                                if self.conf.leak_off_after_bias_en:
                                    if isinstance(layer.act,lib_snn.neurons.Neuron):
                                        #if isinstance(layer.act, lib_snn.neurons.Neuron) and (layer.name!='predictions'):
                                        layer.act.set_leak_const(tf.ones(layer.act.leak_const.shape))

                                    if 'block' in layer.name:
                                        conv_block_name = layer.name.split('_')
                                        conv_name = conv_block_name[2]
                                        conv_block_name = conv_block_name[0] + '_' + conv_block_name[1]

                                        if 'conv2' in conv_name:
                                            conv_block_out = conv_block_name + '_out'
                                            layer_conv_block_out = self.get_layer(conv_block_out)
                                            layer_conv_block_out.act.set_leak_const(
                                                tf.ones(layer_conv_block_out.act.leak_const.shape))


                                if isinstance(layer, lib_snn.layers.Conv2D):
                                    ctrl = tf.expand_dims(layer.f_bias_ctrl, axis=1)
                                    ctrl = tf.expand_dims(ctrl, axis=2)
                                    ctrl = tf.expand_dims(ctrl, axis=3)
                                elif isinstance(layer, lib_snn.layers.Dense):
                                    ctrl = tf.expand_dims(layer.f_bias_ctrl, axis=1)
                                elif len(prev_layer.act.dim) == 4:
                                    ctrl = tf.expand_dims(layer.f_bias_ctrl, axis=1)
                                    ctrl = tf.expand_dims(ctrl, axis=2)
                                    ctrl = tf.expand_dims(ctrl, axis=3)
                                else:
                                    assert False

                                bias_batch = tf.broadcast_to(layer.bias, layer.bias_ctrl_sub.shape)

                                # layer.bias_ctrl_sub = tf.where(layer.f_bias_ctrl,layer)
                                layer.bias_ctrl_sub = tf.where(ctrl, bias_batch, tf.zeros(layer.bias_ctrl_sub.shape))


            elif bias_control_level == 'channel':
                assert False, 'only vgg implemented'
                for idx_layer, layer in enumerate(self.layers_bias_control):
                    if layer.use_bias == tf.reduce_any(layer.f_bias_ctrl):
                        prev_layer = self.layers_bias_control[idx_layer - 1]

                        if isinstance(prev_layer, lib_snn.layers.Conv2D):
                            axis_reduce_batch = [1, 2]
                            axis = [1, 2]

                            spike = tf.math.count_nonzero(prev_layer.act.spike_count_int, dtype=tf.float32, axis=axis_reduce_batch)

                            n_neurons = tf.gather(prev_layer.act.dim, axis)
                            n_neurons = tf.reduce_prod(n_neurons)
                            n_neurons = tf.cast(n_neurons, dtype=tf.float32)

                        elif isinstance(prev_layer, lib_snn.layers.Dense):
                            axis_reduce_batch = [1]
                            axis = [1]

                            spike = prev_layer.act.spike_count_int

                            n_neurons = prev_layer.act.dim[1]
                        else:
                            assert False

                        # spike = tf.reduce_sum(self.layers_w_neuron[idx_layer-1].act.spike_count_int,axis=axis)
                        #spike = tf.reduce_sum(prev_layer.act.spike_count_int, axis=axis_reduce_batch)
                        #spike = tf.math.count_nonzero(prev_layer.act.spike_count_int, dtype=tf.float32, axis=axis_reduce_batch)

                        #assert False


                        #r_spike = tf.expand_dims(spike/n_neurons,axis=0)
                        r_spike = spike/n_neurons
                        f_spike = tf.greater(r_spike, self.bias_control_th_ch[prev_layer.name])

                        # layer.f_bias_ctrl = tf.greater(spike/n_neurons,rate_bias_on)

                        #print(f_spike)
                        print(layer.name)
                        print(f_spike.shape)
                        # print(layer.f_bias_ctrl)
                        # assert False

                        if tf.reduce_any(f_spike):
                            # if layer.f_bias_ctrl
                            #print('{} - {}: bias on - control off'.format(glb_t.t, layer.name))
                            # layer.use_bias = f_spike
                            layer.bias_en_time = glb_t.t
                            layer.f_bias_ctrl = tf.math.logical_not(f_spike)

                            if isinstance(layer, lib_snn.layers.Conv2D):
                                ctrl = tf.expand_dims(layer.f_bias_ctrl, axis=1)
                                ctrl = tf.expand_dims(ctrl, axis=2)
                                #ctrl = tf.expand_dims(ctrl, axis=3)
                            elif isinstance(layer, lib_snn.layers.Dense):
                                ctrl = layer.f_bias_ctrl
                            #    ctrl = tf.expand_dims(layer.f_bias_ctrl, axis=1)
                            #else:
                            #    assert False

                            #assert False

                            bias_batch = tf.broadcast_to(layer.bias, layer.bias_ctrl_sub.shape)

                            # layer.bias_ctrl_sub = tf.where(layer.f_bias_ctrl,layer)
                            layer.bias_ctrl_sub = tf.where(ctrl, bias_batch, tf.zeros(layer.bias_ctrl_sub.shape))

            else:
                assert False

    #
    def plot_layer_neuron_act(self,plot):
        for idx, layer_name in enumerate(plot.layers):
            layer = self.get_layer(layer_name)
            idx_neuron = plot.idx_neurons[idx]
            axe = plot.axes.flatten()[idx]
            out = layer.act.get_spike_count_int().numpy().flatten()[idx_neuron]  # spike
            lib_snn.util.plot(glb_t.t, out / (glb_t.t-layer.bias_en_time), axe=axe, mark=plot.mark)
    #
    def plot_layer_neuron_vmem(self,plot):
        for idx, layer_name in enumerate(plot.layers):
            layer = self.get_layer(layer_name)
            idx_neuron = plot.idx_neurons[idx]
            axe = plot.axes.flatten()[idx]
            vmem = layer.act.vmem.numpy().flatten()[idx_neuron]
            lib_snn.util.plot(glb_t.t, vmem, axe=axe, mark=plot.mark)

    #
    def plot_layer_neuron_out(self,plot):
        for idx, layer_name in enumerate(plot.layers):
            layer = self.get_layer(layer_name)
            idx_neuron = plot.idx_neurons[idx]
            axe = plot.axes.flatten()[idx]
            out = layer.act.out.numpy().flatten()[idx_neuron]
            lib_snn.util.plot(glb_t.t, out, axe=axe, mark=plot.mark)

    #
    def plot_layer_neuron_input(self, plot):
        for idx, layer_name in enumerate(plot.layers):
            layer = self.get_layer(layer_name)
            idx_neuron = plot.idx_neurons[idx]
            axe = plot.axes.flatten()[idx]
            inputs = layer.act.inputs.numpy().flatten()[idx_neuron]
            lib_snn.util.plot(glb_t.t, inputs, axe=axe, mark=plot.mark)

            # plot bias
            if glb_t.t==1:
                idx_bias = idx_neuron % layer.bias.shape[0]
                axe.axhline(y=layer.bias[idx_bias], color='m')


    def plot_logit_t_and_accum(self, plot):

        axe_logit_t = plot.axes.flatten()[0]
        axe_logit_accum = plot.axes.flatten()[1]

        logit_idx_start=0
        logit_idx_end=6

        #logit_idx_start=0
        #logit_idx_end=9

        logit_t = self.get_layer('predictions').record_logit[self.conf.verbose_visual_idx].numpy().flatten()[logit_idx_start:logit_idx_end+1]
        logit_accum = self.get_layer('predictions').record_output[self.conf.verbose_visual_idx].numpy().flatten()[logit_idx_start:logit_idx_end+1]

        #colors = matplotlib.cm.rainbow(np.linspace(0,1,len(logit_t)))
        colors = np.arange(logit_idx_start,logit_idx_end+1)
        t = np.full(len(logit_t),glb_t.t)


        #lib_snn.util.plot(glb_t.t, logit_t, axe=axe_logit_t, mark=plot.mark)
        #lib_snn.util.plot(glb_t.t, logit_accum, axe=axe_logit_accum, mark=plot.mark)
        scatter_logit_t = lib_snn.util.scatter(t, logit_t, axe=axe_logit_t, s=10, color=colors, marker='o')
        scatter_logit_accum = lib_snn.util.scatter(t, logit_accum, axe=axe_logit_accum, s=10, color=colors, marker='o')

        legend_handle_logit_t, legend_label_logit_t = scatter_logit_t.legend_elements(prop="colors")
        legend_logit_t = axe_logit_t.legend(legend_handle_logit_t,legend_label_logit_t)

        legend_handle_logit_accum, legend_label_logit_accum = scatter_logit_accum.legend_elements(prop="colors")
        legend_logit_accum = axe_logit_accum.legend(legend_handle_logit_accum,legend_label_logit_accum)


        #legend = [str(i) for i in range(0,len(logit_t))]
        #axe_logit_t.legend(colors,legend)
        #axe_logit_accum.add_artist()
        #if glb_t.t==0:
        #axe_logit_t.legend(legend)
        #axe_logit_accum.legend(legend)


    # this function is based on Model.test_step in training.py
    # TODO: override Model.test_step
    def test_step_a(self, data):

        ret = {
            'ANN': self.test_step_ann,
            'SNN': self.test_step_snn,
        }[self.nn_mode](data)

        return ret

    # from keras training.py
    def test_step_ann(self, data):
        """The logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.

        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.

        Args:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def test_step_snn(self, data):
        """The logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.

        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.

        Args:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned.
        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # for test during SNN inference
        self.y = y
        self.sample_weight=sample_weight
        y_pred = self(x, training=False)
        #y_pred = y_pred[:,-1,:]
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


    # this function is based on Model.train_step in training.py
    def train_step(self, data):

        ret = {
            'ANN': self.train_step_ann,
            'SNN': self.train_step_snn,
            #'SNN': self.train_step_ann,
        }[self.nn_mode](data)


        #
        if self.conf.verbose_visual:
            import matplotlib.pyplot as plt
            plt.show()

        return ret

    def train_step_ann(self, data):
        """The logic for one training step.

        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happends in fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.

        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Args:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)

            # train_vars_w = [v for v in self.trainable_variables if ('neuron' not in v.name) and ('temporal_kernel' not in v.name)]
            # print(train_vars_w)

            # grads = tape.gradient(loss, train_vars_w)
            # grads_and_vars = zip(grads, train_vars_w)
            # self.grads=grads
            # self.grads_and_vars=grads_and_vars
            # optimizer.apply_gradients(grads_and_vars)

            # print(self.grads_and_vars)

        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        # from keras.optimizers.optimizer_v2.optimizer_v2.py
        #self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        grad_loss=None
        name=None

        grads_and_vars = self.optimizer._compute_gradients(loss=loss, var_list=self.trainable_variables, grad_loss=grad_loss, tape=tape)
        #optimizer = keras.optimizers.optimizer_v2.optimizer_v2.OptimizerV2
        self.optimizer.apply_gradients(grads_and_vars, name=name)


        if self.conf.debug_mode:

            from lib_snn.sim import glb_plot_gradient_kernel
            from lib_snn.sim import glb_plot_gradient_gamma
            from lib_snn.sim import glb_plot_gradient_beta

            print('\n grads')
            for grad_accum, var in grads_and_vars:
                print('{: <10}: - max {:.3e}, min {:.3e}, mean {:.3e}, var {:.3e}'
                      .format(var.name,tf.reduce_max(grad_accum),tf.reduce_min(grad_accum),
                              tf.reduce_mean(grad_accum),tf.math.reduce_variance(grad_accum)))

                if True:
                    if 'kernel' in var.name:
                        lib_snn.util.plot_hist(glb_plot_gradient_kernel,grad_accum,1000,norm_fit=True)
                    elif 'gamma' in var.name:
                        lib_snn.util.plot_hist(glb_plot_gradient_gamma,grad_accum,1000,norm_fit=True)
                    elif 'beta' in var.name:
                        lib_snn.util.plot_hist(glb_plot_gradient_beta,grad_accum,1000,norm_fit=True)



        return self.compute_metrics(x, y, y_pred, sample_weight)

    def train_step_snn(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        #print(data)
        #print(x)
        #y=tf.reshape(y,[x.shape[0],y.shape[1]])

        # Run forward pass.
        last_ts = self.conf.time_step
        range_ts = range(1,last_ts+1)
        #range_ts = range(1,2+1)
        #range_ts = range(1,1+1)
        grad_loss = None
        name = None

        var_list = self.trainable_variables
        #grads = None

        grads_accum = [tf.zeros_like(v) for v in var_list]

        f_grad_accum=False
        #f_grad_accum=True

        if f_grad_accum:
            for t in range_ts:
                #with tf.GradientTape(persistent=True) as tape:
                #print(t)
                #print(last_ts)
                #if t==self.conf.time_step:
                if t<last_ts+1:
                #if t>2:
                    #assert False
                    with tf.GradientTape() as tape:
                        y_pred = self._run_internal_graph(x, training=True, mask=None)
                        loss = self.compute_loss(x, y, y_pred, sample_weight)
                    grads = tape.gradient(loss, var_list, grad_loss)
                    self._validate_target_and_loss(y, loss)
                    grads_accum = [(grad_accum + grad) for grad_accum, grad in zip(grads_accum, grads)]
                    #print(grads[-1][0])
                else:
                    y_pred = self._run_internal_graph(x, training=True)

                # end of time step - increase global time
                glb_t()

                #print(loss)

                #print(grads[-1])
                #assert False
                #grads = [grad * (t / last_ts) for grad in grads]
                #grads = [grad * (t / last_ts)*(t / last_ts) for grad in grads]
                #grads = [tf.math.multiply(grad,(t/last_ts)) for grad in grads]
                #grads = [tf.math.multiply(grad,(t/last_ts)*(t/last_ts)) for grad in grads]

                #time_norm_c = t/last_ts
                #time_norm = tf.constant(time_norm_c,shape=grads.shape)
                #grads = tf.math.multiply(grads,time_norm)

                #grads = grads*t/last_ts
                #self._grads = _grads
                # integration gradients
                #if grads == None:
                #~    grads = _grads
                #else:
                #    grads = tf.add(grads,_grads)

                #grads_accum = [tf.math.add(grad_accum,grad) for grad_accum,grad in zip(grads_accum,grads)]
                #grads_accum = grads
                #grads_and_vars = list(zip(grads, var_list))
                #self.optimizer.apply_gradients(grads_and_vars)

            # for debug
            #self.grads_accum = grads_accum
            #self.grads = grads
            #self.tape = tape


            # assert False
            #grads_accum = [grad_accum / self.conf.time_step for grad_accum in grads_accum]

            grads_accum_and_vars = list(zip(grads_accum, var_list))
            self.optimizer.apply_gradients(grads_accum_and_vars)

            # self.grads_and_vars = grads_accum_and_vars
            # self.optimizer.apply_gradients(grads_accum_and_vars,name=name)
        else:
            if False:
            #if True:
                with tf.GradientTape(persistent=True) as tape:
                    y_pred = self(x, training=True)
                    loss_0 = self.compute_loss(x, y, y_pred[:,0,:], sample_weight)
                    loss_1 = self.compute_loss(x, y, y_pred[:,1,:], sample_weight)
                    loss_2 = self.compute_loss(x, y, y_pred[:,2,:], sample_weight)
                    loss_3 = self.compute_loss(x, y, y_pred[:,3,:], sample_weight)

                    loss = loss_0+loss_1+loss_2+loss_3
                    #loss = loss_2+loss_3
                    #loss = loss_3

                    #grads_2 = tape.gradient(loss_2, var_list, grad_loss)
                    #grads_3 = tape.gradient(loss_3, var_list, grad_loss)
                    #print(grads_2)
                    #print(grads_3)
                    #grads = grads_2+grads_3
                    #assert False

                    #grads_accum = [(grad_accum + grad) for grad_accum, grad in zip(grads_accum, grads)]

                self._validate_target_and_loss(y, loss)

                grads_and_vars = self.optimizer._compute_gradients(loss, var_list=var_list, grad_loss=grad_loss,
                                                                   tape=tape)
                self.optimizer.apply_gradients(grads_and_vars, name=name)

                return self.compute_metrics(x, y, y_pred[:, -1, :], sample_weight)
            elif True:
                if self.conf.snn_training_spatial_first:
                    tape_prev = None
                    loss_prev = None
                    var_list_prev = None
                    glb_t.reset()
                    for t in range_ts:
                        #with tf.GradientTape() as tape:
                        #with tf.GradientTape(persistent=True) as tape:
                        with tf.GradientTape() as tape:
                            y_pred = self(x, training=True)
                            loss = self.compute_loss(x, y, y_pred, sample_weight)

                        self._validate_target_and_loss(y, loss)

                        # drop out or skip update - p percent
                        if False:
                        #if True:
                            rand = tf.random.uniform(shape=[],minval=0,maxval=1.0)
                            drop_prop = 0.5
                            grads = tape.gradient(loss, var_list, grad_loss)
                            #grads = tf.cond(rand<drop_prop,
                                            #lambda: tf.zeros(shape=grads.shape),
                                            #lambda: grads)
                            grads = tf.where(rand<drop_prop,[0]*len(grads),grads)

                        grads = tape.gradient(loss, var_list, grad_loss)

                        # spike norm - grad

                        #if tape_prev is not None:
                        #    grads_prev = tape_prev.gradient(loss, var_list_prev, grad_loss)
                        #    grads = grads+grads_prev
                        #grads = [grad * ((t+1) / last_ts) for grad in grads]
                        #grads = [grad/self.conf.time_step for grad in grads]
                        #if grads is not None:

                        #
                        #for grad, var in zip(grads,var_list):
                        #    print(var.name+' - ' +str(tf.reduce_mean(tf.math.abs(grad)).numpy()))

                        # gradient accum
                        #f_stochastic_gradient_accum=True
                        f_stochastic_gradient_accum=False
                        if f_stochastic_gradient_accum:
                            rand = tf.random.uniform(shape=[],minval=0,maxval=1.0)
                            drop_prop = 0.5
                            grads_accum = tf.cond(rand<drop_prop,lambda: grads_accum,lambda: [(grad_accum + grad) for grad_accum, grad in zip(grads_accum, grads)])
                        else:
                            grads_accum = [(grad_accum + grad) for grad_accum, grad in zip(grads_accum, grads)]


                        #
                        if False:
                            print()
                            for layer in self.layers:
                                if hasattr(layer,'kernel'):
                                    print('kernel> {} max {:.3e}, mean {:.3e}'.format(layer.name,tf.reduce_max(layer.kernel),tf.reduce_mean(layer.kernel)))

                                if hasattr(layer,'bias'):
                                    print('bias> {} max {:.3e}, mean {:.3e}'.format(layer.name,tf.reduce_max(layer.bias),tf.reduce_mean(layer.bias)))

                        #tape_prev = tape
                        #loss_prev = loss
                        #var_list_prev = var_list

                        if False:
                        #if True:
                            print('y')
                            print(y[0])
                            print('y_pred')
                            print(y_pred[0])
                            print('grads')
                            #print(grads)

                            for idx, grad in enumerate(grads):
                                print('grad <{:}> - min - max: {:} - {:}'.format(
                                    var_list[idx].name, tf.reduce_min(grad),
                                    tf.reduce_max(grad)))
                            #
                            #print(tf.reduce_max(grads[0]))
                            #print(tf.reduce_max(grads[-1]))
                            print(grads[-1])
                            print()

                        #assert False
                        glb_t()

                    #
                    grads_accum = [grad_accum/self.conf.time_step for grad_accum in grads_accum]
                    grads_accum_and_vars = list(zip(grads_accum, var_list))
                else:
                    #sample_weight = data_adapter.unpack_x_y_sample_weight(data)
                    # Run forward pass.
                    with tf.GradientTape() as tape:
                        y_pred = self(x, training=True)
                        loss = self.compute_loss(x, y, y_pred, sample_weight)

                    self._validate_target_and_loss(y, loss)
                    # Run backwards pass.
                    # from keras.optimizers.optimizer_v2.optimizer_v2.py
                    #self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
                    grad_loss=None
                    name=None

                    grads_accum_and_vars = self.optimizer._compute_gradients(loss=loss, var_list=self.trainable_variables, grad_loss=grad_loss, tape=tape)


                #
                self.optimizer.apply_gradients(grads_accum_and_vars)


                nan_test = [tf.reduce_any(tf.math.is_nan(grad_accum)) for grad_accum in grads_accum]
                #if tf.reduce_any(nan_test):
                #if tf.executing_eagerly() and tf.reduce_any(nan_test):
                if tf.executing_eagerly():
                    #
                    #if False:
                    #if True:
                    if self.conf.verbose_snn_train:

                        self.print_snn_train(grads_accum_and_vars)

                    #if tf.reduce_any(nan_test) or (loss > 100):
                    if tf.reduce_any(nan_test) or (loss > 1000):
                        print(loss)
                        print(tf.reduce_any(nan_test))
                        #print('here')
                        assert False


                return self.compute_metrics(x, y, y_pred, sample_weight)

            else:
                if True:
                    for t in range_ts:
                        #print(t)
                        with tf.GradientTape() as tape:
                            y_pred = self._run_internal_graph(x, training=True)
                            loss = self.compute_loss(x, y, y_pred, sample_weight)

                        #print(x)

                        #if t==self.conf.time_step:
                        #    y_pred=_y_pred

                        # end of time step - increase global time
                        glb_t()
                else:
                        y_pred_1 = self._run_internal_graph(x, training=True)
                        glb_t()
                        loss_1 = self.compute_loss(x, y, y_pred_1, sample_weight)

                        y_pred_2 = self._run_internal_graph(x, training=True)
                        glb_t()
                        loss_2 = self.compute_loss(x, y, y_pred_2, sample_weight)
                        y_pred_3 = self._run_internal_graph(x, training=True)
                        glb_t()
                        loss_3 = self.compute_loss(x, y, y_pred_3, sample_weight)
                        y_pred = self._run_internal_graph(x, training=True)
                        glb_t()
                        loss_4 = self.compute_loss(x, y, y_pred, sample_weight)

                #loss = self.compute_loss(x, y, y_pred, sample_weight)
                #loss = loss + loss_1 + loss_2 + loss_3
                #loss = loss_4 + loss_3

                #self.loss=loss
                #self.loss_4=loss_4
                #self.loss_3=loss_3

                self._validate_target_and_loss(y, loss)

                #self.optimizer.minimize(loss, var_list, tape=tape)
                grads_and_vars = self.optimizer._compute_gradients(loss, var_list=var_list, grad_loss=grad_loss, tape=tape)
                self.optimizer.apply_gradients(grads_and_vars,name=name)
                #grads = tape.gradient(loss, var_list, grad_loss)
                #grads_accum = grads
                #assert False

        return self.compute_metrics(x, y, y_pred, sample_weight)
        #return ret

    # TODO: move other part
    def record_acc_spike_time_point(self,inputs,outputs):
        #print('record_acc_spike_time_point')

        y_pred=outputs
        # accuracy
        # Updates stateful loss metrics.

        #self.compiled_metrics.update_state(self.y, y_pred, self.sample_weight)
        #self.compiled_metrics.update_state(self.y, y_pred, self.sample_weight)

        #loss=self.compiled_loss(self.y, y_pred, self.sample_weight, regularization_losses=self.losses)
        losses = self.loss_metrics[self.count_accuracy_time_point]

        losses.reset_state()
        loss = losses(self.y, y_pred, self.sample_weight, regularization_losses=self.losses)

        #if False:
        metrics=self.accuracy_metrics[self.count_accuracy_time_point]
        metrics.update_state(self.y, y_pred, self.sample_weight)

        ## update loss
        #metrics[0](self.y, y_pred, self.sample_weight, regularization_losses=self.losses)
        #metrics[1:].update_state(self.y, y_pred, self.sample_weight)
        # Collect metrics to return

        #print(metrics)

        #print(self.compiled_loss.metrics)
        #print(loss)
        #assert False

        #print(loss)

        if tf.math.is_nan(loss):

            #for metric in self.compiled_loss.metrics:
            #    metric.reset_state()

            #self.loss_metrics[self.count_accuracy_time_point].reset_state()
            losses.reset_state()

            loss_metrics = metrics.metrics
        else:
            loss_metrics = losses.metrics + metrics.metrics

        #assert False
        #self.reset_metrics()
        return_metrics = {}
        #metrics = self.accuracy_time_point[self.count_accuracy_time_point]
        #for metric in metrics:
        #for metric in metrics.metrics:
        #for metric in self.metrics:
        for metric in loss_metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        self.accuracy_results[self.count_accuracy_time_point]=return_metrics


        # spike count - layer wise
        for layer_name in self.total_spike_count_int.keys():
            #print(layer_name)
            #print(self.count_accuracy_time_point)
            #print(tf.reduce_sum(self.get_layer(layer_name).act_snn.spike_count_int))

            #self.total_spike_count_int[layer_name][self.count_accuracy_time_point]=\
            #    tf.reduce_sum(self.get_layer(layer_name).act_snn.spike_count_int)

            #self.total_spike_count[layer_name][self.count_accuracy_time_point]= \
            #    tf.reduce_sum(self.get_layer(layer_name).act_snn.spike_count)

            #self.total_residual_vmem[layer_name][self.count_accuracy_time_point]= \
            #    tf.reduce_sum(self.get_layer(layer_name).act_snn.vmem)

            self.total_spike_count_int[layer_name] = tf.tensor_scatter_nd_update(
                self.total_spike_count_int[layer_name],
                [[self.count_accuracy_time_point]],
                [tf.reduce_sum(self.get_layer(layer_name).act_snn.spike_count_int)])

            self.total_spike_count[layer_name] = tf.tensor_scatter_nd_update(
                self.total_spike_count[layer_name],
                [[self.count_accuracy_time_point]],
                [tf.reduce_sum(self.get_layer(layer_name).act_snn.spike_count)])

            self.total_residual_vmem[layer_name] = tf.tensor_scatter_nd_update(
                self.total_residual_vmem[layer_name],
                [[self.count_accuracy_time_point]],
                [tf.reduce_sum(self.get_layer(layer_name).act_snn.vmem)])


        #
        self.count_accuracy_time_point+=1


    ###########################################################################
    ## processing - pre-processing
    ###########################################################################
    def preproc(self, inputs, training, f_val_snn=False):
        preproc_sel= {
            'ANN': self.preproc_ann,
            'SNN': self.preproc_snn
        }

        if f_val_snn:
            self.preproc_snn(inputs,training)
        else:
            preproc_sel[self.nn_mode](inputs, training)


    def preproc_snn(self,inputs,training):
        # reset for sample
        self.reset_snn_sample()

        if self.f_done_preproc == False:
            self.f_done_preproc = True
            #self.print_model_conf()
            self.reset_snn_run()
            self.preproc_ann_to_snn()

            # snn validation mode
            if self.conf.f_surrogate_training_model:
                self.load_temporal_kernel_para()

        if self.conf.f_comp_act:
            lib_snn.anal.save_ann_act(self,inputs,training)

        # gradient-based optimization of TC and td in temporal coding (TTFS)
        if self.en_opt_time_const_T2FSNN:
            self.call_ann(inputs,training)

    def preproc_ann(self, inputs, training):
        if self.f_done_preproc == False:
            self.f_done_preproc=True
            # here
            ##self.print_model_conf()
            self.preproc_ann_norm()

            # surrogate DNN model for training SNN with temporal information
            if self.conf.f_surrogate_training_model:
                self.preproc_surrogate_training_model()

        self.f_skip_bn=self.conf.f_fused_bn


    def preproc_ann_to_snn(self):
        if self.conf.verbose:
            print('preprocessing: ANN to SNN')

        if self.conf.f_fused_bn or ((self.nn_mode=='ANN')and(self.conf.f_validation_snn)):
            #self.fused_bn()
            self.bn_fusion()

        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()

        #self.print_act_after_w_norm()

    def preproc_surrogate_training_model(self):
        if self.f_loss_enc_spike_dist:
            self.dist_beta_sample_func()


    #
    def preproc_ann_norm(self):
        if self.conf.f_fused_bn:
            #self.fused_bn()
            self.bn_fusion()

        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()


    ###########################################################
    # init
    ###########################################################

    def init(self,model_ann=None):

        # common init
        self.model_ann=model_ann

        #for layer in self.model.layers:
        for layer in self.layers:
            if hasattr(layer, 'kernel'):
                self.layers_w_kernel.append(layer)

        self.layers_w_act = []
        for layer in self.layers:
            #if hasattr(layer, 'act'):
            if isinstance(layer, lib_snn.activations.Activation):
                #if not isinstance(layer,lib_snn.layers.InputGenLayer):
                #if not layer.act_dnn is None:
                #if not (layer.activation is None):
                #if not (layer.act_dnn is None):
                #    self.layers_w_act.append(layer)
                #    #print(layer.name)

                self.layers_w_act.append(layer)

        #assert False

        self.layers_w_bias= []
        #for layer in self.layers_w_kernel:
        #    if hasattr(layer, 'bias'):
        #        self.layers_bias_control.append(layer)
        for layer in self.layers:
            if hasattr(layer, 'bias'):
                self.layers_w_bias.append(layer)

        if self.conf.bias_control:
            #self.layers_bias_control = self.layers_w_bias
            self.layers_bias_control = []

            if False:
                for layer in self.layers_w_bias:
                    if not isinstance(layer, lib_snn.layers.Add):
                        self.layers_bias_control.append(layer)

            self.layers_bias_control = self.layers_w_kernel
            #self.layers_bias_control = self.layers_w_neuron


        #self.en_record_output = self.model._run_eagerly and (
        #(self.model.nn_mode == 'ANN' and self.conf.f_write_stat) or self.conf.debug_mode)
        self.en_record_output = self._run_eagerly and \
                                ((self.nn_mode == 'ANN' and self.conf.f_write_stat) or self.conf.en_record_output) or \
                                (self.nn_mode == 'SNN' and self.conf.vth_search)

        if self.en_record_output:
            #self.layers_record = self.layers_w_kernel
            #self.layers_record = self.layers_w_act
            #self.layers_record = self.layers[1:]
            self.layers_record = []
            for layer in self.layers:
                if (layer in self.layers_w_kernel) or (layer in self.layers_w_act):
                    self.layers_record.append(layer)

            #for layer in self.layers_record:
            #    layer.init_record_output()

            #for layer in self.layers_record:
            #    print(layer.name)
            #assert False

            #self.layers_record=[]
            #for layer in self.layers:
            #    #if not isinstance(layer,tf.python.keras.engine.input_layer.InputLayer):
            #    #if not isinstance(layer,lib_snn.layers.InputLayer):
            #    if not isinstance(layer,tf.keras.layers.InputLayer):
            #        self.layers_record.append(layer)


            # partially
            if False:
            #if True and (self.conf.f_write_stat):
                self.layers_record = []
                #for layer in self.layers_w_act[:3]:
                #for layer in self.layers_w_act[3:6]:
                #for layer in self.layers_w_act[6:9]:
                #for layer in self.layers_w_act[9:12]:
                #for layer in self.layers_w_act[12:16]:
                #for layer in self.layers_w_act[16:22]:
                for layer in self.layers_w_act[22:33]:
                    self.layers_record.append(layer)
                ##for layer in self.layers_w_kernel[4:10]:
                ##for layer in self.layers_w_kernel[10:15]:
                ##for layer in self.layers_w_kernel[15:20]:
                ##for layer in self.layers_w_kernel[25:30]:
                #for layer in self.layers_w_kernel[35:40]:
                #    self.layers_record.append(layer)

            self.set_en_record_output()


        # TODO: add condition
        #if self.conf.f_w_norm_data or self.bias_control:
        #if (self.en_snn) and ('ResNet' in self.name):
        #if 'ResNet' in self.name:
        f_dnn_to_snn = False
        if 'ResNet' in self.name and f_dnn_to_snn:
            self.block_norm_set_resnet()


        if self.nn_mode=='ANN':
            self.init_ann()
        elif self.nn_mode == 'SNN':
            self.init_snn(model_ann)
        else:
            assert False


    #
    def init_ann(self):
        pass


    #
    def init_snn(self, model_ann):

        #
        #self.set_layers_w_neuron()

        #
        self.set_layers_w_neuron()

        #
        self.spike_max_pool_setup()

        # init - neuron
        for layer in self.layers_w_neuron:
            #print(layer.name)
            #layer.act_snn.init()
            if hasattr(layer.act,'init'):
                layer.act.init()


        #
        # dummy run
        #self.compiled_metrics.update_state(self.y, self.y, self.sample_weight)

        #if self._is_compiled:
        #metrics = []
        #if self.compiled_loss is not None:
            #metrics += self.compiled_loss.metrics
        #if self.compiled_metrics is not None:
            #metrics += self.compiled_metrics.metrics

        #compiled_loss = compile_utils.LossesContainer(
            #self.loss, None, output_names=self.output_names)
        #metrics += [compiled_loss._loss_metric]
        #metrics += self.compiled_metrics._metrics
        metrics = self.compiled_metrics._metrics

        #print(metrics)
        #for metric in metrics:
            #print(metric)
            #print(metric.name)
        #assert False

        for idx in range(self.num_accuracy_time_point):

            self.loss_metrics[idx] = compile_utils.LossesContainer(self.loss, None, output_names=self.output_names)
            self.loss_metrics[idx].reset_state()

            #self.accuracy_metrics[idx] = self.compiled_metrics
            self.accuracy_metrics[idx] = compile_utils.MetricsContainer(
                                    metrics, None, output_names=self.output_names, from_serialized=False)
            self.accuracy_metrics[idx].reset_state()


        #print(self.compiled_metrics.metrics)
        #print(self._is_compiled)
        #assert False

        # total spike count init - layer wise at each accuracy time point
        # TODO: np -> tf.Variables, tf.constant? - nesseccary?
        self.total_spike_count=collections.OrderedDict()
        self.total_spike_count_int=collections.OrderedDict()
        self.total_residual_vmem=collections.OrderedDict()

        for layer in self.layers_w_neuron:
            #if isinstance(layer.act_snn,lib_snn.neurons.Neuron):
            #self.total_spike_count_int[layer.name]=np.zeros([self.num_accuracy_time_point])
            #self.total_spike_count_int[layer.name]=np.empty_like([self.num_accuracy_time_point],dtype=object)

            self.total_spike_count[layer.name]=tf.zeros([self.num_accuracy_time_point])
            self.total_spike_count_int[layer.name]=tf.zeros([self.num_accuracy_time_point])
            self.total_residual_vmem[layer.name]=tf.zeros([self.num_accuracy_time_point])


        #
        if self.bias_control:
            self.set_bias_control_th(model_ann)



        #
        self.init_done=True

    #
    def set_en_record_output(self):
        # for layer in self.model.layers:
        for layer in self.layers_record:
            layer.en_record_output = True
            layer.init_record_output()

        #for layer in self.model.layers:
        for layer in self.layers:
            if isinstance(layer, lib_snn.layers.InputGenLayer):
                layer.en_record_output = True
                layer.init_record_output()

        self.dict_stat_w = collections.OrderedDict()

    #
    def block_norm_set_resnet(self):

        # block_norm_in set
        self.block_norm_in_name = collections.OrderedDict()
        self.block_norm_out_name = collections.OrderedDict()
        self.prev_layer_name = collections.OrderedDict()

        #if self.conf.bias_control:
        self.prev_layer_name_bias_control = collections.OrderedDict()

        for idx_l, l in enumerate(self.layers_w_act):
            if not ('conv' in l.name):
                continue

            #print(l.name)
            conv_block_name = l.name.split('_')
            conv_block_name = conv_block_name[0] + '_' + conv_block_name[1]

            #print(conv_block_name)

            if (idx_l==0) and (not 'block' in conv_block_name):
                self.block_norm_in_name[conv_block_name] = None
                self.block_norm_out_name[conv_block_name] = conv_block_name
                next_block_norm_in_name = conv_block_name
            else:
                if not (conv_block_name in self.block_norm_in_name.keys()):
                    self.block_norm_in_name[conv_block_name] = next_block_norm_in_name
                    self.block_norm_out_name[conv_block_name] = conv_block_name+'_out'

                    next_block_norm_in_name = self.block_norm_out_name[conv_block_name]

        for idx_l, l in enumerate(self.layers_w_kernel):
            conv_block_name = None
            if idx_l==0:
                #prev_layer_name = l.name
                pass

            elif ('conv' in l.name) and ('block') in l.name:
                conv_block_name = l.name.split('_')
                conv_name = conv_block_name[2]
                conv_block_name = conv_block_name[0] + '_' + conv_block_name[1]

                if 'conv0' in conv_name:
                    self.prev_layer_name[l.name] = self.block_norm_in_name[conv_block_name]
                    self.prev_layer_name_bias_control[l.name] = conv_block_name+'_conv1'
                    #self.prev_layer_name_bias_control[l.name] = self.prev_layer_name[l.name]
                elif 'conv1' in conv_name:
                    self.prev_layer_name[l.name] = self.block_norm_in_name[conv_block_name]

                    prev_layer = self.get_layer(self.prev_layer_name[l.name])
                    if hasattr(prev_layer,'bias'):
                        self.prev_layer_name_bias_control[l.name] = self.prev_layer_name[l.name]
                    else:
                        prev_layer_conv_block_name = self.prev_layer_name[l.name].split('_')
                        prev_layer_conv_block_name = prev_layer_conv_block_name[0]+'_'+prev_layer_conv_block_name[1]
                        self.prev_layer_name_bias_control[l.name] = prev_layer_conv_block_name+'_conv1'

                elif 'conv2' in conv_name:
                    self.prev_layer_name[l.name] = conv_block_name + '_conv1'
                    self.prev_layer_name_bias_control[l.name] = self.prev_layer_name[l.name]
                elif 'conv3' in conv_name:
                    self.prev_layer_name[l.name] = conv_block_name + '_conv2'
                    self.prev_layer_name_bias_control[l.name] = self.prev_layer_name[l.name]
                else:
                    print(l.name)
                    assert False
            else:
                if prev_conv_block_name in prev_layer_name:
                    self.prev_layer_name[l.name] = self.block_norm_out_name[prev_conv_block_name]
                    self.prev_layer_name_bias_control[l.name] = self.prev_layer_name[l.name]
                else:
                    assert False
                    self.prev_layer_name[l.name]=prev_layer_name
                    self.prev_layer_name_bias_control[l.name] = self.prev_layer_name[l.name]

                prev_layer = self.get_layer(self.prev_layer_name[l.name])
                if hasattr(prev_layer,'bias'):
                    self.prev_layer_name_bias_control[l.name] = self.prev_layer_name[l.name]
                else:
                    prev_layer_conv_block_name = self.prev_layer_name[l.name].split('_')
                    prev_layer_conv_block_name = prev_layer_conv_block_name[0]+'_'+prev_layer_conv_block_name[1]
                    self.prev_layer_name_bias_control[l.name] = prev_layer_conv_block_name+'_conv1'


            prev_layer_name = l.name
            prev_conv_block_name = conv_block_name


        #for layer_name, prev_layer_name in self.prev_layer_name_bias_control.items():
        #    print('layer - {}, prev_layer - {}'.format(layer_name,prev_layer_name))
        #assert False



    ###########################################################
    # reset snn
    ###########################################################
    #
    def reset_snn_run(self):
        self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.list_layer_name)+1])
        self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.list_layer_name)+1])

    #
    def reset_snn_sample(self):
        self.reset_snn_time_step()
        self.reset_snn_neuron()
        #self.snn_output = np.zeros((self.num_accuracy_time_point,)+self.list_neuron['fc1'].get_spike_count().numpy().shape)
        self.snn_output.assign(tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_neuron.dim)))
        self.count_accuracy_time_point=0

    #
    def reset_snn_time_step(self):
        Model.t = 0

    def reset_snn(self):
        # reset count accuracy_time_point
        #if self.verbose:
            #print('reset_snn')
        self.count_accuracy_time_point=0

        #
        self.reset_snn_neuron()

        # sptr
        self.reset_snn_sptr()


    #
    def reset_snn_neuron(self):
        #print('reset_snn_neuron')
        for layer in self.layers_w_neuron:
            layer.reset()

    # sptr - only for training
    def reset_snn_sptr(self):
        if self.conf.sptr:
            for layer in self.layers:
                if hasattr(layer, 'input_accum'):
                    layer.input_accum.assign(tf.zeros(layer.input_accum.shape))



    # set layers with neuron
    def set_layers_w_neuron(self):
        if False: #old
            #self.layers_w_neuron = []
            for layer in self.layers:
                #if hasattr(layer, 'act_snn'):
                if hasattr(layer, 'act'):
                    #if layer.act_snn is not None:
                    #if isinstance(layer.act,lib_snn.neurons.Neuron):
                    if not (layer.act_dnn is None):
                        #print(layer.name)
                        self.layers_w_neuron.append(layer)
        for layer in self.layers:
            if isinstance(layer, lib_snn.activations.Activation):
                if isinstance(layer.act, lib_snn.neurons.Neuron):
                    self.layers_w_neuron.append(layer)


    ###########################################################
    # BN fusion
    ###########################################################
    #
    def bn_fusion(self):
        assert False
        print('---- BN Fusion ----')

        for name_l in self.list_layer_name:
            layer = self.model.get_layer(name=name_l)
            layer.bn_fusion()

        print('---- BN Fusion Done ----')

    #
    def bn_defusion(self):
        assert False
        #print('---- BN DeFusion ----')

        for name_l in self.list_layer_name:
            layer = self.model.get_layer(name=name_l)
            layer.bn_defusion()

        #print('---- BN DeFusion Done ----')


    ###########################################################
    # Weight normalization
    ###########################################################
    #
    def w_norm_layer_wise(self):
        print('layer-wise normalization')
        f_norm=np.max

        #for idx_l, l in enumerate(self.list_layer_name):
        for idx_l, l in enumerate(self.list_layer):
            if idx_l==0:
                self.norm[l.name]=f_norm(self.dict_stat_r[l.name])
            else:
                self.norm[l.name]=f_norm(list(self.dict_stat_r.values())[idx_l])/f_norm(list(self.dict_stat_r.values())[idx_l-1])

            self.norm_b[l.name]=f_norm(self.dict_stat_r[l.name])

        # print
        print('norm weight')
        for k, v in self.norm.items():
            print(k +': '+str(v))

        print('norm bias')
        for k, v in self.norm_b.items():
            print(k +': '+str(v))

        #
        #for name_l in self.list_layer_name:
        for layer in self.list_layer:
            #layer = self.model.get_layer(name=name_l)
            layer.kernel = layer.kernel/self.norm[layer.name]
            layer.bias = layer.bias/self.norm_b[layer.name]


        # TODO: move
        if self.conf.noise_en:
            assert False, 'not modified yet'
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

        #for idx_l, l in enumerate(self.list_layer_name):
        for idx_l, l in enumerate(self.list_layer):
            key=l.name+'_'+stat

            f_name_stat = f_name_stat_pre+'_'+key
            f_name=os.path.join(path_stat,f_name_stat)
            f_stat[key]=open(f_name,'r')
            r_stat[key]=csv.reader(f_stat[key])

            # TODO: np -> tf.Variable
            for row in r_stat[key]:
                #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
                self.dict_stat_r[l.name]=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

        self.w_norm_layer_wise()




    ###########################################################################
    ## processing - post-processing
    ###########################################################################
    def postproc(self,inputs):
        postproc_sel= {
            'ANN': self.postproc_ann,
            'SNN': self.postproc_snn
        }
        postproc_sel[self.nn_mode](inputs)

    def postproc_ann(self,inputs):

        #
        if self.en_opt_time_const_T2FSNN:
            lib_snn.util.recording_dnn_act(self,inputs)

        # write stat for data-based weight normalization
        if self.conf.f_write_stat:
            lib_snn.util.collect_dnn_act(self,inputs)


    def postproc_snn(self,inputs):

        # output zero check
        #spike_zero = tf.reduce_sum(self.snn_output,axis=[0,2])
        #if np.any(spike_zero.numpy() == 0.0):
        #    print('spike count 0')

        # calculating total residual vmem
        #self.get_total_residual_vmem()

        #
        if self.conf.f_entropy:
            assert False, 'not verified yet'
            #self.cal_entropy()

        # visualization - first spike time
        if self.conf.f_record_first_spike_time and self.conf.f_visual_record_first_spike_time:
            assert False, 'not verified yet'
            lib_snn.util.visual_first_spike_time(self)


    # postprocessing - SNN at each time step
    def postproc_snn_time_step(self):

        #
        if not self.f_load_model_done:
            return

        # recording snn output
        if self.t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
            self.recording_ret_val()



        # raster plot
        #if self.f_debug_visual:
        if flags._run_for_visual_debug:
            lib_snn.util.debug_visual_raster(self,self.t)

        # compare activation - DNN vs. SNN
        if self.conf.f_comp_act:
            assert False, 'not verified yet'
           #lib_snn.anal.comp_act(self)

        # ISI
        if self.conf.f_isi:
            assert False, 'not verified yet'
        #    self.total_isi += self.get_total_isi()
        #    self.total_spike_amp += self.get_total_spike_amp()
        #    self.f_out_isi(t)

        # entropy - spike train
        if self.conf.f_entropy:
            assert False, 'not verified yet'
        #    for idx_l, l in enumerate(self.list_layer_name):
        #        if l !='fc3':
        #            self.dict_stat_w[l][t] = self.list_neuron[l].out.numpy()


    ###########################################################
    ## SNN output
    ###########################################################
    #
    def recording_ret_val(self):
        output=self.snn_output_func()
        self.snn_output.scatter_nd_update([self.count_accuracy_time_point],tf.expand_dims(output,0))

        tc_int, tc = self.get_total_spike_count()
        self.total_spike_count_int[self.count_accuracy_time_point]+=tc_int
        self.total_spike_count[self.count_accuracy_time_point]+=tc

        self.count_accuracy_time_point+=1

        #num_spike_count = tf.cast(tf.reduce_sum(self.snn_output,axis=[2]),tf.int32)

    #
    def snn_output_func(self):
        snn_output_func_sel = {
            "SPIKE": self.snn_output_neuron.spike_counter,
            "VMEM": self.snn_output_neuron.vmem,
            "FIRST_SPIKE_TIME": self.snn_output_neuron.first_spike_time
        }
        return snn_output_func_sel[self.conf.snn_output_type]


    #
    def get_total_spike_count(self):
        len=self.total_spike_count.shape[1]
        spike_count = np.zeros([len,])
        spike_count_int = np.zeros([len,])

        #for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
        for idx, layer in enumerate(self.list_layer):
            n = layer.act
            spike_count_int[idx]=tf.reduce_sum(n.get_spike_count_int())
            spike_count_int[len-1]+=spike_count_int[idx]
            spike_count[idx]=tf.reduce_sum(n.get_spike_count())
            spike_count[len-1]+=spike_count[idx]

            #print(nn+": "+str(spike_count_int[idx]))


        #print("total: "+str(spike_count_int[len-1])+"\n")

        return [spike_count_int, spike_count]



    ###########################################################################
    ##
    ###########################################################################
    def get_total_residual_vmem(self):
        assert False, 'not verified yet'
        #len=self.total_residual_vmem.shape[0]
        #for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            #idx=idx_n-1
            #if nn!='in' or nn!='fc3':
                #self.total_residual_vmem[idx]+=tf.reduce_sum(tf.abs(n.vmem))
                #self.total_residual_vmem[len-1]+=self.total_residual_vmem[idx]


    ###########################################
    # bias control - new
    ###########################################
    def set_bias_control_th(self, model_ann):
        self.bias_control_th = collections.OrderedDict()
        self.bias_control_th_ch = collections.OrderedDict()

        if False:   # manual set
        #if True:  # manual set
            for idx_layer, layer in enumerate(self.layers_w_neuron):
                #self.bias_control_th[layer.name] = 0.005
                self.bias_control_th[layer.name] = 0.01
                #self.bias_control_th_ch[layer.name] = tf.constant(0.002,shape=layer.f_bias_ctrl.shape)
                self.bias_control_th_ch[layer.name] = tf.constant(0.01,shape=layer.f_bias_ctrl.shape)
                #self.bias_control_th[layer.name] = 0.0
        elif True:  # non zero ratio-based (DNN)
            #for idx_layer, layer in enumerate(self.layers_w_kernel):
            for idx_layer, layer in enumerate(self.layers_bias_control):
            #for idx_layer, layer in enumerate(self.layers_w_act):
            #for idx_layer, layer in enumerate(self.layers_w_neuron):

                #print('here')
                #print(layer.name)

                layer_ann = model_ann.get_layer(layer.name)

                if isinstance(layer, lib_snn.layers.InputGenLayer):
                    continue

                #print(layer.name)
                #if False:
                if True:
                    if isinstance(layer, lib_snn.layers.Conv2D):
                        axis = [1,2,3]
                        axis_ch = [1,2]
                    elif isinstance(layer, lib_snn.layers.Dense):
                        axis = [1]
                        axis_ch = [1]
                    else:
                        print(layer)
                        assert False

                non_zero = tf.math.count_nonzero(layer_ann.record_output, dtype=tf.float32, axis=axis)
                non_zero_r = non_zero / tf.cast(tf.reduce_prod(layer_ann.record_output.shape[1:]), tf.float32)

                #self.bias_control_th[layer.name] = tf.reduce_mean(non_zero_r)*0.0001
                #self.bias_control_th[layer.name] = tf.reduce_mean(non_zero_r)*0.001
                #self.bias_control_th[layer.name] = tf.reduce_mean(non_zero_r)*0.01
                #self.bias_control_th[layer.name] = tf.reduce_mean(non_zero_r)*0.1
                #self.bias_control_th[layer.name] = tf.reduce_mean(non_zero_r)
                #self.bias_control_th[layer.name] = 0.1  # ResNet-32 + leak_off_bias_en
                #self.bias_control_th[layer.name] = 0.01
                #self.bias_control_th[layer.name] = 0.01*((16-idx_layer)/16*0.5+0.5)
                #self.bias_control_th[layer.name] = 0.001   # VGG
                #self.bias_control_th[layer.name] = 0.0001
                #self.bias_control_th[layer.name] = 0.00001
                #self.bias_control_th[layer.name] = 0.00
                #self.bias_control_th[layer.name] = non_zero_r
                self.bias_control_th[layer.name] = non_zero_r * 0.1    # ts-64, 94.35
                #self.bias_control_th[layer.name] = non_zero_r * 0.01    # ts-64, 94.29, 94.32
                #self.bias_control_th[layer.name] = non_zero_r * 0.01* ((16-idx_layer)/16*0.5+0.5)   # ts-64, 94.29, 94.25, 94.38
                #self.bias_control_th[layer.name] = non_zero_r*0.001 # ts-64, 94.4, 94.29
                #self.bias_control_th[layer.name] = non_zero_r*0.0001 # ts-64, 94.36

                print('< bias_control_th >')
                print('{:} - {:}'.format(layer.name,self.bias_control_th[layer.name]))

                channel = False
                if channel:
                    assert False
                    # channel-wise
                    non_zero_ch = tf.math.count_nonzero(layer_ann.record_output, dtype=tf.float32, axis=axis_ch)
                    non_zero_ch_r = non_zero_ch / tf.cast(tf.reduce_prod(layer_ann.record_output.shape[1:]), tf.float32)

                    non_zero_ch_r = tf.expand_dims(non_zero_ch_r,axis=0)
                    self.bias_control_th_ch[layer.name] = tf.broadcast_to(non_zero_ch_r, shape=layer.f_bias_ctrl.shape)


            #self.bias_control_th_ch[layer.name] = tf.constant(0.01, shape=layer.f_bias_ctrl.shape)

        #self.bias_control_th['fc1'] = 0.05




    ###########################################
    # bias control - old
    ###########################################
    # TODO: bias control
    def bias_control_old(self,t):
        if self.conf.neural_coding == "RATE":
            if t == 0:
                self.bias_enable()
            else:
                self.bias_disable()
        else:
            assert False

    def bias_control_old(self,t):
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
        #for k, l in self.list_layer.items():
        #    if not 'bn' in k:
        #        l.use_bias = True
        for layer in self.layers:
            if hasattr(layer, 'use_bias'):
                layer.use_bias = True


    def bias_disable(self):
        #for k, l in self.list_layer.items():
            #if not 'bn' in k:
                #l.use_bias = False
        for layer in self.layers:
            if hasattr(layer, 'use_bias'):
                layer.use_bias = False
                #layer.bias = tf.zeros(tf.shape(layer.bias))
                #layer.bias = tf.ones(tf.shape(layer.bias))

    def bias_restore(self):
        if self.conf.use_bias:
            self.bias_enable()
        else:
            self.bias_disable()


    ###########################################################
    ## Print
    ###########################################################
    #
    def print_model_conf(self):
        self.model.summary()




    ###########################################################
    # load weights
    ###########################################################
    def load_weights_dnn_to_snn(self, model_ann):

        for layer_ann in model_ann.layers:
            if isinstance(layer_ann,lib_snn.layers.Conv2D) or isinstance(layer_ann, lib_snn.layers.Dense):
                layer_name = layer_ann.name
                layer_snn = self.get_layer(layer_name)
                layer_snn.kernel = layer_ann.kernel
                layer_snn.bias = layer_ann.bias
                if layer_snn.bn is not None:
                    layer_snn.bn.set_weights(layer_ann.bn.get_weights())

        #for layer in self.layers:
            #if isinstance(layer,lib_snn.layers.Conv2D):
                #print(layer.name)
                #assert False

            #self.bn = tf.keras.layers.BatchNormalization(name=name_bn)

        #self.model.get_layer('conv1').set_weights(pre_model.get_layer('vgg16').get_layer('block1_conv1').get_weights())



    ###########################################################
    # run internal graph - SNN, temporal first
    # based on _run_internal_graph() function in keras.engine.functional.py
    ###########################################################
    def _run_internal_graph_snn_t_first(self, inputs, training=None, mask=None):
        """ run internal graph - SNN, temporal first
            Computes output tensors for new inputs.

        # Note:
            - Can be run on non-Keras tensors.

        Args:
            inputs: Tensor or nested structure of Tensors.
            training: Boolean learning phase.
            mask: (Optional) Tensor or nested structure of Tensors.

        Returns:
            output_tensors
        """
        inputs = self._flatten_to_reference_inputs(inputs)
        if mask is None:
            masks = [None] * len(inputs)
        else:
            masks = self._flatten_to_reference_inputs(mask)
        for input_t, mask in zip(inputs, masks):
            input_t._keras_mask = mask

        # Dictionary mapping reference tensors to computed tensors.
        tensor_dict = {}
        tensor_usage_count = self._tensor_usage_count
        for x, y in zip(self.inputs, inputs):
            y = self._conform_to_reference_input(y, ref_input=x)
            x_id = str(id(x))
            tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

        nodes_by_depth = self._nodes_by_depth
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        # sspark
        # SNN training, temporal first
        #if self.conf.nn_mode=='SNN' and not self.conf.snn_training_spatial_first:
        #if False:
        #if True:
        for depth in depth_keys:
            nodes = nodes_by_depth[depth]
            for node in nodes:
                if node.is_input:

                    _input = tensor_dict[x_id][0]
                    #args, kwargs = node.map_arguments(tensor_dict)
                    layer_out = tf.TensorArray(
                        dtype=tf.float32,
                        size=self.conf.time_step,
                        element_shape=_input.shape,
                        clear_after_read=False,
                        tensor_array_name='out_arr')

                    glb_t.reset()
                    for t in range(1,self.conf.time_step+1):
                        layer_out = layer_out.write(t-1,_input)
                        glb_t()

                    #continue  # Input tensors already exist.

                else:

                    #
                    #print(node.layer)

                    if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
                        assert False
                        continue  # Node is not computable, try skipping.

                    #layer_in = layer_out

                    args, kwargs = node.map_arguments(tensor_dict)
                    layer_in = args[0]
                    #print(layer_in)

                    f_temporal_reduction = False
                    #
                    layers_temporal_reduction = []
                    layers_temporal_reduction.append(lib_snn.layers.BatchNormalization)
                    #layers_temporal_reduction.append(lib_snn.layers.MaxPool2D)
                    #layers_temporal_reduction.append(lib_snn.layers.AveragePooling2D)

                    if type(node.layer) in layers_temporal_reduction:
                        f_temporal_reduction = True

                    if f_temporal_reduction:

                        _layer_in = tf.reduce_mean(layer_in.stack(),axis=0)

                        _layer_out = node.layer(_layer_in)

                        #
                        layer_out = tf.TensorArray(
                            dtype=tf.float32,
                            size=self.conf.time_step,
                            element_shape=_layer_out.shape,
                            clear_after_read=False,
                            tensor_array_name='out_arr')

                        glb_t.reset()
                        for t in range(1,self.conf.time_step+1):
                            layer_out= layer_out.write(t-1,_layer_out)

                            glb_t()

                    else:
                        glb_t.reset()
                        for t in range(1,self.conf.time_step+1):

                            if isinstance(layer_in,list) :
                                _layer_in = [_layer_in.read(t-1) for _layer_in in layer_in]
                            else:
                                _layer_in = layer_in.read(t-1)

                            _layer_out = node.layer(_layer_in)

                            if t-1==0:
                                layer_out = tf.TensorArray(
                                    dtype=tf.float32,
                                    size=self.conf.time_step,
                                    element_shape=_layer_out.shape,
                                    clear_after_read=False,
                                    tensor_array_name='out_arr')

                            layer_out = layer_out.write(t-1,_layer_out)

                            glb_t()

                #for x_id, y in zip(node.flat_output_ids, tf.nest.flatten(layer_out.read(self.conf.time_step-1))):
                for x_id, y in zip(node.flat_output_ids, tf.nest.flatten(layer_out)):
                    tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

        output_tensors = []
        for x in self.outputs:
            x_id = str(id(x))
            assert x_id in tensor_dict, 'Could not compute output ' + str(x)
            #output_tensors.append(tensor_dict[x_id].pop())

            output_tensors.append(tensor_dict[x_id].pop().read(self.conf.time_step-1))

        return tf.nest.pack_sequence_as(self._nested_outputs, output_tensors)

    ###########################################################
    # run internal graph - SNN, spatial first
    # based on _run_internal_graph() function in keras.engine.functional.py
    ###########################################################
    def _run_internal_graph_snn_s_first(self, inputs, training=None, mask=None):
        range_ts = range(1, self.conf.time_step + 1)

        glb_t.reset()
        for t in range_ts:
            ret = self._run_internal_graph(inputs, training=training, mask=mask)
            glb_t()

        return ret



    def print_snn_train(self,grads_accum_and_vars,path_root=None):
        import pandas as pd

        print('\n grads_accum')
        for grad_accum, var in grads_accum_and_vars:
            #print('{: <10}: - sum {:.3e}, mean {:.3e}'.format(var.name,tf.reduce_sum(grad_accum),tf.reduce_mean(grad_accum)))
            print('{: <10}: - max {:.3e}, min {:.3e}, mean {:.3e}, var {:.3e}'
                  .format(var.name,tf.reduce_max(grad_accum),tf.reduce_min(grad_accum),
                          tf.reduce_mean(grad_accum),tf.math.reduce_std(grad_accum)))

            if self.conf.verbose_visual:
                if 'kernel' in var.name:
                    lib_snn.util.plot_hist(glb_plot_gradient_kernel,grad_accum,1000)
                elif 'gamma' in var.name:
                    lib_snn.util.plot_hist(glb_plot_gradient_gamma,grad_accum,1000)
                elif 'beta' in var.name:
                    lib_snn.util.plot_hist(glb_plot_gradient_beta,grad_accum,1000)

            try:
                dout_ga
            except NameError:
                dout_ga = pd.DataFrame(columns=['max','min','mean','std'])
            dout_ga.loc[var.name,'max'] = tf.reduce_max(grad_accum).numpy()
            dout_ga.loc[var.name,'min'] = tf.reduce_min(grad_accum).numpy()
            dout_ga.loc[var.name,'mean'] = tf.reduce_mean(grad_accum).numpy()
            dout_ga.loc[var.name,'std'] = tf.math.reduce_std(grad_accum).numpy()

        print('\n spike count')
        for layer in self.layers:
            if hasattr(layer,'act') and isinstance(layer.act,lib_snn.neurons.Neuron):
                spike_count = layer.act.spike_count_int
                #spike = layer.act.out
                print('{: <10}: - max {:.3e}, mean {:.3e}, non-zero percent {:.3e}'
                      .format(layer.name,tf.reduce_max(spike_count),tf.reduce_mean(spike_count)
                              ,tf.math.count_nonzero(spike_count,dtype=tf.float32)/tf.cast(tf.reduce_prod(spike_count.shape),dtype=tf.float32)))

                try:
                    dout_sc
                except NameError:
                    dout_sc = pd.DataFrame(columns=['max','min','mean','std'])
                dout_sc.loc[layer.name,'max'] = tf.reduce_max(spike_count).numpy()
                dout_sc.loc[layer.name,'min'] = tf.reduce_min(spike_count).numpy()
                dout_sc.loc[layer.name,'mean'] = tf.reduce_mean(spike_count).numpy()
                dout_sc.loc[layer.name,'std'] = tf.math.reduce_std(spike_count).numpy()

        print('\n vmem residual')
        for layer in self.layers:
            if hasattr(layer,'act') and isinstance(layer.act,lib_snn.neurons.Neuron):
                vmem = layer.act.vmem.read(self.conf.time_step-1)
                print('{: <10}: - max {:.3e}, min {:.3e}, mean {:.3e}, var {:.3e}'
                      .format(layer.name,tf.reduce_max(vmem),tf.reduce_min(vmem),
                              tf.reduce_mean(vmem),tf.math.reduce_variance(vmem)))
                try:
                    dout_vr
                except NameError:
                    dout_vr = pd.DataFrame(columns=['max','min','mean','std'])
                dout_vr.loc[layer.name,'max'] = tf.reduce_max(vmem).numpy()
                dout_vr.loc[layer.name,'min'] = tf.reduce_min(vmem).numpy()
                dout_vr.loc[layer.name,'mean'] = tf.reduce_mean(vmem).numpy()
                dout_vr.loc[layer.name,'std'] = tf.math.reduce_std(vmem).numpy()



                #print('spike - sum {:.3e}, mean {:.3e}'.format(tf.reduce_sum(spike),tf.reduce_mean(spike)))

        #
        print('\n kernel ')
        for layer in self.layers:
            if hasattr(layer,'kernel'):
                print('{: <10} max {:.3e}, min {:.3e}, mean {:.3e}, sq {:.3e}'
                      .format(layer.name,
                              tf.reduce_max(layer.kernel),
                              tf.reduce_min(layer.kernel),
                              tf.reduce_mean(layer.kernel),
                              tf.reduce_sum(tf.math.pow(layer.kernel,2))))

                try:
                    dout_k
                except NameError:
                    dout_k = pd.DataFrame(columns=['max','min','mean','std'])
                dout_k.loc[layer.name,'max'] = tf.reduce_max(layer.kernel).numpy()
                dout_k.loc[layer.name,'min'] = tf.reduce_min(layer.kernel).numpy()
                dout_k.loc[layer.name,'mean'] = tf.reduce_mean(layer.kernel).numpy()
                dout_k.loc[layer.name,'std'] = tf.math.reduce_std(layer.kernel).numpy()

        #
        print('\n bias ')
        for layer in self.layers:
            if hasattr(layer,'bias'):
                print('{: <10} max {:.3e}, min {:.3e}, mean {:.3e}, sq {:.3e}'
                      .format(layer.name,
                              tf.reduce_max(layer.bias),
                              tf.reduce_min(layer.bias),
                              tf.reduce_mean(layer.bias),
                              tf.reduce_sum(tf.math.pow(layer.bias,2))))

                try:
                    dout_b
                except NameError:
                    dout_b = pd.DataFrame(columns=['max','min','mean','std'])
                dout_b.loc[layer.name,'max'] = tf.reduce_max(layer.bias).numpy()
                dout_b.loc[layer.name,'min'] = tf.reduce_min(layer.bias).numpy()
                dout_b.loc[layer.name,'mean'] = tf.reduce_mean(layer.bias).numpy()
                dout_b.loc[layer.name,'std'] = tf.math.reduce_std(layer.bias).numpy()


        #
        print('\n bn - moving mean')
        for layer in self.layers:
            #if hasattr(layer,'bn') and layer.bn is not None:
            if isinstance(layer,lib_snn.layers.BatchNormalization):
                bn_var = layer.moving_mean
                print('{: <10} max {:.3e}, min {:.3e}, mean {:.3e}, sq {:.3e}'
                      .format(layer.name,
                              tf.reduce_max(bn_var),
                              tf.reduce_min(bn_var),
                              tf.reduce_mean(bn_var),
                              tf.reduce_sum(tf.math.pow(bn_var,2))))
                try:
                    dout_bn_m
                except NameError:
                    dout_bn_m = pd.DataFrame(columns=['max','min','mean','std'])
                dout_bn_m.loc[layer.name,'max'] = tf.reduce_max(bn_var).numpy()
                dout_bn_m.loc[layer.name,'min'] = tf.reduce_min(bn_var).numpy()
                dout_bn_m.loc[layer.name,'mean'] = tf.reduce_mean(bn_var).numpy()
                dout_bn_m.loc[layer.name,'std'] = tf.math.reduce_std(bn_var).numpy()
        #
        print('\n bn - moving variance')
        for layer in self.layers:
            if isinstance(layer,lib_snn.layers.BatchNormalization):
                bn_var = layer.moving_variance
                print('{: <10} max {:.3e}, min {:.3e}, mean {:.3e}, sq {:.3e}'
                      .format(layer.name,
                              tf.reduce_max(bn_var),
                              tf.reduce_min(bn_var),
                              tf.reduce_mean(bn_var),
                              tf.reduce_sum(tf.math.pow(bn_var,2))))
                try:
                    dout_bn_v
                except NameError:
                    dout_bn_v = pd.DataFrame(columns=['max','min','mean','std'])
                dout_bn_v.loc[layer.name,'max'] = tf.reduce_max(bn_var).numpy()
                dout_bn_v.loc[layer.name,'min'] = tf.reduce_min(bn_var).numpy()
                dout_bn_v.loc[layer.name,'mean'] = tf.reduce_mean(bn_var).numpy()
                dout_bn_v.loc[layer.name,'std'] = tf.math.reduce_std(bn_var).numpy()

        #
        print('\n bn - gamma')
        for layer in self.layers:
            if isinstance(layer,lib_snn.layers.BatchNormalization):
                bn_var = layer.gamma
                print('{: <10} max {:.3e}, min {:.3e}, mean {:.3e}, sq {:.3e}'
                      .format(layer.name,
                              tf.reduce_max(bn_var),
                              tf.reduce_min(bn_var),
                              tf.reduce_mean(bn_var),
                              tf.reduce_sum(tf.math.pow(bn_var,2))))
                try:
                    dout_bn_g
                except NameError:
                    dout_bn_g = pd.DataFrame(columns=['max','min','mean','std'])
                dout_bn_g.loc[layer.name,'max'] = tf.reduce_max(bn_var).numpy()
                dout_bn_g.loc[layer.name,'min'] = tf.reduce_min(bn_var).numpy()
                dout_bn_g.loc[layer.name,'mean'] = tf.reduce_mean(bn_var).numpy()
                dout_bn_g.loc[layer.name,'std'] = tf.math.reduce_std(bn_var).numpy()

        #
        print('\n bn - beta')
        for layer in self.layers:
            if isinstance(layer,lib_snn.layers.BatchNormalization):
                bn_var = layer.beta
                print('{: <10} max {:.3e}, min {:.3e}, mean {:.3e}, sq {:.3e}'
                      .format(layer.name,
                              tf.reduce_max(bn_var),
                              tf.reduce_min(bn_var),
                              tf.reduce_mean(bn_var),
                              tf.reduce_sum(tf.math.pow(bn_var,2))))
                try:
                    dout_bn_b
                except NameError:
                    dout_bn_b = pd.DataFrame(columns=['max','min','mean','std'])
                dout_bn_b.loc[layer.name,'max'] = tf.reduce_max(bn_var).numpy()
                dout_bn_b.loc[layer.name,'min'] = tf.reduce_min(bn_var).numpy()
                dout_bn_b.loc[layer.name,'mean'] = tf.reduce_mean(bn_var).numpy()
                dout_bn_b.loc[layer.name,'std'] = tf.math.reduce_std(bn_var).numpy()


        if path_root is None:
            path_root = 'snn_train_log'

        os.makedirs(path_root,exist_ok=True)

        prefix = 'tdbn_tf_ts-'+str(self.conf.time_step)
        postfix = 'vth-'+str(self.conf.n_init_vth)+'.csv'

        #
        file_dout_ga='ga'
        file_dout_sc='sc'
        file_dout_vr='vr'
        file_dout_k='k'
        file_dout_b='b'
        file_dout_bn_m='bn_m'
        file_dout_bn_v='bn_v'
        file_dout_bn_g='bn_g'
        file_dout_bn_b='bn_b'

        #
        files = collections.OrderedDict()
        files['dout_ga']=[dout_ga,file_dout_ga]
        files['dout_sc']=[dout_sc,file_dout_sc]
        files['dout_vr']=[dout_vr,file_dout_vr]
        files['dout_k']=[dout_k,file_dout_k]
        files['dout_b']=[dout_b,file_dout_b]
        files['dout_bn_m']=[dout_bn_m,file_dout_bn_m]
        files['dout_bn_v']=[dout_bn_v,file_dout_bn_v]
        files['dout_bn_g']=[dout_bn_g,file_dout_bn_g]
        files['dout_bn_b']=[dout_bn_b,file_dout_bn_b]

        #
        for key, [dout,file] in files.items():
            f = prefix+'_'+file+'_'+postfix
            f = os.path.join(path_root,f)
            dout.to_csv(f)


    #
    def summary(self,
                line_length=None,
                positions=None,
                print_fn=None,
                expand_nested=False,
                show_trainable=False):
        super(Model, self).summary(line_length=line_length, positions=positions,
                           print_fn=print_fn, expand_nested=expand_nested, show_trainable=show_trainable)

        if self.total_num_neurons == None:
            self.cal_total_num_neurons()

        self.print_total_num_neurons()


    #
    def cal_total_num_neurons(self):
        total_num_neurons = 0
        #for l in self.layers_w_neuron:
        for l in self.layers:
            if hasattr(l, 'act'):
                if isinstance(l.act, lib_snn.neurons.Neuron):
                    total_num_neurons += l.act.num_neurons

        self.total_num_neurons = total_num_neurons

    def print_total_num_neurons(self):
        print('total num neurons: {:}'.format(self.total_num_neurons))


    # stdp - pathway
    #def stdp_pathway(self):
    #    for layer_name, (pre_name, post_name) in self.syn_pre_post.items():
            #pre = self.get_layer(pre_name).act.spike_trace
            #post = self.get_layer(post_name).act.spike_trace

