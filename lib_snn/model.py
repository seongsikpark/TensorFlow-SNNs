
import tensorflow as tf

import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers

#
import os
import csv

#
import numpy as np
import collections

#
import lib_snn



#
class Model(tf.keras.Model):
    count=0
    def __init__(self, input_shape, data_format, num_class, conf, **kwargs):
        #print("lib_SNN - Layer - init")


        lmb = kwargs.pop('lmb', None)
        n_dim_classifier = kwargs.pop('n_dim_classifier', None)


        #
        super(Model, self).__init__(**kwargs)

        #
        Model.count += 1
        #assert Model.count==1, 'We have only one Model instance'

        #
        self.verbose = conf.verbose

        #
        Model.conf = conf
        Model.data_format = data_format
        self.kernel_size = None                 # for conv layer
        #self.num_class = conf.num_class
        self.num_class = num_class
        Model.use_bias = conf.use_bias

        #
        Model.f_1st_iter = True
        self.f_1st_iter_stat = True
        Model.f_load_model_done = False
        self.f_debug_visual = conf.verbose_visual
        self.f_done_preproc = False
        #self.f_skip_bn = False      # for the 1st iteration
        Model.f_skip_bn = False      # for the 1st iteration
        Model.f_dummy_run=True

        # keras model
        self.model = None

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


        # input
        #self._input_shape = [-1]+input_shape.as_list()
        self._input_shape = [-1]+list(input_shape)
        self.in_shape = [self.conf.batch_size]+self._input_shape[1:]
        self.in_shape_snn = [self.conf.batch_size] + self._input_shape[1:]


        # output
        if False:
            self.count_accuracy_time_point=0
            self.accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
            self.accuracy_time_point.append(conf.time_step)
            self.num_accuracy_time_point = len(self.accuracy_time_point)
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

        regularizer_type = {
            #'L1': regularizers.l1(conf.lmb),
            #'L2': regularizers.l2(conf.lmb)
            'L1': regularizers.l1(lmb),
            'L2': regularizers.l2(lmb),
        }

        Model.kernel_regularizer = regularizer_type[self.conf.regularizer]

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
        self.en_train = self.conf.en_train

        # SNN mode
        Model.en_snn = (self.conf.nn_mode == 'SNN' or self.conf.f_validation_snn)

        # DNN-to-SNN conversion, save dist. act. of DNN
        self.en_write_stat = (self.conf.nn_mode=='ANN' and self.conf.f_write_stat)

        # SNN, temporal coding, time const. training after DNN-to-SNN conversion (T2FSNN + GO)
        self.en_opt_time_const_T2FSNN = (self.conf.nn_mode=='SNN' and not self.conf.en_train \
                                         and self.conf.neural_coding=='TEMPORAL' and self.conf.f_train_tk)

        # comparison activation - ANN vs. SNN
        self.en_comp_act = (self.conf.nn_mode=='SNN' and self.conf.f_comp_act)



        # data-based weight normalization
        if self.conf.f_w_norm_data:
            self.norm=collections.OrderedDict()
            self.norm_b=collections.OrderedDict()

        # debugging
        if self.f_debug_visual:
            #self.debug_visual_threads = []
            self.debug_visual_axes = []
            self.debug_visual_list_neuron = collections.OrderedDict()




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

        # TODO: np -> tf.Variables
        self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.list_layer_name)+1])
        self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.list_layer_name)+1])
        self.total_residual_vmem=np.zeros(len(self.list_layer_name)+1)


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

            self.snn_output = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_neuron.dim)),dtype=tf.float32,trainable=False)
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

    def call(self, inputs, training=False, epoch=-1, f_val_snn=False):

        ret_val = self.call_ann(inputs, training)

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
                ret_val = self.run_mode[self.conf.nn_mode](inputs,training,self.conf.time_step,epoch)

                # training time constant
                self.train_time_const()
            else:
                # inference - rate, phase, and burst coding
                if f_val_snn:
                    assert False, 'f_val_snn mode is not validated yet'
                    ret_val = self.call_snn(inputs,training,self.conf.time_step,epoch)
                else:
                    ret_val = self.run_mode[self.conf.nn_mode](inputs,training,self.conf.time_step,epoch)


            # post-processing
            self.postproc(inputs)
        else:
            print('Dummy run')

            if self.conf.nn_mode=='SNN' and self.conf.f_surrogate_training_model:
                ret_val = self.call_ann_surrogate_training(inputs,False,self.conf.time_step,epoch)

            # validation on SNN
            if self.conf.en_train and self.conf.f_validation_snn:
                ret_val = self.call_snn(inputs,False,1,0)

            #ret_val = self.run_mode_load_model[self.conf.nn_mode](inputs,training,self.conf.time_step,epoch)
            #ret_val = self.run_mode_load_model[self.conf.nn_mode](inputs,False,self.conf.time_step,epoch)
            ret_val = self.run_mode_load_model[self.conf.nn_mode](inputs,False,2,epoch)

            Model.f_load_model_done=True

            #
            #if self.f_1st_iter and self.conf.nn_mode=='ANN':
                #print('1st iter - dummy run')
                #self.f_1st_iter = False
                #self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)

            self.f_1st_iter = False

            Model.f_skip_bn = (self.conf.nn_mode=='ANN' and self.conf.f_fused_bn) or (self.conf.nn_mode=='SNN')

        return ret_val

    #
    def call_ann(self,inputs,training, tw=0, epoch=0):
#
        #a_in = tf.reshape(inputs,self._input_shape)
        a_in = inputs
        a_out = self.model(a_in,training=training)

        return a_out

    #
    def call_snn(self,inputs,training, tw, epoch):

        #
        for t in range(tw):
            self.t=t
            #a_in = tf.reshape(inputs,self._input_shape)
            a_in = inputs
            a_out = self.model(a_in,training=training)

            #
            self.postproc_snn_time_step()

        return self.snn_output


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
            preproc_sel[self.conf.nn_mode](inputs, training)


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

        if self.conf.f_fused_bn or ((self.conf.nn_mode=='ANN')and(self.conf.f_validation_snn)):
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

    #
    def reset_snn_neuron(self):
        for layer in self.list_layer:
            layer.act.reset()


    ###########################################################
    # BN fusion
    ###########################################################
    #
    def bn_fusion(self):
        print('---- BN Fusion ----')

        for name_l in self.list_layer_name:
            layer = self.model.get_layer(name=name_l)
            layer.bn_fusion()

        print('---- BN Fusion Done ----')

    #
    def bn_defusion(self):
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
        postproc_sel[self.conf.nn_mode](inputs)

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
        if self.f_debug_visual:
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


    ###########################################################
    ## Print
    ###########################################################
    #
    def print_model_conf(self):
        self.model.summary()




