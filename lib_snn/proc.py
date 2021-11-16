
#
import tensorflow as tf

#
import collections
import os
import csv
import numpy as np
import pandas as pd

#
import matplotlib.pyplot as plt

#
import lib_snn

from lib_snn.sim import glb_t
from lib_snn.sim import glb_plot


########################################
# preprocessing (on_test_begin)
########################################

########################################
# setting (on_test_begin)
########################################
def preproc(self):
    # print summary model
    print('summary model')
    self.model.summary()

    # initialization
    set_init(self)

    #
    preproc_sel={
        'ANN': preproc_ann,
        'SNN': preproc_snn,
    }
    preproc_sel[self.conf.nn_mode](self)



#
def reset_batch_ann(self):
    pass

def reset_batch_snn(self):

    # TODO: move to model.py
    self.model.reset_snn()
    self.model.reset_snn_neuron()
    reset_snn_time_step(self)

    if self.conf.calibration_vmem:
        lib_snn.calibration.vmem_calibration(self)

#
def set_init(self):
    #print('SNNLIB callback initialization')

    for layer in self.model.layers:
        if hasattr(layer, 'kernel'):
            self.layers_w_kernel.append(layer)

    self.en_record_output = self.model._run_eagerly and ((self.conf.nn_mode=='ANN' and self.conf.f_write_stat) or self.conf.debug_mode)
    #self.en_record_output = True

    if self.en_record_output:
        self.layers_record=self.layers_w_kernel

       #self.layers_record = []
        #for layer in self.layers_w_kernel[:2]:
            #self.layers_record.append(layer)

        set_en_record_output(self)

    #
    if self.conf.nn_mode=='SNN':
        init_snn_run(self)

#
def init_snn_run(self):
    self.model.init_snn()

    #self.model.set_layers_w_neuron()

    # spike max pool setup
    #self.model.spike_max_pool_setup()

#
def set_en_record_output(self):
    #for layer in self.model.layers:
    for layer in self.layers_record:
        layer.en_record_output = True

    for layer in self.model.layers:
        if isinstance(layer, lib_snn.layers.InputGenLayer):
            layer.en_record_output = True

    self.dict_stat_w=collections.OrderedDict()

#
def preproc_ann(self):
    ##self.print_model_conf()
    preproc_ann_norm(self)

    # surrogate DNN model for training SNN with temporal information
    if self.conf.f_surrogate_training_model:
        assert False
        #self.preproc_surrogate_training_model()

#
def preproc_snn(self):

    #
    preproc_ann_to_snn(self)

#
def preproc_ann_norm(self):
    if self.conf.f_fused_bn:
        bn_fusion(self)

    # data-based weight normalization
    if self.conf.f_w_norm_data:
        w_norm_data(self)

    #print(self.model.get_layer('conv1').bias)
    ## for debug
    #self.model.bias_disable()
    #print(self.model.get_layer('conv1').bias)


#
def preproc_ann_to_snn(self):
    if self.conf.verbose:
        print('preprocessing: ANN to SNN')

    if self.conf.f_fused_bn or ((self.conf.nn_mode == 'ANN') and (self.conf.f_validation_snn)):
        bn_fusion(self)

    # data-based weight normalization
    if self.conf.f_w_norm_data:
        w_norm_data(self)



########################################
# reset (on_test_batch_begin)
########################################
def preproc_batch(self):
    #print('on_test_batch_begin')

    # reset
    reset_batch_sel={
        'ANN': reset_batch_ann,
        'SNN': reset_batch_snn,
    }
    reset_batch_sel[self.conf.nn_mode](self)

    #
    preproc_batch_sel={
        'ANN': preproc_batch_ann,
        'SNN': preproc_batch_snn,
    }
    preproc_batch_sel[self.conf.nn_mode](self)

def preproc_batch_ann(self):
    pass

def preproc_batch_snn(self):
    pass
    #reset_snn_sample(self)

#
def reset_snn_sample(self):
    assert False
    reset_snn_time_step(self)
    self.reset_snn_neuron()
    #self.snn_output = np.zeros((self.num_accuracy_time_point,)+self.list_neuron['fc1'].get_spike_count().numpy().shape)
    self.snn_output.assign(tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_neuron.dim)))
    self.count_accuracy_time_point=0

#
def reset_snn_time_step(self):
    glb_t.reset()


########################################
# (on_test_batch_end)
########################################
def postproc_batch(self):
    if self.en_record_output:
        collect_record_output(self)



#
def collect_record_output(self):
    with tf.device('CPU:0'):
        for layer in self.layers_record:
            #print(layer)

            #print(layer.record_output)

            if not (layer.name in self.dict_stat_w.keys()):
                #self.dict_stat_w[layer.name] = layer.record_output.numpy()
                self.dict_stat_w[layer.name] = tf.Variable(layer.record_output)
            else:
                self.dict_stat_w[layer.name] = tf.concat([self.dict_stat_w[layer.name],layer.record_output],0)
                #prev = self.dict_stat_w[layer.name]
                #new = layer.record_output.numpy()
                #self.dict_stat_w[layer.name] = np.concatenate((prev,new),axis=0)
                #self.dict_stat_w[layer.name].concate

    #print(self.dict_stat_w['conv1'].shape)
    #assert False, 'start here'

    #assert False



########################################
# (on_test_end)
########################################
def postproc(self):

    #
    postproc_sel={
        'ANN': postproc_ann,
        'SNN': postproc_snn,
    }
    postproc_sel[self.conf.nn_mode](self)

def postproc_ann(self):
    #
    if self.conf.f_write_stat:
        lib_snn.weight_norm.save_act_stat(self)
        return

    # TODO
    dnn_snn_compare=True
    if (not self.conf.full_test) and dnn_snn_compare:
        #idx=7
        plot_dnn_act(self)
        #plot_act_dist(self)
        pass

#
def plot_act_dist(self,fig=None):

    if fig is None:
        fig = glb_plot

    for layer in self.layers_w_kernel:
        axe = fig.axes.flatten()[layer.depth]
        if self.conf.nn_mode=='ANN':
            act = layer.record_output.numpy().flatten()
            bins = np.arange(0,1,0.03)
        else:
            act = layer.act.spike_count_int.numpy().flatten()
            bins = np.arange(0,self.conf.time_step,1)

        (n, bins, patches) = axe.hist(act,bins=bins)
        axe.axvline(x=np.max(act), color='b')
        axe.set_ylim([0,n[10]])
        axe.set_title(layer.name)

    plt.show()


# TODO: move
def plot_dnn_act(self, layers=None, idx=None):

    #
    for idx_layer, layer in enumerate(glb_plot.layers):
        layer = self.model.get_layer(layer)
        axe = glb_plot.axes.flatten()[idx_layer]
        idx_neuron = glb_plot.idx_neurons[idx_layer]

        act = layer.record_output.numpy().flatten()[idx_neuron]
        axe.axhline(y=act, color='b')

        idx_bias = idx_neuron % layer.bias.shape[0]
        axe.axhline(y=layer.bias[idx_bias], color='m')

        #if self.conf.f_w_norm_data:
            #axe.axhline(y=(self.dict_stat_r[layer.name]/self.norm_b[layer.name]).flatten()[idx_neuron], color='r')

    if False:
        # layers in model
        idx=glb_plot.idx
        for layer in self.layers_w_kernel:
            #print(layer.name)
            #print(layer.depth)

            axe = glb_plot.axes.flatten()[layer.depth]

            # idx=0,0,0,0
            #spike = layer.act.get_spike_count_int().numpy().flatten()[idx]
            # print('{} - {}, {}'.format(glb_t.t,spike,spike/glb_t.t))
            #lib_snn.util.plot(glb_t.t, spike / glb_t.t, axe=axe)
            act = layer.record_output.numpy().flatten()[idx]
            axe.axhline(y=act, color='b')


def postproc_snn(self):
    # results
    cal_results(self)

    # print results
    print_results(self)

    #
    if self.conf.full_test:
        save_results(self)

    #
    dnn_snn_compare=True
    if (not self.conf.full_test) and dnn_snn_compare:
        fig = lib_snn.sim.GLB_PLOT()
        plot_act_dist(self, fig)

#
def cal_results(self):
    self.results_acc = np.zeros(self.model.num_accuracy_time_point)
    self.results_spike = np.zeros(self.model.num_accuracy_time_point)

    for idx in range(self.model.num_accuracy_time_point):
        self.results_acc[idx] = self.model.accuracy_results[idx]['acc'].numpy()

    for layer_spike in self.model.total_spike_count_int.values():
        self.results_spike += layer_spike

    self.results_df = pd.DataFrame({'time step': self.model.accuracy_time_point, 'accuracy': self.results_acc,
                                    'spike count': self.results_spike / self.test_ds_num})
    self.results_df.set_index('time step', inplace=True)


def print_results(self):
    pd.set_option('display.float_format', '{:.4g}'.format)

    #
    # df=pd.DataFrame({'time step': model.accuracy_time_point, 'spike count': list(model.total_spike_count[:,-1]),'accuracy': accuracy_result})
    # df=pd.DataFrame({'time step': model.accuracy_time_point, 'accuracy': accuracy_result, 'spike count': model.total_spike_count_int[:,-1]})
    print(self.results_df)

def save_results(self):

    model_dataset = self.conf.model+'_'+self.conf.dataset

    # TODO: modify
    if self.conf.f_w_norm_data:
        _norm = 'M999'
    else:
        _norm = 'NO'

    _n = self.conf.n_type
    _in = self.conf.input_spike_mode
    _nc = self.conf.neural_coding
    _ts = self.conf.time_step
    _tsi = self.conf.time_step_save_interval
    _vth = self.conf.n_init_vth

    config = 'norm-{}_n-{}_in-{}_nc-{}_ts-{}-{}_vth-{}'.format(_norm,_n,_in,_nc,_ts,_tsi,_vth)

    # bias control
    if self.conf.bias_control:
        config += '_bc'
    #
    if self.conf.vth_toggle:
        config += '_vth-tg-'+str(self.conf.vth_toggle_init)

    file = config+'.xlsx'

    # set path
    path = self.conf.root_results
    path = os.path.join(path,model_dataset)
    f_name_result = os.path.join(path,file)

    #
    os.makedirs(path,exist_ok=True)


    #
    print('output file: {}'.format(f_name_result))
    self.results_df.to_excel(f_name_result)


########################################
# processing kernel
########################################
def bn_fusion(self):
    print('---- BN Fusion ----')
    # bn fusion
    if (self.conf.nn_mode == 'ANN' and self.conf.f_fused_bn) or (self.conf.nn_mode == 'SNN'):
        for layer in self.model.layers:
            if hasattr(layer, 'bn') and (layer.bn is not None):
                layer.bn_fusion()

    print('---- BN Fusion Done ----')


def w_norm_data(self):
    # weight normalization - data based

    print('---- Data-based weight normalization ----')

    self.dict_stat_r = collections.OrderedDict()

    self.norm = collections.OrderedDict()
    self.norm_b = collections.OrderedDict()

    #path_stat=self.conf.path_stat
    ##f_name_stat_pre=self.conf.prefix_stat
    #model_dataset = self.conf.model+'_'+self.conf.dataset
    #f_name_stat_pre=model_dataset
    #path_stat = os.path.join(path_stat,model_dataset)

    path_stat = os.path.join(self.path_model,self.conf.path_stat)

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
    #stat='max_80'
    #stat='max_75'
    #stat='max_25'
    #stat='max_60'
    #stat='max_40'
    #stat='median'
    #stat='mean'

    #if self.conf.calibration_weight:?
        #stat='max_90'

    #for idx_l, l in enumerate(self.list_layer_name):
    #for idx_l, l in enumerate(self.list_layer):
    for idx_l, l in enumerate(self.layers_w_kernel):
        key=l.name+'_'+stat

        #f_name_stat = f_name_stat_pre+'_'+key
        f_name_stat = key
        f_name=os.path.join(path_stat,f_name_stat)
        f_stat[key]=open(f_name,'r')
        r_stat[key]=csv.reader(f_stat[key])

        # TODO: np -> tf.Variable
        for row in r_stat[key]:
            #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
            self.dict_stat_r[l.name]=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

        f_stat[key].close()

    #
    f_norm = np.max
    #f_norm = np.median
    #f_norm = np.mean
    #f_norm = lambda x: np.percentile(x,80)

    #w_norm_data_layer_wise(self,f_norm)
    w_norm_data_channel_wise(self,f_norm)

    #
    if self.conf.nn_mode=='SNN':
        if self.conf.calibration_vth:
            lib_snn.calibration.vth_calibration(self,f_norm,stat)

        if self.conf.vth_toggle:
            lib_snn.calibration.vth_toggle(self)

    #
    if self.conf.calibration_weight:
        lib_snn.calibration.weight_calibration(self)
        #lib_snn.calibration.weight_calibration_inv_vth(self)

    #
    if self.conf.calibration_bias:
        lib_snn.calibration.bias_calibration(self)


#
def w_norm_data_layer_wise(self, f_norm):

    print('layer-wise normalization')


    # for idx_l, l in enumerate(self.list_layer_name):
    #for idx_l, l in enumerate(self.list_layer):
    if 'VGG' in self.conf.model:
        for idx_l, l in enumerate(self.layers_w_kernel):
            stat = self.dict_stat_r[l.name]
            norm = f_norm(stat)
            norm = np.where(norm==0, 1, norm)

            if idx_l == 0:
                self.norm[l.name] = norm
            else:
                self.norm[l.name] = norm / self.norm_b[prev_name]

            #self.norm_b[l.name] = f_norm(self.dict_stat_r[l.name])
            self.norm_b[l.name] = norm
            prev_name=l.name
    else:
        assert False


    # print
    #print('norm weight')
    #for k, v in self.norm.items():
        #print(k + ': ' + str(v))

    #print('norm bias')
    #for k, v in self.norm_b.items():
        #print(k + ': ' + str(v))

    #
    # for name_l in self.list_layer_name:
    for layer in self.layers_w_kernel:
        # layer = self.model.get_layer(name=name_l)
        layer.kernel = layer.kernel / self.norm[layer.name]
        layer.bias = layer.bias / self.norm_b[layer.name]



#
def w_norm_data_channel_wise(self, f_norm):

    print('channel-wise normalization')

    #
    f_norm = lambda x: np.max(x, axis=0)

    # for idx_l, l in enumerate(self.list_layer_name):
    #for idx_l, l in enumerate(self.list_layer):
    if 'VGG' in self.conf.model:
        for idx_l, l in enumerate(self.layers_w_kernel):
            stat = self.dict_stat_r[l.name]
            stat = stat.reshape(-1,stat.shape[-1])
            norm = f_norm(stat)
            norm = np.where(norm==0, 1, norm)

            if idx_l == 0:
                self.norm[l.name] = norm
            else:
                #stat = self.dict_stat_r[l.name]
                #stat = stat.reshape(-1, stat.shape[-1])
                #norm = f_norm(stat)

                #print(stat)
                #print(stat.shape)
                #print(l.kernel.shape)
                #print(self.norm[l.name])

                #assert False

                #self.norm[l.name] = (f_norm(stat).T / self.norm[prev_name]).T
                #self.test = np.expand_dims(norm,axis=0).T
                self.norm[l.name] = norm / np.expand_dims(self.norm_b[prev_name],axis=0).T

                #print(self.norm[l.name])
                #assert False

            #self.norm_b[l.name] = f_norm(self.dict_stat_r[l.name])
            self.norm_b[l.name] = norm
            prev_name=l.name

        #
        # for visual debug
        #lib_snn.util.print_act_stat_r(self)

        #assert False
    else:
        assert False


    # print
    #print('norm weight')
    #for k, v in self.norm.items():
        #print(k + ': ' + str(v))

    #print('norm bias')
    #for k, v in self.norm_b.items():
        #print(k + ': ' + str(v))

    #
    # for name_l in self.list_layer_name:
    for layer in self.layers_w_kernel:
        # layer = self.model.get_layer(name=name_l)
        layer.kernel = layer.kernel / self.norm[layer.name]
        layer.bias = layer.bias / self.norm_b[layer.name]



