
import tensorflow as tf

import numpy as np


########################################
# write activation distributions
# for data-based weight normalization
# during DNN-to-SNN conversion
########################################

def init_write_stat(self):
    self.f_1st_iter_stat = True

    for l_name in self.list_layer_name_write_stat:
        #self.dict_stat_w[l_name]=tf.Variable(initial_value=tf.zeros(self.list_shape[l_name]),trainable=None)
        output_shape = self.model.get_layer(l_name).output_shape
        print(output_shape)
        #output_shape = self.list_shape[l_name]
        self.dict_stat_w[l_name]=tf.Variable(initial_value=tf.zeros(output_shape),trainable=None)


########################################
# comparision acitvation - ANN vs. SNN
########################################
def init_anal_comp_act(self):
    # TODO: change data type, np -> tf.Variable
    self.total_comp_act=np.zeros([self.conf.time_step,len(self.layer_name)+1])

    self.f_1st_iter_stat = True

    for l_name in self.list_layer_name_write_stat:
        self.dict_stat_w[l_name]=tf.Variable(initial_value=tf.zeros(self.list_shape[l_name]),trainable=None)

#
def save_ann_act(self,inputs,f_training):
    self.call_ann(inputs,f_training)
    _save_ann_act(self)

#
# TODO: change data type, np -> tf.Variable
def _save_ann_act(self):
    for l_name in self.list_layer_name:
        self.dict_stat_w[l_name] = self.dnn_act_list[l_name].numpy()

#
def comp_act(self):
    if self.conf.neural_coding=='RATE':
        self.comp_act_rate(t)
    elif self.conf.neural_coding=='WEIGHTED_SPIKE':
        self.comp_act_ws(t)
    elif self.conf.neural_coding=='BURST':
        self.comp_act_burst(t)
    else:
        assert False, 'not supported neural coding {}'.format(self.conf.neural_coding)

#
# TODO: change data type, np -> tf.Variable
def comp_act_rate(self,t):
    self.total_comp_act[t,-1]=0.0
    for idx_l, l in enumerate(self.layer_name):
        if l !='fc3':
            self.total_comp_act[t,idx_l]=np.mean(np.abs(self.list_neuron[l].spike_counter.numpy().flatten()/(float)(t+1)-self.dict_stat_w[l].flatten()))
            self.total_comp_act[t,-1]+=self.total_comp_act[t,idx_l]


    #l='conv1'
    #print(self.list_neuron[l].spike_counter.numpy().flatten())
    #print(self.dict_stat_w[l].flatten())

#
def comp_act_ws(self,t):
    self.total_comp_act[t,-1]=0.0
    for idx_l, l in enumerate(self.layer_name):
        if l !='fc3':
            #self.total_comp_act[t,idx_l]=np.mean(np.abs(self.list_neuron[l].spike_counter.numpy().flatten()/((float)(t+1)/(float)(self.conf.p_ws))-self.dict_stat_w[l].flatten()))
            self.total_comp_act[t,idx_l]=np.mean(np.abs(self.list_neuron[l].spike_counter.numpy().flatten()/((float)(t+1)/(float)(self.conf.p_ws))-self.dict_stat_w[l].flatten()))
            self.total_comp_act[t,-1]+=self.total_comp_act[t,idx_l]

#
def comp_act_burst(self,t):
    self.comp_act_ws(t)





########################################
# inter-spike-interval (ISI)
########################################
def init_anal_isi(self):
    self.total_isi=np.zeros(self.conf.time_step)

    type_spike_amp_kind = {
        'RATE': 1,
        'WEIGHTED_SPIKE': self.conf.p_ws,
        'BURST': int(5-np.log2(self.conf.n_init_vth))
    }

    self.spike_amp_kind = type_spike_amp_kind[self.conf.neural_coding]+1
    self.total_spike_amp=np.zeros(self.spike_amp_kind)

    type_spike_amp_bin = {
        'RATE': np.power(0.5,range(0,self.spike_amp_kind+1)),
        'WEIGHTED_SPIKE': np.power(0.5,range(0,self.spike_amp_kind+1)),
        'BURST': 32*np.power(0.5,range(0,self.spike_amp_kind+1))
    }

    self.spike_amp_bin=type_spike_amp_bin[self.conf.neural_coding]
    self.spike_amp_bin=self.spike_amp_bin[::-1]
    self.spike_amp_bin[0]=0.0


########################################
# spike train entropy
########################################
def init_anal_entropy(self):
    # TODO: change data type, np -> tf.Variable

    for l_name in self.list_layer_name_write_stat:
        self.dict_stat_w[l_name] = np.zeros([self.conf.time_step,]+self.list_shape[l_name][1:])

    self.arr_length_entropy = [2,3,4,5,8,10]
    self.total_entropy=np.zeros([len(self.arr_length_entropy),len(self.layer_name)+1])





