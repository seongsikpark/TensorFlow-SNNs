
import numpy as np
import collections
import csv
import os

import tensorflow as tf

import matplotlib.pyplot as plt

#
def vth_calibration_test(self,f_norm, stat):


    #
    path_stat = os.path.join(self.path_model,self.conf.path_stat)
    stat = 'max'
    stat = 'max_999'
    #stat = 'max_99'
    for idx_l, l in enumerate(self.layers_w_kernel):
        #print(l.name)
        key=l.name+'_'+stat

        #f_name_stat = f_name_stat_pre+'_'+key
        f_name_stat = key
        f_name=os.path.join(path_stat,f_name_stat)
        f_stat=open(f_name,'r')
        r_stat=csv.reader(f_stat)

        for row in r_stat:
            #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
            stat_r=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

        #print(self.dict_stat_r[l.name])
        #print(np.median(self.dict_stat_r[l.name]))
        #stat = self.dict_stat_r[l.name]

        #represent_stat = f_norm(self.dict_stat_r[l.name])
        represent_stat = self.norm_b[l.name]

        #
        #stat_r = np.where(stat_r==0, )
        vth_cal = represent_stat / (stat_r+ 1e-10)
        #vth_cal = stat_r / represent_stat
        vth_cal = np.where(vth_cal==0, 1, vth_cal)
        #vth_cal = np.where(vth_cal>1, 1, vth_cal)
        vth_cal = np.expand_dims(vth_cal, axis=0)
        vth_cal = np.broadcast_to(vth_cal, l.act.dim)
        vth_cal = vth_cal*l.act.vth

        vth_cal_w_comp = np.mean(vth_cal)

        #l.act.set_vth_init(vth_cal)

        #
        #vth_cal_w_comp = np.mean(stat_r,axis=[0,1])/represent_stat
        #vth_cal_w_comp = np.mean(stat_r,axis=(0, 1))/represent_stat
        # TODO: tmp
        #vth_cal_w_comp = np.mean(stat_r)/np.mean(represent_stat)
        #print(vth_cal_w_comp)
        #print(vth_cal_w_comp.shape)

        # weight compensation
        if idx_l != 0:
            scale = prev_vth_cal
            l.kernel = l.kernel*scale

        prev_vth_cal = vth_cal_w_comp

#
def vth_calibration(self,f_norm, stat):

    vth_cal = collections.OrderedDict()
    const = 1/1.2
    if False:
        vth_cal['conv1'] = 0.5
        vth_cal['conv1_1'] = 0.5
        vth_cal['conv2'] = 0.8
        vth_cal['conv2_1'] = 1.0
        vth_cal['conv3'] = 0.5
        vth_cal['conv3_1'] = 0.7
        vth_cal['conv3_2'] = 0.5
        vth_cal['conv4'] = 0.8
        vth_cal['conv4_1'] = 0.5
        vth_cal['conv4_2'] = 0.5
        vth_cal['conv5'] = 0.7
        vth_cal['conv5_1'] = 0.7
        vth_cal['conv5_2'] = 0.7
        vth_cal['fc1'] = 0.7
        vth_cal['fc2'] = 0.7
        vth_cal['predictions'] = 0.5
    else:
        vth_cal['conv1'] = const
        vth_cal['conv1_1'] = const
        vth_cal['conv2'] = const
        vth_cal['conv2_1'] = const
        vth_cal['conv3'] = const
        vth_cal['conv3_1'] = const
        vth_cal['conv3_2'] = const
        vth_cal['conv4'] = const
        vth_cal['conv4_1'] = const
        vth_cal['conv4_2'] = const
        vth_cal['conv5'] = const
        vth_cal['conv5_1'] = const
        vth_cal['conv5_2'] = const
        vth_cal['fc1'] = const
        vth_cal['fc2'] = const
        vth_cal['predictions'] = const

    for idx_l, l in enumerate(self.layers_w_kernel):
        l.act.set_vth_init(vth_cal[l.name])


    # scale - vth
    for idx_l, l in enumerate(self.layers_w_kernel):
        if idx_l != 0:
            scale = prev_vth_cal
            l.kernel = l.kernel*scale

        prev_vth_cal = vth_cal[l.name]

# TODO: move
def read_stat(self,layer,stat):

    path_stat = os.path.join(self.path_model,self.conf.path_stat)

    key = layer.name + '_' + stat

    # f_name_stat = f_name_stat_pre+'_'+key
    f_name_stat = key
    f_name = os.path.join(path_stat, f_name_stat)
    f_stat = open(f_name, 'r')
    r_stat = csv.reader(f_stat)

    for row in r_stat:
        # self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
        stat_r = np.asarray(row, dtype=np.float32).reshape(layer.output_shape_fixed_batch[1:])

    return stat_r

#
def vth_toggle(self):

    stat = 'max_999'
    for idx_l, l in enumerate(self.layers_w_kernel):
        #

        #
        # simple toggle
        #vth_schedule = [self.conf.vth_toggle_init, 2-self.conf.vth_toggle_init]

        #
        # stat based toggle
        self.stat_r = read_stat(self,l,stat)
        stat_r = self.stat_r

        #represent_stat = f_norm(self.dict_stat_r[l.name])
        represent_stat = self.norm_b[l.name]

        vth_toggle_init = stat_r/self.norm_b[l.name]
        #vth_schedule = [vth_toggle_init, 2-vth_toggle_init]
        #a = vth_toggle_init*1.1
        a = vth_toggle_init
        b = 2-vth_toggle_init

        #b = vth_toggle_init
        #a = 2-vth_toggle_init

        #vth_schedule = np.stack([a,b],axis=-1)
        vth_schedule = tf.stack([a,b],axis=-1)
        vth_schedule = tf.reshape(vth_schedule,shape=[-1,2])
        # batch
        vth_schedule = tf.tile(vth_schedule,[l.act.vth.shape[0],1])

        # vth schedule update
        l.act.vth_schedule = vth_schedule
        l.act.set_vth_init(vth_schedule[:,0])

        print('{} - vth toggle set done '.format(l.name))

        #assert False
        #vth_schedule_init = tf.tile(vth_schedule,[l.act.vth.shape[0],1])[:,0]
        #l.act.set_vth_init(vth_schedule_init)


        #l.act.set_vth_init(l.act.vth*self.conf.vth_toggle_init)
        #l.act.set_vth_init(l.act.vth*0.3)



#
def vth_calibration_old(self,f_norm, stat):

    #
    path_stat = os.path.join(self.path_model,self.conf.path_stat)
    stat = 'max'
    stat = 'max_999'
    #stat = 'max_99'
    for idx_l, l in enumerate(self.layers_w_kernel):
        #print(l.name)
        key=l.name+'_'+stat

        #f_name_stat = f_name_stat_pre+'_'+key
        f_name_stat = key
        f_name=os.path.join(path_stat,f_name_stat)
        f_stat=open(f_name,'r')
        r_stat=csv.reader(f_stat)

        for row in r_stat:
            #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
            stat_r=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

        #print(self.dict_stat_r[l.name])
        #print(np.median(self.dict_stat_r[l.name]))
        #stat = self.dict_stat_r[l.name]

        #represent_stat = f_norm(self.dict_stat_r[l.name])
        represent_stat = self.norm_b[l.name]

        #
        vth_cal = represent_stat / (stat_r+ 1e-10)
        vth_cal = np.where(vth_cal>1, 1, vth_cal)
        vth_cal = np.expand_dims(vth_cal, axis=0)
        vth_cal = np.broadcast_to(vth_cal, l.act.dim)

        l.act.set_vth_init(vth_cal)


    #assert False

#
def bias_calibration(self):

    bias_cal = collections.OrderedDict()

    const = 1.3

    bias_cal['conv1'] = const
    bias_cal['conv1_1'] = const
    bias_cal['conv2'] = const
    bias_cal['conv2_1'] = const
    bias_cal['conv3'] = const
    bias_cal['conv3_1'] = const
    bias_cal['conv3_2'] = const
    bias_cal['conv4'] = const
    bias_cal['conv4_1'] = const
    bias_cal['conv4_2'] = const
    bias_cal['conv5'] = const
    bias_cal['conv5_1'] = const
    bias_cal['conv5_2'] = const
    bias_cal['fc1'] = const
    bias_cal['fc2'] = const
    bias_cal['predictions'] = const

    #
    for layer in self.layers_w_kernel:
        layer.bias = layer.bias*bias_cal[layer.name]


# weight calibration - resolve information bottleneck
def weight_calibration(self):

    #
    stat = None

    norm_wc = collections.OrderedDict()
    norm_b_wc = collections.OrderedDict()

    #norm = [0.5, 0.5, 0.5, ]
    norm = collections.OrderedDict()

    const = 0.7

    # layer-wise norm, max_90
    #if stat=='max_90':
    if False:
        norm['conv1']   = 0.3
        norm['conv1_1'] = 0.3
        norm['conv2']   = 0.75
        norm['conv2_1'] = 0.9
        norm['conv3']   = 1.0
        norm['conv3_1'] = 0.9
        norm['conv3_2'] = 1.0
        norm['conv4']   = 1.0
        norm['conv4_1'] = 1.0
        norm['conv4_2'] = 1.0
        norm['conv5']   = 1.0
        norm['conv5_1'] = 0.9
        norm['conv5_2'] = 0.4
        norm['fc1']     = 1.0
        norm['fc2']     = 0.4
        norm['predictions'] = 0.1

    # stat=='max_99', channel-wise
    #elif True:
    elif False:
        norm['conv1']   = 0.3
        norm['conv1_1'] = 0.9
        norm['conv2']   = 0.8
        norm['conv2_1'] = 0.8
        norm['conv3']   = 1.0
        norm['conv3_1'] = 1.0
        norm['conv3_2'] = 0.9
        norm['conv4']   = 0.9
        norm['conv4_1'] = 0.9
        norm['conv4_2'] = 1.0
        norm['conv5']   = 1.0
        norm['conv5_1'] = 1.0
        norm['conv5_2'] = 1.0
        norm['fc1']     = 1.0
        norm['fc2']     = 1.0
        norm['predictions'] = 1.0

    elif True:
        norm['conv1']   = const
        #norm['conv1']   = 0.6
        norm['conv1_1'] = const
        norm['conv2']   = const
        norm['conv2_1'] = const
        norm['conv3']   = const
        norm['conv3_1'] = const
        norm['conv3_2'] = const
        norm['conv4']   = const
        norm['conv4_1'] = const
        norm['conv4_2'] = const
        norm['conv5']   = const
        norm['conv5_1'] = const
        norm['conv5_2'] = const
        norm['fc1']     = const
        norm['fc2']     = const
        norm['predictions'] = const

    elif False:
        norm['conv1']   = 0.3
        norm['conv1_1'] = 0.9
        norm['conv2']   = 1.0
        norm['conv2_1'] = 0.8
        norm['conv3']   = 1.0
        norm['conv3_1'] = 1.0
        norm['conv3_2'] = 0.9
        norm['conv4']   = 0.9
        norm['conv4_1'] = 0.9
        norm['conv4_2'] = 1.0
        norm['conv5']   = 1.0
        norm['conv5_1'] = 1.0
        norm['conv5_2'] = 1.0
        norm['fc1']     = 1.0
        norm['fc2']     = 1.0
        norm['predictions'] = 1.0

    else:
        #
        path_stat = os.path.join(self.path_model,self.conf.path_stat)
        #stat = 'max_999'
        #stat = 'max'
        #stat = 'max_75'
        stat = 'median'
        for idx_l, l in enumerate(self.layers_w_kernel):
            #print(l.name)
            key=l.name+'_'+stat

            #f_name_stat = f_name_stat_pre+'_'+key
            f_name_stat = key
            f_name=os.path.join(path_stat,f_name_stat)
            f_stat=open(f_name,'r')
            r_stat=csv.reader(f_stat)

            for row in r_stat:
                #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
                stat_r=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

            #norm[l.name] = np.median(stat_r)
            norm[l.name] = 1/np.max(stat_r)
            print(norm[l.name])

    #
    #norm_wc['conv1'] = norm[0]
    #norm_b_wc['conv1'] = norm[0]
    #
    #    norm_wc['conv1_1'] = norm[1]/norm[0]
    #    norm_b_wc['conv1_1'] = norm[1]
    #

    if 'VGG' in self.conf.model:
        for idx_l, l in enumerate(self.layers_w_kernel):
            if idx_l == 0:
                norm_wc[l.name] = norm[l.name]
            else:
                norm_wc[l.name] = norm[l.name]/norm[prev_layer_name]

            prev_layer_name = l.name
            norm_b_wc[l.name] = norm[l.name]

    for layer in self.layers_w_kernel:
        # layer = self.model.get_layer(name=name_l)
        if layer.name in norm_wc.keys():
            layer.kernel = layer.kernel / norm_wc[layer.name]
        if layer.name in norm_b_wc.keys():
            layer.bias = layer.bias / norm_b_wc[layer.name]


def weight_calibration_inv_vth(self):
    #
    # scale - inv. vth
    for idx_l, l in enumerate(self.layers_w_kernel):
        if idx_l != 0:
            scale = self.conf.n_init_vth
            l.kernel = l.kernel*scale


# TODO: move
def vmem_calibration(self):

    #
    path_stat = os.path.join(self.path_model,self.conf.path_stat)
    #stat = 'max_999'
    stat = 'max'
    #stat = 'max_90'
    #stat = 'max_50'
    for idx_l, l in enumerate(self.layers_w_kernel):
        #print(l.name)
        key=l.name+'_'+stat

        #f_name_stat = f_name_stat_pre+'_'+key
        f_name_stat = key
        f_name=os.path.join(path_stat,f_name_stat)
        f_stat=open(f_name,'r')
        r_stat=csv.reader(f_stat)

        for row in r_stat:
            #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
            stat_r=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])
        stat_r_max = np.max(stat_r)


        #represent_stat = np.max(dict_stat_r[l.name])

        #
        vmem_cal_norm = stat_r / stat_r_max
        #print(stat_r_max)
        #print(stat_r.shape)
        #print(vmem_cal_norm.shape)
        #assert False
        #vmem_cal = 0.7*(1-vmem_cal_norm)*l.act.vth
        vmem_cal = 0.7*(1-vmem_cal_norm)*l.act.vth
        #vmem_cal = 0.7*np.power((1-vmem_cal_norm),2)*l.act.vth

        #vth_cal = np.where(vth_cal>1, 1, vth_cal)
        #vmem_cal = np.expand_dims(vmem_cal, axis=0)
        #vmem_cal = np.broadcast_to(vmem_cal, l.act.dim)

        #l.act.set_vth_init(vth_cal)

        #if l.name=='conv1':
        l.act.reset_vmem(vmem_cal)

    #vmem_cal_norm
    #conv1_n = self.model.get_layer('conv1').act
    #conv1_n.reset_vmem()
