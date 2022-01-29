
import numpy as np
import collections
import csv
import os

import tensorflow as tf

import tensorflow_probability as tfp

import matplotlib.pyplot as plt

import lib_snn

#
from lib_snn.sim import glb_ig_attributions
from lib_snn.sim import glb_rand_vth
from lib_snn.sim import glb_vth_search_err
from lib_snn.sim import glb_vth_init
from lib_snn.sim import glb_bias_comp

#
def vth_calibration_stat(self):
    #
    path_stat = os.path.join(self.path_model,self.conf.path_stat)
    #stat = 'max'
    stat = 'max_999'
    #stat = 'max_99'
    for idx_l, l in enumerate(self.model.layers_w_kernel):
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
        #vth_cal = represent_stat / (stat_r+ 1e-10)
        #vth_cal = represent_stat / (stat_r)
        #self.vth_cal = stat_r / represent_stat
        vth_cal = stat_r / represent_stat
        #vth_cal = stat_r / represent_stat
        vth_cal = np.where(vth_cal==0, 1, vth_cal)
        #vth_cal = np.where(vth_cal>1, 1, vth_cal)
        #vth_cal = 0.1*(vth_cal*l.act.vth) + 0.9*(l.act.vth)
        vth_cal = 0.2*(vth_cal) + 0.8*(tf.reduce_mean(l.act.vth,axis=0))
        #vth_cal = 0.3*(vth_cal) + 0.7
        vth_cal_one_batch = vth_cal

        #vth_cal = np.expand_dims(vth_cal, axis=0)
        #vth_cal = np.broadcast_to(vth_cal, l.act.dim)
        vth_cal = tf.expand_dims(vth_cal, axis=0)
        vth_cal = tf.broadcast_to(vth_cal, l.act.dim)
        l.act.set_vth_init(vth_cal)


        if isinstance(l, lib_snn.layers.Conv2D):
            axis = [0, 1, 2]
        elif isinstance(l, lib_snn.layers.Dense):
            axis = [0]
        else:
            assert False

        #vth_cal_w_comp = np.mean(vth_cal,axis=axis)
        vth_cal_w_comp = tf.reduce_mean(vth_cal_one_batch,axis=axis)

        #
        #vth_cal_w_comp = np.mean(stat_r,axis=[0,1])/represent_stat
        #vth_cal_w_comp = np.mean(stat_r,axis=(0, 1))/represent_stat
        # TODO: tmp
        #vth_cal_w_comp = np.mean(stat_r)/np.mean(represent_stat)
        #print(vth_cal_w_comp)
        #print(vth_cal_w_comp.shape)

        # weight compensation
        if idx_l != 0:
            #scale = prev_vth_cal
            l.kernel = l.kernel*scale_next_layer

        #prev_vth_cal = vth_cal_w_comp
        scale_next_layer = vth_cal_w_comp


#
#def vth_calibration(self,f_norm, stat):
def vth_calibration_manual(self):

    vth_cal = collections.OrderedDict()
    const = 1/1.3
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

    for idx_l, l in enumerate(self.model.layers_w_kernel):
        l.act.set_vth_init(vth_cal[l.name])


    # scale - vth
    for idx_l, l in enumerate(self.model.layers_w_kernel):
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
    for idx_l, l in enumerate(self.model.layers_w_kernel):
        #

        #
        # simple toggle
        if True:
        #if False:
            vth = tf.reduce_mean(l.act.vth,axis=0)
            vth_toggle_init = self.conf.vth_toggle_init*vth
            #vth_schedule = tf.stack([self.conf.vth_toggle_init, 2-self.conf.vth_toggle_init])

            # vth schedule update
            #l.act.vth_schedule = vth_schedule
            #l.act.set_vth_init(vth_schedule[0])

            if isinstance(l, lib_snn.layers.Conv2D):
                shape = l.act.vth.shape[1:4]
            elif isinstance(l, lib_snn.layers.Dense):
                shape = l.act.vth.shape[1]
            else:
                assert False

            a = tf.constant(vth_toggle_init,shape=shape)
            #b = tf.constant(2-vth_toggle_init,shape=shape)
            b = a/(2*a-1)       # harmonic mean

        #
        # stat based toggle
        if False:
        #if True:
            self.stat_r = read_stat(self,l,stat)
            stat_r = self.stat_r

            #represent_stat = f_norm(self.dict_stat_r[l.name])
            represent_stat = self.norm_b[l.name]

            vth_toggle_init = stat_r/self.norm_b[l.name]
            #vth_schedule = [vth_toggle_init, 2-vth_toggle_init]
            #a = vth_toggle_init*1.1

            #alpha = 0.9
            alpha = self.conf.vth_toggle_init
            a = (1-alpha)*vth_toggle_init+alpha
            b = a/(2*a-1)       # harmonic mean
            #b = 2-vth_toggle_init

            #b = vth_toggle_init
            #a = 2-vth_toggle_init

        #vth_schedule = np.stack([a,b],axis=-1)
        vth_schedule = tf.stack([a,b],axis=-1)
        vth_schedule = tf.reshape(vth_schedule,shape=[-1,2])


        # batch
        vth_schedule = tf.tile(vth_schedule,[l.act.vth.shape[0],1])

        # vth schedule update
        l.act.vth_schedule = vth_schedule
        #l.act.set_vth_init(vth_schedule[:,0])
        l.act.vth_toggle_init = vth_schedule[:,0]
        l.act.set_vth_init(l.act.vth_toggle_init)

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
    for idx_l, l in enumerate(self.model.layers_w_kernel):
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
    for layer in self.model.layers_w_kernel:
        layer.bias = layer.bias*bias_cal[layer.name]


# weight calibration - resolve information bottleneck
def weight_calibration(self):
    #
    stat = None

    norm_wc = collections.OrderedDict()
    norm_b_wc = collections.OrderedDict()

    #norm = [0.5, 0.5, 0.5, ]
    norm = collections.OrderedDict()

    #const = 0.95
    #const = 0.5
    #const = 0.6
    const = 0.7

    #
    weight_only_norm = False

    #
    self.cal=collections.OrderedDict()

    # layer-wise norm, max_90
    #if stat=='max_90':
    #if True:
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

    elif False:
    #elif True:
        #norm['conv1']   = const
        norm['conv1']   = 0.95
        #norm['conv1']   = 0.9
        #norm['conv1']   = [0.7]*64
        #norm['conv1'][5] = 0.5
        norm['conv1_1'] = const
        #norm['conv1_1'] = 0.5
        norm['conv2']   = const
        #norm['conv2']   = 0.5
        #norm['conv2_1'] = const
        norm['conv2_1'] = 0.9
        #norm['conv3']   = const
        norm['conv3']   = 0.8
        #norm['conv3_1'] = const
        norm['conv3_1'] = 0.8
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
    #elif True:
        #norm['conv1']   = const
        norm['conv1']   = 0.95
        #norm['conv1']   = 0.7
        #norm['conv1']   = [0.7]*64
        #norm['conv1'][5] = 0.5
        norm['conv1_1'] = const
        #norm['conv1_1'] = 0.5
        norm['conv2']   = const
        #norm['conv2']   = 0.5
        #norm['conv2_1'] = const
        norm['conv2_1'] = 0.9
        #norm['conv3']   = const
        norm['conv3']   = 0.8
        #norm['conv3_1'] = const
        norm['conv3_1'] = 0.8
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

    # current best - 1208
    #elif False:
    elif True:
        #for idx_l, l in enumerate(self.model.layers_w_kernel):
            #norm[l.name]=0.8


        depth_l = len(self.model.layers_w_kernel)
        #a = 0.3
        #a = 0.5
        a = 0.7
        a = 0.8
        #a = 0.9
        a = 1.0
        b = 1.0
        #b = 0.99
        #b = 0.9
        #b = 0.6
        #b = 0.5
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            norm[l.name] = a + (1 - a) * (depth_l - idx_l) / (depth_l)
            norm[l.name] *= b
            # norm[l.name]=a*(depth_l-idx_l)/(depth_l)



            #vth_init = norm[l.name]
            #l.kernel = l.kernel * self.conf.n_init_vth * vth_init
            #l.act.set_vth_init(vth_init)

        # set vth - act mean
        #if True:
        if False:

            path_stat = os.path.join(self.path_model,self.conf.path_stat)
            #stat = 'median'
            stat = 'mean'
            for idx_l, l in enumerate(self.model.layers_w_kernel):
                # print(l.name)
                key = l.name + '_' + stat

                # f_name_stat = f_name_stat_pre+'_'+key
                f_name_stat = key
                f_name = os.path.join(path_stat, f_name_stat)
                f_stat = open(f_name, 'r')
                r_stat = csv.reader(f_stat)

                for row in r_stat:
                    # self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
                    stat_r = np.asarray(row, dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

                stat_r_m = tf.reduce_mean(stat_r)
                #stat_r_m = tf.reduce_max(stat_r)

                vth_init = stat_r_m

                if idx_l < 20:
                    print(vth_init)
                    l.kernel = l.kernel * self.conf.n_init_vth * vth_init
                    l.act.set_vth_init(vth_init)

    # weight cal - due to vth
    elif True:
        #weight_only_norm = True
        #for idx_l, l in enumerate(self.model.layers_w_kernel):
            #norm[l.name] = 1.0

        for idx_l, l in enumerate(self.model.layers_w_kernel):
            #norm[l.name] = 1.0
            if idx_l!=0 :
                l.kernel = l.kernel * self.conf.n_init_vth
                ##norm[l.name] /= self.conf.n_init_vth
            else:
                pass
                #vth_in = 0.8
                #l.kernel = l.kernel * vth_in

                ##vth_init = self.conf.n_init_vth*0.8
                #l.act.set_vth_init(vth_in)

            #else:
                #vth_init = self.conf.n_init_vth*0.8
                #l.act.set_vth_init(vth_init)


    elif False:
    #elif True:
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
        for idx_l, l in enumerate(self.model.layers_w_kernel):
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

    #if True:
    if False:
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            if idx_l!=0:

                #stat_mean = read_stat(self,l,'mean')*0.001
                #stat_mean = read_stat(self,prev_l,'mean')*0.01
                #stat_mean = read_stat(self,prev_l,'mean')*r_sat
                stat_mean = read_stat(self,prev_l,'mean')

                #stat_mean *= 0.005
                #stat_mean *= 0.01
                #stat_mean *= 0.001
                stat_mean *= 0.0001
                #stat_mean *= r_sat*0.01
                #stat_mean *= r_sat*0.02
                #stat_mean *= r_sat*0.1
                #stat_mean *= r_sat
                #print('{} - r_sat: {}'.format(l.name,r_sat))
                #print('{} - stat_mean*r_sat: {}'.format(l.name,stat_mean))

                if isinstance(l, lib_snn.layers.Conv2D):
                    stat_mean = tf.expand_dims(stat_mean,axis=0)
                    bias_comp = tf.nn.conv2d(stat_mean,l.kernel,strides=l.strides,padding=l.padding.upper())
                    bias_comp = tf.reduce_mean(bias_comp,axis=[0,1,2])
                elif isinstance(l, lib_snn.layers.Dense):

                    #if l.name=='fc1':
                    #    print(stat_mean)
                    #    stat_mean = tf.reduce_max(stat_mean,axis=[0,1])

                    #
                    if isinstance(prev_l,lib_snn.layers.Conv2D):
                        stat_mean = tf.reduce_mean(stat_mean,axis=[0,1])

                    #print(stat_mean)
                    #print(l.kernel)

                    bias_comp = tf.linalg.matvec(l.kernel,stat_mean,transpose_a=True)
                else:
                    assert False


                print('{} - bias_comp (avg): {}, bias_comp: {}'.format(l.name,tf.reduce_mean(bias_comp),bias_comp))

                l.bias = l.bias + bias_comp

            prev_l = l


    #
    #norm_wc['conv1'] = norm[0]
    #norm_b_wc['conv1'] = norm[0]
    #
    #    norm_wc['conv1_1'] = norm[1]/norm[0]
    #    norm_b_wc['conv1_1'] = norm[1]
    #

    if 'VGG' in self.conf.model:
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            if idx_l == 0:
                norm_wc[l.name] = norm[l.name]
            else:
                #norm_wc[l.name] = norm[l.name]/norm[prev_layer_name]
                #norm_wc[l.name] = norm[l.name] / np.expand_dims(norm_b_wc[prev_layer_name],axis=0).T
                norm_wc[l.name] = norm[l.name] / np.expand_dims(norm[prev_layer_name],axis=0).T

            prev_layer_name = l.name

            if not weight_only_norm:
                norm_b_wc[l.name] = norm[l.name]

    for layer in self.model.layers_w_kernel:
        # layer = self.model.get_layer(name=name_l)
        if layer.name in norm_wc.keys():
            layer.kernel = layer.kernel / norm_wc[layer.name]
        if layer.name in norm_b_wc.keys():
            layer.bias = layer.bias / norm_b_wc[layer.name]

    for layer in self.model.layers_w_kernel:
        print(layer.name)
        print(norm_wc[layer.name])



# weight calibration - resolve information bottleneck
def weight_calibration_act_based(self):
    print('\nweight_calibration_act_based')
    #
    stat = None

    norm_wc = collections.OrderedDict()
    norm_b_wc = collections.OrderedDict()

    #norm = [0.5, 0.5, 0.5, ]
    norm = collections.OrderedDict()

    #const = 0.95
    #const = 0.5
    #const = 0.6
    const = 0.7

    #
    self.cal=collections.OrderedDict()


    #if True:
    if False:
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            norm[l.name]=1.0

        #norm['conv1']=0.9
        #norm['conv1_1']=0.8

    # act (DNN) / act (SNN)
    #elif True:
    elif False:

        error_level = 'layer'
        # error_level = 'channel'

        for idx_l, l in enumerate(self.model.layers_w_kernel):

            if idx_l == len(self.model.layers_w_kernel)-1:
                norm[l.name]=1.0
                continue

            if self.conf.bias_control:
                time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
            else:
                time = self.conf.time_step

            ann_act = self.model_ann.get_layer(l.name).record_output
            snn_act = l.act.spike_count_int/time

            if error_level == 'layer':
                if isinstance(l, lib_snn.layers.Conv2D):
                    axis = [0, 1, 2, 3]
                elif isinstance(l, lib_snn.layers.Dense):
                    axis = [0, 1]
                else:
                    assert False
            elif error_level == 'channel':
                if isinstance(l, lib_snn.layers.Conv2D):
                    axis = [0, 1, 2]
                elif isinstance(l, lib_snn.layers.Dense):
                    axis = [0]
                else:
                    assert False
            else:
                assert False

            ann_act = tf.reduce_mean(ann_act,axis=axis)
            snn_act = tf.reduce_mean(snn_act,axis=axis)

            error_r = snn_act / ann_act

            norm[l.name] = error_r

    # firing rate -> to 1
    elif True:
    #elif False:

        #error_level = 'layer'
        error_level = 'channel'

        for idx_l, l in enumerate(self.model.layers_w_kernel):

            #if idx_l == len(self.model.layers_w_kernel)-1:
            #    norm[l.name]=1.0
            #    continue

            if self.conf.bias_control:
                time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
            else:
                time = self.conf.time_step

            snn_act = l.act.spike_count_int/time

            #ann_act = self.model_ann.get_layer(l.name).record_output
            #ann_act_s = self.model_ann.get_layer(l.name).bias
            #ann_act_d =

            #print(norm_s)
            #assert False


            if error_level == 'layer':
                if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
                    axis = [0, 1, 2, 3]
                elif isinstance(l, lib_snn.layers.Dense):
                    axis = [0, 1]
                else:
                    assert False
            elif error_level == 'channel':
                if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
                    axis = [0, 1, 2]
                elif isinstance(l, lib_snn.layers.Dense):
                    axis = [0]
                else:
                    assert False
            else:
                assert False


            #fire_r_m = tf.reduce_mean(snn_act,axis=axis)
            #fire_r_m = tf.reduce_max(snn_act,axis=axis)
            #fire_r_m = tfp.stats.percentile(snn_act,99.91,axis=axis)
            fire_r_m = tfp.stats.percentile(snn_act,99.9,axis=axis)
            #fire_r_m = tfp.stats.percentile(snn_act,99,axis=axis)
            #fire_r_m = tf.where(fire_r_m==0,tf.ones(fire_r_m.shape),fire_r_m)

            #time_r = time/self.conf.time_step
            #fire_r_m = fire_r_m/time_r

            #
            #norm[l.name] = fire_r_m

            #for idx_l, l in enumerate(self.model.layers_w_kernel):
            #if idx_l != 0:

            #l.kernel = l.kernel * fire_r_m

            #norm[l.name] = 1.0
            norm[l.name] = fire_r_m

            if idx_l == len(self.model.layers_w_kernel)-1:
                norm[l.name]=1.0
            #    continue

            # vth calibration - manual search
            #if False:
            if True:
                norm[l.name] = 1.0
                #if idx_l == 0:

                dnn_act = self.model_ann.get_layer(l.name).record_output

                if self.conf.f_w_norm_data:
                    stat_max = tf.reduce_max(dnn_act, axis=axis)
                    stat_max = tfp.stats.percentile(dnn_act, 99.9, axis=axis)
                    stat_max = tf.ones(stat_max.shape)
                    #stat_max = tf.reduce_max(dnn_act, axis=axis)
                    #stat_max = read_stat(self, l, 'max')
                    #stat_max = stat_max / self.norm_b[l.name]
                    #stat_max = tf.expand_dims(stat_max,axis=0)
                    #stat_max = tf.reduce_max(stat_max,axis=axis)
                else:
                    stat_max = read_stat(self, l, 'max')
                    stat_max = tf.expand_dims(stat_max,axis=0)
                    stat_max = tf.reduce_max(stat_max,axis=axis)



                num_range = 100
                #num_range = 250
                #num_range = 125
                #num_range = 133
                #num_range = 200
                #num_range = 333
                #num_range = 1000
                #num_range = 5000
                #
                #range = tf.random.normal(shape=[num_range],mean=0.0,stddev=0.2,dtype=tf.float32)
                #range = 1.0-tf.math.abs(range)
                #range = tf.random.uniform(shape=[num_range],minval=0.5,maxval=1,dtype=tf.float32)
                #range = tf.random.uniform(shape=[num_range],minval=0.0,maxval=1,dtype=tf.float32)
                #range = tf.range(1/num_range,1+1/num_range,1/num_range,dtype=tf.float32)


                if not l.name in glb_rand_vth.keys():
                    glb_rand_vth[l.name] = tf.random.uniform(shape=[num_range], minval=0.0, maxval=1, dtype=tf.float32)

                range = glb_rand_vth[l.name]

                #range_vth = range*self.conf.n_init_vth
                #range_vth = range*stat_max

                #print(stat_max)
                #print(stat_max.shape)
                #print(stat_max.shape[0])
                #print(range.shape[0])

                #len_range = range.shape[0]
                if error_level=='layer':
                    errs = tf.zeros([num_range])
                elif error_level=='channel':
                    errs = tf.zeros([num_range,stat_max.shape[0]])
                else:
                    assert False

                #assert False
                #errs = []

                if self.conf.bias_control:
                    time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
                else:
                    time = self.conf.time_step
                #time = self.conf.time_step

                #

                for idx, vth_scale in enumerate(range):

                    vth = vth_scale*stat_max

                    #clip_max = vth*self.conf.time_step
                    #clip_max = vth*time
                    clip_max = time

                    #dnn_act_clip_floor = tf.math.floor(dnn_act/vth)
                    dnn_act_clip_floor = tf.math.floor(dnn_act/vth*time)
                    dnn_act_clip_floor = tf.clip_by_value(dnn_act_clip_floor,0,clip_max)
                    dnn_act_clip_floor = dnn_act_clip_floor*vth/time

                    #print(vth)
                    err = dnn_act - dnn_act_clip_floor
                    err = tf.math.square(err)
                    #err = tf.math.square(err)*1/dnn_act

                    # integrated gradients
                    if self.conf.vth_search_ig:
                        #if isinstance(l, lib_snn.layers.Conv2D):
                            #ig_attributions = tf.reduce_mean(glb_ig_attributions[l.name],axis=[0,1])
                        #elif isinstance(l, lib_snn.layers.Dense):
                            #ig_attributions = glb_ig_attributions[l.name]
                        #else:
                            #assert False
                        #print(err.shape)
                        ig_attributions = glb_ig_attributions[l.name]
                        eps = 0.01
                        #print(err.shape)
                        #print(ig_attributions.shape)
                        #err = err*(1+ig_attributions)
                        #err = err*(10+ig_attributions/tf.reduce_max(ig_attributions))
                        #err = err*(2+ig_attributions/tf.reduce_max(ig_attributions))
                        #err = err*(1+ig_attributions/tf.reduce_max(ig_attributions))
                        #err = err*(0.5+ig_attributions/tf.reduce_max(ig_attributions))
                        #err = err*(0.1+ig_attributions/tf.reduce_max(ig_attributions))
                        #err = err*(0.01+ig_attributions/tf.reduce_max(ig_attributions))    # old best?
                        #err = err*(0.001+ig_attributions/tf.reduce_max(ig_attributions))
                        #err = err*(tf.reduce_min(ig_attributions)+ig_attributions)
                        #alpha = 0.5
                        #err = alpha*err/tf.reduce_max(err,axis=axis)+(1-alpha)*1/(ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                        #err = err*ig_attributions
                        #err = err*(ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                        #jerr = err*(eps+ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                        err = err*(eps+ig_attributions/tf.reduce_max(ig_attributions))
                        #err = err*(tf.reduce_min(ig_attributions)+ig_attributions/tf.reduce_max(ig_attributions))
                        #err = err*(tf.reduce_mean(ig_attributions)+ig_attributions/tf.reduce_max(ig_attributions))
                        #err = err*(eps+ig_attributions/tf.reduce_max(ig_attributions))    # old best?
                        #err = err*(ig_attributions/tf.reduce_max(ig_attributions))
                        #print(err.shape)

                    #err = err * snn_act
                    #print(err)

                    #err = tf.math.abs(err)

                    #err = err * snn_act
                    #print(err)
                    #err = tf.math.square(err)
                    err = tf.reduce_mean(err,axis=axis)
                    #print(err)

                    #if error_level == 'layer':
                    #    errs = tf.tensor_scatter_nd_update(errs,[[idx]],[err])
                    #elif error_level == 'channel':
                    #    errs = tf.tensor_scatter_nd_update(errs,[[idx]],[err])


                    # old
                    #errs = tf.tensor_scatter_nd_update(errs,[[idx]],[err])

                    if not l.name in glb_vth_search_err.keys():
                        glb_vth_search_err[l.name] = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)


                    vth_err_arr = glb_vth_search_err[l.name]
                    #if idx==0:


                    if vth_err_arr.size() < num_range:
                        glb_vth_search_err[l.name] = vth_err_arr.write(idx, err)
                    else:
                        glb_vth_search_err[l.name] = vth_err_arr.write(idx, vth_err_arr.read(idx) + err)



                    #print(errs)
                    #assert False
                    #errs.append(err)

                #errs_accum = glb_vth_search_err[l.name].stack()
                #return
                #break

#
#
#                #
#                # set
#                #
#
#                #vth_idx_min_err = tf.math.argmin(errs)
#                vth_idx_min_err = tf.math.argmin(errs_accum)
#                #print(vth_idx_min_err)
#
#                #vth_min_err = tf.gather(range_vth,vth_idx_min_err)
#                vth_min_err_scale = tf.gather(range,vth_idx_min_err)
#                vth_min_err = vth_min_err_scale*stat_max
#
#                #print(stat_max)
#                ##print(range_vth)
#                #print(errs)
#                #print(vth_idx_min_err)
#                #print(vth_min_err)
#                #print(vth_min_err_scale)
#                #print(vth_min_err)
#                #print(range)
#
#
#                #
#                # TODO: parameterize
#                vth_init = vth_min_err
#                #vth_init = glb_vth_init[l.name]
#
#                #vth_init = tf.where(vth_init==0,tf.ones(vth_init.shape),vth_init)
#
#                vth_init_fm = vth_init
#
#                print('{} - vth_init - {}'.format(l.name,vth_init))
#
#                #assert False
#
#
#                ##
#                if error_level=='channel':
#                    if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
#                        vth_init_fm = tf.expand_dims(vth_init_fm, axis=0)
#                        vth_init_fm = tf.expand_dims(vth_init_fm, axis=1)
#                        vth_init_fm = tf.expand_dims(vth_init_fm, axis=2)
#                    elif isinstance(l, lib_snn.layers.Dense):
#                        vth_init_fm = tf.expand_dims(vth_init_fm, axis=0)
#                    else:
#                        assert False
#                    #print(l.name)
#                    #print(stat_max.shape)
#                    #print(vth_init_fm.shape)
#                    #print(l.act.vth.shape)
#                    vth_init_fm = tf.broadcast_to(vth_init_fm,shape=l.act.vth.shape)
#
#
#                # prev vth
#                if error_level=='channel':
#                    if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
#                        prev_vth = tf.reduce_mean(l.act.vth,axis=[0,1,2])
#                    elif isinstance(l, lib_snn.layers.Dense):
#                        prev_vth = tf.reduce_mean(l.act.vth, axis=0)
#                    else:
#                        assert False
#                else:
#                    prev_vth = tf.reduce_mean(l.act.vth)
#
#
#                #
#                l.act.set_vth_init(vth_init_fm)
#
#                #print(vth_init_fm)
#
#                #
#                #norm[l.name]=vth_init
#
#                #if idx_l==0:
#                #    norm[l.name]=1.0
#                #stat_max = read_stat(self, l, 'max_999')
#                #stat_max = tf.expand_dims(stat_max,axis=0)
#                #stat_max = tf.reduce_max(stat_max,axis=axis)
#                #norm[l.name]=stat_max
#                #
#
#                if True:
#                #if False:
#
#                    if idx_l!=0:
#                        #l.kernel = l.kernel*prev_fire_r_m
#                        #l.kernel = l.kernel*((1-alpha) + alpha*prev_fire_r_m)
#                        #print(prev_vth_init.shape)
#                        #print(l.kernel.shape)
#
#                        if error_level=='channel':
#                            if isinstance(l, lib_snn.layers.Conv2D):
#                                prev_layer_vth_init = tf.expand_dims(prev_layer_vth_init, axis=0)
#                                prev_layer_vth_init = tf.expand_dims(prev_layer_vth_init, axis=1)
#                                prev_layer_vth_init = tf.expand_dims(prev_layer_vth_init, axis=3)
#
#                                prev_prev_layer_vth_init = tf.expand_dims(prev_prev_layer_vth_init, axis=0)
#                                prev_prev_layer_vth_init = tf.expand_dims(prev_prev_layer_vth_init, axis=1)
#                                prev_prev_layer_vth_init = tf.expand_dims(prev_prev_layer_vth_init, axis=3)
#
#                            elif isinstance(l, lib_snn.layers.Dense):
#                                prev_layer_vth_init = tf.expand_dims(prev_layer_vth_init, axis=1)
#
#                                prev_prev_layer_vth_init = tf.expand_dims(prev_prev_layer_vth_init, axis=1)
#
#                        #l.kernel = l.kernel*prev_layer_vth_init/prev_prev_layer_vth_init
#                        l.kernel = self.model_ann.get_layer(l.name).kernel*prev_layer_vth_init
#
#                        #if idx_l != len(self.model.layers_w_kernel) - 1:
#                            #l.act.set_vth_init(vth_init_fm)
#                            ##norm[l.name] = vth_init
#
#
#                prev_layer_vth_init = vth_init
#                prev_prev_layer_vth_init = prev_vth
#
#                #snn_act = l.act.spike_count_int/(self.conf.time_step-l.bias_en_time)
#
#
#
#
#
#            # vth calibration
#            if False:
#            #if True:
#                norm[l.name] = 1.0
#                fire_r_m_c = fire_r_m
#                fire_r_m_next_layer_kernel = fire_r_m
#
#                if error_level=='channel':
#                    if isinstance(l, lib_snn.layers.Conv2D):
#                        fire_r_m = tf.expand_dims(fire_r_m,axis=0)
#                        fire_r_m = tf.expand_dims(fire_r_m,axis=0)
#                        fire_r_m = tf.expand_dims(fire_r_m,axis=0)
#                        fire_r_m = tf.broadcast_to(fire_r_m,shape=l.act.vth.shape)
#                    elif isinstance(l, lib_snn.layers.Dense):
#                        fire_r_m = tf.expand_dims(fire_r_m,axis=0)
#                        fire_r_m = tf.broadcast_to(fire_r_m,shape=l.act.vth.shape)
#
#                    #
#                    if idx_l != len(self.model.layers_w_kernel):
#                        next_layer = self.model.layers_w_kernel[idx_l+1]
#
#                        if isinstance(next_layer, lib_snn.layers.Conv2D):
#                            fire_r_m_next_layer_kernel = tf.expand_dims(fire_r_m_next_layer_kernel,axis=0)
#                            fire_r_m_next_layer_kernel = tf.expand_dims(fire_r_m_next_layer_kernel,axis=1)
#                            fire_r_m_next_layer_kernel = tf.expand_dims(fire_r_m_next_layer_kernel,axis=3)
#                        elif isinstance(next_layer, lib_snn.layers.Dense):
#                            fire_r_m_next_layer_kernel = tf.expand_dims(fire_r_m_next_layer_kernel,axis=1)
#
#                        fire_r_m_next_layer_kernel = tf.broadcast_to(fire_r_m_next_layer_kernel,next_layer.kernel.shape)
#
#                alpha = 0.1
#                #vth_scale = prev_fire_r_m
#                #vth_sacle = ((1-alpha)+alpha*prev_fire_r_m)
#
#                #vth_init = self.conf.n_init_vth*fire_r_m
#                #vth_init = l.act.vth*fire_r_m
#                #vth_init = l.act.vth*vth_scale
#                vth_init = l.act.vth*((1-alpha)+alpha*fire_r_m)
#
#                l.act.set_vth_init(vth_init)
#
#                if idx_l!=0:
#                    #l.kernel = l.kernel*prev_fire_r_m
#                    l.kernel = l.kernel*((1-alpha) + alpha*prev_fire_r_m)
#
#                prev_fire_r_m = fire_r_m_next_layer_kernel
#
#                #
#                print('fire r')
#                print('{} - fire_r_m: {}'.format(l.name,fire_r_m_c))
#
#            #
#            print('bias enable time')
#            print('{} - t bias en: {}'.format(l.name, tf.reduce_mean(l.bias_en_time)))
#
#    #elif True:
#    elif False:
#        stat='max_999'
#        #stat='max'
#
#        #error_level = 'layer'
#        error_level = 'channel'
#
#        for idx_l, l in enumerate(self.model.layers_w_kernel):
#
#            if idx_l == len(self.model.layers_w_kernel)-1:
#                norm[l.name]=1.0
#                continue
#
#            if self.conf.bias_control:
#                time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
#            else:
#                time = self.conf.time_step
#
#            self.stat_r = read_stat(self, l, stat)
#            stat_r = self.stat_r
#
#            # represent_stat = f_norm(self.dict_stat_r[l.name])
#            #represent_stat = self.norm_b[l.name]
#
#            norm_sat = stat_r / self.norm_b[l.name]
#            norm_sat_e = norm_sat - 1
#            norm_sat_e = tf.where(norm_sat_e > 0, norm_sat, tf.zeros(norm_sat.shape))
#
#            if error_level == 'layer':
#                if isinstance(l, lib_snn.layers.Conv2D):
#                    axis = [0, 1, 2, 3]
#                elif isinstance(l, lib_snn.layers.Dense):
#                    axis = [0, 1]
#                else:
#                    assert False
#            elif error_level == 'channel':
#                if isinstance(l, lib_snn.layers.Conv2D):
#                    axis = [0, 1, 2]
#                elif isinstance(l, lib_snn.layers.Dense):
#                    axis = [0]
#                else:
#                    assert False
#            else:
#                assert False
#
#            #norm_sat_e = tf.reduce_mean(norm_sat_e,axis=axis)
#
#            #
#            ann_act = self.model_ann.get_layer(l.name).record_output
#            ann_act_s = self.model_ann.get_layer(l.name).bias
#            ann_act_d = ann_act - ann_act_s
#            ann_act_r = ann_act_d / ann_act_s
#
#            print(l.name)
#            print('ann_act_r')
#            print(ann_act_r)
#            print(ann_act_s)
#
#            bias_comp_sub = norm_sat_e/ann_act_r
#            bias_comp_sub = tf.where(tf.equal(ann_act,0.0), tf.zeros(bias_comp_sub.shape), bias_comp_sub)
#            bias_comp_sub = tf.reduce_mean(bias_comp_sub,axis=axis)
#
#            print('bias_comp_sub')
#            print(bias_comp_sub)
#
#            l.bias -= bias_comp_sub*0.001
#
#
#        return
#
#    #if True:
#    if False:
#        #
#        path_stat = os.path.join(self.path_model,self.conf.path_stat)
#        #stat = 'max_999'
#        #stat = 'max'
#        #stat = 'max_75'
#        stat = 'median'
#        for idx_l, l in enumerate(self.model.layers_w_kernel):
#            #print(l.name)
#            key=l.name+'_'+stat
#
#            #f_name_stat = f_name_stat_pre+'_'+key
#            f_name_stat = key
#            f_name=os.path.join(path_stat,f_name_stat)
#            f_stat=open(f_name,'r')
#            r_stat=csv.reader(f_stat)
#
#            for row in r_stat:
#                #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
#                stat_r=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])
#
#            #norm[l.name] = np.median(stat_r)
#            norm[l.name] = 1/np.max(stat_r)
#            print(norm[l.name])
#
#    #
#    #norm_wc['conv1'] = norm[0]
#    #norm_b_wc['conv1'] = norm[0]
#    #
#    #    norm_wc['conv1_1'] = norm[1]/norm[0]
#    #    norm_b_wc['conv1_1'] = norm[1]
#    #
#
#    if 'VGG' in self.conf.model:
#        for idx_l, l in enumerate(self.model.layers_w_kernel):
#            if idx_l == 0:
#                norm_wc[l.name] = norm[l.name]
#            else:
#                #norm_wc[l.name] = norm[l.name]/norm[prev_layer_name]
#                norm_wc[l.name] = norm[l.name] / np.expand_dims(norm_b_wc[prev_layer_name],axis=0).T
#
#            prev_layer_name = l.name
#            norm_b_wc[l.name] = norm[l.name]
#
#    for layer in self.model.layers_w_kernel:
#        # layer = self.model.get_layer(name=name_l)
#        if layer.name in norm_wc.keys():
#            layer.kernel = layer.kernel / norm_wc[layer.name]
#        if layer.name in norm_b_wc.keys():
#            layer.bias = layer.bias / norm_b_wc[layer.name]
#
#    for layer in self.model.layers_w_kernel:
#        print(layer.name)
#        print(norm_wc[layer.name])
#
#    #print('end')



#
def calibration_bias_set(self):
    print('calibration_bias_set')
    for l in self.model.layers_w_kernel:
        if not l.name is 'predictions':
            #print(l.name)
            #print(l.bias)
            l.bias = l.bias + glb_bias_comp[l.name]/self.conf.calibration_num_batch


#
def vth_search(self):

    #error_level = 'layer'
    error_level = 'channel'

    for idx_l, l in enumerate(self.model.layers_w_kernel):

        if error_level == 'layer':
            if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
                axis = [0, 1, 2, 3]
            elif isinstance(l, lib_snn.layers.Dense):
                axis = [0, 1]
            else:
                assert False
        elif error_level == 'channel':
            if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
                axis = [0, 1, 2]
            elif isinstance(l, lib_snn.layers.Dense):
                axis = [0]
            else:
                assert False
        else:
            assert False

        dnn_act = l.record_output

        if self.conf.f_w_norm_data:
            stat_max = tf.reduce_max(dnn_act, axis=axis)
            stat_max = tfp.stats.percentile(dnn_act, 99.9, axis=axis)
            stat_max = tf.ones(stat_max.shape)
            #stat_max = tf.reduce_max(dnn_act, axis=axis)
            #stat_max = read_stat(self, l, 'max')
            #stat_max = stat_max / self.norm_b[l.name]
            #stat_max = tf.expand_dims(stat_max,axis=0)
            #stat_max = tf.reduce_max(stat_max,axis=axis)
        else:
            stat_max = read_stat(self, l, 'max')
            #stat_max = stat_max/2
            #stat_max = read_stat(self, l, 'max_999')
            stat_max = tf.expand_dims(stat_max,axis=0)
            #stat_max = dnn_act
            #stat_max = tf.where(stat_max==0.0,tf.ones(stat_max.shape),stat_max)
            stat_max = tf.reduce_max(stat_max,axis=axis)

            #stat_max = tf.reduce_max(dnn_act, axis=axis)

            #stat_max = tf.where(stat_max==0,tf.ones(stat_max.shape),stat_max)


        #num_range = 10
        num_range = 100
        #num_range = 250
        #num_range = 125
        #num_range = 133
        #num_range = 200
        #num_range = 333
        #num_range = 1000
        #num_range = 5000
        #
        #range = tf.random.normal(shape=[num_range],mean=0.0,stddev=0.2,dtype=tf.float32)
        #range = 1.0-tf.math.abs(range)
        #range = tf.random.uniform(shape=[num_range],minval=0.5,maxval=1,dtype=tf.float32)
        #range = tf.random.uniform(shape=[num_range],minval=0.0,maxval=1,dtype=tf.float32)
        #range = tf.range(1/num_range,1+1/num_range,1/num_range,dtype=tf.float32)


        if not l.name in glb_rand_vth.keys():
            #glb_rand_vth[l.name] = tf.random.uniform(shape=[num_range], minval=0.0, maxval=1, dtype=tf.float32)
            #glb_rand_vth[l.name] = tf.random.normal(shape=[num_range],mean=0.0,stddev=0.2,dtype=tf.float32)
            #glb_rand_vth[l.name] = tf.random.uniform(shape=[num_range], minval=0.5, maxval=1, dtype=tf.float32)
            glb_rand_vth[l.name] = tf.range(1/num_range,1+1/num_range,1/num_range,dtype=tf.float32)

            glb_rand_vth[l.name] = tf.expand_dims(glb_rand_vth[l.name],axis=1) * tf.expand_dims(stat_max,axis=0)



        range = glb_rand_vth[l.name]

        #range_vth = range*self.conf.n_init_vth
        #range_vth = range*stat_max

        #print(stat_max)
        #print(stat_max.shape)
        #print(stat_max.shape[0])
        #print(range.shape[0])

        if self.conf.bias_control:
            time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
        else:
            time = self.conf.time_step
        #time = self.conf.time_step

        #

        for idx, vth in enumerate(range):

            #vth = vth_scale*stat_max

            #clip_max = vth*self.conf.time_step
            #clip_max = vth*time
            clip_max = time

            #dnn_act_clip_floor = tf.math.floor(dnn_act/vth)
            dnn_act_clip_floor = tf.math.floor(dnn_act/vth*time)
            dnn_act_clip_floor = tf.clip_by_value(dnn_act_clip_floor,0,clip_max)
            dnn_act_clip_floor = dnn_act_clip_floor*vth/time

            #print(vth)
            err = dnn_act - dnn_act_clip_floor
            err = tf.math.square(err)
            #err = tf.math.square(err)*1/dnn_act

            #print('vth')
            #print(vth)
            #print('time')
            #print(time)
            #print('dnn_act')
            #print(dnn_act)


            # integrated gradients
            if self.conf.vth_search_ig:
                #if isinstance(l, lib_snn.layers.Conv2D):
                #ig_attributions = tf.reduce_mean(glb_ig_attributions[l.name],axis=[0,1])
                #elif isinstance(l, lib_snn.layers.Dense):
                #ig_attributions = glb_ig_attributions[l.name]
                #else:
                #assert False
                #print(err.shape)
                ig_attributions = glb_ig_attributions[l.name]
                eps = 0.01
                #print(err.shape)
                #print(ig_attributions.shape)
                #err = err*(1+ig_attributions)
                #err = err*(10+ig_attributions/tf.reduce_max(ig_attributions))
                #err = err*(2+ig_attributions/tf.reduce_max(ig_attributions))
                #err = err*(1+ig_attributions/tf.reduce_max(ig_attributions))
                #err = err*(0.5+ig_attributions/tf.reduce_max(ig_attributions))
                #err = err*(0.1+ig_attributions/tf.reduce_max(ig_attributions))
                #err = err*(0.01+ig_attributions/tf.reduce_max(ig_attributions))    # old best?
                #err = err*(0.001+ig_attributions/tf.reduce_max(ig_attributions))
                #err = err*(tf.reduce_min(ig_attributions)+ig_attributions)
                #alpha = 0.5
                #err = alpha*err/tf.reduce_max(err,axis=axis)+(1-alpha)*1/(ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                #err = err*ig_attributions
                #err = err*(ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                #jerr = err*(eps+ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                err = err*(eps+ig_attributions/tf.reduce_max(ig_attributions))
                #err = err*(tf.reduce_min(ig_attributions)+ig_attributions/tf.reduce_max(ig_attributions))
                #err = err*(tf.reduce_mean(ig_attributions)+ig_attributions/tf.reduce_max(ig_attributions))
                #err = err*(eps+ig_attributions/tf.reduce_max(ig_attributions))    # old best?
                #err = err*(ig_attributions/tf.reduce_max(ig_attributions))
                #print(err.shape)

            #err = err * snn_act
            #print(err)

            #err = tf.math.abs(err)

            #err = err * snn_act
            #print(err)
            #err = tf.math.square(err)
            err = tf.reduce_mean(err,axis=axis)
            #print(err)

            #if error_level == 'layer':
            #    errs = tf.tensor_scatter_nd_update(errs,[[idx]],[err])
            #elif error_level == 'channel':
            #    errs = tf.tensor_scatter_nd_update(errs,[[idx]],[err])


            # old
            #errs = tf.tensor_scatter_nd_update(errs,[[idx]],[err])
            if not l.name in glb_vth_search_err.keys():
                glb_vth_search_err[l.name] = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)


            vth_err_arr = glb_vth_search_err[l.name]


            #if idx==0:
            if vth_err_arr.size() < num_range:
                glb_vth_search_err[l.name] = vth_err_arr.write(idx, err)
            else:
                glb_vth_search_err[l.name] = vth_err_arr.write(idx, vth_err_arr.read(idx) + err)

            print('here')
            print(vth_err_arr.read(idx))
            print(err)


            #print(errs)
            #assert False
            #errs.append(err)

        #errs_accum = glb_vth_search_err[l.name].stack()
        #return
        #break

def vth_set_and_norm_a(self):

    self.norm = collections.OrderedDict()
    self.norm_b = collections.OrderedDict()
    lib_snn.proc.w_norm_data_channel_wise(self,None,'max_999')


def vth_set_and_norm(self):
    print('vth_set_and_norm')
    #
    stat = None

    dict_stat = collections.OrderedDict()

    norm_wc = collections.OrderedDict()
    norm_b_wc = collections.OrderedDict()

    # norm = [0.5, 0.5, 0.5, ]
    norm = collections.OrderedDict()

    #
    self.cal = collections.OrderedDict()

    #
    #error_level = 'layer'
    error_level = 'channel'



    for idx_l, l in enumerate(self.model.layers_w_kernel):

        if error_level == 'layer':
            if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
                axis = [0, 1, 2, 3]
            elif isinstance(l, lib_snn.layers.Dense):
                axis = [0, 1]
            else:
                assert False
        elif error_level == 'channel':
            if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
                axis = [0, 1, 2]
            elif isinstance(l, lib_snn.layers.Dense):
                axis = [0]
            else:
                assert False
        else:
            assert False

        if False:
            # if idx_l == len(self.model.layers_w_kernel)-1:
            #    norm[l.name]=1.0
            #    continue

            if self.conf.bias_control:
                time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
            else:
                time = self.conf.time_step

            # vth calibration - manual search
            # if False:
            norm[l.name] = 1.0
            # if idx_l == 0:

            dnn_act = self.model_ann.get_layer(l.name).record_output

            if self.conf.f_w_norm_data:
                stat_max = tf.reduce_max(dnn_act, axis=axis)
                stat_max = tfp.stats.percentile(dnn_act, 99.9, axis=axis)
                stat_max = tf.ones(stat_max.shape)
                # stat_max = tf.reduce_max(dnn_act, axis=axis)
                # stat_max = read_stat(self, l, 'max')
                # stat_max = stat_max / self.norm_b[l.name]
                # stat_max = tf.expand_dims(stat_max,axis=0)
                # stat_max = tf.reduce_max(stat_max,axis=axis)
            else:
                stat_max = read_stat(self, l, 'max')
                stat_max = tf.expand_dims(stat_max, axis=0)
                stat_max = tf.reduce_max(stat_max, axis=axis)

            num_range = 100
            # num_range = 250
            # num_range = 125
            # num_range = 133
            # num_range = 200
            # num_range = 333
            # num_range = 1000
            # num_range = 5000
            #
            # range = tf.random.normal(shape=[num_range],mean=0.0,stddev=0.2,dtype=tf.float32)
            # range = 1.0-tf.math.abs(range)
            # range = tf.random.uniform(shape=[num_range],minval=0.5,maxval=1,dtype=tf.float32)
            # range = tf.random.uniform(shape=[num_range],minval=0.0,maxval=1,dtype=tf.float32)
            range = tf.range(1 / num_range, 1 + 1 / num_range, 1 / num_range, dtype=tf.float32)

            # range_vth = range*self.conf.n_init_vth
            # range_vth = range*stat_max

            # print(stat_max)
            # print(stat_max.shape)
            # print(stat_max.shape[0])
            # print(range.shape[0])

            # len_range = range.shape[0]
            if error_level == 'layer':
                errs = tf.zeros([num_range])
            elif error_level == 'channel':
                errs = tf.zeros([num_range, stat_max.shape[0]])
            else:
                assert False

            # assert False
            # errs = []

            if self.conf.bias_control:
                time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
            else:
                time = self.conf.time_step
            # time = self.conf.time_step

            #

            for idx, vth_scale in enumerate(range):

                vth = vth_scale * stat_max

                # clip_max = vth*self.conf.time_step
                # clip_max = vth*time
                clip_max = time

                # dnn_act_clip_floor = tf.math.floor(dnn_act/vth)
                dnn_act_clip_floor = tf.math.floor(dnn_act / vth * time)
                dnn_act_clip_floor = tf.clip_by_value(dnn_act_clip_floor, 0, clip_max)
                dnn_act_clip_floor = dnn_act_clip_floor * vth / time

                # print(vth)
                err = dnn_act - dnn_act_clip_floor
                err = tf.math.square(err)
                # err = tf.math.square(err)*1/dnn_act

                # integrated gradients
                if self.conf.vth_search_ig:
                    # if isinstance(l, lib_snn.layers.Conv2D):
                    # ig_attributions = tf.reduce_mean(glb_ig_attributions[l.name],axis=[0,1])
                    # elif isinstance(l, lib_snn.layers.Dense):
                    # ig_attributions = glb_ig_attributions[l.name]
                    # else:
                    # assert False
                    # print(err.shape)
                    ig_attributions = glb_ig_attributions[l.name]
                    eps = 0.01
                    # print(err.shape)
                    # print(ig_attributions.shape)
                    # err = err*(1+ig_attributions)
                    # err = err*(10+ig_attributions/tf.reduce_max(ig_attributions))
                    # err = err*(2+ig_attributions/tf.reduce_max(ig_attributions))
                    # err = err*(1+ig_attributions/tf.reduce_max(ig_attributions))
                    # err = err*(0.5+ig_attributions/tf.reduce_max(ig_attributions))
                    # err = err*(0.1+ig_attributions/tf.reduce_max(ig_attributions))
                    # err = err*(0.01+ig_attributions/tf.reduce_max(ig_attributions))    # old best?
                    # err = err*(0.001+ig_attributions/tf.reduce_max(ig_attributions))
                    # err = err*(tf.reduce_min(ig_attributions)+ig_attributions)
                    # alpha = 0.5
                    # err = alpha*err/tf.reduce_max(err,axis=axis)+(1-alpha)*1/(ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                    # err = err*ig_attributions
                    # err = err*(ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                    # jerr = err*(eps+ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                    err = err * (eps + ig_attributions / tf.reduce_max(ig_attributions))
                    # err = err*(tf.reduce_min(ig_attributions)+ig_attributions/tf.reduce_max(ig_attributions))
                    # err = err*(tf.reduce_mean(ig_attributions)+ig_attributions/tf.reduce_max(ig_attributions))
                    # err = err*(eps+ig_attributions/tf.reduce_max(ig_attributions))    # old best?
                    # err = err*(ig_attributions/tf.reduce_max(ig_attributions))
                    # print(err.shape)

                # err = err * snn_act
                # print(err)

                # err = tf.math.abs(err)

                # err = err * snn_act
                # print(err)
                # err = tf.math.square(err)
                err = tf.reduce_mean(err, axis=axis)
                # print(err)

                if error_level == 'layer':
                    errs = tf.tensor_scatter_nd_update(errs, [[idx]], [err])
                elif error_level == 'channel':
                    errs = tf.tensor_scatter_nd_update(errs, [[idx]], [err])

                # print(errs)
                # assert False
                # errs.append(err)

        if False:
            if self.conf.f_w_norm_data:
                if self.model_ann is None:
                    dnn_act = l.record_output
                else:
                    dnn_act = self.model_ann.get_layer(l.name).record_output

                stat_max = tf.reduce_max(dnn_act, axis=axis)
                stat_max = tfp.stats.percentile(dnn_act, 99.9, axis=axis)
                stat_max = tf.ones(stat_max.shape)
                # stat_max = tf.reduce_max(dnn_act, axis=axis)
                # stat_max = read_stat(self, l, 'max')
                # stat_max = stat_max / self.norm_b[l.name]
                # stat_max = tf.expand_dims(stat_max,axis=0)
                # stat_max = tf.reduce_max(stat_max,axis=axis)
            else:
                stat_max = read_stat(self, l, 'max')
                #stat_max = read_stat(self, l, 'max_999')
                stat_max = tf.expand_dims(stat_max, axis=0)
                #stat_max = dnn_act
                #stat_max = tf.where(stat_max==0.0,tf.ones(stat_max.shape),stat_max)
                stat_max = tf.reduce_max(stat_max, axis=axis)

        errs_accum = glb_vth_search_err[l.name].stack()
        vth_idx_min_err = tf.math.argmin(errs_accum, output_type=tf.int32)

        # print(vth_idx_min_err)

        # vth_min_err = tf.gather(range_vth,vth_idx_min_err)
        #vth_min_err_scale = tf.gather(glb_rand_vth[l.name], vth_idx_min_err)
        #vth_min_err = vth_min_err_scale * stat_max
        #vth_min_err = tf.gather(glb_rand_vth[l.name], vth_idx_min_err)

        vth_min_err = tf.gather_nd(glb_rand_vth[l.name], tf.transpose(tf.stack([vth_idx_min_err,tf.range(len(vth_idx_min_err))])))

        # print(stat_max)
        ##print(range_vth)
        # print(errs)
        # print(vth_idx_min_err)
        # print(vth_min_err)
        # print(vth_min_err_scale)
        #print(vth_min_err)
        # print(range)

        # TODO: parameterize
        vth_init = vth_min_err
        #vth_init = glb_vth_init[l.name]

        # vth_init = tf.where(vth_init==0,tf.ones(vth_init.shape),vth_init)

        print('{} - vth_init - {}'.format(l.name, vth_init))

        ##
        if False:
            vth_init_fm = vth_init
            if error_level == 'channel':
                if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
                    vth_init_fm = tf.expand_dims(vth_init_fm, axis=0)
                    vth_init_fm = tf.expand_dims(vth_init_fm, axis=1)
                    vth_init_fm = tf.expand_dims(vth_init_fm, axis=2)
                elif isinstance(l, lib_snn.layers.Dense):
                    vth_init_fm = tf.expand_dims(vth_init_fm, axis=0)
                else:
                    assert False
                # print(l.name)
                # print(stat_max.shape)
                # print(vth_init_fm.shape)
                # print(l.act.vth.shape)
                vth_init_fm = tf.broadcast_to(vth_init_fm, shape=l.act.vth.shape)

        #
        #l.act.set_vth_init(vth_init_fm)

        # print(vth_init_fm)

        #
        norm[l.name]=vth_init

        if False:
            stat_max = read_stat(self, l, 'max_999')

            #stat_max = tf.expand_dims(stat_max, axis=0)
            # stat_max = dnn_act
            # stat_max = tf.where(stat_max==0.0,tf.ones(stat_max.shape),stat_max)
            #stat_max = tf.reduce_max(stat_max, axis=axis)

            stat_max = stat_max.reshape(-1, stat_max.shape[-1])
            #print('a')
            #print(stat_max)

            if isinstance(l, lib_snn.layers.InputGenLayer):
                norm_t = 1
            else:
                norm_t = np.max(stat_max,axis=0)
                # norm = np.where(norm == 0, 1, norm)

            if isinstance(l,lib_snn.layers.Dense):
                norm_t = np.max(norm_t,axis=0)

            #print('stat max')
            #print(stat_max)


            #
            norm[l.name]=norm_t
        #norm[l.name]=stat_max
        #norm[l.name]=tf.reduce_max(stat_max)
        #norm[l.name]=1.0

        # if idx_l==0:
        #    norm[l.name]=1.0
        # stat_max = read_stat(self, l, 'max_999')
        # stat_max = tf.expand_dims(stat_max,axis=0)
        # stat_max = tf.reduce_max(stat_max,axis=axis)
        # norm[l.name]=stat_max
        #

        #if True:
        if False:

            if idx_l != 0:
                # l.kernel = l.kernel*prev_fire_r_m
                # l.kernel = l.kernel*((1-alpha) + alpha*prev_fire_r_m)
                # print(prev_vth_init.shape)
                # print(l.kernel.shape)

                if error_level == 'channel':
                    if isinstance(l, lib_snn.layers.Conv2D):
                        prev_vth_init = tf.expand_dims(prev_vth_init, axis=0)
                        prev_vth_init = tf.expand_dims(prev_vth_init, axis=1)
                        prev_vth_init = tf.expand_dims(prev_vth_init, axis=3)
                    elif isinstance(l, lib_snn.layers.Dense):
                        prev_vth_init = tf.expand_dims(prev_vth_init, axis=1)

                #l.kernel = l.kernel * prev_vth_init
                l.kernel = self.model_ann.get_layer(l.name).kernel*prev_vth_init

                # if idx_l != len(self.model.layers_w_kernel) - 1:
                # l.act.set_vth_init(vth_init_fm)
                ##norm[l.name] = vth_init

        prev_vth_init = vth_init

        # snn_act = l.act.spike_count_int/(self.conf.time_step-l.bias_en_time)

        #
        #print('bias enable time')
        #print('{} - t bias en: {}'.format(l.name, tf.reduce_mean(l.bias_en_time)))

    if error_level == 'channel':
        if 'VGG' in self.conf.model:
            for idx_l, l in enumerate(self.model.layers_w_kernel):
                if idx_l == 0:
                    norm_wc[l.name] = norm[l.name]
                else:
                    # norm_wc[l.name] = norm[l.name]/norm[prev_layer_name]
                    norm_wc[l.name] = norm[l.name] / np.expand_dims(norm_b_wc[prev_layer_name], axis=0).T

                prev_layer_name = l.name
                norm_b_wc[l.name] = norm[l.name]
    else:
        assert False

    for layer in self.model.layers_w_kernel:
        # layer = self.model.get_layer(name=name_l)
        #if layer.name in norm_wc.keys():
        layer.kernel = layer.kernel / norm_wc[layer.name]
        #if layer.name in norm_b_wc.keys():
        layer.bias = layer.bias / norm_b_wc[layer.name]

    #
    if False:
        for l in self.model.layers_w_kernel:
            if l.name in norm_b_wc.keys():
                vth_init_fm = norm_b_wc[l.name]

                ##
                if error_level == 'channel':
                    if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
                        vth_init_fm = tf.expand_dims(vth_init_fm, axis=0)
                        vth_init_fm = tf.expand_dims(vth_init_fm, axis=1)
                        vth_init_fm = tf.expand_dims(vth_init_fm, axis=2)
                    elif isinstance(l, lib_snn.layers.Dense):
                        vth_init_fm = tf.expand_dims(vth_init_fm, axis=0)
                    else:
                        assert False
                    # print(l.name)
                    # print(stat_max.shape)
                    # print(vth_init_fm.shape)
                    # print(l.act.vth.shape)
                    vth_init_fm = tf.broadcast_to(vth_init_fm, shape=l.act.vth.shape)

                #
                l.act.set_vth_init(vth_init_fm)

    print('layer norm')
    for layer in self.model.layers_w_kernel:
        print(layer.name)
        print(norm_wc[layer.name])
        print(norm_b_wc[layer.name])


def weight_calibration_inv_vth(self):
    #
    # scale - inv. vth
    for idx_l, l in enumerate(self.model.layers_w_kernel):
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
    for idx_l, l in enumerate(self.model.layers_w_kernel):
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


################################################
# reproduce previous work - ICML-21, ICLR-21
################################################

#
def bias_calibration_ICLR_21(self):
    print('bias_calibration_ICLR_21')

    for idx_l, l in enumerate(self.model.layers_w_kernel):
        if isinstance(l, lib_snn.layers.Conv2D):
            axis = [0,1,2]
        elif isinstance(l, lib_snn.layers.Dense):
            axis = [0]
        else:
            assert False

        time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)

        vth_channel = tf.reduce_mean(l.act.vth,axis=axis)
        bias_comp = vth_channel/(2*time)

        l.bias = l.bias + bias_comp

        print('bias_comp: {:}'.format(bias_comp))

    print('- Done')

# light pipeline
def bias_calibration_ICML_21(self):
    print('\nbias_calibration_ICML_21')

    for idx_l, l in enumerate(self.model.layers_w_kernel):
        if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
            axis = [0,1,2]
        elif isinstance(l, lib_snn.layers.Dense):
            #axis = [0,1]
            axis = [0]
        else:
            assert False

        dnn_act = self.model_ann.get_layer(l.name).record_output

        dnn_act_mean = tf.reduce_mean(dnn_act, axis=axis)
        # snn_act_mean = self.conf.n_init_vth*tf.reduce_mean(snn_act,axis=axis)/self.conf.time_step

        #time=self.conf.time_step
        ##assert False
        time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
        #time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time,axis=axis), tf.float32)

        if l.name == 'predictions':
            continue
            if self.conf.snn_actput_type is 'SPIKE':
                snn_act = l.act.spike_count_int
            elif self.conf.snn_actput_type is 'VMEM':
                snn_act = l.act.vmem
            else:
                assert False
            snn_act = tf.nn.softmax(snn_act/time)
            #assert False
        else:
            snn_act = l.act.spike_count_int/time

        snn_act_mean = self.conf.n_init_vth*tf.reduce_mean(snn_act,axis=axis)

        #if False:  # ICML-20, calibration through bias
        if True:  # ICML-20, calibration through bias

            #print(dnn_act.shape)
            #print(snn_act.shape)

            err_act = dnn_act-snn_act

            #if l.name == 'conv1':
            #    print(tf.reduce_mean(dnn_act))

            # test
            #err_act = tf.where(err_act>0, err_act, tf.zeros(err_act.shape))
            #err_act = tf.where(err_act>0, err_act, err_act/2)

            # integrated gradients
            #if self.conf.vth_search_ig:
                #ig_attributions = glb_ig_attributions[l.name]
                #err_act = err_act * (0.01+ig_attributions / tf.reduce_max(ig_attributions))

            err_act_m = tf.reduce_mean(err_act,axis=axis)

            #bias_comp = dnn_act_mean - snn_act_mean
            bias_comp = err_act_m

            #bias_comp = tf.reduce_mean(dnn_act_mean - snn_act_mean)


            # print
            #print('bias_comp (pre mean): {:}'.format(tf.reduce_mean(bias_comp)))
            #print('bias_comp: {:}'.format(tf.reduce_mean(bias_comp)))

            #print('bias_comp: {:}'.format(bias_comp))

            # test
            #r = tf.random.uniform(shape=bias_comp.shape,minval=0,maxval=1)
            #bias_comp = bias_comp*(1.5+0.5*r)
            #bias_comp *= 0.5
            #bias_comp *= 1.5
            #bias_comp *= 2
            #bias_comp *= 3
            #bias_comp *= 4  # T=128, WP+B-ML
            #bias_comp *= 5  #
            #bias_comp *= 6  #
            #bias_comp *= 8
            #bias_comp *= 10

            #bias_comp = (dnn_act_mean - snn_act_mean)/self.conf.time_step
            #bias_comp = dnn_act_mean - self.conf.n_init_vth*snn_act_mean/self.conf.time_step

            #print(l.name)
            #print(bias_comp)

            #l.bias = l.bias + bias_comp
            #l.bias = l.bias + bias_comp/self.conf.calibration_num_batch

            if not l.name in glb_bias_comp.keys():
                glb_bias_comp[l.name] = tf.zeros(l.bias.shape)

            glb_bias_comp[l.name] = glb_bias_comp[l.name] + bias_comp

            # residual vmem comp
            if False:
            #if True:
                res_vmem = l.act.vmem
                res_vmem = tf.where(res_vmem > 0, res_vmem, tf.zeros(res_vmem.shape))
                res_vmem = tf.reduce_mean(res_vmem,axis=0)
                res_vmem = tf.expand_dims(res_vmem,axis=0)
                res_vmem = tf.broadcast_to(res_vmem,shape=l.act.vmem.shape)

                #vmem_init = res_vmem
                vmem_init = res_vmem*(1-snn_act)
                #vmem_init = res_vmem*(1-snn_act)*0.1
                #l.act.set_vmem_init(res_vmem)
                #l.act.set_vmem_init(res_vmem)
                #r = 0.01
                #vmem_init = tf.random.uniform(l.act.vth.shape,minval=0,maxval=r)*l.act.vth
                #vmem_init = tf.random.uniform(l.act.vth.shape,minval=0,maxval=r)*res_vmem

                l.act.set_vmem_init(vmem_init)
                #l.act.set_vmem_init(-res_vmem)
                #l.act.set_vmem_init(-res_vmem*0.0001)
                #l.act.vmem_init = l.act.vmem_init - res_vmem
                #l.act.vmem_init = l.act.vmem_init + res_vmem

        #elif True:    # calibration through bias and weight (static and dynamic)
        elif False:    # calibration through bias and weight (static and dynamic)
            dnn_act_s = l.bias
            dnn_act_d = dnn_act - dnn_act_s
            dnn_act_s = tf.math.abs(dnn_act_s)
            dnn_act_d = tf.math.abs(dnn_act_d)

            dnn_act_r_s = dnn_act_s/(dnn_act_s+dnn_act_d)
            dnn_act_r_d = dnn_act_d/(dnn_act_s+dnn_act_d)

            err_act = dnn_act-snn_act
            #err_act = tf.where(dnn_act==0,tf.zeros(dnn_act.shape),dnn_act-snn_act)
            #err_act_m = tf.reduce_mean(err_act)

            print(dnn_act_r_s)
            print(dnn_act_r_d)

            calib_s = err_act*dnn_act_r_s

            if False:
                #calib_d = err_act_m*dnn_act_r_d/tf.reduce_mean(snn_act)
                err_act_batchmean = tf.reduce_mean(err_act,axis=0)
                dnn_act_r_d_batchmean = tf.reduce_mean(dnn_act_r_d,axis=0)
                snn_act_batch_mean = tf.reduce_mean(l.act.spike_count_int,axis=0)
                calib_d_batchmean = err_act_batchmean*dnn_act_r_d_batchmean/snn_act_batch_mean
                calib_d = tf.reduce_mean(calib_d_batchmean)


            #calib_d = err_act*dnn_act_r_d/l.act.spike_count_int

            if idx_l ==0:
                spike_avg = time
            else:
                spike_avg = tf.reduce_mean(self.model.layers_w_kernel[idx_l-1].act.spike_count_int)

            #dnn_act_r_d = tf.where(tf.equal(dnn_act_r_d,0.5),tf.zeros(dnn_act_r_d.shape),dnn_act_r_d)
            #calib_d = err_act*dnn_act_r_d/time
            calib_d = err_act*dnn_act_r_d/self.conf.time_step
            #calib_d = err_act*dnn_act_r_d
            #calib_d = err_act*dnn_act_r_d/spike_avg


            calib_s = tf.reduce_mean(calib_s,axis=axis)
            calib_d = tf.reduce_mean(calib_d,axis=axis)

            #print(err_act)
            #print('pp')
            #print(dnn_act-snn_act)
            #print(calib_s)
            print('calib_d: {:}'.format(calib_d))


            weight_comp = calib_d
            #bias_comp = calib_s
            bias_comp = calib_s

            #weight_comp = tf.broadcast_to(weight_comp,shape=l.kernel.shape)

            #
            l.kernel = l.kernel +weight_comp
            #l.kernel *= tf.reduce_mean(dnn_act)
            #l.kernel *= tf.reduce_mean(dnn_act)/time
            l.bias = l.bias + bias_comp

        # mean calib
        #elif True:
        elif False:
        #if True:
            if idx_l!=0:
                #print(l.name)

                ann_prev_l = self.model_ann.get_layer(prev_l.name)
                ann_prev_l_act = ann_prev_l.record_output
                num_sat = tf.where(tf.math.greater_equal(ann_prev_l_act,tf.ones(ann_prev_l_act.shape))
                                   ,tf.ones(ann_prev_l_act.shape),tf.zeros(ann_prev_l_act.shape))

                if isinstance(prev_l, lib_snn.layers.Conv2D):
                    num_sat = tf.reduce_sum(num_sat,axis=[0,1,2])
                    num_n = tf.reduce_prod(num_sat.shape[:3])
                elif isinstance(prev_l, lib_snn.layers.Dense):
                    num_sat = tf.reduce_sum(num_sat,axis=[0])
                    num_n = tf.reduce_prod(num_sat.shape[0])

                r_sat = num_sat/tf.cast(num_n,dtype=tf.float32)


                #stat_mean = read_stat(self,l,'mean')*0.001
                #stat_mean = read_stat(self,prev_l,'mean')*0.01
                #stat_mean = read_stat(self,prev_l,'mean')*r_sat
                stat_mean = read_stat(self,prev_l,'mean')

                #stat_mean *= 0.005
                stat_mean *= 0.001
                #stat_mean *= 0.01
                #stat_mean = stat_mean*r_sat*0.01
                #stat_mean = stat_mean*r_sat*0.02
                #stat_mean = stat_mean*0.01*(0.8+0.2*r_sat)
                #stat_mean = stat_mean*r_sat
                print('{} - r_sat: {}'.format(l.name,r_sat))
                #print('{} - stat_mean*r_sat: {}'.format(l.name,stat_mean))

                if isinstance(l, lib_snn.layers.Conv2D):
                    stat_mean = tf.expand_dims(stat_mean,axis=0)
                    bias_comp = tf.nn.conv2d(stat_mean,l.kernel,strides=l.strides,padding=l.padding.upper())
                    bias_comp = tf.reduce_mean(bias_comp,axis=[0,1,2])
                elif isinstance(l, lib_snn.layers.Dense):

                    #if l.name=='fc1':
                    #    print(stat_mean)
                    #    stat_mean = tf.reduce_max(stat_mean,axis=[0,1])

                    #
                    if isinstance(prev_l,lib_snn.layers.Conv2D):
                        stat_mean = tf.reduce_mean(stat_mean,axis=[0,1])

                    #print(stat_mean)
                    #print(l.kernel)

                    bias_comp = tf.linalg.matvec(l.kernel,stat_mean,transpose_a=True)
                else:
                    assert False



                if False:
                    if isinstance(l, lib_snn.layers.Conv2D):
                        axis = [0, 1]
                        bias_comp = tf.reduce_mean(stat_mean, axis=axis)
                        e_kernel = tf.reduce_mean(l.kernel, )
                        bias_comp = tf.math.multiply(bias_comp)
                    elif isinstance(l, lib_snn.layers.Dense):
                        #axis = [0]
                        bias_comp = stat_mean
                    else:
                        assert False


                print('{} - bias_comp (avg): {}, bias_comp: {}'.format(l.name,tf.reduce_mean(bias_comp),bias_comp))

                l.bias = l.bias + bias_comp

            prev_l = l


    print('- Done')


# adv. pipeline
def vmem_calibration_ICML_21(self):
    print('vmem_calibration_ICML_21')

    for idx_l, l in enumerate(self.model.layers_w_kernel):
        l_ann = self.model.get_layer(l.name)

        if l.name == 'predictions':
            continue

        dnn_act = l_ann.record_output
        snn_act = l.act.spike_count_int

        if self.conf.bias_control:
            time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
        else:
            time=self.conf.time_step

        error = dnn_act - self.conf.n_init_vth*snn_act/time
        error = tf.reduce_mean(error,axis=0)
        vmem_comp = error*time
        #vmem_comp = error
        #vmem_comp = error*self.conf.time_step
        #vmem_comp = error*(self.conf.time_step*0.01)
        #vmem_comp = error
        #vmem_comp = tf.where(vmem_comp>0,vmem_comp,tf.zeros(vmem_comp.shape))
        vmem_comp = tf.expand_dims(vmem_comp,axis=0)
        vmem_comp = tf.broadcast_to(vmem_comp, l.act.dim)

        #print(l.name)
        #print(vmem_comp)

        l.act.set_vmem_init(vmem_comp)


        #print(l.name)
        #print(bias_comp)

        #l.bias = l.bias + bias_comp

    print('- Done')


