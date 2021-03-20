from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#import tensorflow.contrib.eager as tfe

from tensorflow.python.framework import ops

#
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt

from collections import OrderedDict


import functools
import itertools

#import backprop as bp_sspark

#
import lib_snn


#
cce = tf.keras.losses.CategoricalCrossentropy()
kl = tf.keras.losses.KLDivergence()

# TODO: remove
def loss(predictions, labels):

    return tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=predictions,labels=labels,axis=1))


def loss_cross_entoropy(predictions, labels):

    logits = tf.nn.softmax(predictions)
    loss_value = cce(labels,logits)

    return loss_value


def loss_kl(predictions, labels):
    return kl(labels,predictions)



def loss_mse(predictions, labels):
    loss_value = tf.losses.mean_squared_error(labels, predictions)
    #delta = predictions - labels

    #return loss_value, delta
    return loss_value

def loss_kld(predictions, labels):
    return kl(labels,predictions)
    #loss_value = tf.losses.KLDivergence(labels, predictions)
    #return loss_value

#
def compute_accuracy(predictions, labels):
    return tf.reduce_sum(
                tf.cast(
                    tf.equal(
                        tf.argmax(predictions, axis=1, output_type=tf.int64),
                        tf.argmax(labels, axis=1, output_type=tf.int64)
                    ), dtype=tf.float32
                )
            ) / float(predictions.shape[0].value)



#
def train_one_epoch(neural_coding):

    train_snn_func_sel = {
        'RATE': train_one_epoch_rate,
        'WEIGHTED_SPIKE': train_one_epoch_ws,
        'BURST': train_one_epoch_burst,
        'TEMPORAL': train_one_epoch_ttfs
        #'TEMPORAL': train_one_epoch_ttfs_test
    }
    train_snn_func=train_snn_func_sel[neural_coding]

    return train_snn_func


###############################################################################
##
###############################################################################
def train_one_epoch_rate(model, optimizer, dataset):
    print("train_snn.py - train_one_epoch_rate > not yet implemented")
    assert(False)

def train_one_epoch_ws(model, optimizer, dataset):
    print("train_snn.py - train_one_epoch_ws > not yet implemented")
    assert(False)

def train_one_epoch_burst(model, optimizer, dataset):
    print("train_snn.py - train_one_epoch_burst > not yet implemented")
    assert(False)




def train_one_epoch_ttfs_test(model, optimizer, dataset):


    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')

    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')

    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
        grads_and_vars = []

        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, f_training=True)

            loss_value = loss(predictions, labels)
            avg_loss(loss_value)

            accuracy(tf.argmax(predictions,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss_value, trainable_vars)
        #print(trainable_vars)

        grads_and_vars = zip(grads,trainable_vars)

        optimizer.apply_gradients(grads_and_vars)




    return avg_loss.result(), 100*accuracy.result()



fig_glob, axs_glob= plt.subplots(1,2)
axs_glob[0].set_yscale('log')
plt.yscale("log")

#
def reg_tc_para(model):


    train_vars_tk = [v for v in model.trainable_variables if ('temporal_kernel' in v.name)]
    #print(train_vars_tk)

    # l2 norm
    ret = tf.reduce_sum(tf.math.square(train_vars_tk))

    return ret


#
def train_one_epoch_ttfs(model, optimizer, dataset, epoch):
    #global_step = tf.train.get_or_create_global_step()

    #print("epoch: {}".format(global_step))

    #avg_loss = tfe.metrics.Mean('loss_total')
    #avg_loss_pred = tfe.metrics.Mean('loss_pred')
    #avg_loss_enc_st = tfe.metrics.Mean('loss_enc_st')

    #accuracy = tfe.metrics.Accuracy('accuracy')

    avg_loss = tf.keras.metrics.Mean('loss_total')
    avg_loss_pred = tf.keras.metrics.Mean('loss_pred')
    avg_loss_enc_st = tf.keras.metrics.Mean('loss_enc_st')

    accuracy = tf.keras.metrics.Accuracy('accuracy')


    # TODO: parametrize

    #f_loss_enc_spike = True
    #f_loss_enc_spike = False

    f_loss_enc_spike = model.conf.f_loss_enc_spike and (epoch > model.conf.epoch_start_loss_enc_spike)
    f_reg_tc_para = model.conf.f_train_tk_reg and (epoch > model.conf.epoch_start_train_tk)

    #
    if f_loss_enc_spike:
        if epoch % 50 == 0:
            model.dist_beta_sample_func()

    #
    f_plot_done = False


    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    #for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
    for (images, labels) in dataset:


        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, f_training=True, epoch=epoch)
            #predictions = predictions[-1]

            # labels_st: labels (first spike time) for TTFS coding
            #labels_st = lib_snn.ground_truth_in_spike_time(labels,1,model.list_neuron['in'].init_first_spike_time)
            #labels_st = lib_snn.ground_truth_in_spike_time(labels,1,model.init_first_spike_time)

            #
            # label - decoding value
            #labels = model.temporal_decoding(labels_st)

            # label - spike time
            #labels = labels_st

            #softmax = tf.nn.softmax(predictions)
            #print(softmax[0,:].numpy())
            #print(labels[0,:].numpy())

            #print(predictions)
            #print(labels)
            #print(labels_st)

            loss_list=OrderedDict()
            avg_loss_list=OrderedDict()
            loss_weight=OrderedDict()
            #loss_name=['prediction','enc_st','max_enc_st','min_enc_st', 'max_tk_rep']
            #loss_name=['prediction','enc_st','max_enc_st','min_enc_st']
            #loss_name=['prediction', 'max_enc_st']



            if f_loss_enc_spike:
                loss_name=['prediction', 'enc_st']
            else:
                loss_name=['prediction']

            # regularizer - temporal kernel paras.
            #f_reg_tc_para = True
            #f_reg_tc_para = False
            if f_reg_tc_para:
                loss_name.append('reg_tc_para')

            #
            # loss - prediction
            #loss_list['prediction'] = loss(predictions, labels)
            loss_list['prediction'] = loss_cross_entoropy(predictions, labels)

            # regularizer temporal kernel paras - reg_tc_para
            if f_reg_tc_para:
                loss_list['reg_tc_para'] = reg_tc_para(model)


#            #
#            # loss - encoded spike time
#            #print(model.t_conv1)
#            #print(tf.round(model.t_conv1))
#
#            #enc_st = model.t_conv1
#
#            loss_tmp=0
#            #for _, (name, enc_st) in enumerate(model.list_st.items()):
#            #for name in model.layer_name:
#
#                #enc_st=model.list_st[name]
#
#
#                #print(name)
#                #print(enc_st)
#
#                #enc_st = model.list_v[name]
#
#            #enc_st = model.list_st['conv1']
#            enc_st = model.list_tk['conv1'].out_enc
#            round_enc_st = tf.constant(tf.round(enc_st),shape=enc_st.shape)
#            loss_tmp += loss_mse(enc_st,round_enc_st)
#
#            #enc_st = model.list_st['conv2']
#            enc_st = model.list_tk['conv2'].out_enc
#            round_enc_st = tf.constant(tf.round(enc_st),shape=enc_st.shape)
#            loss_tmp += loss_mse(enc_st,round_enc_st)
##
##            #enc_st = model.list_st['fc1']
##            enc_st = model.list_tk['fc1'].out_enc
##            round_enc_st = tf.constant(tf.round(enc_st),shape=enc_st.shape)
##            loss_tmp += loss_mse(enc_st,round_enc_st)
#
#            loss_list['enc_st'] = loss_tmp
#            #loss_list['enc_st'] += loss_mse(enc_st,round_enc_st)


            # TODO: here - KL divergence loss
            if f_loss_enc_spike:
                if model.conf.model_name=='cnn_mnist_train_ANN_surrogate':

                    # loss - encoded spike time (KL-divergence)
                    loss_tmp = 0
                    enc_st = model.list_tk['conv1'].out_enc
                    enc_st_target_end = 200

                    #enc_st = tf.where(enc_st>enc_st_target_end,enc_st_target_end)
                    enc_st = tf.clip_by_value(enc_st,0,enc_st_target_end)
                    #tmp = tf.where(tf.equal(enc_st,0),100,enc_st)
                    #print(tf.reduce_min(tmp))
                    enc_st = tf.round(enc_st)
                    #enc_st = tf.histogram_fixed_width(enc_st, [0,model.list_tk['conv1'].tw])
                    #enc_st = tf.histogram_fixed_width(enc_st, [0,enc_st_target_end], dtype=tf.float32)
                    enc_st = tf.histogram_fixed_width(enc_st, [0,enc_st_target_end])
                    enc_st = tf.cast(enc_st,tf.float32)
                    #max_enc_target = tf.where(t > max_enc_st*max_border,\
                    #dist = tfd.Beta(1,3)
                    #dist = tfd.Beta(0.9,0.1)
                    dist = tfd.Beta(0.1,0.9)
                    dist_sample = dist.sample(enc_st.shape)
                    #dist_sample = tf.multiply(dist_sample,model.conf.time_window)
                    dist_sample = tf.multiply(dist_sample,enc_st_target_end)
                    dist_sample = tf.round(dist_sample)
                    loss_tmp += loss_kld(enc_st,dist_sample)

                    #
                    #print(enc_st)
                    #plt.figure()
                    #plt.hist(tf.reshape(dist_sample,[-1]))
                    #plt.figure()
                    #plt.hist(tf.reshape(enc_st,[-1]))
                    #plt.show()
                    #assert(False)


                    enc_st = model.list_tk['conv2'].out_enc
                    enc_st = tf.clip_by_value(enc_st,0,enc_st_target_end)
                    enc_st = tf.round(enc_st)
                    #enc_st = tf.histogram_fixed_width(enc_st, [0,model.list_tk['conv2'].tw])
                    #enc_st = tf.histogram_fixed_width(enc_st, [0,200])
                    #enc_st = tf.histogram_fixed_width(enc_st, [0,enc_st_target_end], dtype=tf.float32)
                    enc_st = tf.histogram_fixed_width(enc_st, [0,enc_st_target_end])
                    enc_st = tf.cast(enc_st,tf.float32)
                    #max_enc_target = tf.where(t > max_enc_st*max_border,\
                    #dist = tfd.Beta(1,3)
                    #dist = tfd.Beta(0.9,0.1)
                    dist = tfd.Beta(0.1,0.9)
                    dist_sample = dist.sample(enc_st.shape)
                    #dist_sample = tf.multiply(dist_sample,model.conf.time_window)
                    dist_sample = tf.multiply(dist_sample,enc_st_target_end)
                    dist_sample = tf.round(dist_sample)
                    loss_tmp += loss_kld(enc_st,dist_sample)


    #            #print(type(dist_sample))
    #            #fig, axs = plt.subplots(1,2)
    #
    #            #axs_tmp[0].hist(tf.reshape(dist_sample[0,:,:,:],shape=-1))
    #            #axs_tmp[1].hist(tf.reshape(enc_st[0,:,:,:],shape=-1))
    #            #plt.hist(tf.reshape(dist_sample[0,:,:,:],shape=-1))
    #            plt.hist(tf.reshape(enc_st[0,:,:,:],shape=-1),bins=100)
    #
    #            plt.draw()
    #            plt.pause(0.0000000000000001)
                else:

                    # loss - encoded spike time (KL-divergence)
                    loss_tmp = 0

                    #alpha = 0.1
                    #beta = 0.9

                    #dist = tfd.Beta(alpha,beta)

                    for l_name, tk in model.list_tk.items():
                        #print(tk)
                        #print(l_name)
                        #enc_st = model.list_tk['conv1'].out_enc
                        enc_st = tk.out_enc
                        #enc_st_target_end = 200
                        #enc_st_target_end = tk.tw*10

                        enc_st = tf.clip_by_value(enc_st,0,model.enc_st_target_end)
                        #enc_st = tf.round(enc_st)

                        enc_st = tf.reshape(enc_st, [-1])

                        enc_st = tf.histogram_fixed_width(enc_st, [0,model.enc_st_target_end], nbins=model.enc_st_target_end)

                        enc_st = tf.cast(enc_st,tf.float32)

                        #enc_st = tf.cast(enc_st,tf.float32)

                        #dist_sample = dist.sample(enc_st.shape)
                        #dist_sample = enc_st
                        dist_sample = model.dist_beta_sample[l_name]

                        loss_tmp += loss_kld(enc_st,dist_sample)


                        #print(enc_st)
                        #print(dist_sample)

                        #fig, axs = plt.subplots(1,2)

#                        if (not f_plot_done) and (epoch % 1==0) and (l_name=='conv2'):
#                            f_plot_done = True
#                            axs_glob[0].plot(enc_st)
#                            axs_glob[1].plot(dist_sample)
#
#
#                            plt.draw()
#                            plt.pause(0.0000000000000001)

                loss_list['enc_st'] = loss_tmp

#            else:
#                for l_name, tk in model.list_tk.items():
#                    if (not f_plot_done) and (epoch % 10 == 0) and (epoch!=0) and (l_name == 'conv2'):
#
#                        f_plot_done = True
#
#                        enc_st = tk.out_enc
#                        enc_st = tf.clip_by_value(enc_st, 0, model.enc_st_target_end)
#                        enc_st = tf.reshape(enc_st, [-1])
#                        enc_st = tf.histogram_fixed_width(enc_st, [0, model.enc_st_target_end],
#                                                          nbins=model.enc_st_target_end)
#
#                        enc_st = tf.cast(enc_st, tf.float32)
#
#                        axs_glob[0].plot(enc_st)
#
#                        plt.draw()
#                        plt.pause(0.0000000000000001)




            #            # debug
            #            for l_name, tk in model.list_tk.items():
#                enc_st = tk.out_enc
#                #enc_st = tf.clip_by_value(enc_st,0,model.enc_st_target_end)
#                #enc_st = tf.clip_by_value(enc_st,0,model.conf.time_window)
#                enc_st = tf.reshape(enc_st, [-1])
#                enc_st = tf.histogram_fixed_width(enc_st, [0,model.enc_st_target_end], nbins=model.enc_st_target_end)
#                enc_st = tf.cast(enc_st,tf.float32)
#
#                axs_glob[0].plot(enc_st)
#
#                plt.draw()
#                plt.pause(0.0000000000000001)


#
#            # TODO: Is it really needed? -> regularizer
#            # loss - maximum encoded spike time
#
#            #loss_tmp=0
#            #for _, (name, enc_st) in enumerate(model.list_st.items()):
#                #v = model.list_v[name]
#            #min_val = model.temporal_encoding(model.v_conv1)
#
#            #
#            loss_tmp=0
#            #max_border = 4.0/5.0
#            max_border = 1.0
#
#            #v=model.list_v['conv1']
#            #t=model.list_st['conv1']
#            v=model.list_tk['conv1'].in_enc
#            t=model.list_tk['conv1'].out_enc
#            max_enc_st = tf.constant(model.conf.time_window,dtype=tf.float32,shape=v.shape)
#            #non_bounded_enc_st = model.list_tk['conv1'].call_encoding_kernel(v)
#            max_enc_target = tf.where(t > max_enc_st*max_border,\
#                                      max_enc_st,\
#                                      t)
#            #loss_tmp += loss_mse(non_bounded_enc_st, max_enc_target)
#            loss_tmp += loss_mse(t, max_enc_target)
#
#            #v=model.list_v['conv2']
#            #t=model.list_st['conv2']
#            v=model.list_tk['conv2'].in_enc
#            t=model.list_tk['conv2'].out_enc
#            max_enc_st = tf.constant(model.conf.time_window,dtype=tf.float32,shape=v.shape)
#            #non_bounded_enc_st = model.list_tk['conv2'].call_encoding_kernel(v)
#            #max_enc_target = tf.where(max_enc_st < non_bounded_enc_st, \
#            max_enc_target = tf.where(t > max_enc_st*max_border,\
#                                      max_enc_st,\
#                                      t)
#            #loss_tmp += loss_mse(non_bounded_enc_st, max_enc_target)
#            loss_tmp += loss_mse(t, max_enc_target)

            #loss_list['max_enc_st'] = loss_tmp


#
#
#            #
#            # loss - minimum encoded spike time
#            #enc_st = model.t_conv1
#            loss_tmp=0
#
#            #t = model.list_st['conv1']
#            t=model.list_tk['conv1'].out_enc
#            min_target_value = model.conf.tc        # 1 tau target
#            min_enc_st_target = tf.constant(min_target_value,dtype=tf.float32,shape=t.shape)
#            min_enc_target = tf.where(min_enc_st_target < t,\
#                                      min_enc_st_target,\
#                                      t)
#            loss_tmp += loss_mse(t, min_enc_target)
#
#
#            #t = model.list_st['conv2']
#            t=model.list_tk['conv2'].out_enc
#            min_target_value = model.conf.tc        # 1 tau target
#            min_enc_st_target = tf.constant(min_target_value,dtype=tf.float32,shape=t.shape)
#            min_enc_target = tf.where(min_enc_st_target < t, \
#                                      min_enc_st_target, \
#                                      t)
#            loss_tmp += loss_mse(t, min_enc_target)
#            #print(t)
#            #print(min_target_value)


#            #t = model.list_st['fc1']
#            t=model.list_tk['fc1'].out_enc
#            min_target_value = model.conf.tc        # 1 tau target
#            min_enc_st_target = tf.constant(min_target_value,dtype=tf.float32,shape=t.shape)
#            min_enc_target = tf.where(min_enc_st_target < t, \
#                                      min_enc_st_target, \
#                                      t)
#            loss_tmp += loss_mse(t, min_enc_target)
#
#            loss_list['min_enc_st'] = loss_tmp

            # TODO: modifiy decoding dim
            # max_tk_rep
            #loss_tmp=[0]
            #v = model.list_tk['in'].in_enc
            #max_tk_rep = model.list_tk['in'].call_decoding(tf.constant(0.0,shape=v.shape,dtype=tf.float32),0,True)
            #max_tk_rep_target = tf.where(v > max_tk_rep,max_tk_rep,v)
            #loss_tmp += loss_mse(v,max_tk_rep_target)
            #
            #v = model.list_tk['conv1'].in_enc
            #max_tk_rep = model.list_tk['conv1'].call_decoding(tf.constant(0.0,shape=v.shape,dtype=tf.float32),0,True)
            #max_tk_rep_target = tf.where(v > max_tk_rep,max_tk_rep,v)
            #loss_tmp += loss_mse(v,max_tk_rep_target)
            #
            #v = model.list_tk['conv2'].in_enc
            #max_tk_rep = model.list_tk['conv2'].call_decoding(tf.constant(0.0,shape=v.shape,dtype=tf.float32),0,True)
            #max_tk_rep_target = tf.where(v > max_tk_rep,max_tk_rep,v)
            #loss_tmp += loss_mse(v,max_tk_rep_target)


            #print(v>max_tk_rep)
            #print(max_tk_rep)

            #loss_list['max_tk_rep'] = loss_tmp







            #
            #
            loss_weight['prediction']=1.0

            if f_loss_enc_spike:
                #loss_weight['enc_st']=0.001
                loss_weight['enc_st']=model.conf.w_loss_enc_spike

            if f_reg_tc_para:
                loss_weight['reg_tc_para']=model.conf.w_train_tk_reg

            #loss_weight['max_enc_st']=0.0
            ##loss_weight['min_enc_st']=0.1
            #loss_weight['min_enc_st']=0.0
            #loss_weight['max_tk_rep']=0.0


            #
            #if epoch > 100:
            #if True:
                #loss_weight['enc_st']=0.01
                #loss_weight['max_enc_st']=0.00001

            #if epoch > 100:
            #    #loss_weight['enc_st']=0.1
            #    loss_weight['min_enc_st']=0.00001

            #
            #if epoch > model.list_tk['in'].epoch_start_train_tk:
                #loss_weight['min_enc_st']=0.0000001
                #loss_weight['max_enc_st']=0.00000001
                #loss_weight['max_tk_rep']=0.01
                #loss_weight['max_tk_rep']=1.0

            #
            #loss_total = loss_pred
            #loss_total = loss_pred + loss_enc_st
            #loss_total = loss_pred + loss_enc_st + loss_max_enc_st
            #loss_total = loss_pred + loss_enc_st + loss_max_enc_st + loss_min_enc_st

            loss_total=0
            for l_name in loss_name:
                loss_total = loss_total + loss_weight[l_name]*loss_list[l_name]

            #
            #avg_loss(loss_total)
            #avg_loss_pred(loss_list['prediction'])



            avg_loss = loss_total
            avg_loss_pred = loss_list['prediction']

            avg_loss_enc_st=0.0
            avg_loss_max_enc_st=0.0
            avg_loss_min_enc_st=0.0
            avg_loss_max_tk_rep=0.0



            if f_loss_enc_spike:
                if isinstance(loss_list['enc_st'],int):
                    avg_loss_enc_st=0.0
                else:
                    avg_loss_enc_st=loss_list['enc_st'].numpy()

            #avg_loss_max_enc_st=loss_list['max_enc_st'].numpy()
            #avg_loss_min_enc_st=loss_list['min_enc_st'].numpy()
            #avg_loss_max_tk_rep=loss_list['max_tk_rep'].numpy()


            #print(loss_enc_st)


            #
            #print("loss - {:0.3f}, loss_pred - {:0.3f}, loss_enc_st - {:0.3f}"\
            #      .format(avg_loss.result(),avg_loss_pred.result(),avg_loss_enc_st))

            #print("loss - {:0.3f}, l_pred - {:0.3f}, l_enc_st - {:0.3f}, l_max_enc_st - {:0.3f}"\
            #      .format(avg_loss.result(),avg_loss_pred.result(),avg_loss_enc_st,avg_loss_max_enc_st))


            #print("loss: {:0.3f}, l_pred: {:0.3f}, l_enc_st: {:0.3f}, l_max_enc_st: {:0.3f}, l_min_enc_st: {:0.3f}"\
            #      .format(avg_loss.result(),avg_loss_pred.result(),avg_loss_enc_st,avg_loss_max_enc_st,avg_loss_min_enc_st))



            accuracy(tf.argmax(predictions,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
            #accuracy(tf.argmin(predictions,axis=1,output_type=tf.int64), tf.argmin(labels,axis=1,output_type=tf.int64))

            #print("acc: {}, loss: {}".format(accuracy.result(), avg_loss.result()))


        #print(tape.gradient(loss_total, model.list_layer['fc1'].kernel))

        #
        #assert(False)


        #
        # weight

        #trainable_vars = model.trainable_variables
        train_vars_w = [v for v in model.trainable_variables if ('neuron' not in v.name) and ('temporal_kernel' not in v.name)]

        #for vars in train_vars_w:
            #print(vars.name)

        #trainable_vars_tmporal_kernel=[]
        #for l_name in model.layer_name:
            #trainable_vars_tmporal_kernel.append(model.list_tc[l_name])
            #trainable_vars_tmporal_kernel.append(model.list_td[l_name])
            #trainable_vars_tmporal_kernel.append(model.list_ta[l_name])

        #print(trainable_vars)

        #print(tf.reduce_mean(model.list_ta['conv1']))
        #assert(False)


        #for v in train_vars_tk:
            #print(v.name)

        #
        #grads = tape.gradient(loss_total, trainable_vars)

        #
        if epoch < model.list_tk['in'].epoch_start_train_tk:
            grads = tape.gradient(loss_list['prediction'], train_vars_w)
            grads_and_vars = zip(grads,train_vars_w)
            optimizer.apply_gradients(grads_and_vars)
        else:

            f_train_w = False
            f_train_tk = False

            # temporal kernel
            train_vars_tk = []

            if model.train_tk_strategy == 'N':
                train_vars_tk = [v for v in model.trainable_variables if ('temporal_kernel' in v.name)]

                f_train_w = True
                f_train_tk = True

            elif model.train_tk_strategy == 'R':
                #print(model.train_tk_strategy)
                #print(model.train_tk_strategy_coeff)

                #epoch_mod = int(epoch/model.train_tk_strategy_coeff) % model.train_tk_strategy_coeff_x3
                epoch_mod = int(epoch/model.train_tk_strategy_coeff) % 3

                #print(model.conf.train_tk_strategy)
                #print(epoch_mod)

                if epoch_mod == 0:
                    f_train_w = True
                    f_train_tk = False
                elif epoch_mod == 1:
                    train_vars_tk = [v for v in model.trainable_variables if (('temporal_kernel' in v.name) and ('tc' in v.name))]
                    f_train_w = False
                    f_train_tk = True

                    #print([v.name for v in model.trainable_variables if (('temporal_kernel' in v.name) and ('tc' in v.name))])
                elif epoch_mod == 2:
                    train_vars_tk = [v for v in model.trainable_variables if (('temporal_kernel' in v.name) and ('td' in v.name))]
                    f_train_w = False
                    f_train_tk = True

                else:
                    assert False

            elif model.train_tk_strategy == 'I':
                epoch_mod = epoch % (model.train_tk_strategy_coeff+2)

                if epoch_mod < model.train_tk_strategy_coeff:
                    f_train_w = True
                    f_train_tk = False
                elif epoch_mod == model.train_tk_strategy_coeff:
                    train_vars_tk = [v for v in model.trainable_variables if (('temporal_kernel' in v.name) and ('tc' in v.name))]
                    f_train_w = False
                    f_train_tk = True

                    #print([v.name for v in model.trainable_variables if (('temporal_kernel' in v.name) and ('tc' in v.name))])
                elif epoch_mod == model.train_tk_strategy_coeff+1:
                    train_vars_tk = [v for v in model.trainable_variables if (('temporal_kernel' in v.name) and ('td' in v.name))]
                    f_train_w = False
                    f_train_tk = True

                else:
                    assert False

            else:
                assert False

            if (f_train_w==True) and (f_train_tk == True):
                train_vars = train_vars_w + train_vars_tk

                grads_w = tape.gradient(loss_total, train_vars_w)
                grads_tk = tape.gradient(loss_total, train_vars_tk)

                grads = grads_w + [x*model.conf.w_train_tk for x in grads_tk if x is not None]

                f_all_none = False

                #print(grads_tk)
                #print(grads_w[0])

            elif (f_train_w==True) and (f_train_tk == False):
                train_vars = train_vars_w
                grads = tape.gradient(loss_total, train_vars)

                f_all_none = False

            elif (f_train_w==False) and (f_train_tk == True):
                train_vars = train_vars_tk
                grads = tape.gradient(loss_total, train_vars)

                f_all_none = (grads == [None] * len(grads))

            else:
                assert False

            if not f_all_none:
                grads_and_vars = zip(grads,train_vars)
                optimizer.apply_gradients(grads_and_vars)

            #grads = tape.gradient(loss_total, train_vars)
            #grads_and_vars = zip(grads,train_vars)
            #optimizer.apply_gradients(grads_and_vars)

            #print(grads_w[10])
            #print(grads_tk[10])
            #print(grads_tk)


            #print(grads_tk)
            #print([x*0.1 if x is not None else x for x in grads_tk])
            #print(type(grads_tk))
            #print(type(grads))
            ##print(np.ndarray(grads).shape)
            #print(len(grads))
            #print(len(grads_t))
            #assert False

            #grads_and_vars_w = zip(grads_w,train_vars_w)
            #grads_and_vars_tk = zip(grads_tk,train_vars_tk)

            #optimizer.apply_gradients(grads_and_vars_w)
            #optimizer.apply_gradients(grads_and_vars_tk)


            # train_vars_tk
            #
            #grads_t_kernel = tape.gradient(loss_total,train_vars_tk)
            #grads_and_vars_tk = zip(grads_t_kernel,train_vars_tk)
            #optimizer.apply_gradients(grads_and_vars_tk)


        #
        # make encoded spike time integer
        #
        #with tf.GradientTape(persistent=True) as tape:
        #    predictions = model(images, f_training=True)



    #return avg_loss.result(), 100*accuracy.result()
    #return avg_loss.result(), avg_loss_pred.result(),avg_loss_enc_st,\
    #       avg_loss_max_enc_st,avg_loss_min_enc_st,avg_loss_max_tk_rep,100*accuracy.result()
    return avg_loss, avg_loss_pred, avg_loss_enc_st,\
           avg_loss_max_enc_st,avg_loss_min_enc_st,avg_loss_max_tk_rep,100*accuracy.result()



###############################################################################
##
###############################################################################

# ANN training function
#@profile
#def train_one_epoch(model, optimizer, dataset):
#    tf.train.get_or_create_global_step()
#    avg_loss = tfe.metrics.Mean('loss')
#    accuracy = tfe.metrics.Accuracy('accuracy')
#
#    def model_loss(labels, images):
#        prediction = model(images, f_training=True)
#        loss_value = loss(prediction, labels)
#        avg_loss(loss_value)
#        accuracy(tf.argmax(prediction,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
#        #tf.contrib.summary.scalar('loss', loss_value)
#        #tf.contrib.summary.scalar('accuracy', compute_accuracy(prediction, labels))
#        return loss_value
#
#    #builder = tf.profiler.ProfileOptionBuilder
#    #opts = builder(builder.time_and_memory()).order_by('micros').build()
#
#    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
#        #print(batch)
#        #print(images)
#        #print(labels)
#        #with tf.contrib.summary.record_summaries_every_n_global_steps(10):
#
#        batch_model_loss = functools.partial(model_loss, labels, images)
#        optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())
#
#        #batch_model_loss = functools.partial(model_loss, labels, images)
#        #optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())
#
#    #print('Train set: Accuracy: %4f%%\n'%(100*accuracy.result()))
#    return avg_loss.result(), 100*accuracy.result()




def train_ann_one_epoch_mnist_cnn(model, optimizer, dataset, conf):
    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')

    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')


    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
        grads_and_vars = []

        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, f_training=True)

            loss_value = loss(predictions, labels)
            avg_loss(loss_value)

            accuracy(tf.argmax(predictions,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


        # backprop
        delta = []
        delta_tmp = loss_value - labels
        delta.append(delta_tmp)

        #delta_tmp = bp_Dense(model.list_l[3],delta_tmp)
        #delta_tmp = bp_relu(model.list_a[3], delta_tmp)
        #delta.append(delta_tmp)


        # gradient
        layer = model.list_l[3]
        d_local = model.list_a[3]
        delt = delta[0]

        #print(tf.shape(delt))
        #print(tf.shape(d_local))
        grad_t, grad_b_t = grad_Dense(layer,delt,d_local)

        grad = tape.gradient(loss_value, layer.kernel)
        grad_b = tape.gradient(loss_value, layer.bias)




        del_in = delta_tmp
        d_s_w = d_local

        del_in_e = tf.tile(tf.expand_dims(del_in,1), [1,tf.shape(layer.kernel)[0],1])
        d_s_w_e = tf.tile(tf.expand_dims(d_s_w,2), [1,1,tf.shape(layer.kernel)[1]])

        grad_t = tf.reduce_sum(tf.multiply(del_in_e, d_s_w_e),0)
        grad_b_t = tf.reduce_sum(del_in,0)


        #grad_b = tape.gradient(loss_value,model.list_a[4])

        #print(grad_b_t)
        #3print(grad_b)

        #diff = grad_t - grad
        #diff = grad_b_t - grad_t
        #print('max: %.3f, sum: %.3f'%(tf.reduce_max(diff),tf.reduce_sum(diff)))



        grads_and_vars.append((grad,layer.kernel))
        grads_and_vars.append((grad_b,layer.bias))

        layer = model.list_l[2]
        #d_local = model.p_flat
        #delt = delta[1]
        #grad, grad_b = grad_Dense(layer,delt,d_local)
        grad = tape.gradient(loss_value, layer.kernel)
        grad_b = tape.gradient(loss_value, layer.bias)

        grads_and_vars.append((grad,layer.kernel))
        grads_and_vars.append((grad_b,layer.bias))


        layer = model.list_l[1]
        grad = tape.gradient(loss_value, layer.kernel)
        grad_b = tape.gradient(loss_value, layer.bias)

        grads_and_vars.append((grad, layer.kernel))
        grads_and_vars.append((grad_b, layer.bias))


        layer = model.list_l[0]
        grad = tape.gradient(loss_value, layer.kernel)
        grad_b = tape.gradient(loss_value, layer.bias)

        grads_and_vars.append((grad, layer.kernel))
        grads_and_vars.append((grad_b, layer.bias))


        optimizer.apply_gradients(grads_and_vars)

        #grads = tape.gradient(loss_value, model.variables)
        #optimizer.apply_gradients(zip(grads, model.variables))

    return avg_loss.result(), 100*accuracy.result()

def train_snn_one_epoch_mnist_cnn(model, optimizer, dataset, conf):
    print('not defined yet')

def train_ann_one_epoch_mnist(model, optimizer, dataset, conf):
    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')


    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')


    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
        grads_and_vars = []

        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, f_training=True)


            #loss_value, delta = loss_cross_entropy(predictions, labels)
            loss_value, delta = loss_mse(predictions, labels)

            avg_loss(loss_value)

            accuracy(tf.argmax(predictions,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


        #del_last = loss_value - labels_expandded

        #softmax = tf.nn.softmax(predictions)

        #del_last = tf.cast(softmax,tf.float32) - tf.cast(labels,tf.float32)
        #del_last = tf.cast(predictions,tf.float32) - tf.cast(labels,tf.float32)

        del_last = delta

        idx_layer_last = 1

        for idx_layer in range(idx_layer_last,-1,-1):

            layer = model.list_l[idx_layer]
            act = model.list_a[idx_layer]
            neuron = model.list_n[idx_layer]

            # backprop
            if idx_layer == idx_layer_last:
                del_in = del_last
            else:
                del_in = del_out

            del_out = bp_Dense(layer, del_in)
            del_out = bp_relu(act, del_out)

            #print(tape.gradient(model.list_s[-1], model.list_a[-1]))
            #print(tape.gradient(loss_value,model.list_a[-1]))


            #test = tfe.gradients_function(model.list_l[1].call)
            #test = tfe.implicit_gradients(model.list_l[1].call)

            #print(test(del_in))


            # weight update
            d_s_w = act

            del_in_e = tf.tile(tf.expand_dims(del_in,1), [1,tf.shape(layer.kernel)[0],1])
            d_s_w_e = tf.tile(tf.expand_dims(d_s_w,2), [1,1,tf.shape(layer.kernel)[1]])

            grad_b = tf.reduce_sum(del_in,0)
            grad = tf.reduce_sum(tf.multiply(del_in_e, d_s_w_e),0)

            grads_and_vars.append((grad, layer.kernel))
            grads_and_vars.append((grad_b, layer.bias))

        optimizer.apply_gradients(grads_and_vars)

    return avg_loss.result(), 100*accuracy.result()



def train_snn_one_epoch_mnist_psp_ver(model, optimizer, dataset, conf):
    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')


    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')


    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
        grads_and_vars = []

        #with tf.GradientTape(persistent=True) as tape:
        predictions_times = model(images, f_training=True)

        if predictions_times.shape[1] != labels.numpy().shape[0]:
            predictions_times_trimmed = predictions_times[:,0:labels.numpy().shape[0],:]
            d = predictions_times.shape[1] - labels.numpy().shape[0]

            labels_expandded = tf.concat([labels, tf.zeros((d,)+(labels.numpy().shape[1],))],axis=0)
            f_exp = True
        else:
            predictions_times_trimmed = predictions_times
            labels_expandded = labels
            f_exp = False


        predictions_trimmed = predictions_times_trimmed[-1]

        loss_value = loss(predictions_trimmed, labels)

        avg_loss(loss_value)

        #accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
        accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


        #predictions = predictions_times[-1]

        #del_last = loss_value - labels_expandded

        softmax = tf.nn.softmax(predictions_trimmed)

        del_last = tf.cast(softmax,tf.float32) - tf.cast(labels,tf.float32)
        #del_last = tf.cast(predictions_trimmed,tf.float32) - tf.cast(labels,tf.float32)

        if f_exp :
            del_last = tf.concat([del_last, tf.zeros((d,)+(del_last.numpy().shape[1],))],axis=0)

        idx_layer_last = 1

        for idx_layer in range(idx_layer_last,-1,-1):

            layer = model.list_l[idx_layer]
            act = model.list_a[idx_layer]
            neuron_pri = model.list_n[idx_layer]
            neuron_post = model.list_n[idx_layer+1]
            tpsp = model.list_tpsp[idx_layer]

            # backprop
            if idx_layer == idx_layer_last:
                del_in = del_last
            else:
                del_in = del_out

            del_out = bp_Dense(layer, del_in)
            #del_out = tf.multiply(del_out,neuron.get_spike_rate())
            #del_out = bp_relu(act, del_out)
            tpsp_m_post = tf.reduce_sum(tpsp,2)
            tpsp_m_post = tf.divide(tf.cast(tpsp_m_post,tf.float32),tf.cast(neuron_pri.spike_counter_int*conf.n_init_vth,tf.float32))
            del_out = tf.multiply(del_out,tpsp_m_post)

            #print(tape.gradient(model.list_s[-1], model.list_a[-1]))
            #print(tape.gradient(loss_value,model.list_a[-1]))
            #print(tape.gradient(loss_value,model.list_l[-1].kernel))


            # weight update
            #d_s_w = act
            #d_s_w = neuron.get_spike_rate()
            #d_s_w = neuron.get_tot_psp()/neuron.get_spike_count_int()
            #d_s_w = neuron.get_spike_rate()

            tpsp_m_pre = tf.multiply(layer.kernel, tpsp)
            tpsp_m_pre = tf.reduce_sum(tpsp_m_pre,1)
            tpsp_m_pre = tf.divide(tpsp_m_pre,neuron_post.spike_counter_int*conf.n_init_vth)
            tpsp_m_pre = 1.0 + tpsp_m_pre.numpy()

            d_s_w = tpsp

            del_in_e = tf.multiply(del_in, tpsp_m_pre)
            del_in_e = tf.tile(tf.expand_dims(del_in_e,1), [1,tf.shape(layer.kernel)[0],1])
            #d_s_w_e = tf.tile(tf.expand_dims(d_s_w,2), [1,1,tf.shape(layer.kernel)[1]])
            d_s_w_e = d_s_w

            grad_b = tf.reduce_sum(del_in,0)
            grad = tf.reduce_sum(tf.multiply(del_in_e, d_s_w_e),0)

            grads_and_vars.append((grad, layer.kernel))
            grads_and_vars.append((grad_b, layer.bias))

        optimizer.apply_gradients(grads_and_vars)

    return avg_loss.result(), 100*accuracy.result()


def train_snn_one_epoch_mnist(model, optimizer, dataset, conf):
    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')


    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')


    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
        grads_and_vars = []

        with tf.GradientTape(persistent=True) as tape:
            predictions_times = model(images, f_training=True)

            if predictions_times.shape[1] != labels.numpy().shape[0]:
                predictions_times_trimmed = predictions_times[:,0:labels.numpy().shape[0],:]
                d = predictions_times.shape[1] - labels.numpy().shape[0]

                labels_expandded = tf.concat([labels, tf.zeros((d,)+(labels.numpy().shape[1],))],axis=0)
                f_exp = True
            else:
                predictions_times_trimmed = predictions_times
                labels_expandded = labels
                f_exp = False


            predictions_trimmed = predictions_times_trimmed[-1]

            #loss_value = loss(predictions_trimmed, labels)
            #loss_value, delta = loss_cross_entropy(predictions_trimmed, labels)
            loss_value, delta = loss_mse(predictions_trimmed, labels)

            avg_loss(loss_value)

            #accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
            accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


            #predictions = predictions_times[-1]

            #del_last = loss_value - labels_expandded

            #softmax = tf.nn.softmax(predictions_trimmed)

        #del_last = tf.cast(softmax,tf.float32) - tf.cast(labels,tf.float32)
        #del_last = tf.cast(predictions_trimmed,tf.float32) - tf.cast(labels,tf.float32)

        del_last = delta

        if f_exp :
            del_last = tf.concat([del_last, tf.zeros((d,)+(del_last.numpy().shape[1],))],axis=0)

        idx_layer_last = 1

        for idx_layer in range(idx_layer_last,-1,-1):

            layer = model.list_l[idx_layer]
            act = model.list_a[idx_layer]
            neuron = model.list_n[idx_layer]
            neuron_post = model.list_n[idx_layer+1]

            # backprop
            if idx_layer == idx_layer_last:
                del_in = del_last
            else:
                del_in = del_out

            del_out = bp_Dense(layer, del_in)
            del_out = bp_relu(act, del_out)
            #del_out = tf.multiply(del_out,1.0/conf.n_init_vth)

            # weight update
            #d_s_w = act
            #d_s_w = neuron.get_spike_rate()
            #d_s_w = neuron.get_tot_psp()/neuron.get_spike_count_int()
            d_s_w = neuron.get_spike_rate()

            del_in_e = del_in
            del_in_e = tf.tile(tf.expand_dims(del_in_e,1), [1,tf.shape(layer.kernel)[0],1])
            d_s_w_e = tf.tile(tf.expand_dims(d_s_w,2), [1,1,tf.shape(layer.kernel)[1]])

            grad_b = tf.reduce_sum(del_in,0)
            grad = tf.reduce_sum(tf.multiply(del_in_e, d_s_w_e),0)

            grads_and_vars.append((grad, layer.kernel))
            grads_and_vars.append((grad_b, layer.bias))

        optimizer.apply_gradients(grads_and_vars)

    return avg_loss.result(), 100*accuracy.result()


# backprop
def bp_Dense(layer, del_in):
    return tf.matmul(del_in, layer.kernel, transpose_b=True)

def bp_relu(act, del_in):
    return tf.multiply(del_in,act)

def grad_Dense(layer, delta, d_local):
    delta_e = tf.tile(tf.expand_dims(delta,1), [1,tf.shape(layer.kernel)[0],1])
    d_local_e = tf.tile(tf.expand_dims(d_local,2), [1,1,tf.shape(layer.kernel)[1]])

    grad_b = tf.reduce_sum(delta,0)
    grad = tf.reduce_sum(tf.multiply(delta_e, d_local_e),0)

    return grad, grad_b


# temporal coding - trainting time_const, time_delay
def train_time_const_delay_tmeporal_coding(model, dataset, conf):
    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')


    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')

    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):

        #with tf.GradientTape(persistent=True) as tape:
        predictions_times = model(images, f_training=True)

        if predictions_times.shape[1] != labels.numpy().shape[0]:
            predictions_times_trimmed = predictions_times[:,0:labels.numpy().shape[0],:]
            d = predictions_times.shape[1] - labels.numpy().shape[0]

        else:
            predictions_times_trimmed = predictions_times


        predictions_trimmed = predictions_times_trimmed[-1]

        loss_value = loss(predictions_trimmed, labels)

        avg_loss(loss_value)

        #accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
        accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


        #predictions = predictions_times[-1]

        softmax = tf.nn.softmax(predictions_trimmed)



    return avg_loss.result(), 100*accuracy.result()
