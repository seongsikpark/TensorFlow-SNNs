from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.framework import ops

from collections import OrderedDict


import functools
import itertools

import backprop as bp_sspark

#
import lib_snn



#
def loss(predictions, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions,labels=labels))


def loss_cross_entoropy(predictions, labels):
    loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions,labels=labels))
    delta = tf.softmax(predictions) - labels

    return loss_value, delta



def loss_mse(predictions, labels):
    loss_value = tf.losses.mean_squared_error(labels, predictions)
    #delta = predictions - labels

    #return loss_value, delta
    return loss_value



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
    avg_loss = tfe.metrics.Mean('loss')
    accuracy = tfe.metrics.Accuracy('accuracy')

    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
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





def train_one_epoch_ttfs(model, optimizer, dataset):
    #print("train_snn.py - train_one_epoch_ttfs > not yet implemented")
    #assert(False)

    tf.train.get_or_create_global_step()
    avg_loss = tfe.metrics.Mean('loss_total')
    avg_loss_pred = tfe.metrics.Mean('loss_pred')
    avg_loss_enc_st = tfe.metrics.Mean('loss_enc_st')

    accuracy = tfe.metrics.Accuracy('accuracy')

    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):

        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, f_training=True)
            #predictions = predictions[-1]

            # labels_st: labels (first spike time) for TTFS coding
            #labels_st = lib_snn.ground_truth_in_spike_time(labels,1,model.list_neuron['in'].init_first_spike_time)
            labels_st = lib_snn.ground_truth_in_spike_time(labels,1,model.init_first_spike_time)

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
            loss_name=['prediction','enc_st','max_enc_st','min_enc_st']

            #
            # loss - prediction
            loss_list['prediction'] = loss(predictions, labels)

            #
            # loss - encoded spike time
            #print(model.t_conv1)
            #print(tf.round(model.t_conv1))

            #enc_st = model.t_conv1

            loss_tmp=0
            #for _, (name, enc_st) in enumerate(model.list_st.items()):
            #for name in model.layer_name:

                #enc_st=model.list_st[name]


                #print(name)
                #print(enc_st)

                #enc_st = model.list_v[name]

            enc_st = model.list_st['conv1']
            round_enc_st = tf.constant(tf.round(enc_st),shape=enc_st.shape)
            loss_tmp += loss_mse(enc_st,round_enc_st)

            enc_st = model.list_st['conv2']
            round_enc_st = tf.constant(tf.round(enc_st),shape=enc_st.shape)
            loss_tmp += loss_mse(enc_st,round_enc_st)

            enc_st = model.list_st['fc1']
            round_enc_st = tf.constant(tf.round(enc_st),shape=enc_st.shape)
            loss_tmp += loss_mse(enc_st,round_enc_st)

            loss_list['enc_st'] = loss_tmp
            #loss_list['enc_st'] += loss_mse(enc_st,round_enc_st)

            #
            # loss - maximum encoded spike time

            loss_tmp=0
            #for _, (name, enc_st) in enumerate(model.list_st.items()):
                #v = model.list_v[name]
            #min_val = model.temporal_encoding(model.v_conv1)
            v_conv1=model.list_v['conv1']
            t_conv1=model.list_st['conv1']


            max_enc_st = tf.constant(model.conf.time_window,dtype=tf.float32,shape=v_conv1.shape)
            non_bounded_enc_st = model.temporal_encoding_kernel(v_conv1)
            max_enc_target = tf.where(max_enc_st < non_bounded_enc_st,\
                                      max_enc_st,\
                                      t_conv1)
            loss_list['max_enc_st'] = loss_mse(non_bounded_enc_st, max_enc_target)

            #
            # loss - minimum encoded spike time
            #enc_st = model.t_conv1
            enc_st = model.list_st['conv1']
            v_conv1=model.list_v['conv1']

            min_target_value = model.conf.tc
            min_enc_st_target = tf.constant(min_target_value,dtype=tf.float32,shape=v_conv1.shape)
            min_enc_target = tf.where(min_enc_st_target < enc_st,\
                                      min_enc_st_target,\
                                      enc_st)
            loss_list['min_enc_st'] = loss_mse(enc_st, min_enc_target)

            #
            #
            loss_weight['prediction']=1.0
            loss_weight['enc_st']=0.05
            loss_weight['max_enc_st']=0.1
            #loss_weight['min_enc_st']=0.1
            loss_weight['min_enc_st']=0.0

            #
            #loss_total = loss_pred
            #loss_total = loss_pred + loss_enc_st
            #loss_total = loss_pred + loss_enc_st + loss_max_enc_st
            #loss_total = loss_pred + loss_enc_st + loss_max_enc_st + loss_min_enc_st

            loss_total=0
            for l_name in loss_name:
                loss_total = loss_total + loss_weight[l_name]*loss_list[l_name]

            #
            avg_loss(loss_total)
            avg_loss_pred(loss_list['prediction'])
            #avg_loss_enc_st(loss_enc_st)
            avg_loss_enc_st=loss_list['enc_st'].numpy()
            avg_loss_max_enc_st=loss_list['max_enc_st'].numpy()
            avg_loss_min_enc_st=loss_list['min_enc_st'].numpy()

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
        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss_total, trainable_vars)
        #print(trainable_vars)

        grads_and_vars = zip(grads,trainable_vars)

        optimizer.apply_gradients(grads_and_vars)


        #
        # make encoded spike time integer
        #
        #with tf.GradientTape(persistent=True) as tape:
        #    predictions = model(images, f_training=True)



    #return avg_loss.result(), 100*accuracy.result()
    return avg_loss.result(), avg_loss_pred.result(),avg_loss_enc_st,\
           avg_loss_max_enc_st,avg_loss_min_enc_st,100*accuracy.result()



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
    avg_loss = tfe.metrics.Mean('loss')
    accuracy = tfe.metrics.Accuracy('accuracy')

    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
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
        diff = grad_b_t - grad_t
        print('max: %.3f, sum: %.3f'%(tf.reduce_max(diff),tf.reduce_sum(diff)))



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
    avg_loss = tfe.metrics.Mean('loss')
    accuracy = tfe.metrics.Accuracy('accuracy')

    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
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
    avg_loss = tfe.metrics.Mean('loss')
    accuracy = tfe.metrics.Accuracy('accuracy')

    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
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
    avg_loss = tfe.metrics.Mean('loss')
    accuracy = tfe.metrics.Accuracy('accuracy')

    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
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
    avg_loss = tfe.metrics.Mean('loss')
    accuracy = tfe.metrics.Accuracy('accuracy')

    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):

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
