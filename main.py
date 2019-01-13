from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import sys
sys.path.insert(0,'./')
#sys.path.insert(0,'/home/sspark/Projects/05_SNN/')

# sspark
import models
import model_cnn_mnist

#en_gpu=False
en_gpu=True

gpu_number=0

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)

if en_gpu==False:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf

import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.framework import ops


# resnet
import resnet

# imagenet
from datasets import imagenet  # from tensornets
from data import ImageNetDataset


# cifar-100

import shutil

#
import pprint
import matplotlib.pyplot as plt

from tqdm import tqdm

import math

import csv
import collections
import re

#

#
np.set_printoptions(threshold=np.inf, linewidth=1000, precision=4)

#
pp = pprint.PrettyPrinter().pprint

#
flags = tf.app.flags
tf.app.flags.DEFINE_string('date','','date')

tf.app.flags.DEFINE_integer('epoch', 300, 'Number os epochs')
tf.app.flags.DEFINE_string('gpu_fraction', '1/3', 'define the gpu fraction used')
tf.app.flags.DEFINE_integer('batch_size', 100, '')
tf.app.flags.DEFINE_string('activation', 'ReLU', '')
tf.app.flags.DEFINE_string('optim_type', 'adam', '[exp_decay, adam]')

#tf.app.flags.DEFINE_string('nn_mode', 'SNN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')
tf.app.flags.DEFINE_string('nn_mode', 'ANN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')

tf.app.flags.DEFINE_string('output_dir', './tensorboard', 'Directory to write TensorBoard summaries')
tf.app.flags.DEFINE_string('checkpoint_dir', './ckpt_tmp', 'Directory to save checkpoints')
tf.app.flags.DEFINE_string('checkpoint_load_dir', './ckpt_tmp', 'Directory to load checkpoints')
tf.app.flags.DEFINE_bool('en_load_model', False, 'Enable to load model')

#
tf.app.flags.DEFINE_boolean('en_train', False, 'enable training')

#
tf.app.flags.DEFINE_float('n_init_vth', 0.7, 'initial value of vth')
tf.app.flags.DEFINE_float('n_in_init_vth', 0.7, 'initial value of vth of n_in')
tf.app.flags.DEFINE_float('n_init_vinit', 0.0, 'initial value of vinit')
tf.app.flags.DEFINE_float('n_init_vrest', 0.0, 'initial value of vrest')

# exponetial decay
'''
tf.app.flags.DEFINE_float('init_lr', 0.1, '')
tf.app.flags.DEFINE_float('decay_factor', 0.1, '')
tf.app.flags.DEFINE_integer('num_epoch_per_decay', 350, '')
'''
# adam optimizer
#tf.app.flags.DEFINE_float('init_lr', 1e-5, '')
# SGD
tf.app.flags.DEFINE_float('lr', 0.1, 'learning rate')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum')
# ADAM
#tf.app.flags.DEFINE_float('lr', 0.0001, '')

# l2 norm
tf.app.flags.DEFINE_float('lamb',0.0001, 'lambda')

tf.app.flags.DEFINE_float('lr_decay', 2.0, '')
tf.app.flags.DEFINE_integer('lr_decay_step', 50, '')

tf.app.flags.DEFINE_integer('time_step', 10, 'time steps per sample in SNN')


tf.app.flags.DEFINE_integer('num_test_dataset', 10000, 'number of test datset')
tf.app.flags.DEFINE_integer('size_test_batch', 1, 'size of test batch') # not used now

tf.app.flags.DEFINE_string('pooling', 'max', 'max or avg, only for CNN')

tf.app.flags.DEFINE_integer('save_interval', 10, 'save interval of model')

tf.app.flags.DEFINE_bool('en_remove_output_dir', False, 'enable removing output dir')


#
tf.app.flags.DEFINE_boolean('use_bias', True, 'use bias')
tf.app.flags.DEFINE_string('regularizer', 'L2', 'L2 or L1 regularizer')


tf.app.flags.DEFINE_string('model_name', 'no_name', 'model name')

tf.app.flags.DEFINE_string('n_type', 'LIF', 'LIF or IF: neuron type')

#
tf.app.flags.DEFINE_string('dataset', 'MNIST', 'dataset')
tf.app.flags.DEFINE_string('ann_model', 'MLP', 'neural network model')

#
tf.app.flags.DEFINE_boolean('verbose',True, 'verbose mode')
tf.app.flags.DEFINE_boolean('verbose_visual',True, 'verbose visual mode')

#
tf.app.flags.DEFINE_integer('time_step_save_interval',10,'snn test save interval')

#
tf.app.flags.DEFINE_bool('f_fused_bn',False,'f_fused_bn')

#
tf.app.flags.DEFINE_bool('f_stat_train_mode',False,'f_stat_train_mode')
tf.app.flags.DEFINE_bool('f_real_value_input_snn',False,'f_real_value_input_snn')
tf.app.flags.DEFINE_bool('f_vth_conp',False,'f_vth_conp')
tf.app.flags.DEFINE_bool('f_spike_max_pool',False,'f_spike_max_pool')
tf.app.flags.DEFINE_bool('f_w_norm_data',False,'f_w_norm_data')
tf.app.flags.DEFINE_bool('f_ws',False,'wieghted synapse')
tf.app.flags.DEFINE_integer('p_ws',8,'period of wieghted synapse')

tf.app.flags.DEFINE_integer('num_class',0,'number_of_class (do not touch)')

tf.app.flags.DEFINE_string('input_spike_mode','POISSON','input spike mode - POISSON, WEIGHTED_SPIKE, ROPOSED')
tf.app.flags.DEFINE_string('neural_coding','RATE','neural coding - RATE, WEIGHTED_SPIKE, PROPOSED')

tf.app.flags.DEFINE_bool('f_positive_vmem',True,'positive vmem')

tf.app.flags.DEFINE_bool('f_isi',False,'isi stat')
tf.app.flags.DEFINE_bool('f_refractory',False,'refractory mode')

tf.app.flags.DEFINE_bool('f_comp_act',False,'compare activation')
tf.app.flags.DEFINE_bool('f_entropy',False,'entropy test')
tf.app.flags.DEFINE_bool('f_write_stat',False,'write stat')

# data.py - imagenet data
tf.app.flags.DEFINE_string('data_path_imagenet', './imagenet', 'data path imagenet')
tf.app.flags.DEFINE_integer('k_pathces', 5, 'patches for test (random crop)')
tf.app.flags.DEFINE_integer('input_size', 28, 'input image width / height')


#
tf.app.flags.DEFINE_string('path_stat','./stat/', 'path stat')
tf.app.flags.DEFINE_string('prefix_stat','act_n_train', 'prefix of stat file name')




# tmp
cifar_10_crop_size=36

#
conf = flags.FLAGS

data_path_imagenet='/home/sspark/Datasets/ILSVRC2012'
conf.data_path_imagenet = data_path_imagenet

# ImageNet datapath

#def imagenet_load(data_dir, resize_wh, crop_wh, crops):
def imagenet_load():
    data_dir = data_path_imagenet
    resize_wh = 256
    ##resize_wh = 224
    #crop_wh = 224
    crop_wh = conf.input_size
    crops = 1
    return imagenet.load(
        data_dir, 'val', batch_size=conf.batch_size,
        resize_wh=resize_wh,
        crop_locs=10 if crops == 10 else 4,
        crop_wh=crop_wh,
        total_num= conf.num_test_dataset
    )


#
def loss(predictions, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions,labels=labels))

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
def train_one_epoch(model, optimizer, dataset):
    tf.train.get_or_create_global_step()
    avg_loss = tfe.metrics.Mean('loss')
    accuracy = tfe.metrics.Accuracy('accuracy')

    def model_loss(labels, images):
        prediction = model(images, f_training=True)
        loss_value = loss(prediction, labels)
        avg_loss(loss_value)
        accuracy(tf.argmax(prediction,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
        #tf.contrib.summary.scalar('loss', loss_value)
        #tf.contrib.summary.scalar('accuracy', compute_accuracy(prediction, labels))
        return loss_value

    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
        #print(batch)
        #print(images)
        #print(labels)
        #with tf.contrib.summary.record_summaries_every_n_global_steps(10):
        batch_model_loss = functools.partial(model_loss, labels, images)
        optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())

    #print('Train set: Accuracy: %4f%%\n'%(100*accuracy.result()))
    return avg_loss.result(), 100*accuracy.result()

def plot_dist_activation_vgg16(model):
        plt.subplot2grid((6,3),(0,0))
        plt.hist(model.stat_a_conv1)
        plt.subplot2grid((6,3),(0,1))
        plt.hist(model.stat_a_conv1_1)
        plt.subplot2grid((6,3),(1,0))
        plt.hist(model.stat_a_conv2)
        plt.subplot2grid((6,3),(1,1))
        plt.hist(model.stat_a_conv2_1)
        plt.subplot2grid((6,3),(2,0))
        plt.hist(model.stat_a_conv3)
        plt.subplot2grid((6,3),(2,1))
        plt.hist(model.stat_a_conv3_1)
        plt.subplot2grid((6,3),(2,2))
        plt.hist(model.stat_a_conv3_1)
        plt.subplot2grid((6,3),(3,0))
        plt.hist(model.stat_a_conv4)
        plt.subplot2grid((6,3),(3,1))
        plt.hist(model.stat_a_conv4_1)
        plt.subplot2grid((6,3),(3,2))
        plt.hist(model.stat_a_conv4_2)
        plt.subplot2grid((6,3),(4,0))
        plt.hist(model.stat_a_conv5)
        plt.subplot2grid((6,3),(4,1))
        plt.hist(model.stat_a_conv5_1)
        plt.subplot2grid((6,3),(4,2))
        plt.hist(model.stat_a_conv5_2)
        plt.subplot2grid((6,3),(5,0))
        plt.hist(model.stat_a_fc1)
        plt.subplot2grid((6,3),(5,1))
        plt.hist(model.stat_a_fc2)
        plt.subplot2grid((6,3),(5,2))
        plt.hist(model.stat_a_fc3)
        plt.show()


# distribution of activation - layer-wise
def save_dist_activation_vgg16(model):
    layer_name=[
        #model.stat_a_conv1,
        #model.stat_a_conv1_1
        #model.stat_a_conv2,
        #model.stat_a_conv2_1,
        #model.stat_a_conv3,
        #model.stat_a_conv3_1,
        #model.stat_a_conv3_2,
        #model.stat_a_conv4,
        #model.stat_a_conv4_1,
        #model.stat_a_conv4_2,
        #model.stat_a_conv5,
        #model.stat_a_conv5_1,
        #model.stat_a_conv5_2,
        #model.stat_a_fc1,
        #model.stat_a_fc2,
        model.stat_a_fc3
    ]

    f = open('./stat/dist_act_neuron_trainset_fc_'+conf.model_name,'wb')
    wr = csv.writer(f)
    wr.writerow(['max','min','mean','std','99.9','99','98'])

    for _, s_layer in enumerate(layer_name):
        wr.writerow([np.max(s_layer),np.min(s_layer),np.mean(s_layer),np.std(s_layer),np.nanpercentile(s_layer,99.9),np.nanpercentile(s_layer,99),np.nanpercentile(s_layer,98)])
    f.close()


# distribution of activation - neuron-wise
def save_dist_activation_neuron_vgg16(model):
    # delete later
    layer_name=[
    #    'conv1',
    #    'conv1_1',
    #    'conv2',
    #    'conv2_1'
    #    'conv3',
    #    'conv3_1',
    #    'conv3_2',
    #    'conv4',
    #    'conv4_1',
    #    'conv4_2'
        'conv5',
        'conv5_1',
        'conv5_2',
        'fc1',
        'fc2',
        'fc3'
    ]

    path_stat='/home/sspark/Projects/05_SNN/stat/'
    #f_name_stat='act_n_train_after_w_norm_max_999'
    f_name_stat='act_n_train'
    #stat_conf=['max_999']
    #stat_conf=['max','mean','max_999','max_99','max_98']
    stat_conf=['max_95','max_90']
    #stat_conf=['max','mean','min','max_75','max_25']
    f_stat=collections.OrderedDict()
    wr_stat=collections.OrderedDict()

    for idx_l, l in enumerate(layer_name):
        for idx_c, c in enumerate(stat_conf):
            key=l+'_'+c
            f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+conf.model_name,'w')
            wr_stat[key]=csv.writer(f_stat[key])

    #f = open('./stat/dist_act_neuron_trainset_test_'+stat_conf+conf.model_name,'wb')

    #wr = csv.writer(f,quoting=csv.QUOTE_NONE,escapechar='\n')
    #wr = csv.writer(f)
    #wr.writerow(['max','min','mean','std','99.9','99','98'])
    #wr.writerow(['max','min','mean','99.9','99','98'])

    for idx_l, l in enumerate(layer_name):
        s_layer=model.dict_stat_w[l]
        #print(np.shape(s_layer))
        #print(np.shape(s_layer)[1:])
        #shape_n=np.shape(s_layer)[1:]

        # before
        #max=np.max(s_layer,axis=0).flatten()
        #mean=np.mean(s_layer,axis=0).flatten()
        #max_999=np.nanpercentile(s_layer,99.9,axis=0).flatten()
        #max_99=np.nanpercentile(s_layer,99,axis=0).flatten()
        #max_98=np.nanpercentile(s_layer,98,axis=0).flatten()
        max_95=np.nanpercentile(s_layer,95,axis=0).flatten()
        max_90=np.nanpercentile(s_layer,90,axis=0).flatten()
        #wr_stat[l+'_max'].writerow(max)
        #wr_stat[l+'_mean'].writerow(mean)
        #wr_stat[l+'_max_999'].writerow(max_999)
        #wr_stat[l+'_max_99'].writerow(max_99)
        #wr_stat[l+'_max_98'].writerow(max_98)
        wr_stat[l+'_max_95'].writerow(max_95)
        wr_stat[l+'_max_90'].writerow(max_90)


        #min=np.min(s_layer,axis=0).flatten()
        #max=np.max(s_layer).flatten()
        #print(max)
        #print(np.nanpercentile(s_layer,99.9))
        #print(np.nanpercentile(s_layer,99))
        #print(np.nanpercentile(s_layer,98))
        #print(np.nanpercentile(s_layer,95))
        #print(np.nanpercentile(s_layer,90))
        #print(np.mean(s_layer))
        #plt.hist(s_layer.flatten(), log=True, bins=int(max*2))
        #plt.show()

        # for after w norm stat
        #max=np.max(s_layer,axis=0).flatten()
        #mean=np.mean(s_layer,axis=0).flatten()
        #min=np.mean(s_layer,axis=0).flatten()
        #max_25=np.nanpercentile(s_layer,25,axis=0).flatten()
        #max_75=np.nanpercentile(s_layer,75,axis=0).flatten()
        #hist=np.histogram(s_layer)

        #wr_stat[l+'_max'].writerow(max)
        #wr_stat[l+'_mean'].writerow(mean)
        #wr_stat[l+'_min'].writerow(min)
        #wr_stat[l+'_max_75'].writerow(max_25)
        #wr_stat[l+'_max_25'].writerow(max_75)
        #wr_stat[l+'_hist'].writerow()


        #print(np.shape(max))

        #np.savetxt('a',max)

        #wr.writerow([np.max(s_layer,axis=0),np.min(s_layer,axis=0),np.mean(s_layer,axis=0),np.std(s_layer,axis=0),np.nanpercentile(s_layer,99.9,axis=0),np.nanpercentile(s_layer,99,axis=0),np.nanpercentile(s_layer,98,axis=0)])
        #wr.writerow([np.max(s_layer,axis=0),np.min(s_layer,axis=0),np.mean(s_layer,axis=0),np.nanpercentile(s_layer,99.9,axis=0),np.nanpercentile(s_layer,99,axis=0),np.nanpercentile(s_layer,98,axis=0)])
        #wr.writerow([max,min,mean,max_999,max_99,max_98])

    for idx_l, l in enumerate(layer_name):
        for idx_c, c in enumerate(stat_conf):
            key=l+'_'+c
            f_stat[key].close()

#
def test(model, dataset):
    avg_loss = tfe.metrics.Mean('loss')

    if conf.nn_mode=='SNN':
        #accuracy_times = np.array((2,))
        accuracy_times = []
        accuracy_result = []

        if conf.dataset == 'ImageNet':
            accuracy_times_top5 = []
            accuracy_result_top5 = []

        accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
        accuracy_time_point.append(conf.time_step)
        argmax_axis_predictions=1


        num_accuracy_time_point=len(accuracy_time_point)

        print('accuracy_time_point')
        print(accuracy_time_point)

        for i in range(num_accuracy_time_point):
            accuracy_times.append(tfe.metrics.Accuracy('accuracy'))

            if conf.dataset == 'ImageNet':
                accuracy_times_top5.append(tfe.metrics.Mean('accuracy_top5'))


        num_batch=int(math.ceil(float(conf.num_test_dataset)/float(conf.batch_size)))

        pbar = tqdm(range(1,num_batch+1),ncols=80)
        #pbar.set_description("test batch")
        pbar.set_description("batch")

        for (idx_batch, (images, labels_one_hot)) in enumerate(tfe.Iterator(dataset)):
            #print('idx: %d'%(idx_batch))
            #print('image')
            #print(images.shape)
            #print(images)
            #print('label')
            #print(labels)
            labels = tf.argmax(labels_one_hot,axis=1,output_type=tf.int32)

            if idx_batch!=-1:
                predictions_times = model(images, labels=labels, f_training=False)
                tf.reshape(predictions_times,(-1,)+labels.numpy().shape)

                for i in range(num_accuracy_time_point):
                    predictions=predictions_times[i]
                    accuracy = accuracy_times[i]
                    accuracy(tf.argmax(predictions,axis=argmax_axis_predictions,output_type=tf.int32), labels)


                    if conf.dataset == 'ImageNet':
                        accuracy_top5 = accuracy_times_top5[i]
                        with tf.device('/cpu:0'):
                            accuracy_top5(tf.cast(tf.nn.in_top_k(predictions,labels,5),tf.int32))

                predictions = predictions_times[-1]
                avg_loss(loss(predictions,labels_one_hot))

                if conf.verbose:
                    print(predictions-labels*conf.time_step)

            pbar.update()

        for i in range(num_accuracy_time_point):
            accuracy_result.append(accuracy_times[i].result().numpy())
            accuracy_result_top5.append(accuracy_times_top5[i].result().numpy())

        print('')
        print('accruacy')
        print(accuracy_result)
        print(accuracy_result_top5)

        plt.plot(accuracy_time_point,accuracy_result)
        plt.show()

        #print('Test set: Average loss: %.4f, Accuracy: %4f%%\n'%(avg_loss.result(), 100*accuracy.result()))
        #with tf.contrib.summary.always_record_summaries():
            #tf.contrib.summary.scalar('loss', avg_loss.result())
            #tf.contrib.summary.scalar('accuracy', accuracy.result())
            #tf.contrib.summary.scalar('w_conv1', model.variables)
        ret_accu = 100*accuracy_result[-1]
        ret_accu_top5 = 100*accuracy_result_top5[-1]


        print('total spike count - int')
        print(model.total_spike_count_int)
        print('total spike count - float')
        print(model.total_spike_count)
        #print('total residual vmem')
        #print(model.total_residual_vmem)

        if conf.f_comp_act:
            print('compare act')
            print(model.total_comp_act)

        if conf.f_isi:
            print('total isi')
            print(model.total_isi)

            print('spike amplitude')
            print(model.total_spike_amp)

            plt.subplot(211)
            plt.bar(np.arange(conf.time_step)[1:],model.total_isi[1:])
            plt.subplot(212)
            plt.bar(np.arange(model.spike_amp_bin[1:-1].size),model.total_spike_amp[1:],tick_label=model.spike_amp_bin[1:-1])

            plt.show()

        if conf.f_entropy:
            print('total_entropy')
            print(model.total_entropy)

        print('f write date: '+conf.date)

        #plt.plot(accuracy_time_point,model.total_spike_count)
        #plt.show()

    else:
        accuracy=tfe.metrics.Accuracy('accuracy')

        if conf.dataset == 'ImageNet':
            accuracy_top5=tfe.metrics.Mean('accuracy_top5')

        num_batch=int(math.ceil(float(conf.num_test_dataset)/float(conf.batch_size)))

        pbar = tqdm(range(1,num_batch+1),ncols=80)
        pbar.set_description("batch")

        for (idx_batch, (images, labels_one_hot)) in enumerate(tfe.Iterator(dataset)):
        #for idx_batch in range(0,2):
            #images, labels = tfe.Iterator(dataset).next()
            #print('idx: %d'%(idx_batch))
            #print('image')
            #print(images.shape)
            #print(images[0,0,0:10])
            #print('label')
            #print(labels)
            #print(tf.argmax(labels,axis=1))

            labels = tf.argmax(labels_one_hot,axis=1,output_type=tf.int32)

            if idx_batch!=-1:
                #model=tfe.defun(model)
                predictions = model(images, f_training=False)

                #print(predictions.shape)
                #print(str(tf.argmax(predictions,axis=1))+' : '+str(tf.argmax(labels,axis=1)))
                accuracy(tf.argmax(predictions,axis=1,output_type=tf.int32), labels)

                if conf.dataset == 'ImageNet':
                    with tf.device('/cpu:0'):
                        accuracy_top5(tf.cast(tf.nn.in_top_k(predictions,labels,5),tf.int32))
                avg_loss(loss(predictions,labels_one_hot))

            pbar.update()

        ret_accu = 100*accuracy.result()
        ret_accu_top5 = 100*accuracy_top5.result()

        #plt.hist(model.stat_a_fc3)
        #plt.show()
        #plot_dist_activation_vgg16(model)
        #save_dist_activation_vgg16(model)

        #print(model.stat_a_fc3)
        #print(model.stat_a_fc3.shape)
        #print(tf.reduce_min(model.stat_a_fc3))
        #print(tf.reduce_max(model.stat_a_fc3,axis=0))
        #print(np.max(model.stat_a_fc3,axis=0))

        # should include the class later
        #if conf.f_write_stat:
        #    save_dist_activation_neuron_vgg16(model)

        if conf.f_write_stat:
            if conf.ann_model=='ResNet50' and conf.dataset=='ImageNet':
                model.save_activation()


        #print(tf.reduce_max(model.stat_a_fc2))
        #print(tf.reduce_max(model.stat_a_fc3))


    return avg_loss.result(), ret_accu, ret_accu_top5



def preprocess_cifar10_train(img, label):

    img_p = tf.image.convert_image_dtype(img,tf.float32)
    #img_p = tf.image.resize_image_with_crop_or_pad(img_p,cifar_10_crop_size,cifar_10_crop_size)

    img_p = tf.image.resize_image_with_crop_or_pad(img_p,36,36)
    img_p = tf.random_crop(img_p,[32,32,3])


    #img_p = tf.cast(img,tf.float32)
    #img_p = tf.image.convert_image_dtype(img,tf.float32)
    #img_p = tf.image.rgb_to_hsv(img_p)

    #img_p = tf.image.resize_image_with_crop_or_pad(img_p,cifar_10_crop_size,cifar_10_crop_size)
    img_p = tf.image.random_flip_left_right(img_p)

    #
    #img_p = tf.image.random_brightness(img_p,max_delta=63)
    #img_p = tf.image.random_brightness(img_p,max_delta=83)
    #img_p = tf.image.random_brightness(img_p,max_delta=103)

    #
    #img_p = tf.image.random_contrast(img_p,lower=0.2,upper=1.8)
    #img_p = tf.image.random_contrast(img_p,lower=0.1,upper=2.0)

    #
    #img_p = tf.image.per_image_standardization(img_p)
    #img_p.set_shape([cifar_10_crop_size,cifar_10_crop_size,3])
    img_p.set_shape([32,32,3])

    return img_p, label

def preprocess_cifar10_test(img, label):

    img_p = tf.image.convert_image_dtype(img,tf.float32)
    #img_p = tf.image.resize_image_with_crop_or_pad(img_p,cifar_10_crop_size,cifar_10_crop_size)

    #img_p = tf.cast(img,tf.float32)
    #img_p = tf.image.convert_image_dtype(img,tf.float32)
    #img_p = tf.image.rgb_to_hsv(img_p)
    #img_p = tf.image.resize_image_with_crop_or_pad(img_p,cifar_10_crop_size,cifar_10_crop_size)

    # test 180711
    #img_p = tf.image.per_image_standardization(img_p)

    #img_p.set_shape([cifar_10_crop_size,cifar_10_crop_size,3])
    img_p.set_shape([32,32,3])

    return img_p, label


# convert class labels from scalars to one-hot vectors
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arrange(num_labels) * num_classes

def load_data():
    print("load MNIST dataset")
    #data = input_data.read_data_sets(data_dir, one_hot=True)
    data = input_data.read_data_sets("MNIST-data/", one_hot=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((data.train.images,data.train.labels))
    train_dataset = train_dataset.shuffle(60000).batch(conf.batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((data.validation.images,data.validation.labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((data.test.images[:conf.num_test_dataset], data.test.labels[:conf.num_test_dataset]))


    #print(data.test.images.shape)

    print(train_dataset)

    return train_dataset, val_dataset, test_dataset

def load_data_cifar10():
    print("load CIFAR10 dataset")
    (img_train,label_train), (img_test, label_test) = tf.contrib.keras.datasets.cifar10.load_data()

    #print(type(img_train))
    img_train = img_train.astype(float)
    img_test = img_test.astype(float)

    #print(img_train[0])
    print(tf.reduce_min(img_train))
    print(tf.reduce_min(img_test))

    img_train = img_train / 255.0
    img_test = img_test / 255.0

    #print(type(img_train))

    #print(img_train[0])

    #img_train[:,:,:,0] = (img_train[:,:,:,0]-np.mean(img_train[:,:,:,0]))/np.std(img_train[:,:,:,0])
    #img_train[:,:,:,1] = (img_train[:,:,:,1]-np.mean(img_train[:,:,:,1]))/np.std(img_train[:,:,:,1])
    #img_train[:,:,:,2] = (img_train[:,:,:,2]-np.mean(img_train[:,:,:,2]))/np.std(img_train[:,:,:,2])

    #img_test[:,:,:,0] = (img_test[:,:,:,0]-np.mean(img_test[:,:,:,0]))/np.std(img_test[:,:,:,0])
    #img_test[:,:,:,1] = (img_test[:,:,:,1]-np.mean(img_test[:,:,:,1]))/np.std(img_test[:,:,:,1])
    #img_test[:,:,:,2] = (img_test[:,:,:,2]-np.mean(img_test[:,:,:,2]))/np.std(img_test[:,:,:,2])


    print(tf.reduce_min(img_train))
    print(tf.reduce_min(img_test))

    label_test=label_test[:conf.num_test_dataset]
    img_test=img_test[:conf.num_test_dataset,:,:,:]


    #print('################################ test start at idx1 - should modify ######################')
    #label_test=label_test[1:conf.num_test_dataset]
    #img_test=img_test[1:conf.num_test_dataset,:,:,:]

    #train_dataset = tf.data.Dataset.from_tensor_slices((img_train[:45000],tf.squeeze(tf.one_hot(label_train[:45000],10))))

    #val_dataset = tf.data.Dataset.from_tensor_slices((img_train[45000:],tf.squeeze(tf.one_hot(label_train[45000:],10))))
    #val_dataset = val_dataset.map(preprocess_cifar10_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,10))))

    #val_dataset = tf.data.Dataset.from_tensor_slices((img_test[:conf.num_test_dataset],tf.squeeze(tf.one_hot(label_test[:conf.num_test_dataset],10))))
    val_dataset = tf.data.Dataset.from_tensor_slices((img_test,tf.squeeze(tf.one_hot(label_test,10))))
    val_dataset = val_dataset.map(preprocess_cifar10_test)

    #test_dataset = tf.data.Dataset.from_tensor_slices((img_test[:conf.num_test_dataset],tf.squeeze(tf.one_hot(label_test[:conf.num_test_dataset],10))))
    test_dataset = tf.data.Dataset.from_tensor_slices((img_test,tf.squeeze(tf.one_hot(label_test,10))))
    test_dataset = test_dataset.map(preprocess_cifar10_test)


    # for stat of train dataset
    if conf.f_stat_train_mode:
        test_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,10))))
        test_dataset = test_dataset.map(preprocess_cifar10_test)



    print(train_dataset)


    return train_dataset, val_dataset, test_dataset

def load_data_cifar100():
    print("load CIFAR100 dataset")
    (img_train,label_train), (img_test, label_test) = tf.contrib.keras.datasets.cifar100.load_data()

    #print(type(img_train))
    img_train = img_train.astype(float)
    img_test = img_test.astype(float)

    #print(img_train[0])
    print(tf.reduce_min(img_train))
    print(tf.reduce_min(img_test))

    img_train = img_train / 255.0
    img_test = img_test / 255.0

    #print(type(img_train))

    #print(img_train[0])

    #img_train[:,:,:,0] = (img_train[:,:,:,0]-np.mean(img_train[:,:,:,0]))/np.std(img_train[:,:,:,0])
    #img_train[:,:,:,1] = (img_train[:,:,:,1]-np.mean(img_train[:,:,:,1]))/np.std(img_train[:,:,:,1])
    #img_train[:,:,:,2] = (img_train[:,:,:,2]-np.mean(img_train[:,:,:,2]))/np.std(img_train[:,:,:,2])

    #img_test[:,:,:,0] = (img_test[:,:,:,0]-np.mean(img_test[:,:,:,0]))/np.std(img_test[:,:,:,0])
    #img_test[:,:,:,1] = (img_test[:,:,:,1]-np.mean(img_test[:,:,:,1]))/np.std(img_test[:,:,:,1])
    #img_test[:,:,:,2] = (img_test[:,:,:,2]-np.mean(img_test[:,:,:,2]))/np.std(img_test[:,:,:,2])


    print(tf.reduce_min(img_train))
    print(tf.reduce_min(img_test))

    label_test=label_test[:conf.num_test_dataset]
    img_test=img_test[:conf.num_test_dataset,:,:,:]
    #train_dataset = tf.data.Dataset.from_tensor_slices((img_train[:45000],tf.squeeze(tf.one_hot(label_train[:45000],10))))

    #val_dataset = tf.data.Dataset.from_tensor_slices((img_train[45000:],tf.squeeze(tf.one_hot(label_train[45000:],10))))
    #val_dataset = val_dataset.map(preprocess_cifar10_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,100))))

    #val_dataset = tf.data.Dataset.from_tensor_slices((img_test[:conf.num_test_dataset],tf.squeeze(tf.one_hot(label_test[:conf.num_test_dataset],10))))

    val_dataset = tf.data.Dataset.from_tensor_slices((img_test,tf.squeeze(tf.one_hot(label_test,100))))
    val_dataset = val_dataset.map(preprocess_cifar10_test)

    #test_dataset = tf.data.Dataset.from_tensor_slices((img_test[:conf.num_test_dataset],tf.squeeze(tf.one_hot(label_test[:conf.num_test_dataset],10))))
    test_dataset = tf.data.Dataset.from_tensor_slices((img_test,tf.squeeze(tf.one_hot(label_test,100))))
    test_dataset = test_dataset.map(preprocess_cifar10_test)


    # for stat of train dataset
    if conf.f_stat_train_mode:
        test_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,100))))
        test_dataset = test_dataset.map(preprocess_cifar10_test)



    print(train_dataset)

    return train_dataset, val_dataset, test_dataset

def load_data_imagenet(conf):
    print('ImageNet load')

    #
    #train_dataset = ImageNetDataset(conf, 'train')
    #valid_dataset = ImageNetDataset(conf, 'val')
    #test_dataset = ImageNetDataset(conf, 'val')

    #print(test_dataset)



    # tensornets
    #data_imgs, data_labels = imagenet.get_files(conf.data_path_imagenet,'val', None)

    #dataset = tf.data.Dataset.from_tensor_slices((data_imgs, data_labels))
    #dataset = dataset.map(imagenet.load_and_crop, resize_wh=256, crop_locs=4, crop_wh=224).batch(conf.batch_size)
    #dataset = dataset.map(imagenet.load_and_crop).batch(conf.batch_size)

    #(data_imgs, data_labels) = tf.data.Dataset.from_generator\
    test_dataset = tf.data.Dataset.from_generator\
        (
            #lambda: imagenet_load(data_dir=conf.data_path_imagenet,resize_wh=256,crop_wh=224,crops=1),
            imagenet_load,
            #(tf.float32, tf.int32),
            (tf.float32, tf.float32),
            #((None,224,224,3),(None,1000)),
            #((None,224,224,3),(None,1000)),
            #((None,224,224,3),(None,1000)),
            ((None,conf.input_size,conf.input_size,3),(None,conf.num_class)),
        )

    test_dataset = test_dataset.map(imagenet.keras_imagenet_preprocess, num_parallel_calls=2)
    test_dataset = test_dataset.prefetch(1)

    train_dataset = test_dataset
    valid_dataset = test_dataset

    return train_dataset, valid_dataset, test_dataset


def train_data_augmentation(train_dataset):
    train_dataset_p = train_dataset.map(preprocess_cifar10_train)

    train_dataset_p = train_dataset_p.shuffle(50000).batch(conf.batch_size)

    return train_dataset_p

def bypass(input):

    return input


def data_preprocessing(train_dataset):
    func_pre={
        'MNIST': bypass,
        'CIFAR-10': train_data_augmentation,
        'CIFAR-100': train_data_augmentation
    }

    train_dataset_p = func_pre[conf.dataset](train_dataset)

    return train_dataset_p

# noinspection PyUnboundLocalVariable
def main(_):
    #print(device_lib.list_local_devices())
    print('main start')

    # remove output dir
    if conf.en_remove_output_dir:
        print('remove output dir: %s' % conf.output_dir)
        shutil.rmtree(conf.output_dir,ignore_errors=True)

    tfe.enable_eager_execution()

    # ImageNet train data load
    #print('imagenet train dataset load test')
    #path_imagenet_root = '/home/sspark/Datasets/ILSVRC2012'
    #val_label_file = os.path.join(path_imagenet_root,'data/ILSVRC2012_label')
    #metadata_file = os.path.join(path_imagenet_root,'data/meta.mat')
    #imagenet._build_synset_lookup(metadata_file)

    #imagenet._find_image_files_train_2012(path_imagenet_root, val_label_file)


    # print flags
    for k, v in flags.FLAGS.flag_values_dict().items():
        print(k+': '+str(v))

    print(tfe.num_gpus())
    #(device, data_format) = ('/gpu:0', 'channels_first')
    #if tfe.num_gpus() <= 0:
    #    (device, data_format) = ('/cpu:0', 'channels_last')
    data_format = 'channels_last'

    if en_gpu==True:
        (device, data_format) = ('/gpu:0', data_format)
    else:
        (device, data_format) = ('/cpu:0', data_format)

    print ('Using device %s, and data format %s.' %(device, data_format))

    #with tf.device('/cpu:0'):
    with tf.device('/gpu:0'):
        # load dataset
        if conf.dataset=='MNIST':
            (train_dataset, val_dataset, test_dataset) = load_data()
        elif conf.dataset=='CIFAR-10':
            (train_dataset, val_dataset, test_dataset) = load_data_cifar10()
        elif conf.dataset=='CIFAR-100':
            (train_dataset, val_dataset, test_dataset) = load_data_cifar100()
        elif conf.dataset=='ImageNet':
            print('only test dataset supported yet')
            (train_dataset, val_dataset, test_dataset) = load_data_imagenet(conf)



    if conf.dataset!='ImageNet':
        val_dataset = val_dataset.batch(conf.batch_size)
        val_dataset = val_dataset.range(conf.num_test_dataset)

        n_test_batch_sel = {
            'ANN': conf.batch_size,
            'SNN': conf.batch_size
            #'SNN': 1
        }
        n_test_batch = n_test_batch_sel[conf.nn_mode]

        test_dataset = test_dataset.batch(n_test_batch)
        test_dataset = test_dataset.range(conf.num_test_dataset)

    #print('> Dataset info.')
    #print(train_dataset)
    #print(test_dataset)
    #

    if conf.ann_model=='MLP':
        if conf.dataset=='MNIST':
            model = models.MNISTModel_MLP(data_format,conf)
    elif conf.ann_model=='CNN':
        if conf.dataset=='MNIST':
            #model = models.MNISTModel_CNN(data_format,conf)
            model = model_cnn_mnist.MNISTModel_CNN(data_format,conf)
        elif conf.dataset=='CIFAR-10':
            model = models.CIFARModel_CNN(data_format,conf)
        elif conf.dataset=='CIFAR-100':
            model = models.CIFARModel_CNN(data_format,conf)
    elif conf.ann_model=='ResNet50':
        if conf.dataset=='ImageNet':
            model = resnet.Resnet50(data_format,conf)



    en_train_sel = {
        'ANN': conf.en_train,
        'SNN': False
    }

    en_train = en_train_sel[conf.nn_mode]

    #
    lr = conf.lr
    momentum = conf.momentum
    #optimizer = tf.train.GradientDescentOptimizer(lr)
    optimizer = tf.train.MomentumOptimizer(lr,momentum)
    #optimizer = tf.train.AdamOptimizer(lr)


    if conf.output_dir:
        output_dir = os.path.join(conf.output_dir,conf.model_name+'_'+conf.nn_mode)

        if conf.nn_mode == 'SNN':
            output_dir = os.path.join(output_dir,conf.n_type+'_time_step_'+str(conf.time_step)+'_vth_'+str(conf.n_init_vth))

        train_dir = os.path.join(output_dir,'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')

        tf.gfile.MakeDirs(output_dir)
    else:
        train_dir = None
        val_dir = None
        test_dir = None

    summary_writer = tf.contrib.summary.create_file_writer(train_dir,flush_millis=100)
    val_summary_writer = tf.contrib.summary.create_file_writer(val_dir,flush_millis=100,name='val')
    test_summary_writer = tf.contrib.summary.create_file_writer(test_dir,flush_millis=100,name='test')
    checkpoint_dir = os.path.join(conf.checkpoint_dir,conf.model_name)
    checkpoint_load_dir = os.path.join(conf.checkpoint_load_dir,conf.model_name)

    print('model load path: %s' % checkpoint_load_dir)
    print('model save path: %s' % checkpoint_dir)

    if not os.path.isdir(checkpoint_dir):
        #os.mkdir(checkpoint_dir)
        tf.gfile.MakeDirs(checkpoint_dir)

    if not os.path.isdir(checkpoint_load_dir):
        print('there is no load dir: %s' % checkpoint_load_dir)
        sys.exit(1)

    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    #
    with tf.device(device):
        #if conf.en_train:
        if en_train:
            print('Train Phase >')

            acc_val_best = 0.0
            #with tfe.restore_variables_on_create(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)):

            with tf.device('/cpu:0'):
                train_dataset_p = data_preprocessing(train_dataset)
            #train_dataset_p = data_preprocessing(train_dataset)

            images_0, _ = tfe.Iterator(train_dataset_p).get_next()

            if conf.en_load_model:
                #images_0, _ = tfe.Iterator(test_dataset).get_next()


                #if conf.dataset=='MNIST':
                #    # noinspection PyUnboundLocalVariable
                #    images_0, _ = tfe.Iterator(train_dataset).get_next()
                #elif conf.dataset=='CIFAR-10':
                #    with tf.device('/cpu:0'):
                #        # noinspection PyUnboundLocalVariable
                #        train_dataset_p = train_data_augmentation(train_dataset)
                #        images_0, _ = tfe.Iterator(train_dataset_p).get_next()


                # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable
                model(images_0,False)

                #tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_load_dir))

                #restore_variables = (model.trainable_weights + optimizer.variables() + [epoch])
                restore_variables = (model.trainable_weights + optimizer.variables())
                #restore_variables = (model.trainable_weights)

                #print(restore_variables)

                print('load model')
                print(tf.train.latest_checkpoint(checkpoint_load_dir))

                saver = tfe.Saver(restore_variables)
                saver.restore(tf.train.latest_checkpoint(checkpoint_load_dir))

                epoch_start = 1
            else:
                epoch_start = 1

            #with tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_load_dir)):
            #for epoch in range(1,11):
            for epoch in range(epoch_start,conf.epoch+1):
                global_step = tf.train.get_or_create_global_step()
                #start = time.time()

                if conf.dataset=='CIFAR-10' or conf.dataset=='CIFAR-100':
                    with tf.device('/cpu:0'):
                        train_dataset_p = train_data_augmentation(train_dataset)
                #train_dataset_p = train_data_augmentation(train_dataset)

                with summary_writer.as_default():
                    # learning rate decay
                    #if epoch%conf.lr_decay_step == 0:
                    #    lr = lr/conf.lr_decay
                    #    #optimizer = tf.train.GradientDescentOptimizer(lr)
                        #optimizer = tf.train.MomentumOptimizer(lr,momentum)
                        #optimizer = tf.train.AdamOptimizer(lr)

                    # testing on Canyon
                    #if conf.dataset=='MNIST':
                        #loss_train, acc_train = train_one_epoch(model, optimizer, train_dataset)
                    #elif conf.dataset=='CIFAR-10':
                        ## noinspection PyUnboundLocalVariable
                        #loss_train, acc_train = train_one_epoch(model, optimizer, train_dataset_p)

                    loss_train, acc_train = train_one_epoch(model, optimizer, train_dataset_p)

                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar('loss', loss_train, step=epoch)
                        tf.contrib.summary.scalar('accuracy', acc_train, step=epoch)

                #end = time.time()
                #print('\nTrain time for epoch #%d (global step %d): %f' % (epoch, global_step.numpy(), end-start))

                #
                with val_summary_writer.as_default():
                    loss_val, acc_val, _ = test(model, val_dataset)

                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar('loss', loss_val, step=epoch)
                        tf.contrib.summary.scalar('accuracy', acc_val, step=epoch)

                    if acc_val_best < acc_val and epoch > 100:
                        acc_val_best = acc_val

                        # save model
                        #if epoch%conf.save_interval==0:
                        all_variables = (model.variables + optimizer.variables() + [global_step])
                        #print([v.name for v in all_variables])
                        #tfe.Saver(all_variables).save(checkpoint_prefix, global_step=global_step)
                        tfe.Saver(all_variables).save(checkpoint_prefix, global_step=epoch)
                        #print(all_variables)
           #            print('save model > global_step: %d'%(global_step.numpy()))

                print('[%3d] train(loss: %.3f, acc: %.3f), valid(loss: %.3f, acc: %.3f, best: %.3f)'%(epoch,loss_train,acc_train,loss_val,acc_val,acc_val_best))


        print(' Test Phase >')
        #print(model.variables)
        #test(model, test_dataset)
        #print(test_dataset.range(0))

        #if conf.en_train == False:
        if en_train == False:
            if conf.dataset=='ImageNet':
                #images_0, _ = test_dataset.next()  # tensornets
                images_0, _ = tfe.Iterator(test_dataset).get_next()
            else:
                images_0, _ = tfe.Iterator(test_dataset).get_next()
            model(images_0,False)
            #print('image 0 done')

            restore_variables = [v for v in model.variables if 'neuron' not in v.name]
            #restore_variables = model.trainable_weights
            #restore_variables = (model.trainable_weights + optimizer.variables() + [epoch])
            #restore_variables = (model.trainable_weights + optimizer.variables())
            #restore_variables = optimizer.variables()


            #print('restore variables')
            var_dict=collections.OrderedDict()
            for var in restore_variables:
                v_name=var.name[:-2]
                v_name=re.sub('kernel','weights',v_name)
                v_name=re.sub('bias','biases',v_name)
                v_name=re.sub('_block','/block',v_name)
                var_dict[v_name] = var
                #print(v_name)
            #print([v.name for v in restore_variables])
            #print(var_list)
            #print(var_dict['resnet50/conv1/conv/kernel'])

            #restore_var_list = []
            #restore_var_list = restore_var_list.append(var_dict['resnet50/conv1/conv/kernel'])
            #print(restore_var_list)



            print('load model')
            print(tf.train.latest_checkpoint(checkpoint_dir))

            #print(tf.get_variable("mnist_model_mlp/dense_2/kernel"))

            # temporally off
            #saver = tfe.Saver(restore_variables)
            #saver = tfe.Saver(restore_var_list)
            saver = tfe.Saver(var_dict)
            #print(model.variables)

            #saver = tfe.Saver(
            #    var_list=
            #    #{'resnet50/conv1/conv/weights': kernel}
            #    {'resnet50/conv1/conv/weights': }
            #)

            ckpt_var_list = []
            #(ckpt_var_list, _) = tf.contrib.framework.list_variables(checkpoint_dir)
            #ckpt_var_list, ckpt_shape_list = tf.contrib.framework.list_variables(checkpoint_dir)
            #print(ckpt_var_list)


            for (var, shape) in tf.contrib.framework.list_variables(checkpoint_dir):
                ckpt_var_list.append(var)
            #    print(var)
            #    print(shape)
            #    saver = tfe.Saver(var)
            #    saver.restore(checkpoint_dir)

            #print(ckpt_var_list)
            #print(len(ckpt_var_list))
            #print(len(restore_variables))
            #
            saver.restore(tf.train.latest_checkpoint(checkpoint_dir))

            #tf.contrib.framework.init_from_checkpoint(
            #    checkpoint_dir,
            #    {'*/weights' : '*/kernel'}
            #)

            #print(tf.get_variable('resnet50/conv1/kernel:0'))

            print('load model done')

            #print('print model')
            #for var in restore_variables:
            #    print(var)


        with test_summary_writer.as_default():
            loss_test, acc_test, acc_test_top5 = test(model, test_dataset)
            if conf.dataset == 'ImageNet':
                print('loss_test: %f, acc_test: %f, acc_test_top5: %f'%(loss_test,acc_test,acc_test_top5))
            else:
                print('loss_test: %f, acc_test: %f'%(loss_test,acc_test))

        print('end')


        os._exit(0)




if __name__ == '__main__':
    tf.app.run()

