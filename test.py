from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import math

import train
from tqdm import tqdm

import pandas as pd

#
def test(model, dataset, conf, f_val=False):
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

        if f_val==False:
            print('accuracy_time_point')
            print(accuracy_time_point)

        for i in range(num_accuracy_time_point):
            accuracy_times.append(tfe.metrics.Accuracy('accuracy'))

            if conf.dataset == 'ImageNet':
                accuracy_times_top5.append(tfe.metrics.Mean('accuracy_top5'))


        num_batch=int(math.ceil(float(conf.num_test_dataset)/float(conf.batch_size)))


        if f_val==False:
            pbar = tqdm(range(1,num_batch+1),ncols=80)
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
                predictions_times = model(images, f_training=False)

                if predictions_times.shape[1] != labels.numpy().shape[0]:
                    predictions_times = predictions_times[:,0:labels.numpy().shape[0],:]

                tf.reshape(predictions_times,(-1,)+labels.numpy().shape)

                if f_val:
                    predictions = predictions_times[-1]
                    accuracy = accuracy_times[-1]
                    accuracy(tf.argmax(predictions,axis=argmax_axis_predictions,output_type=tf.int32), labels)

                else:
                    for i in range(num_accuracy_time_point):
                        predictions=predictions_times[i]
                        accuracy = accuracy_times[i]
                        #print(tf.shape(predictions))
                        accuracy(tf.argmax(predictions,axis=argmax_axis_predictions,output_type=tf.int32), labels)

                        if conf.dataset == 'ImageNet':
                            accuracy_top5 = accuracy_times_top5[i]
                            with tf.device('/cpu:0'):
                                accuracy_top5(tf.cast(tf.nn.in_top_k(predictions,labels,5),tf.int32))

                predictions = predictions_times[-1]
                avg_loss(train.loss(predictions,labels_one_hot))

                if conf.verbose:
                    print(predictions-labels*conf.time_step)

            if f_val==False:
                pbar.update()


        if f_val == False:
            for i in range(num_accuracy_time_point):
                accuracy_result.append(accuracy_times[i].result().numpy())

                if conf.dataset == 'ImageNet':
                    accuracy_result_top5.append(accuracy_times_top5[i].result().numpy())

            print('')
            #print('accruacy')
            #print(accuracy_result)
            if conf.dataset == 'ImageNet':
                print(accuracy_result_top5)

            #plt.plot(accuracy_time_point,accuracy_result)
            #plt.show()

            #print('Test set: Average loss: %.4f, Accuracy: %4f%%\n'%(avg_loss.result(), 100*accuracy.result()))
            #with tf.contrib.summary.always_record_summaries():
                #tf.contrib.summary.scalar('loss', avg_loss.result())
                #tf.contrib.summary.scalar('accuracy', accuracy.result())
                #tf.contrib.summary.scalar('w_conv1', model.variables)
            ret_accu = 100*accuracy_result[-1]
        else:
            ret_accu = 100*accuracy_times[-1].result().numpy()

        if conf.dataset == 'ImageNet':
            ret_accu_top5 = 100*accuracy_result_top5[-1]
        else:
            ret_accu_top5 = 0.0

        if f_val == False:
            #print('total spike count - int')
            #print(model.total_spike_count_int)
            #print('total spike count - float')
            #print(model.total_spike_count)
            #print('total residual vmem')
            #print(model.total_residual_vmem)


            #
            #df=pd.DataFrame({'time step': model.accuracy_time_point, 'spike count': list(model.total_spike_count[:,-1]),'accuracy': accuracy_result})
            df=pd.DataFrame({'time step': model.accuracy_time_point, 'accuracy': accuracy_result})
            df.set_index('time step', inplace=True)
            print(df)

            if conf.f_save_result:
                f_name_result = conf.path_result_root+conf.date+'_result.xlsx'
                df.to_excel(f_name_result)


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


        if f_val==False:
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

            #labels = tf.argmax(labels_one_hot,axis=1,output_type=tf.int32)
            #with tf.argmax(labels_one_hot,axis=1,output_type=tf.int32) as labels:

            if idx_batch!=-1:
                #model=tfe.defun(model)
                predictions = model(images, f_training=False)

                #print(predictions.shape)
                #print(str(tf.argmax(predictions,axis=1))+' : '+str(tf.argmax(labels,axis=1)))

                #accuracy(tf.argmax(predictions,axis=1,output_type=tf.int32), labels)
                accuracy(tf.argmax(predictions,axis=1,output_type=tf.int32), tf.argmax(labels_one_hot,axis=1,output_type=tf.int32))

                if conf.dataset == 'ImageNet':
                    with tf.device('/cpu:0'):
                        #accuracy_top5(tf.cast(tf.nn.in_top_k(predictions,labels,5),tf.int32))
                        accuracy_top5(tf.cast(tf.nn.in_top_k(predictions, tf.argmax(labels_one_hot,axis=1,output_type=tf.int32),5),tf.int32))
                avg_loss(train.loss(predictions,labels_one_hot))

            if f_val==False:
                pbar.update()

        ret_accu = 100*accuracy.result()
        if conf.dataset == 'ImageNet':
            ret_accu_top5 = 100*accuracy_top5.result()
        else:
            ret_accu_top5 = 0.0

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


