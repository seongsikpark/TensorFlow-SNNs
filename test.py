from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#import tensorflow.contrib.eager as tfe

import math

import train
from tqdm import tqdm

import pandas as pd


# train tk
train_tk_num_data = None
train_tk_num_data_save_target = None

#
def test(model, dataset, num_dataset, conf, f_val=False, epoch=0, f_val_snn=False):
    #avg_loss = tfe.metrics.Mean('loss')
    avg_loss = tf.keras.metrics.Mean('loss')

    #if conf.nn_mode=='SNN':
    if conf.nn_mode=='SNN' or f_val_snn:
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

            print('num_accuracy_time_point: {:d}'.format(model.num_accuracy_time_point))

        for i in range(num_accuracy_time_point):
            #accuracy_times.append(tfe.metrics.Accuracy('accuracy'))
            accuracy_times.append(tf.keras.metrics.Accuracy('accuracy'))

            if conf.dataset == 'ImageNet':
                #accuracy_times_top5.append(tfe.metrics.Mean('accuracy_top5'))
                accuracy_times_top5.append(tf.keras.metrics.Mean('accuracy_top5'))


        #num_batch=int(math.ceil(float(conf.num_test_dataset)/float(conf.batch_size)))

        num_batch=int(math.ceil(float(num_dataset)/float(conf.batch_size)))

        if conf.f_train_tk:
            global train_tk_num_data
            global train_tk_num_data_save_target
            # 1st run - initial set
            if train_tk_num_data is None:
                if conf.f_load_time_const:
                    train_tk_num_data = conf.time_const_num_trained_data
                else:
                    train_tk_num_data = 0
                train_tk_num_data_save_target=train_tk_num_data+conf.time_const_save_interval

        # TODO: parameterize print_loss
        print_loss = True
        if conf.f_train_tk and print_loss:
            list_loss_prec = list(range(num_batch))
            list_loss_min = list(range(num_batch))
            list_loss_max = list(range(num_batch))

            list_tc = list(range(num_batch))
            list_td = list(range(num_batch))


        #
        if f_val==False:
            pbar = tqdm(range(1,num_batch+1),ncols=80)
            pbar.set_description("batch")

        #
        if f_val_snn:
            model.f_done_preproc = False

        #
        idx_batch=0
        for (images, labels_one_hot) in dataset:
            labels = tf.argmax(labels_one_hot,axis=1,output_type=tf.int32)

            #
            f_resize_output = False
            if conf.batch_size != labels.shape:
                concat_dim = conf.batch_size-labels.numpy().shape[0]
                f_resize_output = True
                batch_size = conf.batch_size-concat_dim

                labels = tf.concat([labels,tf.zeros(shape=[concat_dim],dtype=tf.int32)],0)
                images = tf.concat([images,tf.zeros(shape=(concat_dim,)+tuple(images.shape[1:]),dtype=images.dtype)],0)

            else:
                batch_size = conf.batch_size

            #
            #if idx_batch!=-1:
            #if conf.f_train_tk:
                #for itr_train_time_const in range(0,10):
                #    model(images, f_training=False)

            # run
            # predictions_times - [saved time step, batch, output dim]
            predictions_times = model(images, f_training=False, f_val_snn=f_val_snn, epoch=epoch)


            # post-processing
            if f_resize_output:
                labels = labels[0:batch_size]
                predictions_times = predictions_times[:,batch_size,:]

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
            avg_loss(train.loss_cross_entoropy(predictions,labels_one_hot))


            if conf.f_train_tk and print_loss:
                [loss_prec, loss_min, loss_max] = model.get_time_const_train_loss()

                list_loss_prec[idx_batch]=loss_prec.numpy()
                list_loss_min[idx_batch]=loss_min.numpy()
                list_loss_max[idx_batch]=loss_max.numpy()
            #else:
            #    assert False



            if f_val==False:
                pbar.update()

            if conf.f_train_tk:
                print("\nidx_batch: {:d}".format(idx_batch))
                #train_tk_num_data=(idx_batch+1)*conf.batch_size+conf.time_const_num_trained_data+(epoch)*conf.num_test_dataset
                train_tk_num_data+=batch_size

                print("num_data: {:d}".format(train_tk_num_data))
                print(train_tk_num_data_save_target)
                #if num_data%10000==0:
                #if num_data%conf.time_const_save_interval==0:
                #if train_tk_num_data%conf.time_const_save_interval==0:
                #if num_data%1==0:
                if train_tk_num_data >= train_tk_num_data_save_target:

                    # TODO: move to run.sh and add paramter - train_tk_out_file_name

                    # train_tk_write_output(model,conf)
                    fname = conf.tk_file_name + "_itr-{:d}".format(train_tk_num_data)

                    #fname = conf.time_const_init_file_name + '/' + conf.model_name
                    #if conf.f_tc_based:
                    #    fname+="/tc-{:d}_tw-{:d}_tau_itr-{:d}".format(conf.tc,conf.n_tau_time_window,train_tk_num_data)
                    #else:
                    #    fname+="/tc-{:d}_tw-{:d}_itr-{:d}".format(conf.tc,conf.time_window,train_tk_num_data)

                    if conf.f_train_tk_outlier:
                        fname+="_outlier"

                    print("save time constant: file_name: {:s}".format(fname))
                    f = open(fname,'w')

                    # time const
                    for name_neuron, neuron in model.list_neuron.items():
                        if not ('fc3' in name_neuron):
                            f.write("tc,"+name_neuron+","+str(neuron.time_const_fire.numpy())+"\n")

                    f.write("\n")

                    # time delay
                    for name_neuron, neuron in model.list_neuron.items():
                        if not ('fc3' in name_neuron):
                            f.write("td,"+name_neuron+","+str(neuron.time_delay_fire.numpy())+"\n")

                    f.close()

                    train_tk_num_data_save_target+=conf.time_const_save_interval

            idx_batch += 1

        #
        if f_val_snn:
            model.defused_bn()


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


            pd.set_option('display.float_format','{:.4g}'.format)

            #
            #df=pd.DataFrame({'time step': model.accuracy_time_point, 'spike count': list(model.total_spike_count[:,-1]),'accuracy': accuracy_result})
            #df=pd.DataFrame({'time step': model.accuracy_time_point, 'accuracy': accuracy_result, 'spike count': model.total_spike_count_int[:,-1]})
            df=pd.DataFrame({'time step': model.accuracy_time_point, 'accuracy': accuracy_result, 'spike count': model.total_spike_count_int[:,-1]/num_dataset, 'spike_count_c1':model.total_spike_count_int[:,0]/num_dataset, 'spike_count_c2':model.total_spike_count_int[:,1]/num_dataset})
            df.set_index('time step', inplace=True)
            print(df)

            # TODO: save result - integrate c1 and c2
            # c1
            if conf.f_save_result:
                f_name_result = conf.path_result_root+'/'+conf.date+'_'+conf.neural_coding

                if conf.neural_coding=="TEMPORAL":
                    if conf.f_tc_based:
                        f_name_result = f_name_result+'_tau_tc-'+str(conf.tc)+'_tw-'+str(conf.n_tau_time_window)+'_tfs-'+str(conf.n_tau_fire_start) \
                                        +'_tfd-'+str(conf.n_tau_fire_duration)+'_ts-'+str(conf.time_step)+'_tssi-'+str(conf.time_step_save_interval)
                    else:
                        f_name_result = f_name_result+'_tc-'+str(conf.tc)+'_tw-'+str(conf.time_window)+'_tfs-'+str(conf.time_fire_start)\
                                    +'_tfd-'+str(conf.time_fire_duration)+'_ts-'+str(conf.time_step)+'_tssi-'+str(conf.time_step_save_interval)

                if conf.f_load_time_const:
                    if conf.f_train_tk:
                        f_name_result += '_trained_data-'+str(conf.time_const_num_trained_data+conf.num_test_dataset)
                    else:
                        f_name_result += '_trained_data-'+str(conf.time_const_num_trained_data)


                if conf.f_train_tk_outlier:
                    f_name_result += '_outlier'

                if conf.f_train_tk:
                    f_name_result += '_train-tc'

                if conf.ckpt_epoch!=0:
                    f_name_result += '_ckpt-'+str(conf.ckpt_epoch)

                if conf.f_qvth:
                    f_name_result += '_qvth'

                f_name_result = f_name_result+'.xlsx'

                df.to_excel(f_name_result)
                print("output file: "+f_name_result)


#            # c2
#                # ts: time step
#                # tssi: time step save interval
#                #f_name_result = conf.path_result_root+'/'+conf.date+'_'+conf.neural_coding
#                #f_name_result = conf.path_result_root+'/'+conf.input_spike_mode+conf.neural_coding+'_ts-'+str(conf.time_step)+'_tssi-'+str(conf.time_step_save_interval)
#                f_name_result = '{}/{}_{}_ts-{}_tssi-{}_vth-{}'.format(conf.path_result_root,conf.input_spike_mode,conf.neural_coding,str(conf.time_step),str(conf.time_step_save_interval),conf.n_init_vth)
#
#
#
#                if conf.neural_coding=="TEMPORAL":
#                    f_name_result += outfile_name_temporal(conf)
#
#                if conf.noise_en:
#                    f_name_result += '_nt-'+conf.noise_type+'_np-'+str(conf.noise_pr)
#
#                    if conf.noise_robust_en:
#                        f_name_result += '_nrs-'+str(conf.noise_robust_spike_num)
#
#                        if conf.noise_robust_comp_pr_en:
#                            f_name_result += '_nrc'
#
#                    if conf.rep != -1:
#                        f_name_result += '_rep-'+str(conf.rep)
#
#                #f_name_result = f_name_result+'.xlsx'
#                f_name_result += '.xlsx'
#
#                df.to_excel(f_name_result)
#                print("output file: "+f_name_result)


            if conf.f_train_tk and print_loss:
                df=pd.DataFrame({'loss_prec': list_loss_prec, 'loss_min': list_loss_min, 'loss_max': list_loss_max})
                fname="./time-const-train-loss_b-"+str(conf.batch_size)+"_d-"+str(conf.num_test_dataset)+"_tc-"+str(conf.tc)+"_tw-"+str(conf.time_window)+"_ckpt-"+str(conf.ckpt_epoch)+".xlsx"
                df.to_excel(fname)


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
        if conf.verbose_visual:
            model.figure_hold()

    # DNN (ANN) test
    else:
        accuracy=tf.metrics.Accuracy('accuracy')

        if conf.dataset == 'ImageNet':
            accuracy_top5=tf.metrics.Mean('accuracy_top5')


        if f_val==False:
            num_batch=int(math.ceil(float(conf.num_test_dataset)/float(conf.batch_size)))
            pbar = tqdm(range(1,num_batch+1),ncols=80)
            pbar.set_description("batch")


        #for (idx_batch, (images, labels_one_hot)) in enumerate(tfe.Iterator(dataset)):
        #for (idx_batch, (images, labels_one_hot)) in enumerate(tf.Iterator(dataset)):
        #for idx_batch in range(0,2):
        idx_batch = 0
        for (images, labels_one_hot) in dataset:
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
                predictions = model(images, f_training=False, epoch=epoch)

                #print(predictions.shape)
                #print(str(tf.argmax(predictions,axis=1))+' : '+str(tf.argmax(labels,axis=1)))

                #accuracy(tf.argmax(predictions,axis=1,output_type=tf.int32), labels)
                accuracy(tf.argmax(predictions,axis=1,output_type=tf.int32), tf.argmax(labels_one_hot,axis=1,output_type=tf.int32))

                if conf.dataset == 'ImageNet':
                    with tf.device('/cpu:0'):
                        #accuracy_top5(tf.cast(tf.nn.in_top_k(predictions,labels,5),tf.int32))
                        accuracy_top5(tf.cast(tf.nn.in_top_k(predictions, tf.argmax(labels_one_hot,axis=1,output_type=tf.int32),5),tf.int32))
                avg_loss(train.loss_cross_entoropy(predictions,labels_one_hot))

            if f_val==False:
                pbar.update()

            idx_batch += 1

        ret_accu = 100*accuracy.result()
        if conf.dataset == 'ImageNet':
            ret_accu_top5 = 100*accuracy_top5.result()
        else:
            ret_accu_top5 = 0.0

        #plt.hist(model.stat_a_fc3)
        #plt.show()
        #model.plot_dist_activation_vgg16()
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
            else:
                model.save_activation()



        #print(tf.reduce_max(model.stat_a_fc2))
        #print(tf.reduce_max(model.stat_a_fc3))


    return avg_loss.result(), ret_accu, ret_accu_top5




















############################################
# output file name
############################################

def outfile_name_temporal(conf):

    if conf.f_tc_based:
        f_name_result = '_tau_tc-'+str(conf.tc)+'_tw-'+str(conf.n_tau_time_window)+'_tfs-'+str(int(conf.n_tau_fire_start)) \
                        +'_tfd-'+str(int(conf.n_tau_fire_duration))+'_ts-'+str(conf.time_step)+'_tssi-'+str(conf.time_step_save_interval)
    else:
        f_name_result = '_tc-'+str(conf.tc)+'_tw-'+str(conf.time_window)+'_tfs-'+str(int(conf.time_fire_start)) \
                        +'_tfd-'+str(int(conf.time_fire_duration))

    if conf.f_load_time_const:
        if conf.f_train_tk:
            f_name_result += '_trained_data-'+str(conf.time_const_num_trained_data+conf.num_test_dataset)
        else:
            f_name_result += '_trained_data-'+str(conf.time_const_num_trained_data)

    if conf.f_train_tk:
        if conf.f_train_tk_outlier:
            f_name_result += '_outlier'

        f_name_result += '_train-tc'

    return f_name_result

