
#
# configuration
from config_snn_training_H_direct import config


from config import config
conf = config.flags
# snn library
import lib_snn
import tensorflow as tf
import tensorflow.profiler.experimental as profiler
from tqdm import tqdm
import numpy as np
#
import os
import datasets
import callbacks
from tensorflow.python.distribute import collective_all_reduce_strategy

import pandas as pd
import logging

tf.config.optimizer.set_jit(True)
logger = logging.getLogger()
old_level_logger = logger.level

########################################
# configuration
########################################
dist_strategy = lib_snn.utils.set_gpu()
# dist_strategy = tf.distribute.MirroredStrategy(
#     cross_device_ops=collective_all_reduce_strategy.CollectiveAllReduceStrategy()
# )
# dist_strategy = tf.distribute.MirroredStrategy()


################
# name set
################
#
filepath_save, filepath_load, config_name = lib_snn.utils.set_file_path()

########################################
# load dataset
########################################
train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch = \
    datasets.datasets.load()
#datasets.datasets_bck_eventdata.load()


#
with dist_strategy.scope():

    ########################################
    # build model
    ########################################
    #data_batch = valid_ds.take(1)
    #model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch)
    model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch,valid_ds)
    # model.summary()

    ########################################
    # load model
    ########################################
    if config.load_model:
        model.load_weights(config.load_weight)

    ################
    # Callbacks
    ################
    callbacks_train, callbacks_test = \
        callbacks.callbacks_snn_train(model,train_ds_num,valid_ds,test_ds_num)
    en = 'train'
    # en = 'sample_spike'
    # en = 'visual'
    # en = 'channel_count'
    # en = 'mutual'
    # en = 'count_avg_channel'
    # en='alpha'

    # if True:
    if en == 'train':
        if config.train:
            print('Train mode')

            model.summary()
            print(test_ds_num)
            #train_steps_per_epoch = train_ds_num/batch_size
            train_epoch = config.flags.train_epoch
            init_epoch = config.init_epoch
            # profiler.start(logdir="./tensorflow_pf")
            train_histories = model.fit(train_ds, epochs=train_epoch, steps_per_epoch=train_steps_per_epoch,
                                        initial_epoch=init_epoch, validation_data=valid_ds, callbacks=callbacks_train)
            # profiler.stop()
        else:
            print('Test mode')
            result = model.evaluate(test_ds, callbacks=callbacks_test)
            print('end')
################################################################################################################################
    elif en == 'sample_spike':
        import tensorflow as tf
        import matplotlib.pyplot as plt
        import numpy as np
        import csv
        # save_dir = "/home/ssparknas/SEL_spike_count_int/ResNet20_CIFAR10/base/"
        save_dir = "/home/ssparknas/SEL_spike_count_int/VGG16_CIFAR10/base/"
        num = '4'
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        except:
            print("Error: failed to create the directory")
        fm = []
        layer_names = []
        df_sc = []
        df_1=[]
        df_2=[]
        df_3=[]
        df_4=[]
        df_0=[]
        zero=[]
        one =[]
        two=[]
        three=[]
        four =[]
        result = model.evaluate(test_ds.take(1), callbacks=callbacks_test)
        for layer in model.layers_w_neuron:
            if isinstance(layer.act, lib_snn.neurons.Neuron):
                fm.append(layer.act.spike_count_int)
                layer_names.append(layer.name)
        fm = [fm[1]]
        layer_names = [layer_names[1]]
        for i in range(0, 100):
            result = model.evaluate(test_ds.skip(i).take(1), callbacks=callbacks_test)
            for layer in model.layers_w_neuron:
                if isinstance(layer.act, lib_snn.neurons.Neuron):
                    if layer.name in ['n_conv1','conv1_conv_n']:
                        spike_count_sum = tf.reduce_sum(layer.act.spike_count_int,axis=[1,2]).numpy()
                        spike_count_zero = tf.where(tf.equal(0,layer.act.spike_count_int)).shape
                        spike_count_one = tf.where(tf.equal(1,layer.act.spike_count_int)).shape
                        spike_count_two = tf.where(tf.equal(2,layer.act.spike_count_int)).shape
                        spike_count_three = tf.where(tf.equal(3,layer.act.spike_count_int)).shape
                        spike_count_four = tf.where(tf.equal(4,layer.act.spike_count_int)).shape
                        df_sc.extend(spike_count_sum)
                        zero.extend(spike_count_zero)
                        one.extend(spike_count_one)
                        two.extend(spike_count_two)
                        three.extend(spike_count_three)
                        four.extend(spike_count_four)
        df_1=np.ravel(one)
        df_2=np.ravel(two)
        df_3=np.ravel(three)
        df_4=np.ravel(four)
        df_0=np.ravel(zero)
        df_sc=np.ravel(df_sc)
        # plt.hist(df_sc,bins=10)
        # plt.show()
        df_sc=pd.DataFrame(df_sc)
        df_0= pd.DataFrame(df_0)
        df_1 = pd.DataFrame(df_1)
        df_2 = pd.DataFrame(df_2)
        df_3 = pd.DataFrame(df_3)
        df_4 = pd.DataFrame(df_4)
        df_sc.to_excel(save_dir + '_spike_count_int_'+num+'.xlsx')
        df_0.to_excel(save_dir + '_spike_count_int_0'+num+'.xlsx')
        df_1.to_excel(save_dir + '_spike_count_int_1'+num+'.xlsx')
        df_2.to_excel(save_dir + '_spike_count_int_2'+num+'.xlsx')
        df_3.to_excel(save_dir + '_spike_count_int_3'+num+'.xlsx')
        df_4.to_excel(save_dir + '_spike_count_int_4'+num+'.xlsx')
############################################################################################################################
    elif en=='visual':
        import tensorflow as tf
        import matplotlib.pyplot as plt
        import numpy as np
        import csv


        # folder = "/home/ssparknas/Fig_real_final/"+conf.SEL_model_dataset+'/'+conf.SEL_en+'/'+conf.SEL_num+'/'
        # folder = "/home/ssparknas/Fig_real_final/R20_C10/base/1/"
        # try:
        #     if not os.path.exists(folder):
        #         os.makedirs(folder)
        # except:
        #     print("Error: failed to create the directory")


        # move to prop.py postproc_batch_test()
        if True:
            # if False:
            fm = []
            layer_names = []

            result = model.evaluate(test_ds.take(1), callbacks=callbacks_test)
            for layer in model.layers_w_neuron:
                if isinstance(layer.act, lib_snn.neurons.Neuron):
                    fm.append(layer.act.spike_count_int)
                    layer_names.append(layer.name)

            images_per_row = 16
            img_idx = 0
            layer_idx = 0

            plot_hist = False
            if plot_hist:
                figs_h, axes_h = plt.subplots(4, 4, figsize=(12, 10))

            # only conv1
            fm = [fm[1]]
            layer_names = [layer_names[1]]
            if conf.model == 'ResNet19':
                channel_num = 128
            else:
                channel_num = 64
            # vis_count
            if conf.dataset == 'CIFAR10_DVS':
                s = 48  # for DVS
                sum_spike = 2304  # for DVS
                batch_num = 10
                img_num = 100
            elif conf.dataset == 'ImageNet':
                s = 56
                sum_spike = 50176
                img_num = 200
            else:
                s = 32
                sum_spike = 1024
                batch_num = 100
                img_num = 100

            for i in range(0, 1):
                result = model.evaluate(test_ds.skip(i).take(1), callbacks=callbacks_test)
                for img_idx in range(0, img_num):
                    for layer_name, layer_fm in zip(layer_names, fm):
                        n_features = layer_fm.shape[-1]
                        size = layer_fm.shape[1]

                        n_cols = n_features // images_per_row
                        if n_cols == 0:  # n_in
                            continue

                        if len(layer_fm.shape) == 2:  # fc_layers
                            continue

                        display_grid = np.zeros(((size + 1) * n_cols - 1, images_per_row * (size + 1) - 1))

                        #
                        for col in range(n_cols):
                            for row in range(images_per_row):
                                #
                                channel_index = col * images_per_row + row
                                channel_image = layer_fm[img_idx, :, :, channel_index]


                                if channel_image.numpy().sum() != 0 and channel_image.numpy().std() != 0:
                                    channel_image -= channel_image.numpy().mean()
                                    channel_image /= channel_image.numpy().std()
                                    channel_image *= 64
                                    channel_image += 128


                                elif channel_image.numpy().sum() == 0:
                                    channel_image = np.zeros((s,s))
                                elif channel_image.numpy().sum() == sum_spike:
                                    channel_image = np.full((s,s),63)
                                elif channel_image.numpy().sum() == sum_spike * 2:
                                    channel_image = np.full((s,s),126)
                                elif channel_image.numpy().sum() == sum_spike * 3:
                                    channel_image = np.full((s,s),189)
                                elif channel_image.numpy().sum() == sum_spike * 4:
                                    channel_image = np.full((s, s), 255)
                                channel_image = np.clip(channel_image, 0, 255).astype("uint8")

                                display_grid[
                                col * (size + 1):(col + 1) * size + col,
                                row * (size + 1):(row + 1) * size + row] = channel_image

                        #
                        scale = 1. / size
                        plt.figure(figsize=(scale * display_grid.shape[1],
                                            scale * display_grid.shape[0]))
                        plt.title(layer_name)
                        plt.grid(False)
                        plt.axis("off")

                        plt.imshow(display_grid, aspect="auto", cmap="viridis")
                        plt.show()

                        # channel intensity
                        # image
                        stat_image = False
                        if stat_image:
                            # one image
                            channel_image = layer_fm[img_idx, :, :, :].numpy().copy()
                        else:
                            ## batch
                            channel_image = layer_fm.numpy().copy()

                        channel_intensity = tf.reduce_mean(channel_image, axis=[0, 1])

                        # display_grid_h[layer_idx//4, layer_idx%4] = channel_intensity

                        if plot_hist:
                            axes_h[layer_idx // 4, layer_idx % 4].hist(tf.reshape(channel_intensity, shape=-1),
                                                                       bins=100)

                            ci_mean = tf.reduce_mean(channel_intensity)
                            ci_max = tf.reduce_max(channel_intensity)
                            ci_min = tf.reduce_min(channel_intensity)
                            ci_std = tf.math.reduce_std(channel_intensity)
                            ci_non_zeros = tf.math.count_nonzero(channel_intensity, dtype=tf.int32)
                            ci_non_zeros_r = ci_non_zeros / tf.math.reduce_prod(channel_intensity.shape)

                            print("{:}, mean:{:.3f}, max:{:.3f}, min:{:.3f}, std:{:.4f}, nonz:{:.3f}"
                                  .format(layer_fm.name, ci_mean, ci_max, ci_min, ci_std, ci_non_zeros_r))

                            layer_idx += 1

                    # fname = str(i) + '_' + str(img_idx) + '.png'
                    # plt.savefig(folder + fname)
                    # plt.close()


###############################################################################################################################
    elif en=='channel_count':
        import tensorflow as tf
        import matplotlib.pyplot as plt
        import numpy as np
        import csv

        # move to prop.py postproc_batch_test()
        if True:
            # if False:
            fm = []
            layer_names = []

            result = model.evaluate(test_ds.take(1), callbacks=callbacks_test)
            for layer in model.layers_w_neuron:
                if isinstance(layer.act, lib_snn.neurons.Neuron):
                    fm.append(layer.act.spike_count_int)
                    layer_names.append(layer.name)

            images_per_row = 16
            img_idx = 0
            layer_idx = 0

            plot_hist = False
            if plot_hist:
                figs_h, axes_h = plt.subplots(4, 4, figsize=(12, 10))

            # only conv1
            fm = [fm[1]]
            layer_names = [layer_names[1]]
            if conf.model == 'ResNet19':
                channel_num = 128
            elif conf.model == 'Spikformer':
                channel_num = 48
            else:
                channel_num = 64
            # vis_count
            al_vis_1 = []
            wh_non_vis = []
            wh_al_vis = np.arange(channel_num)
            wh_al_vis_one=[]
            wh_al_vis_two=[]
            wh_al_vis_thr=[]
            wh_al_vis_four=[]
            wh_al_vis_rand=[]
            if conf.dataset == 'CIFAR10_DVS':
                s = 48  # for DVS
                sum_spike = 2304  # for DVS
                batch_num = 10
                img_num = 100
                step = 100
            elif conf.dataset == 'ImageNet':
                # s = 56
                s = 112
                sum_spike = 50176
                img_num = 100 #batch
                step = 100
            else:
                s = 32
                sum_spike = 1024
                batch_num = 100
                img_num = 100
                step = 500

            for i in range(0, step):
                result = model.evaluate(test_ds.skip(i).take(1), callbacks=callbacks_test)
                for img_idx in range(0, img_num):
                    non_vis = []
                    for layer_name, layer_fm in zip(layer_names, fm):
                        n_features = layer_fm.shape[-1]
                        size = layer_fm.shape[1]

                        n_cols = n_features // images_per_row
                        if n_cols == 0:  # n_in
                            continue

                        if len(layer_fm.shape) == 2:  # fc_layers
                            continue

                        display_grid = np.zeros(((size + 1) * n_cols - 1, images_per_row * (size + 1) - 1))

                        #
                        for col in range(n_cols):
                            for row in range(images_per_row):
                                #
                                channel_index = col * images_per_row + row
                                channel_image = layer_fm[img_idx, :, :, channel_index]

                                if tf.reduce_sum(channel_image) == 0:
                                    non_vis.append(channel_index)

                                if channel_image.numpy().sum() != 0 and channel_image.numpy().std() != 0:
                                    channel_image -= channel_image.numpy().mean()
                                    channel_image /= channel_image.numpy().std()
                                    channel_image *= 64
                                    channel_image += 128
                                    if channel_index not in wh_al_vis_rand:
                                        wh_al_vis_rand.append(channel_index)


                                elif channel_image.numpy().sum() == 0:
                                    channel_image = np.zeros((s,s))

                                elif channel_image.numpy().sum() == sum_spike:
                                    channel_image = np.full((s,s),63)
                                    if channel_index not in wh_al_vis_one:
                                        wh_al_vis_one.append(channel_index)

                                elif channel_image.numpy().sum() == sum_spike * 2:
                                    channel_image = np.full((s,s),126)
                                    if channel_index not in wh_al_vis_two:
                                        wh_al_vis_two.append(channel_index)

                                elif channel_image.numpy().sum() == sum_spike * 3:
                                    channel_image = np.full((s,s),189)
                                    if channel_index not in wh_al_vis_thr:
                                        wh_al_vis_thr.append(channel_index)

                                elif channel_image.numpy().sum() == sum_spike * 4:
                                    channel_image = np.full((s, s), 255)
                                    if channel_index not in wh_al_vis_four:
                                        wh_al_vis_four.append(channel_index)

                                channel_image = np.clip(channel_image, 0, 255).astype("uint8")

                                display_grid[
                                col * (size + 1):(col + 1) * size + col,
                                row * (size + 1):(row + 1) * size + row] = channel_image

                        #
                        scale = 1. / size

                        # channel intensity
                        # image

                    # vis, non_vis counting
                    if True:
                        # if False:
                        img_index = str(i) + '_' + str(img_idx)
                        data = {img_index: non_vis}

                        if img_idx == 0:
                            wh_non_vis = non_vis
                            wh_al_vis = list(set(wh_al_vis) - set(non_vis))
                        else:
                            new_non_vis = list(set(wh_non_vis) & set(non_vis))
                            wh_non_vis = new_non_vis
                            wh_non_vis.sort()
                            wh_al_vis = list(set(wh_al_vis) - set(non_vis))

                        if wh_al_vis_one:
                            wh_al_vis_one = list(set(wh_al_vis).intersection(set(wh_al_vis_one)))
                            wh_al_vis_one = list(set(wh_al_vis_one)-set(wh_al_vis_rand))
                        if wh_al_vis_two:
                            wh_al_vis_two = list(set(wh_al_vis).intersection(set(wh_al_vis_two)))
                            wh_al_vis_two = list(set(wh_al_vis_two)-set(wh_al_vis_rand))
                        if wh_al_vis_thr:
                            wh_al_vis_thr = list(set(wh_al_vis).intersection(set(wh_al_vis_thr)))
                            wh_al_vis_thr = list(set(wh_al_vis_thr)-set(wh_al_vis_rand))
                        if wh_al_vis_four:
                            wh_al_vis_four = list(set(wh_al_vis).intersection(set(wh_al_vis_four)))
                            wh_al_vis_four = list(set(wh_al_vis_four)-set(wh_al_vis_rand))
                        # with open('./SEL_data.csv', 'a', newline='') as csvfile:
                        #     writer = csv.writer(csvfile)
                        #
                        #     for img_index, non_vis in data.items():
                        #         writer.writerow([img_index] + non_vis)

            print("non: ",wh_non_vis,len(wh_non_vis))
            # print("all: ",wh_al_vis)
            all_rand = list(set(wh_al_vis)-set(wh_al_vis_one)-set(wh_al_vis_two)-set(wh_al_vis_thr)-set(wh_al_vis_four))
            all_rand.sort()
            blink_vis = np.arange(channel_num)
            blink_vis = list(set(blink_vis) - set(wh_al_vis) - set(wh_non_vis))
            blink_rand = list(set(blink_vis)-set(wh_al_vis_one)-set(wh_al_vis_two)-set(wh_al_vis_thr)-set(wh_al_vis_four))
            blink_ones = list(set(blink_vis).intersection(set(wh_al_vis_one)))
            blink_twos = list(set(blink_vis).intersection(set(wh_al_vis_two)))
            blink_threes = list(set(blink_vis).intersection(set(wh_al_vis_thr)))
            blink_fours = list(set(blink_vis).intersection(set(wh_al_vis_four)))
            blink_vis.sort()
            blink_ones.sort()
            blink_twos.sort()
            blink_threes.sort()
            blink_fours.sort()
            wh_al_vis_one.sort()
            wh_al_vis_two.sort()
            wh_al_vis_thr.sort()
            wh_al_vis_four.sort()
            print("blink: ",blink_vis,len(blink_vis))
            print("blink_ones: ",blink_ones,len(blink_ones))
            print("blink_twos: ",blink_twos,len(blink_twos))
            print("blink_threes: ",blink_threes,len(blink_threes))
            print("blink_fours: ",blink_fours,len(blink_fours))
            print("all_rand: ",all_rand,len(all_rand))
            print("all_ones: ",wh_al_vis_one,len(wh_al_vis_one))
            print("all_twos: ",wh_al_vis_two,len(wh_al_vis_two))
            print("all_threes: ",wh_al_vis_thr,len(wh_al_vis_thr))
            print("all_fours: ",wh_al_vis_four,len(wh_al_vis_four))
# mutual coherence
    elif en == 'mutual':
    # if True:
        logger.setLevel(100)
        import keras
        from lib_snn import grad_cam
        import matplotlib.pyplot as plt

        # from config_snn_training_WTA_SNN import mode

        save_imgs = False
        # save_imgs = True

        # show_imgs = True
        # show_imgs = False

        save_stat = True
        # save_stat = False

        batch_size = 100
        input_shape = (32, 32, 3)
        classes = 10

        # mode = 'normal'
        mode = 'VGG_avlation_VTH'

        # mc - kernel
        #if False:
        if True:
            w = model.get_layer('conv1').kernel
            wc = tf.reshape(w, shape=[3 * 3 * 3, 64])
            wc_l2 = tf.sqrt(tf.reduce_sum(tf.square(wc), axis=[0]))
            wc_l = wc_l2
            #wc_l = wc_l1
            wct = tf.transpose(wc)
            wc_l = tf.expand_dims(wc_l, -1)
            wc_l_mat = wc_l @ tf.transpose(wc_l)
            mc = tf.abs(wct @ wc) / wc_l_mat
            mc_no_diag = tf.linalg.set_diag(mc, tf.zeros(shape=(64)))
            print(tf.reduce_mean(mc_no_diag))
            print(tf.reduce_max(mc_no_diag))

        #
        stats_max = []
        stats_mean = []

        # save directory
        if save_imgs or save_stat:
            save_dir = '/home/ssparknas'
            os.makedirs(save_dir, exist_ok=True)

        #batch_index = conf.sm_batch_index
        #batch_index = 0
        # for batch in test_ds:
        for batch_idx in tqdm(range(0,1)):
        #for batch_idx in tqdm(range(0,100)):
        # for batch_idx in tqdm(range(10, 20)):
        #for batch_idx in tqdm(range(batch_index, batch_index + 1)):

            test_ds_a_batch = test_ds.skip(batch_idx).take(1)
            result = model.evaluate(test_ds_a_batch, callbacks=callbacks_test)

            # mc - encoded spike (spike feature map)
            sc = model.get_layer('n_conv1').act.spike_count
            sc = tf.reshape(sc, shape=[100,32*32, 64])
            sc_l2 = tf.sqrt(tf.reduce_sum(tf.square(sc), axis=[1]))
            sc_l = sc_l2
            sc_l = tf.expand_dims(sc_l,-1)
            sct = tf.transpose(sc,perm=[0,2,1])
            sc_l_mat = sc_l @ tf.transpose(sc_l,perm=[0,2,1])
            mc_sc = tf.math.divide_no_nan(tf.abs(sct @ sc),sc_l_mat)
            mc_sc_no_diag = tf.linalg.set_diag(mc_sc, tf.zeros(shape=(100,64)))

            mc_sc_mean_sample = tf.reduce_mean(mc_sc,axis=[1,2])
            mc_sc_max_sample = tf.reduce_max(mc_sc_no_diag,axis=[1,2])

            if save_stat:
                stats_max.extend(mc_sc_max_sample.numpy())
                stats_mean.extend(mc_sc_mean_sample.numpy())

            #
            if save_stat:
                df_max = pd.DataFrame(stats_max, columns=['mc-max'])
                mean_mc_sc_max = tf.reduce_mean(stats_max)
                df_max = df_max.append(pd.Series({"mc-max":mean_mc_sc_max.numpy()}, index=df_max.columns, name='mean'))
                df_max.to_excel(save_dir + '/MC/' + mode + '_mc_max.xlsx')

                df_mean = pd.DataFrame(stats_mean, columns=['mc-mean'])
                mean_mc_sc_mean = tf.reduce_mean(stats_mean)
                df_mean = df_mean.append(pd.Series({"mc-mean":mean_mc_sc_mean.numpy()}, index=df_mean.columns, name='mean'))
                df_mean.to_excel(save_dir + '/MC/' + mode + '_mc_mean.xlsx')

        if save_imgs:
            #
            fname_w = 'heatmap_' + mode + '_mc_w.png'
            plt.imshow(mc_no_diag)
            plt.savefig(save_dir + '/' + fname_w)

            for sample_idx in range(0, 100):
                fname_sc = 'heatmap_' + mode + '_' + str(sample_idx) + '_mc_sc.png'
                plt.imshow(mc_sc_no_diag[sample_idx])
                plt.savefig(save_dir + '/Heatmap/' + fname_sc)


    elif en == 'count_avg_channel':
        all_count =[]
        if conf.dataset == 'CIFAR10_DVS':
            batch_num = 10
            img_num = 100
        elif conf.dataset == 'ImageNet':
            batch_num = 100
            img_num = 50
        else:
            batch_num = 100
            img_num = 100
        conf.n_conv1_spike_count = True
        for i in range(0, batch_num):
            batch_count = []
            result = model.evaluate(test_ds.skip(i).take(1), callbacks=callbacks_test)
            for layer in model.layers_w_neuron:
                layer_name = layer.name
                if layer_name in ['n_conv1','conv1_conv_n']:
                    spike_count = tf.reduce_sum(layer.act.spike_count_int,axis=[1,2])
                    for i in range(0,img_num):
                        non_zero = tf.math.count_nonzero(spike_count[i,:]).numpy()
                        non_zero = non_zero.astype(np.float32)
                        batch_count.append(non_zero)
                    all_count.append(batch_count)
        print('max : ', tf.reduce_max(all_count))
        print('min : ', tf.reduce_min(all_count))
        print('mean : ', tf.reduce_mean(all_count))
        print('std : ', tf.math.reduce_std(all_count))
        result = model.evaluate(test_ds, callbacks=callbacks_test)


    elif en =='alpha':
        result = model.evaluate(test_ds,callbacks=callbacks_test)
        mode = 'normal3'
        non_ind = [18, 31]
        all_ind = [3, 12, 13, 15, 17, 22, 23, 27, 28, 29, 30, 33, 39, 43, 44, 49, 50, 52, 61]
        bli_ind = [3, 12, 13, 15, 17, 22, 23, 27, 28, 29, 30, 33, 39, 43, 44, 49, 50, 52, 61]
        non=[]
        all=[]
        bli=[]
        bn = model.get_layer('bn_conv1')
        alpha = bn.beta/bn.gamma
        non_alpha = tf.gather(alpha,non_ind).numpy()
        all_alpha = tf.gather(alpha,all_ind).numpy()
        bli_alpha = tf.gather(alpha,bli_ind).numpy()
        non.extend(non_alpha)
        all.extend(all_alpha)
        bli.extend(bli_alpha)
        non=pd.DataFrame(non)
        all=pd.DataFrame(all)
        bli=pd.DataFrame(bli)
        non.to_excel('/home/ssparknas/alpha_value/VGG16_CIFAR10/'+mode+'_non.xlsx')
        all.to_excel('/home/ssparknas/alpha_value/VGG16_CIFAR10/'+mode+'_all.xlsx')
        bli.to_excel('/home/ssparknas/alpha_value/VGG16_CIFAR10/'+mode+'_bli.xlsx')

