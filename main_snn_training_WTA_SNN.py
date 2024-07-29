
import os

#
# configuration
from config_snn_training_WTA_SNN import config

#
import tensorflow as tf

# snn library
import lib_snn

#
import datasets
import callbacks

#
from tqdm import tqdm

#
from config import config
conf = config.flags

#
import pandas as pd
import numpy as np


# for loss_vis
from lib_snn import loss_vis
from keras import callbacks as keras_callbacks

import logging
logger = logging.getLogger()
old_level_logger = logger.level

########################################
# configuration
########################################
dist_strategy = lib_snn.utils.set_gpu()


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

    #if True:
    if False:
        if config.train:
            print('Train mode')

            model.summary()
            #train_steps_per_epoch = train_ds_num/batch_size
            train_epoch = config.flags.train_epoch
            init_epoch = config.init_epoch
            train_histories = model.fit(train_ds, epochs=train_epoch, steps_per_epoch=train_steps_per_epoch,
                                        initial_epoch=init_epoch, validation_data=valid_ds, callbacks=callbacks_train)
        else:
            print('Test mode')

            result = model.evaluate(test_ds, callbacks=callbacks_test)

    # plot kernel
    if False:
        from lib_snn.sim import glb_plot_kernel
        for layer in model.layers:
            if hasattr(layer,'kernel'):
                lib_snn.utils.plot_hist(glb_plot_kernel,layer.kernel,100,norm_fit=True)
                name = layer.name
                mean = tf.reduce_mean(layer.kernel)
                std = tf.math.reduce_std(layer.kernel)

                print('{:} - mean: {:e}, std: {:e} '.format(layer.name,mean,std))




    en_analysis = False
    #en_analysis = True

    if en_analysis:
        # analysis
        import tensorflow as tf
        import numpy as np
        import pandas as pd
        from absl import flags
        conf = flags.FLAGS

        from config_snn_training_WTA_SNN import mode

        save_dir= './result_spike_count_layer_test'
        os.makedirs(save_dir, exist_ok=True)

        #save_dir = os.path.join(save_dir_root,conf.model_dataset_name)

        #os.makedirs(save_dir, exist_ok=True)

        #
        result = model.evaluate(test_ds.take(1), callbacks=callbacks_test)


        #
        l_n = []
        l_sc = []
        for layer in model.layers_w_neuron:
            print(layer.name)
            l_n.append(layer.name)
            spike_count = layer.act.spike_count_int.numpy()
            #hist = tf.histogram_fixed_width(spike_count, [0, conf.time_step+1], nbins=conf.time_step+1)
            hist,_ = np.histogram(spike_count,bins=conf.time_step+1)
            print(hist)
            l_sc.append(hist)

        a_sc = np.vstack(l_sc).T

        df = pd.DataFrame({'name':l_n,'0':a_sc[0],'1':a_sc[1],'2':a_sc[2],'3':a_sc[3],'4':a_sc[4]})

        save_file_name=save_dir+'/'+'spike_count_'+config.model_dataset_name+'_'+mode+".xlsx"
        print('save file: ' + save_file_name)
        df.to_excel(save_file_name)


    #
    # visualization - activation
    #
    if False:
    #if True:
        #psp_mode = True
        psp_mode = False
        import keras
        import matplotlib.pyplot as plt

        # get a image
        imgs_labs, = test_ds.take(1)
        imgs = imgs_labs[0]
        img = imgs[0]
        img = tf.expand_dims(img, axis=0)


        layer_outputs = []
        layer_names = []
        for layer in model.layers:
            if isinstance(layer,lib_snn.activations.Activation):
                if psp_mode:
                    layer_outputs.append(layer.input)
                else:
                    layer_outputs.append(layer.output)
                layer_names.append(layer.name)

        act_model=keras.Model(inputs=model.input,outputs=layer_outputs)

        act = act_model.predict(img)

        plt.matshow(act[1][0,:,:,3])



    # visualization - feature map - spike count
    if False:
    #if True:
        import keras
        import matplotlib.pyplot as plt
        import numpy as np

        from config_snn_training_WTA_SNN import mode

        result = model.evaluate(test_ds.take(1), callbacks=callbacks_test)
        #result = model.evaluate(test_ds.take(10), callbacks=callbacks_test)
        #result = model.evaluate(test_ds, callbacks=callbacks_test)

        # move to proc.py postproc_batch_test()
        if False:
            fm = []
            layer_names = []

            for layer in model.layers_w_neuron:
                if isinstance(layer.act,lib_snn.neurons.Neuron):
                    fm.append(layer.act.spike_count_int)
                    layer_names.append(layer.name)

            #a = fm[13]
            #plt.matshow(a[0,:,:,0])

            images_per_row = 16
            img_idx = 0
            layer_idx = 0

            #display_grid_h = np.zeros((4,4))
            #plt.figure(figsize=(display_grid_h.shape[1], display_grid_h.shape[0]))

            plot_hist = False
            if plot_hist:
                figs_h, axes_h = plt.subplots(4, 4, figsize=(12,10))

            # only conv1
            fm = [fm[1]]
            layer_names = [layer_names[1]]

            #
            for img_idx in range(0,100):
                for layer_name, layer_fm in zip(layer_names,fm):
                    n_features = layer_fm.shape[-1]
                    size = layer_fm.shape[1]

                    n_cols = n_features // images_per_row
                    if n_cols == 0: # n_in
                        continue

                    if len(layer_fm.shape) == 2:    # fc_layers
                        continue

                    display_grid = np.zeros(((size+1)*n_cols-1,images_per_row*(size+1)-1))

                    #
                    for col in range(n_cols):
                        for row in range(images_per_row):
                            #
                            channel_index = col * images_per_row + row
                            channel_image = layer_fm[img_idx,:,:,channel_index].numpy().copy()

                            # normalization
                            if channel_image.sum() != 0:
                                channel_image -= channel_image.mean()
                                channel_image /= channel_image.std()
                                channel_image *= 64
                                channel_image += 128
                            channel_image = np.clip(channel_image,0,255).astype("uint8")

                            display_grid[
                                col * (size+1):(col+1)*size + col,
                                row * (size+1):(row+1)*size + row] = channel_image

                    #
                    scale = 1./size
                    plt.figure(figsize=(scale*display_grid.shape[1],
                                        scale*display_grid.shape[0]))
                    plt.title(layer_name)
                    plt.grid(False)
                    plt.axis("off")

                    plt.imshow(display_grid, aspect="auto", cmap="viridis")


                    # channel intensity
                    # image
                    stat_image= False
                    if stat_image:
                        # one image
                        channel_image = layer_fm[img_idx,:,:,:].numpy().copy()
                    else:
                        ## batch
                        channel_image = layer_fm.numpy().copy()

                    channel_intensity = tf.reduce_mean(channel_image,axis=[0,1])

                    #display_grid_h[layer_idx//4, layer_idx%4] = channel_intensity

                    if plot_hist:
                        axes_h[layer_idx//4,layer_idx%4].hist(tf.reshape(channel_intensity,shape=-1),bins=100)

                        ci_mean = tf.reduce_mean(channel_intensity)
                        ci_max = tf.reduce_max(channel_intensity)
                        ci_min = tf.reduce_min(channel_intensity)
                        ci_std = tf.math.reduce_std(channel_intensity)
                        ci_non_zeros = tf.math.count_nonzero(channel_intensity,dtype=tf.int32)
                        ci_non_zeros_r = ci_non_zeros/tf.math.reduce_prod(channel_intensity.shape)

                        print("{:}, mean:{:.3f}, max:{:.3f}, min:{:.3f}, std:{:.4f}, nonz:{:.3f}"
                              .format(layer_fm.name,ci_mean,ci_max,ci_min,ci_std,ci_non_zeros_r))

                        layer_idx += 1

                fname = mode+'_'+str(img_idx)+'.png'
                plt.savefig('./result_fig_fm_sc/'+fname)


            #plt.show()


    # XAI - integrated gradients
    # batch size should be m_steps+1
    #if False:
    #if True:

        import matplotlib.pyplot as plt

        from config_snn_training_WTA_SNN import mode

        img_save=True

        [imgs, labels], = test_ds.take(1)




        #sample_idx=0   #
        sample_idx=11   #

        #sample_idx=6   # horse -> good
        #sample_idx=7   # airplane
        #sample_idx=8   # cat or dog
        #sample_idx=10   # horse -> good example
        #sample_idx=30   # ? -> good
        #sample_idx=40   # -> good
        #sample_idx=50   # -> good


        for sample_idx in range(0,100):
            img = imgs[sample_idx]
            label = labels[sample_idx]

            baseline = tf.random.uniform(shape=img.shape,minval=0,maxval=1)


            m_steps = 99
            #m_steps = 50
            #label_decoded=386
            label_decoded = tf.argmax(label)

            #image_processed = tf.expand_dims(img,axis=0)
            img_exp = tf.expand_dims(img,axis=0)
            ig_attribution = lib_snn.xai.integrated_gradients(model=model,
                                                              baseline=baseline,
                                                              images=img,
                                                              target_class_idxs=label_decoded,
                                                              m_steps=m_steps)

            #_ = lib_snn.xai.plot_image_attributions(baseline,img_processed,ig_attribution)


            # 5. Get the gradients of the last layer for the predicted label
            grads = lib_snn.integrated_gradients.get_gradients(model,img_exp,top_pred_idx=label_decoded)

            #
            vis = lib_snn.integrated_gradients.GradVisualizer()

            vis.visualize(
                image=img,
                gradients=grads[0].numpy(),
                integrated_gradients=ig_attribution.numpy(),
                clip_above_percentile=99,
                clip_below_percentile=0
            )


            vis.visualize(
                image=img,
                gradients=grads[0].numpy(),
                integrated_gradients=ig_attribution.numpy(),
                clip_above_percentile=95,
                clip_below_percentile=28,
                morphological_cleanup=True,
                outlines=True
            )

            if img_save:
                fname = mode + '_' + str(sample_idx) + '.png'
                plt.savefig('./result_fig_integ_grad/' + fname)

        #plt.show()


    # adversarial attack - FGSM


    # grad_cam, vanilla gradient
    #if False:
    if True:
        logger.setLevel(100)
        import keras
        from lib_snn import grad_cam
        import matplotlib.pyplot as plt

        from config_snn_training_WTA_SNN import mode

        save_imgs = False
        #save_imgs = True

        #show_imgs = True
        show_imgs = False

        save_stat = True


        # normalize heatmap with firing rate proportion of layers
        #f_norm_fr_layer = True
        f_norm_fr_layer = False


        batch_size = 100
        input_shape = (32, 32, 3)
        classes = 10


        if conf.model=='VGG16':
            subplot_w, subplot_h = 4, 4
        elif conf.model=='ResNet20':
            subplot_w, subplot_h = 4, 5


        #model_builder = keras.applications.vgg16.VGG16
        #last_conv_layer_name = "conv1"

        # the local path to our target image

        # display(Image(img_path))
        # prepare image
        #img_array = preprocess_input(grad_cam.get_image_array(img_path, size=img_size))

        # save directory - saliency map
        # plt.savefig('./result_fig_ig_neuron_input/' + fname)
        # plt.savefig('./result_fig_ig_syn_out/' + fname)

        # plt.savefig('./result_fig_grad_cam/'+fname)
        # plt.savefig('./result_fig_syn_out/' + fname)
        # plt.savefig('./result_fig_syn_out_wo_norm/' + fname)    # re

        # plt.savefig('./result_fig_grad_cam_neuron/' + fname)
        # plt.savefig('./result_fig_grad_cam_neuron_x_act/' + fname)     # x
        # plt.savefig('./result_fig_grad_cam_neuron_mean_t/' + fname)
        # plt.savefig('./result_fig_grad_cam_syn_out/' + fname)
        # plt.savefig('./result_fig_grad_cam_syn_out_mean_t/' + fname)
        # plt.savefig('./result_fig_grad_cam_neuron_input/' + fname)
        #
        #save_dir = './result_smap_ga_neuron-mean-t_all'
        #save_dir = './result_smap_ga_neuron'
        #save_dir = './result_smap_ga_neuron_all'
        #save_dir = './result_smap_ga_neuron_scnorm_all'
        #save_dir = './result_smap_ig_neuron-mean-t'
        #save_dir = './result_smap_ig_neuron'
        #save_dir = './result_smap_ig_neuron_all'
        #save_dir = './result_smap_ig_neuron_scnorm_all_new'

        #save_dir = './result_smap_ig_neuron_scnorm_all_new'


        # 240725
        #save_dir = './result_smap_internal_influence'
        #save_dir = './result_amap_cond_n_scnorm'
        save_dir = './result_amap_cond_n'
        #save_dir = './result_amap_cond_n_norm_fr'
        #save_dir = './result_amap_cond_n_scnorm_norm_fr_all'
        #save_dir = './result_amap_cond_n_scnorm_all'
        #save_dir = './result_amap_cond_n_scnorm_ng_all'


        os.makedirs(save_dir,exist_ok=True)


        #
        save_idx=0

        #
        batch_index = conf.sm_batch_index
        #for batch in test_ds:
        #for batch_idx in tqdm(range(0,2)):
        #for batch_idx in tqdm(range(0,100)):
        #for batch_idx in tqdm(range(10, 20)):
        for batch_idx in tqdm(range(batch_index,batch_index+1)):

            if save_stat:
                stats_sample = []
                stats_mean = []
                stats_max = []
                stats_min = []
                stats_std = []
                stats_cv = []

            test_ds_a_batch = test_ds.skip(batch_idx).take(1)
            [imgs, labels], = test_ds_a_batch

            # grad_cam
            for sample_idx in range(0,100):
            #for sample_idx in [0]:
            #for sample_idx in [0,1,2,3,4,5,6,7,31,34]:
            #for sample_idx in [0, 1, 2]:

                sc_layer = []
                heatmap_layers = []

                stats = []

                img = imgs[sample_idx]
                label = labels[sample_idx]
                img_one = tf.expand_dims(img, axis=0)

                interpolated_img = tf.range(0.01,1.01,delta=0.01)
                interpolated_img = tf.expand_dims(interpolated_img,axis=-1)
                interpolated_img = tf.expand_dims(interpolated_img,axis=-1)
                interpolated_img = tf.expand_dims(interpolated_img,axis=-1)
                img_array = tf.expand_dims(img, axis=0)
                img_array = tf.multiply(interpolated_img,img_array)

                #model = keras.Model(inputs=model.input, outputs=model.output)
                #model = lib_snn.model.Model(model.inputs, model.output, batch_size, input_shape, classes, conf)

                model.layers[-1].activation = None
                #preds = model.predict(img_one)
                #preds = model.predict(img_one)
                #preds = model(img_one)
                #preds = model(img_array)
                model.init_snn()
                model.reset()
                preds = model.predict(test_ds_a_batch)
                model.reset()
                pred_index = tf.argmax(preds[-1])

                #fm = []
                #layer_names = []



                #
                #neuron_mode = False
                neuron_mode = True

                def condition(layer, neuron_mode):
                    if neuron_mode:
                        return hasattr(layer, 'act') and isinstance(layer.act, lib_snn.neurons.Neuron) and layer.name != 'n_in' \
                                and len(layer.act.out.read(0).shape) == 4
                    else:
                        return isinstance(layer, lib_snn.layers.Conv2D)

                for layer in model.layers:
                    if condition(layer,neuron_mode):
                        # print(layer.name)
                        model.reset()
                        last_conv_layer_name = layer.name
                        heatmap = grad_cam.make_gradcam_heatmap_snn(img_array, model, last_conv_layer_name, neuron_mode,
                                                                    pred_index=pred_index)
                        #fm.append(heatmap)
                        #layer_names.append(layer.name)

                        if show_imgs or save_imgs:
                            heatmap_layers.append(heatmap)

                        # print
                        if save_stat:
                            mean_heatmap = tf.reduce_mean(heatmap).numpy()
                            max_heatmap = tf.reduce_max(heatmap).numpy()
                            min_heatmap = tf.reduce_min(heatmap).numpy()
                            std_heatmap = tf.math.reduce_std(heatmap).numpy()
                            cv_heatmap = std_heatmap/mean_heatmap
                            #print("{:} - mean: {:.3e}, max: {:.3e}, min: {:.3e}, std : {:.3e}".format(last_conv_layer_name, mean_heatmap, max_heatmap, min_heatmap, std_heatmap))
                            stat = [mean_heatmap,max_heatmap,min_heatmap,std_heatmap,cv_heatmap]
                            stats.append(stat)

                        sc_layer.append(tf.reduce_sum(layer.act.spike_count).numpy())

                #sc_layer_total = tf.reduce_sum(sc_layer)
                norm_fr_layer = tf.keras.utils.normalize(sc_layer,order=1)[0]

                # plot heatmaps
                if show_imgs or save_imgs:
                    figs_h, axes_h = plt.subplots(subplot_h, subplot_w, figsize=(12,10))
                    layer_idx = 0

                    for layer in model.layers:
                        if condition(layer,neuron_mode):

                            axe = axes_h[layer_idx // subplot_w, layer_idx % subplot_w]
                            heatmap=heatmap_layers[layer_idx]
                            if f_norm_fr_layer:
                                heatmap = heatmap*norm_fr_layer[layer_idx]
                            hm = axe.matshow(heatmap)
                            figs_h.colorbar(hm, ax=axe)
                            layer_idx = layer_idx + 1


                if show_imgs or save_imgs:
                    axes_h[layer_idx // subplot_w, layer_idx % subplot_w].matshow(img)

                if save_imgs:
                    fname = 'heatmap_'+mode+'_'+str(sample_idx+batch_idx*100)+'.png'
                    plt.savefig(save_dir + '/' + fname)


                if save_stat:
                    mean_l = [stat[0] for stat in stats]
                    max_l = [stat[1] for stat in stats]
                    min_l = [stat[2] for stat in stats]
                    std_l = [stat[3] for stat in stats]
                    cv_l = [stat[4] for stat in stats]

                    if f_norm_fr_layer:
                    #if False:
                        mean_l = tf.math.multiply(mean_l,norm_fr_layer).numpy()
                        max_l = tf.math.multiply(max_l,norm_fr_layer).numpy()
                        min_l = tf.math.multiply(min_l,norm_fr_layer).numpy()
                        std_l = tf.math.multiply(std_l,norm_fr_layer).numpy()
                        cv_l = tf.math.multiply(cv_l,norm_fr_layer).numpy()

                    mean_layers = tf.reduce_mean(mean_l).numpy()
                    max_layers = tf.reduce_mean(max_l).numpy()
                    min_layers = tf.reduce_mean(min_l).numpy()
                    std_layers = tf.reduce_mean(std_l).numpy()
                    cv_layers = tf.reduce_mean(cv_l).numpy()
                    stats_sample.append([mean_layers,max_layers,min_layers,std_layers,cv_layers])

                    stats_mean.append(mean_l)
                    stats_max.append(max_l)
                    stats_min.append(min_l)
                    stats_std.append(std_l)
                    stats_cv.append(cv_l)


            if save_stat:

                df = pd.DataFrame(stats_sample,columns=['mean','max','min','std','cv'])
                df.to_excel(save_dir+'/'+mode+'_b-'+str(batch_idx)+'.xlsx')

                df = pd.DataFrame(stats_mean)
                df.to_excel(save_dir+'/'+mode+'_b-'+str(batch_idx)+'_mean.xlsx')

                df = pd.DataFrame(stats_min)
                df.to_excel(save_dir+'/'+mode+'_b-'+str(batch_idx)+'_min.xlsx')

                df = pd.DataFrame(stats_max)
                df.to_excel(save_dir+'/'+mode+'_b-'+str(batch_idx)+'_max.xlsx')

                df = pd.DataFrame(stats_std)
                df.to_excel(save_dir+'/'+mode+'_b-'+str(batch_idx)+'_std.xlsx')

                df = pd.DataFrame(stats_cv)
                df.to_excel(save_dir+'/'+mode+'_b-'+str(batch_idx)+'_cv.xlsx')

        #
        logger.setLevel(old_level_logger)



    # mutual coherence
    # cross correlation
    if False:
    #if True:
        logger.setLevel(100)
        import keras
        from lib_snn import grad_cam
        import matplotlib.pyplot as plt

        from config_snn_training_WTA_SNN import mode

        save_imgs = False
        #save_imgs = True

        #show_imgs = True
        show_imgs = False

        save_stat = True
        #save_stat = False

        batch_size = 100
        input_shape = (32, 32, 3)
        classes = 10


        # save directory
        if save_imgs or save_stat:
            #save_dir = './result_encoding_layer_mc'
            save_dir = './result_layer_mc'
            os.makedirs(save_dir, exist_ok=True)


        # mc - kernel
        if False:
        #if True:
            layers_kernel_mean = []
            layers_kernel_max = []
            layers_kernel_name = []

            for layer in model.layers:
                if isinstance(layer, lib_snn.layers.Conv2D):
                    #w = model.get_layer('conv1').kernel
                    w = layer.kernel
                    #wc = tf.reshape(w, shape=[3 * 3 * 3, 64])
                    wc = tf.reshape(w, shape=[-1,w.shape[-1]])
                    wc_l2 = tf.sqrt(tf.reduce_sum(tf.square(wc), axis=[0]))
                    wc_l = wc_l2
                    #wc_l = wc_l1
                    wct = tf.transpose(wc)
                    wc_l = tf.expand_dims(wc_l, -1)
                    wc_l_mat = wc_l @ tf.transpose(wc_l)
                    mc = tf.abs(wct @ wc) / wc_l_mat
                    mc_no_diag = tf.linalg.set_diag(mc, tf.zeros(shape=(w.shape[-1])))
                    mc_mean = tf.reduce_mean(mc_no_diag).numpy()
                    mc_max = tf.reduce_max(mc_no_diag).numpy()

                    #print(layer.name)
                    #print(tf.reduce_mean(mc_no_diag))
                    #print(tf.reduce_max(mc_no_diag))
                    #print()

                    layers_kernel_mean.append(mc_mean)
                    layers_kernel_max.append(mc_max)
                    layers_kernel_name.append(layer.name)

            df_k = pd.DataFrame(columns=layers_kernel_name)
            df_k.loc[0] = layers_kernel_mean
            df_k.loc[1] = layers_kernel_max
            df_k.to_excel(save_dir + '/' + mode + '_mc_kernel.xlsx')


        #batch_index = conf.sm_batch_index
        #batch_index = 0

        #
        range_batch = range(0,100)
        len_range_batch = len(range_batch)

        batch_size = conf.batch_size
        num_entry = batch_size*len_range_batch

        layers = []
        layers_name = []

        for layer in model.layers:
            if isinstance(layer, lib_snn.activations.Activation):
                if isinstance(layer.act, lib_snn.neurons.Neuron):
                    layers.append(layer)
                    layers_name.append(layer.name)

        len_layers = len(layers)


        df_max = pd.DataFrame(np.zeros((num_entry,len_layers)), columns=layers_name)
        df_mean = pd.DataFrame(np.zeros((num_entry,len_layers)), columns=layers_name)


        #
        for batch_idx in tqdm(range_batch):
        #for batch_idx in tqdm(range(0,1)):
        #for batch_idx in tqdm(range(0,100)):
        # for batch_idx in tqdm(range(10, 20)):
        #for batch_idx in tqdm(range(batch_index, batch_index + 1)):

            test_ds_a_batch = test_ds.skip(batch_idx).take(1)
            result = model.evaluate(test_ds_a_batch, callbacks=callbacks_test)


            # mc - encoded spike (spike feature map)
            #for layer in model.layers:
            #    if isinstance(layer, lib_snn.activations.Activation):
            #        if isinstance(layer.act, lib_snn.neurons.Neuron):
            for layer in layers:
                stats_max = []
                stats_mean = []

                #sc = model.get_layer('n_conv1').act.spike_count
                _sc = layer.act.spike_count
                #sc = tf.reshape(sc, shape=[100,32*32, 64])
                sc = tf.reshape(_sc, shape=[100,-1, _sc.shape[-1]])
                sc_l2 = tf.sqrt(tf.reduce_sum(tf.square(sc), axis=[1]))
                sc_l = sc_l2
                sc_l = tf.expand_dims(sc_l,-1)
                sct = tf.transpose(sc,perm=[0,2,1])
                sc_l_mat = sc_l @ tf.transpose(sc_l,perm=[0,2,1])
                mc_sc = tf.math.divide_no_nan(tf.abs(sct @ sc),sc_l_mat)
                mc_sc_no_diag = tf.linalg.set_diag(mc_sc, tf.zeros(shape=(100,sc.shape[-1])))

                #mc_sc_mean_sample = tf.reduce_mean(mc_sc_no_diag,axis=[1,2])
                mc_sc_mean_sample = tf.reduce_sum(mc_sc_no_diag,axis=[1,2])
                mc_sc_no_diag_non_zero = tf.cast(tf.math.count_nonzero(mc_sc_no_diag,axis=[1,2]),tf.float32)
                mc_sc_mean_sample = mc_sc_mean_sample / mc_sc_no_diag_non_zero
                mc_sc_max_sample = tf.reduce_max(mc_sc_no_diag,axis=[1,2])

                #print(layer.name)
                #print(tf.reduce_mean(mc_sc_no_diag))
                #print(tf.reduce_max(mc_sc_no_diag))
                #print()

                if save_stat:
                    stats_max.extend(mc_sc_max_sample.numpy())
                    stats_mean.extend(mc_sc_mean_sample.numpy())

                #
                if save_stat:
                    #df = pd.DataFrame(stats_max, columns=['mc-max'])
                    #df = pd.DataFrame(stats_max, columns=[layer.name])
                    #df[layer.name]=stats_max
                    df_max.loc[batch_idx*batch_size:(batch_idx+1)*batch_size-1,layer.name]=stats_max
                    df_mean.loc[batch_idx*batch_size:(batch_idx+1)*batch_size-1,layer.name]=stats_mean

                    #mean_mc_sc_max = tf.reduce_mean(stats_max)
                    #df = df.append(pd.Series({"mc-max":mean_mc_sc_max.numpy()}, index=df.columns, name='mean'))
                    #df.to_excel(save_dir + '/' + mode + '_mc.xlsx')

                if save_imgs:
                    #
                    fname_w = 'heatmap_' + mode + '_mc_w.png'
                    plt.imshow(mc_no_diag)
                    plt.savefig(save_dir + '/' + fname_w)

                    for sample_idx in range(0, 100):
                        fname_sc = 'heatmap_' + mode + '_' + str(sample_idx) + '_mc_sc.png'
                        plt.imshow(mc_sc_no_diag[sample_idx])
                        plt.savefig(save_dir + '/' + fname_sc)

        #
        df_max.loc['mean']=df_max.mean()
        df_mean.loc['mean']=df_mean.mean()

        #
        df_max.to_excel(save_dir + '/' + mode + '_mc_max.xlsx')
        df_mean.to_excel(save_dir + '/' + mode + '_mc_mean.xlsx')



        assert False
        if False:

            # save directory - saliency map
            save_dir = './result_smap_ig_neuron_scnorm_all_new'

            os.makedirs(save_dir, exist_ok=True)

            batch_index = conf.sm_batch_index
            # for batch in test_ds:
            # for batch_idx in tqdm(range(0,2)):
            # for batch_idx in tqdm(range(0,100)):
            # for batch_idx in tqdm(range(10, 20)):
            for batch_idx in tqdm(range(batch_index, batch_index + 1)):

                if save_stat:
                    stats_sample = []
                    stats_mean = []
                    stats_max = []
                    stats_min = []
                    stats_std = []

                test_ds_a_batch = test_ds.skip(batch_idx).take(1)
                [imgs, labels], = test_ds_a_batch

                # grad_cam
                for sample_idx in range(0, 100):
                    # for sample_idx in [2]:
                    # for sample_idx in [0,1,2,3,4,5,6,7,31,34]:
                    # for sample_idx in [0, 1, 2]:

                    stats = []

                    img = imgs[sample_idx]
                    label = labels[sample_idx]
                    img_one = tf.expand_dims(img, axis=0)

                    interpolated_img = tf.range(0.01, 1.01, delta=0.01)
                    interpolated_img = tf.expand_dims(interpolated_img, axis=-1)
                    interpolated_img = tf.expand_dims(interpolated_img, axis=-1)
                    interpolated_img = tf.expand_dims(interpolated_img, axis=-1)
                    img_array = tf.expand_dims(img, axis=0)
                    img_array = tf.multiply(interpolated_img, img_array)

                    # model = keras.Model(inputs=model.input, outputs=model.output)
                    # model = lib_snn.model.Model(model.inputs, model.output, batch_size, input_shape, classes, conf)

                    model.layers[-1].activation = None
                    # preds = model.predict(img_one)
                    # preds = model.predict(img_one)
                    # preds = model(img_one)
                    # preds = model(img_array)
                    model.init_snn()
                    model.reset()
                    preds = model.predict(test_ds_a_batch)
                    model.reset()
                    pred_index = tf.argmax(preds[-1])

                    # fm = []
                    # layer_names = []

                    if show_imgs:
                        figs_h, axes_h = plt.subplots(4, 4, figsize=(12, 10))
                        layer_idx = 0

                    #
                    # neuron_mode = False
                    neuron_mode = True


                    def condition(layer, neuron_mode):
                        if neuron_mode:
                            return hasattr(layer, 'act') and isinstance(layer.act,
                                                                        lib_snn.neurons.Neuron) and layer.name != 'n_in' \
                                   and len(layer.act.out.read(0).shape) == 4
                        else:
                            return isinstance(layer, lib_snn.layers.Conv2D)


                    for layer in model.layers:
                        if condition(layer, neuron_mode):
                            # print(layer.name)
                            model.reset()
                            last_conv_layer_name = layer.name
                            heatmap = grad_cam.make_gradcam_heatmap_snn(img_array, model, last_conv_layer_name, neuron_mode,
                                                                        pred_index=pred_index)
                            # fm.append(heatmap)
                            # layer_names.append(layer.name)

                            if show_imgs:
                                axe = axes_h[layer_idx // 4, layer_idx % 4]
                                hm = axe.matshow(heatmap)
                                figs_h.colorbar(hm, ax=axe)
                                layer_idx = layer_idx + 1

                            # print
                            if save_stat:
                                mean_heatmap = tf.reduce_mean(heatmap).numpy()
                                max_heatmap = tf.reduce_max(heatmap).numpy()
                                min_heatmap = tf.reduce_min(heatmap).numpy()
                                std_heatmap = tf.math.reduce_std(heatmap).numpy()
                                # print("{:} - mean: {:.3e}, max: {:.3e}, min: {:.3e}, std : {:.3e}".format(last_conv_layer_name, mean_heatmap, max_heatmap, min_heatmap, std_heatmap))
                                stat = [mean_heatmap, max_heatmap, min_heatmap, std_heatmap]
                                stats.append(stat)

                        if show_imgs:
                            axes_h[layer_idx // 4, layer_idx % 4].matshow(img)

                    if save_imgs:
                        fname = 'heatmap_' + mode + '_' + str(sample_idx) + '.png'
                        plt.savefig(save_dir + '/' + fname)

                    if save_stat:
                        mean_l = [stat[0] for stat in stats]
                        max_l = [stat[1] for stat in stats]
                        min_l = [stat[2] for stat in stats]
                        std_l = [stat[3] for stat in stats]

                        mean_layers = tf.reduce_mean(mean_l).numpy()
                        max_layers = tf.reduce_mean(max_l).numpy()
                        min_layers = tf.reduce_mean(min_l).numpy()
                        std_layers = tf.reduce_mean(std_l).numpy()
                        stats_sample.append([mean_layers, max_layers, min_layers, std_layers])

                        stats_mean.append(mean_l)
                        stats_max.append(max_l)
                        stats_min.append(min_l)
                        stats_std.append(std_l)

                if save_stat:
                    df = pd.DataFrame(stats_sample, columns=['mean', 'max', 'min', 'std'])
                    df.to_excel(save_dir + '/' + mode + '_b-' + str(batch_idx) + '.xlsx')

                    df = pd.DataFrame(stats_mean)
                    df.to_excel(save_dir + '/' + mode + '_b-' + str(batch_idx) + '_mean.xlsx')

                    df = pd.DataFrame(stats_min)
                    df.to_excel(save_dir + '/' + mode + '_b-' + str(batch_idx) + '_min.xlsx')

                    df = pd.DataFrame(stats_max)
                    df.to_excel(save_dir + '/' + mode + '_b-' + str(batch_idx) + '_max.xlsx')

                    df = pd.DataFrame(stats_std)
                    df.to_excel(save_dir + '/' + mode + '_b-' + str(batch_idx) + '_std.xlsx')

            #
            logger.setLevel(old_level_logger)



    # Loss landscape
    #if True:
    if False:
        if False:
            training_hist_w = [model.get_weights()]

            collect_weights = keras_callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: training_hist_w.append(model.get_weights()) if (epoch%1)==0 else (None)
            )
            callbacks_train.append(collect_weights)


            model.summary()
            #train_steps_per_epoch = train_ds_num/batch_size
            train_epoch = config.flags.train_epoch
            init_epoch = config.init_epoch
            train_histories = model.fit(train_ds, epochs=train_epoch, steps_per_epoch=train_steps_per_epoch,
                                        initial_epoch=init_epoch, validation_data=valid_ds, callbacks=callbacks_train)


            #
            train_batch, = train_ds.take(1)
            x = train_batch[0]
            y = train_batch[1]

            #
            pcoords = loss_vis.PCACoordinates(training_hist_w)
            loss_surface = loss_vis.LossSurface(model,x,y)
            #loss_surface = loss_vis.LossSurface(model,train_ds)
            #loss_surface.compile(points=30,coords=pcoords,range=0.4)
            loss_surface.compile(points=10,coords=pcoords,range=0.1)
            #
            ax = loss_surface.plot(dpi=150)
            loss_vis.plot_training_path(pcoords, training_hist_w, ax)

            plt.show()


        path_w_root = './models_ckpt_WTA-SNN_e10/VGG16_AP_CIFAR10/'

        # normal
        #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s/'
        # SIM-S
        #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-nwta-sm-0.24_4/'
        # SIM-A
        #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-nwta-sm-2e-05_4/'
        # WTA-1
        path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-sm-3e-06_4/'

        #
        path_w = path_w_root+path_model

        for i in range(10, 301, 10):
            trained_epoch = 'ep-'+f"{i:04}"
            model.load_weights(path_w+trained_epoch+'.hdf5')


            if i==10:
                train_hist_w=[model.get_weights()]
            else:
                train_hist_w.append(model.get_weights())

        #
        train_batch, = train_ds.take(1)
        x = train_batch[0]
        y = train_batch[1]


        #
        pcoords = loss_vis.PCACoordinates(train_hist_w)
        loss_surface = loss_vis.LossSurface(model,x,y)
        #loss_surface = loss_vis.LossSurface(model,train_ds)
        #loss_surface.compile(points=30,coords=pcoords,range=0.4)
        loss_surface.compile(points=30,coords=pcoords,range=0.1)
        #
        ax = loss_surface.plot(dpi=150)
        loss_vis.plot_training_path(pcoords, train_hist_w, ax)

        plt.show()



    # feature - channels
    # implicit inhibition
    if False:
    #if True:
        logger.setLevel(100)
        import keras
        import matplotlib.pyplot as plt

        from config_snn_training_WTA_SNN import mode

        save_imgs = False
        #save_imgs = True

        #show_imgs = True
        show_imgs = False

        save_stat = True
        #save_stat = False

        batch_size = 100
        input_shape = (32, 32, 3)
        classes = 10


        # save directory
        if save_imgs or save_stat:
            save_dir = './result_feature_ch'
            os.makedirs(save_dir, exist_ok=True)

        #
        range_batch = range(0,100)
        len_range_batch = len(range_batch)

        batch_size = conf.batch_size
        num_entry = batch_size*len_range_batch

        layers = []
        layers_name = []

        for layer in model.layers:
            if isinstance(layer, lib_snn.activations.Activation):
                if isinstance(layer.act, lib_snn.neurons.Neuron):
                    layers.append(layer)
                    layers_name.append(layer.name)

        len_layers = len(layers)


        df_feat = pd.DataFrame(np.zeros((num_entry,len_layers)), columns=layers_name)
        df_feat_std = pd.DataFrame(np.zeros((num_entry,len_layers)), columns=layers_name)
        df_feat_mean = pd.DataFrame(np.zeros((num_entry,len_layers)), columns=layers_name)


        #
        for batch_idx in tqdm(range_batch):
            #for batch_idx in tqdm(range(0,1)):
            #for batch_idx in tqdm(range(0,100)):
            # for batch_idx in tqdm(range(10, 20)):
            #for batch_idx in tqdm(range(batch_index, batch_index + 1)):

            test_ds_a_batch = test_ds.skip(batch_idx).take(1)
            result = model.evaluate(test_ds_a_batch, callbacks=callbacks_test)


            # mc - encoded spike (spike feature map)
            #for layer in model.layers:
            #    if isinstance(layer, lib_snn.activations.Activation):
            #        if isinstance(layer.act, lib_snn.neurons.Neuron):
            for layer in layers:
                list_feature_nonz = []
                list_feature_std = []
                list_feature_mean = []

                #sc = model.get_layer('n_conv1').act.spike_count
                _sc = layer.act.spike_count
                #sc = tf.reshape(sc, shape=[100,32*32, 64])
                sc = tf.reshape(_sc, shape=[100,-1, _sc.shape[-1]])

                feature_nonz = tf.math.count_nonzero(tf.reduce_sum(sc,axis=[1]),axis=[1])
                num_feature = sc.shape[-1]
                feature_nonz = feature_nonz/num_feature
                feature_nonz = tf.expand_dims(feature_nonz,-1)

                feature_std = tf.math.reduce_std(sc,axis=[1])
                feature_std = tf.reduce_sum(feature_std,axis=[1])/tf.cast(tf.math.count_nonzero(feature_std,axis=[1]),tf.float32)

                feature_mean = tf.reduce_sum(sc,axis=[1])
                feature_mean = tf.reduce_sum(feature_mean,axis=[1])/tf.cast(tf.math.count_nonzero(feature_mean,axis=[1]),tf.float32)

                if save_stat:
                    list_feature_nonz.extend(feature_nonz.numpy())
                    list_feature_std.extend(feature_std.numpy())
                    list_feature_mean.extend(feature_mean.numpy())

                    df_feat.loc[batch_idx*batch_size:(batch_idx+1)*batch_size-1,layer.name]=list_feature_nonz
                    df_feat_std.loc[batch_idx*batch_size:(batch_idx+1)*batch_size-1,layer.name]=list_feature_std
                    df_feat_mean.loc[batch_idx*batch_size:(batch_idx+1)*batch_size-1,layer.name]=list_feature_mean

                if save_imgs:
                    #
                    fname_w = 'heatmap_' + mode + '_mc_w.png'
                    plt.imshow(mc_no_diag)
                    plt.savefig(save_dir + '/' + fname_w)

                    for sample_idx in range(0, 100):
                        fname_sc = 'heatmap_' + mode + '_' + str(sample_idx) + '_mc_sc.png'
                        plt.imshow(mc_sc_no_diag[sample_idx])
                        plt.savefig(save_dir + '/' + fname_sc)

        #
        df_feat.loc['mean']=df_feat.mean()
        df_feat_std.loc['mean']=df_feat_std.mean()
        df_feat_mean.loc['mean']=df_feat_mean.mean()

        #
        df_feat.to_excel(save_dir + '/nonz_'+conf.model+'_' + mode + '.xlsx')
        df_feat_std.to_excel(save_dir + '/std_'+conf.model+'_' + mode + '.xlsx')
        df_feat_mean.to_excel(save_dir + '/mean_'+conf.model+'_' + mode + '.xlsx')
