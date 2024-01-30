
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


        df.to_excel(config.config_name+".xlsx")


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

        # move to prop.py postproc_batch_test()
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
    if True:

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
    if False:
    #if True:
        import keras
        from lib_snn import grad_cam
        import matplotlib.pyplot as plt

        from config_snn_training_WTA_SNN import mode

        save_imgs = False

        [imgs, labels], = test_ds.take(1)

        #sample_idx=0   #
        #sample_idx=1   #
        sample_idx=10   #
        #sample_idx=6   #
        #sample_idx=9   #



        #model_builder = keras.applications.vgg16.VGG16
        #last_conv_layer_name = "conv1"
        #last_conv_layer_name = "conv2"
        #last_conv_layer_name = "conv5_2"

        # the local path to our target image

        # display(Image(img_path))

        # prepare image
        #img_array = preprocess_input(grad_cam.get_image_array(img_path, size=img_size))

        # grad_cam
        for sample_idx in range(0,100):

            img = imgs[sample_idx]
            label = labels[sample_idx]
            img_array = tf.expand_dims(img, axis=0)

            model = keras.Model(inputs=model.input, outputs=model.output)
            model.layers[-1].activation = None
            preds = model.predict(img_array)
            fm = []
            layer_names = []


            figs_h, axes_h = plt.subplots(4, 4, figsize=(12,10))
            layer_idx = 0

            for layer in model.layers:
                if isinstance(layer, lib_snn.layers.Conv2D):
                    # print(layer.name)
                    last_conv_layer_name = layer.name
                    heatmap = grad_cam.make_gradcam_heatmap(img_array, model, last_conv_layer_name)
                    fm.append(heatmap)
                    layer_names.append(layer.name)

                    axes_h[layer_idx // 4, layer_idx % 4].matshow(heatmap)
                    layer_idx = layer_idx + 1

                    # generate class activation heatmap
                    #heatmap = grad_cam.make_gradcam_heatmap(img_array, model, last_conv_layer_name)

                    # display
                    #plt.matshow(heatmap)


            axes_h[layer_idx // 4, layer_idx % 4].matshow(img)

            if save_imgs:
                fname = 'grad_cam_'+mode+'_'+str(sample_idx)+'.png'
                plt.savefig('./result_figs/'+fname)
        #plt.show()


