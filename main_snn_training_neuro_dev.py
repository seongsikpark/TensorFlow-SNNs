
#
# configuration
from config_snn_training_neuro_dev import config


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

    #
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


        # loss surface
        if True:
            # for loss_vis
            from lib_snn import loss_vis
            import matplotlib.pyplot as plt

            #path_w_root = './models_ckpt_WTA-SNN_e10/VGG16_AP_CIFAR10/'
            path_w_root = './models_ckpt_e10/'

            # normal
            #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s/'
            # SIM-S
            #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-nwta-sm-0.24_4/'
            # SIM-A
            #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-nwta-sm-2e-05_4/'
            # WTA-1
            #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-sm-3e-06_4/'

            # ResNet20 - rtz
            path_model='ResNet20_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-z/'

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
            loss_surface.compile(points=30,coords=pcoords,range=0.15)
            #
            ax = loss_surface.plot(dpi=150)
            loss_vis.plot_training_path(pcoords, train_hist_w, ax)

            plt.show()
            plt.savefig('loss surface.pdf')
        # loss surface
        if True:
            # for loss_vis
            from lib_snn import loss_vis
            import matplotlib.pyplot as plt

            #path_w_root = './models_ckpt_WTA-SNN_e10/VGG16_AP_CIFAR10/'
            path_w_root = './models_ckpt_e10/'

            # normal
            #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s/'
            # SIM-S
            #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-nwta-sm-0.24_4/'
            # SIM-A
            #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-nwta-sm-2e-05_4/'
            # WTA-1
            #path_model='ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-sm-3e-06_4/'

            # VGG16 C10 rtz - proposed method
            path_model='ResNet20_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-z/'

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
            loss_surface.compile(points=30,coords=pcoords,range=0.15)
            #
            ax = loss_surface.plot(dpi=150)
            loss_vis.plot_training_path(pcoords, train_hist_w, ax)

            plt.show()
            plt.savefig('loss surface.pdf')
