
#
# configuration
from config_snn_training_WTA_SNN import config


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
        spike_count = layer.act.spike_count.numpy()
        #hist = tf.histogram_fixed_width(spike_count, [0, conf.time_step+1], nbins=conf.time_step+1)
        hist,_ = np.histogram(spike_count,bins=conf.time_step+1)
        print(hist)
        l_sc.append(hist)

    a_sc = np.vstack(l_sc).T

    df = pd.DataFrame({'name':l_n,'0':a_sc[0],'1':a_sc[1],'2':a_sc[2],'3':a_sc[3],'4':a_sc[4]})


    df.to_excel(config.config_name+".xlsx")