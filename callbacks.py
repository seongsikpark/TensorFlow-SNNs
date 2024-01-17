
import lib_snn
from config import config
conf = config.flags

import tensorflow as tf


def callbacks_snn_train(model,train_ds_num,valid_ds,test_ds_num):

    monitor_cri = config.monitor_cri
    train = config.train
    load_model = config.load_model

    filepath_save = config.filepath_save
    path_tensorboard = config.path_tensorboard



    #if train and load_model and (not f_hp_tune_train):
    if train and load_model:
        print('Evaluate pretrained model')
        assert monitor_cri == 'val_acc', 'currently only consider monitor criterion - val_acc'
        result = model.evaluate(valid_ds)
        idx_monitor_cri = model.metrics_names.index('acc')
        best = result[idx_monitor_cri]
        print('previous best result - {}'.format(best))
    else:
        best = None

    save_freq=conf.save_model_freq_epoch
    if save_freq < 0:
        save_freq = 'epoch'


    # model checkpoint save and resume
    cb_model_checkpoint = lib_snn.callbacks.ModelCheckpointResume(
        filepath=filepath_save + '/ep-{epoch:04d}.hdf5',
        save_weight_only=True,
        #save_best_only=True,
        save_best_only=conf.save_best_model_only,
        save_freq=save_freq,
        monitor=monitor_cri,
        verbose=1,
        best=best,
        log_dir=path_tensorboard,
        # tensorboard_writer=cb_tensorboard._writers['train']
    )
    cb_manage_saved_model = lib_snn.callbacks.ManageSavedModels(filepath=filepath_save,
                                                                max_to_keep=conf.save_models_max_to_keep)
    cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=path_tensorboard, update_freq='epoch')

    cb_libsnn = lib_snn.callbacks.SNNLIB(config.flags,config.filepath_load,train_ds_num,test_ds_num)
    #cb_libsnn = lib_snn.callbacks.SNNLIB(config.flags,config.filepath_load,test_ds_num,model_ann)
    #cb_libsnn_ann = lib_snn.callbacks.SNNLIB(config.flags,config.filepath_load,test_ds_num)

    #
    callbacks_train = []
    if config.flags.save_model:
        callbacks_train.append(cb_model_checkpoint)
        callbacks_train.append(cb_manage_saved_model)
    callbacks_train.append(cb_libsnn)
    callbacks_train.append(cb_tensorboard)

    callbacks_test = []
    # TODO: move to parameters

    callbacks_test = [cb_libsnn]

    return callbacks_train, callbacks_test