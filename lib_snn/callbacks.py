
import os
import shutil

import tensorflow as tf

from tensorflow.python.keras.utils.io_utils import path_to_string

#
import lib_snn

class ManageSavedModels(tf.keras.callbacks.Callback):
    def __init__(self,
                 filepath,
                 max_to_keep=5,
                 **kwargs):
        super(ManageSavedModels, self).__init__()

        self.filepath = path_to_string(filepath)
        self.max_to_keep = max_to_keep

    #
    def check_and_remove(self):
        list_dir = os.listdir(self.filepath)

        if len(list_dir) <= self.max_to_keep:
            return

        mtime = lambda f: os.stat(os.path.join(self.filepath, f)).st_mtime
        list_dir_sorted = list(sorted(os.listdir(self.filepath), key=mtime))

        for d in list_dir_sorted[:-5]:
            target_d = os.path.join(self.filepath,d)
            if os.path.isfile(target_d):
                os.remove(target_d)
            else:
                shutil.rmtree(target_d)


    #def on_train_batch_end(self, batch, logs=None):
        #self.check_and_remove()

    def on_epoch_end(self, epoch, logs=None):
        self.check_and_remove()



# ModelCheckpointResume
# wrapper for keras.callback.ModelCheckpoint
# add "best" argument for resume training
class ModelCheckpointResume(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,
                filepath,
                monitor = 'val_loss',
                verbose = 0,
                save_best_only = False,
                save_weights_only = False,
                mode = 'auto',
                save_freq = 'epoch',
                options = None,
                best = None,
                tensorboard_writer = None,
                log_dir = None,
                ** kwargs):

        if save_freq is not 'epoch':
            assert False, 'only supported save_freq=epoch'

        super(ModelCheckpointResume, self).__init__(
            filepath=filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            options=options,
            **kwargs)

        if best is not None:
            self.best = best

        #tf.summary.create_file_writer()

        #print('ModelCheckpointResume - init - previous best: '.format(self.best))


    # from keras.callbacks.ModelCheckpoint
    def on_epoch_end(self, epoch, logs=None):

        super(ModelCheckpointResume, self).on_epoch_end(epoch=epoch,logs=logs)

        #print(self.best)
        tf.summary.scalar('best_acc_val', data=self.best, step=epoch)
        logs['best_acc_val'] = self.best


#
class TensorboardBestValAcc(tf.keras.callbacks.Callback):
    def __init__(self,
                 best_val_acc,
                 **kwargs):

        self.best_val_acc = best_val_acc
        super(TensorboardBestValAcc, self).__init__(**kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        print('on_epoch_begin')
        print(logs)

    def on_epoch_end(self, epoch, logs=None):
        print('best val_acc')
        #print(cb_model_checkpoint.best)
        print(self.best_val_acc)
        print(logs)

#
class SNNLIB(tf.keras.callbacks.Callback):
    def __init__(self, conf, path_model, test_ds_num, model_ann=None, **kwargs):
        super(SNNLIB, self).__init__(**kwargs)
        self.conf = conf
        self.path_model = path_model
        self.test_ds_num = test_ds_num
        self.model_ann = model_ann


        self.f_skip_bn=False
        #self.layers_w_kernel=[]

        #
        self.init_done = False
        self.bn_fusion_done = False
        self.w_norm_done = False

        # calibration
        self.calibration_static_done = False
        self.calibration_act_based_done = False
        self.run_for_calibration = False

        # compare
        #self.run_for_compare_post_calib = False


    #def build(self):
        #lib_snn.proc.set_init(self)

    def on_test_begin(self, logs=None):
        # initialization
        lib_snn.proc.preproc(self)

        # reset
        lib_snn.proc.reset(self)


    def on_test_end(self, logs=None):
        lib_snn.proc.postproc(self)

    def on_test_batch_begin(self, batch, logs=None):
        #print('on_test_batch_begin')
        lib_snn.proc.preproc_batch(self)

    def on_test_batch_end(self, batch, logs=None):
        #print('on_test_batch_end')
        lib_snn.proc.postproc_batch(self)


#
class DNNtoSNN(tf.keras.callbacks.Callback):
    def __init__(self, conf, **kwargs):
        super(DNNtoSNN, self).__init__(**kwargs)
        self.conf = conf

    def on_test_begin(self, logs=None):
        #print(self.model)

        #print(self.model.model)
        #self.model.get_layer()

        #list_layer = self.model.layers[0:]

        #print(self.model.get_layer('conv1').kernel[0,0,0])
        #print(self.model.get_layer('conv1').bias)


        # bn fusion
        if (self.conf.nn_mode=='ANN' and self.conf.f_fused_bn) or (self.conf.nn_mode=='SNN'):
            for layer in self.model.layers:
                if hasattr(layer, 'bn') and (layer.bn is not None):
                    layer.bn_fusion()

        #assert False

        #print(self.model.get_layer('conv1').kernel[0,0,0])
        #print(self.model.get_layer('conv1').bias)

########
# callback test
class CallbackTest(tf.keras.callbacks.Callback):
    def __init__(self,
                 **kwargs):
        super(CallbackTest, self).__init__(
            **kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        print('on_epoch_begin')
        print(logs)

    def on_epoch_end(self, epoch, logs=None):
        print('on_epoch_begin')
        print(logs)
