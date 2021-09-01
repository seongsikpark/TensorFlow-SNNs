
import os
import shutil

import tensorflow as tf

from tensorflow.python.keras.utils.io_utils import path_to_string


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
                ** kwargs):
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

        print('ModelCheckpointResume - init - previous best: '.format(self.best))


