
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