#
import tensorflow as tf

#from absl import flags
#flags = flags.FLAGS
from config import conf


from tensorflow.python.keras.engine import compile_utils

#
import collections
import os
import csv
import numpy as np
import pandas as pd

#
import matplotlib.pyplot as plt

#
import lib_snn

from lib_snn.sim import glb_t
from lib_snn.sim import glb_plot


########################################
# init (on_train_begin)
#######################################
def preproc(self):
    # print summary model
    #print('summary model')
    #if self.total_num_neurons==0:
    #    cal_total_num_neurons(self)

    if self.conf.verbose:
        self.model.summary()
        #print_total_num_neurons(self)

    # initialization
    #if not self.init_done:
    #    set_init(self)

    #
    preproc_sel={
        'ANN': preproc_ann,
        'SNN': preproc_snn,
    }
    preproc_sel[self.model.nn_mode](self)

#
def preproc_ann(self):

    # initialization
    if not self.init_done:
        lib_snn.proc.set_init(self)


    # quantization fine tuning
    if self.conf.fine_tune_quant:
    #if False:
        for layer in self.model.layers_w_act:
            stat=lib_snn.calibration.read_stat(self,layer,'max_999')
            stat_max = tf.reduce_max(stat)
            layer.quant_max = tf.Variable(stat_max,trainable=False,name='quant_max')
            print('proproc_ann')
            print(layer.name)
            print(layer.quant_max)

    #assert False


#
def preproc_snn(self):
    assert False