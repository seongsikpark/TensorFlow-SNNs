
'''
 Common configurations

'''


import flags

#conf = flag
#conf = flags.conf

from absl import flags
conf = flags.FLAGS


import os

# GPU setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,4"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#
import tensorflow as tf

#
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import matplotlib
matplotlib.use('TkAgg')


def get_train_mode():
    train= (conf.mode=='train') or (conf.mode=='load_and_train')
    return train


