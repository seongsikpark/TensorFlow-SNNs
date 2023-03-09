'''
    Configuration for SNN direct training

'''

#
from absl import flags

#
import config_common

#conf = config_common.conf


FLAGS = flags.FLAGS

#print(FLAGS)
#print('a')


import os

# GPU setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,4"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


FLAGS.debug_mode=False

FLAGS.n_reset_type='reset_to_zero'
