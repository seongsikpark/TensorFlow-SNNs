'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,4"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#
from config import config
conf = config.flags


#
#conf.train_epoch = 100
#conf.num_train_data = 10000

conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'

#conf.n_reset_type = 'reset_by_sub'
conf.n_reset_type = 'reset_to_zero'


conf.vth_rand_static = False
#conf.vth_rand_static = True

conf.vrest = 0.0
#conf.vrest = -0.1
conf.vrest_rand_static = False
#conf.vrest_rand_static = True

conf.adaptive_vth = False
#conf.adaptive_vth = True
conf.adaptive_vth_scale = 1.2

#
conf.sptr = True
conf.sptr_decay = 0.1
conf.snn_training_spatial_first=True

#
#conf.grad_clipnorm = 1.0
conf.grad_clipnorm = 3.0

#
#conf.debug_mode=True

#
config.set()