'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,4"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

#
from config import config
conf = config.flags


#
#conf.train_epoch = 100
#conf.num_train_data = 100

#conf.nn_mode = 'SNN'
conf.nn_mode = 'ANN'

conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'

#conf.n_type = 'IF'
conf.n_type = 'LIF'

# conf.vth_rand_static = False

# conf.vrest = 0.0
#conf.vrest_rand_static = False

# conf.adaptive_vth = True
# conf.adaptive_vth_scale = 1.2

#conf.use_bn=False

#conf.n_init_vth = 0.3


#conf.debug_mode = True
#conf.num_train_data = 200

# conf.leak_const_init = 0.9
# conf.leak_const_train = True
# conf.grad_clipnorm = 1.0

#
config.set()
