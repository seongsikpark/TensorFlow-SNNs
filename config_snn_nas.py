'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

#
from config import config
conf = config.flags

conf.save_model=False

#conf.debug_mode = True

#conf.data_aug_mix='None'

#conf.num_train_data=200

#
#conf.train_epoch = 100
#conf.num_train_data = 100
#conf.train_epoch = 150
#conf.step_decay_epoch = 50

conf.train_epoch_search = 100
conf.step_decay_epoch_search = 50
conf.train_epoch = 300
conf.step_decay_epoch = 50

conf.optimizer = 'ADAM'
conf.learning_rate = 0.001

#
#conf.lr_schedule='COS'

conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'

conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'

#conf.n_type = 'IF'
conf.n_type = 'LIF'

#
conf.pooling_vgg = 'avg'

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
