'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,4"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#
from config import config
conf = config.flags


#
#conf.train_epoch = 100
#conf.num_train_data = 10000

#
#conf.train_epoch = 500
#conf.step_decay_epoch = 150
#conf.learning_rate = 0.1g
#conf.lmb=5.0E-5

#
#conf.pooling_vgg='avg'

#
#conf.model='ResNet20'
#conf.model='ResNet32'

#
#conf.dataset='CIFAR100'

conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'

conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'

#conf.time_step = 2


conf.vth_rand_static = False
#conf.vth_rand_static = True

conf.vrest = 0.0
#conf.vrest = -0.1
conf.vrest_rand_static = False
#conf.vrest_rand_static = True

#conf.adaptive_vth = False
#conf.adaptive_vth = True
conf.adaptive_vth_scale = 1.2

#
#conf.sptr = True
#conf.sptr_decay = 0.1
#conf.snn_training_spatial_first=True

#
#conf.leak_const_train = True

#
#conf.grad_clipnorm = 1.0
#conf.grad_clipnorm = 3.0
#conf.grad_clipnorm = 5.0

#
#conf.debug_mode=True


#conf.en_stdp_pathway=True
conf.stdp_pathway_weight=0.1

#
config.set()
