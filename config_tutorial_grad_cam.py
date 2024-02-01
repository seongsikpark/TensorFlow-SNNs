'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ['NCCL_P2P_DISABLE']='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# for ssl verification error
os.environ['CURL_CA_BUNDLE'] = ''


#
from config import config
conf = config.flags

#
#conf.debug_mode = True
#conf.verbose_snn_train = True


#
#conf.mode='inference'
##conf.batch_size_inf=100
#conf.batch_size=120
#conf.time_step=2


#
#conf.learning_rate = 0.2
#conf.lmb = 1.0E-4
#conf.time_step = 64

#
#conf.train_epoch = 90
#conf.step_decay_epoch = 30
#conf.train_epoch = 10
#conf.train_epoch = 10
#conf.num_train_data = 10000

#conf.model='ResNet20'
#conf.model='ResNet32'

#conf.dataset='CIFAR100'
#conf.dataset='ImageNet'
#conf.dataset='CIFAR10_DVS'

#conf.pooling_vgg = 'avg'

#conf.nn_mode = 'SNN'
conf.nn_mode = 'ANN'

conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'


conf.vth_rand_static = False
conf.vrest = 0.0
#conf.vrest_rand_static = False
#conf.vrest_rand_static = True

#conf.adaptive_vth = False
#conf.adaptive_vth = True
conf.adaptive_vth_scale = 1.2

#conf.use_bn=False

#conf.n_init_vth = 0.3

conf.leak_const_init = 0.9
#conf.leak_const_train = True


#
config.set()