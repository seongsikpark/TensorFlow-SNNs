'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#
from config import config
conf = config.flags

#conf.save_model=False


#
conf.exp_set_name='NAS_test'
conf.root_model_save=conf.exp_set_name


#conf.debug_mode = True

#conf.data_aug_mix='None'

conf.num_train_data=200

#
#conf.train_epoch = 100
#conf.num_train_data = 100
#conf.train_epoch = 150
#conf.step_decay_epoch = 50

#conf.train_epoch_search = 120
conf.train_epoch_search = 1
conf.step_decay_epoch_search = 40
conf.train_epoch = 300
conf.step_decay_epoch = 100

#
conf.optimizer = 'ADAMW'
conf.lr_schedule = 'COS'

conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'

conf.n_init_vth = 1.0

conf.train_epoch = 310
# spikformer
#conf.learning_rate_init = 1E-4
#conf.learning_rate = 5E-3
#conf.weight_decay_AdamW = 2E-2
# VGG-C10
conf.learning_rate_init = 1E-5
conf.learning_rate = 6E-3
conf.weight_decay_AdamW = 2E-2
# VGG-C100
#conf.learning_rate_init = 1E-5
#conf.learning_rate = 6E-3
#conf.weight_decay_AdamW = 2E-2
# R19-C10
#conf.learning_rate_init = 1E-5
#conf.learning_rate = 6E-3
#conf.weight_decay_AdamW = 2E-2
# R19-C100
#conf.learning_rate_init = 1E-4
#conf.learning_rate = 5E-3
#conf.weight_decay_AdamW = 4E-2


conf.batch_size = 100
conf.label_smoothing=0.1
conf.debug_lr = True
conf.lmb=1E-3
conf.regularizer=None
#conf.data_aug_mix='mixup'

conf.mix_off_iter = 500*200
conf.mix_alpha = 0.5

conf.randaug_en = True
conf.randaug_mag = 0.9
conf.randaug_mag_std = 0.4
conf.randaug_n = 1
conf.randaug_rate = 0.5

conf.rand_erase_en = True

#
#conf.lr_schedule='COS'


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



# spike reg
#if True:
if False:
    conf.reg_spike_out=True
    conf.reg_spike_out_const=1E-3
    conf.reg_spike_out_alpha=4  # temperature
    #conf.reg_spike_rate_alpha=8E-1  # coefficient of reg. rate
    conf.reg_spike_out_sc=True
    conf.reg_spike_out_sc_wta=False
    #conf.reg_spike_out_sc_train=True
    conf.reg_spike_out_sc_sm=True
    #conf.reg_spike_out_sc_sq=True
    conf.reg_spike_out_norm=True
    #conf.reg_spike_out_norm_sq=True

#
config.set()
