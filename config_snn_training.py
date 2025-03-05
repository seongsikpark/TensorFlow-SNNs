'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ['NCCL_P2P_DISABLE']='0'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="9"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

#
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # 0: show all, 1: hide info, 2: hide info&warning, 3: hide all (info, warning, error)

#
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_gpu_global_jit'
os.environ['TF_XLA_FLGAS'] = '--vmodule=xla_compilation_cache=1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'


#
from config import config
conf = config.flags

#
#conf.debug_mode = True
#conf.verbose_snn_train = True


#
#conf.mode='inference'
##conf.batch_size_inf=100
#conf.batch_size=400
#conf.batch_size=300
#conf.batch_size=180
#conf.batch_size=120
#conf.time_step=2
#conf.name_model_load='./models/VGG16_AP_CIFAR100/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-z'

# imagenet config - ResNet18 (2 GPU)
#conf.batch_size=200
#conf.train_epoch = 90
#conf.step_decay_epoch = 30

# cifar10-dvs config
#conf.learning_rate = 0.1
#conf.lmb = 1E-3

#
#conf.learning_rate = 0.04
#conf.lmb = 1E-3
#conf.time_step = 10
#conf.optimizer = 'ADAM'
#conf.lr_schedule = None

#
#conf.train_epoch = 10
#conf.train_epoch = 10
#conf.num_train_data = 200

#conf.model='VGG11'
#conf.model='VGG16'
#conf.model='ResNet18'
#conf.model='ResNet19'
#conf.model='ResNet20'
#conf.model='ResNet32'
#conf.model='ResNet20_SEW'   # spike-element-wise block
#conf.model = 'Spikformer'
conf.model = 'Spikformer_tb'

#conf.dataset='CIFAR100'
#conf.dataset='ImageNet'
#conf.dataset='CIFAR10_DVS'

conf.pooling_vgg = 'avg'

conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'

conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'


#conf.use_bn=False

#conf.n_init_vth = 0.3

conf.leak_const_init = 0.9
#conf.leak_const_train = True


conf.exp_set_name='test'




conf.optimizer = 'ADAMW'
conf.lr_schedule = 'COS'

conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'

conf.n_init_vth = 1.0

conf.train_epoch = 310
#
#conf.learning_rate_init = 1E-5
#conf.learning_rate = 6E-3
#conf.weight_decay_AdamW = 2E-2
# spikformer - C10
conf.learning_rate_init = 1E-4
conf.learning_rate = 5E-3
conf.weight_decay_AdamW = 2E-2

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

#conf.rand_erase_en = True

# test
conf.neuron_detach_reset = True

#
if False:
#if True:
    if True:    # proposed method
    #if False:
        conf.reg_spike_out=True
        conf.reg_spike_out_const=8E-6
        conf.reg_spike_out_alpha=4  # temperature
        #conf.reg_spike_rate_alpha=8E-1  # coefficient of reg. rate
        conf.reg_spike_out_sc=True
        #conf.reg_spike_out_sc_wta=False
        #conf.reg_spike_out_sc_train=True
        conf.reg_spike_out_sc_sm=True
        #conf.reg_spike_out_sc_sq=True
        conf.reg_spike_out_norm=True
        #conf.reg_spike_out_norm_sq=True

        #
        #conf.reg_spike_out_sc_sm_wo_tmp=True
        #conf.reg_spike_out_sc_sm_wo_spa=True
    else:   # previous work
        conf.reg_spike_out = True
        conf.reg_spike_out_const = 1E-7
        conf.reg_spike_out_alpha = 4  # temperature
        # conf.reg_spike_rate_alpha=8E-1  # coefficient of reg. rate
        #conf.reg_spike_out_sc = True
        #conf.reg_spike_out_sc_wta=False
        # conf.reg_spike_out_sc_train=True
        #conf.reg_spike_out_sc_sm = True
        # conf.reg_spike_out_sc_sq=True
        conf.reg_spike_out_norm = True
        #conf.reg_spike_out_norm_sq=True

#
config.set()
