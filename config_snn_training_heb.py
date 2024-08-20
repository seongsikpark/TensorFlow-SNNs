'''
    Configuration for SNN direct training

'''

# GPU setting
import os
#os.environ['NCCL_P2P_DISABLE']='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#
from config import config
conf = config.flags

#
#conf.debug_mode = True
#conf.verbose_snn_train = True

#
conf.exp_set_name='SNN_training_hebbian'

#
#conf.save_best_model_only=False
#conf.save_model_freq_epoch=5000       # iterations
#conf.root_model_save='./models_ckpt_e10'
#conf.save_models_max_to_keep=300

#
#conf.mode='inference'
##conf.batch_size_inf=100
#conf.batch_size=400
#conf.batch_size=200
#conf.batch_size=300
#conf.batch_size=180
#conf.batch_size=120
#conf.time_step=2
#conf.name_model_load='./models/VGG16_AP_CIFAR100/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-z'


#
#conf.learning_rate = 0.1
#conf.lmb = 1E-3
#conf.time_step = 4
#conf.optimizer = 'ADAM'
#conf.lr_schedule = None

#
#conf.train_epoch = 150
#conf.step_decay_epoch = 50
#conf.train_epoch = 10
#conf.train_epoch = 10
#conf.num_train_data = 1000

#conf.model='VGG11'
conf.model='VGG16'
#conf.model='VGG_SPECK'  # VGG for Speck HW
#conf.model='ResNet19'
#conf.model='ResNet20'
#conf.model='ResNet32'
#conf.model='ResNet20_SEW'   # spike-element-wise block


#conf.dataset='CIFAR100'
#conf.dataset='ImageNet'
#conf.dataset='CIFAR10_DVS'

#
#conf.cifar10_dvs_img_size = 32
#conf.cifar10_dvs_crop_img_size = 36


conf.pooling_vgg = 'avg'

conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'

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
conf.train_algo_snn='hebbian'



#
config.set()
