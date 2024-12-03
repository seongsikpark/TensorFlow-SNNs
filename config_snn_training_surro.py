'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ['NCCL_P2P_DISABLE']='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


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
#conf.num_train_data = 10000

#conf.model='VGG11'
conf.model='VGG16'
#conf.model='ResNet18'
#conf.model='ResNet19'
#conf.model='ResNet20'
#conf.model='ResNet32'
#conf.model='ResNet20_SEW'   # spike-element-wise block


#conf.dataset='CIFAR100'
#conf.dataset='ImageNet'
#conf.dataset='CIFAR10_DVS'


conf.pooling_vgg = 'avg'

conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'



#
conf.fire_surro_grad_func = 'boxcar'

conf.surro_grad_alpha = 0.5
#conf.surro_grad_beth =

conf.debug_surro_grad = True

conf.exp_set_name='surro_grad'
conf.save_model = False

#
config.set()
