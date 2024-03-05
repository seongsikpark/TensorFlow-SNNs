'''
    Configuration for SNN direct training

'''

# GPU setting
import os
#os.environ['NCCL_P2P_DISABLE']='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,2"
#os.environ["CUDA_VISIBLE_DEVICES"]="4,7"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5,6,7"
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3,6,7"
#os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


#
from config import config
conf = config.flags

#
#conf.debug_mode = True
#conf.verbose_snn_train = True
#conf.verbose_visual = True


#
conf.mode='inference'
##conf.batch_size_inf=100
#conf.batch_size=400
#conf.batch_size=200
#conf.batch_size=300
#conf.batch_size=180
#conf.batch_size=120
#conf.batch_size=51
#conf.batch_size=1

#conf.batch_size_inf = 51

#conf.time_step=4
#conf.name_model_load='./models/VGG16_AP_CIFAR100/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-z'

conf.root_model_load='/home/sspark/models_WTA_SNN_TMP/models_ckpt_WTA-SNN'

#
#conf.learning_rate = 0.1
#conf.lmb = 1.0E-4
#conf.optimizer = 'ADAMW'
#conf.lr_schedule = None

#
#conf.train_epoch = 90
#conf.step_decay_epoch = 30
#conf.train_epoch = 10
#conf.train_epoch = 10
#conf.save_best_model_only=False
#conf.save_model_freq_epoch=10*500
#conf.save_models_max_to_keep=100


#
#conf.num_train_data = 1000
#conf.idx_test_data=0
#conf.num_test_data=1


#conf.model='VGG11'
#conf.model='ResNet20'
#conf.model='ResNet32'

#conf.dataset='CIFAR100'
#conf.dataset='ImageNet'
#conf.dataset='CIFAR10_DVS'


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
#conf.reg_spike_out = True
#conf.reg_spike_out_const = 1E-6
#conf.reg_spike_out_alpha = 4
#conf.reg_spike_out_sc=True
#conf.reg_spike_out_sc_wta=False
## conf.reg_spike_out_sc_train=True
#conf.reg_spike_out_sc_sm=True
##conf.reg_spike_out_sc_sq=True
#conf.reg_spike_out_norm = True

#conf.reg_spike_vis_fmap_sc=True
#conf.debug_neuron_input=True
conf.debug_syn_output=True

#conf.sm_batch_index=1

# trained model
mode=conf.trained_model_reg_spike
mode='NORMAL'
#mode='WTA-1'
#mode='WTA-2'
#mode='SIM-A'
#mode='SIM-S'

# WTA-SNN
#if True:
#if False:

if mode=='NORMAL':
    #conf.name_model_load="./models_ckpt_WTA-SNN/VGG16_AP_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s"
    pass

elif mode=='WTA-1':
    #conf.name_model_load='./models_ckpt_WTA-SNN/VGG16_AP_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-sm-5e-06_4'
    conf.reg_spike_out = True
    conf.reg_spike_out_const = 3E-6
    conf.reg_spike_out_alpha = 4
    conf.reg_spike_out_sc=True
    #conf.reg_spike_out_sc_wta=False
    # conf.reg_spike_out_sc_train=True
    conf.reg_spike_out_sc_sm=True
    #conf.reg_spike_out_sc_sq=True
    conf.reg_spike_out_norm = True
elif mode=='WTA-2':
    #conf.name_model_load='./models_ckpt_WTA-SNN/VGG16_AP_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-sm-2e-05_4'
    conf.reg_spike_out = True
    conf.reg_spike_out_const = 1E-5
    conf.reg_spike_out_alpha = 4
    conf.reg_spike_out_sc=True
    #conf.reg_spike_out_sc_wta=False
    # conf.reg_spike_out_sc_train=True
    conf.reg_spike_out_sc_sm=True
    #conf.reg_spike_out_sc_sq=True
    conf.reg_spike_out_norm = True
elif mode=='SIM-A':
    # spike reg - similar spike
    #conf.name_model_load='./models_ckpt_WTA-SNN/VGG16_AP_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-nwta-sm-2e-05_4'
    conf.reg_spike_out = True
    conf.reg_spike_out_const = 2E-5
    conf.reg_spike_out_alpha = 4
    conf.reg_spike_out_sc=True
    conf.reg_spike_out_sc_wta=False
    # conf.reg_spike_out_sc_train=True
    conf.reg_spike_out_sc_sm=True
    #conf.reg_spike_out_sc_sq=True
    conf.reg_spike_out_norm = True
elif mode=='SIM-S':
    # spike reg - similar acc
    #conf.name_model_load='./models_ckpt_WTA-SNN/VGG16_AP_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-sc-nwta-sm-0.3_4'
    conf.reg_spike_out = True
    conf.reg_spike_out_const = 2.4E-1
    conf.reg_spike_out_alpha = 4
    conf.reg_spike_out_sc=True
    conf.reg_spike_out_sc_wta=False
    # conf.reg_spike_out_sc_train=True
    conf.reg_spike_out_sc_sm=True
    #conf.reg_spike_out_sc_sq=True
    conf.reg_spike_out_norm = True
else:
    assert False






#
#conf.grad_clipnorm = 3.0
#conf.grad_clipnorm = 1.0

#
#conf.en_stdp_pathway = True
conf.stdp_pathway_weight = 0.1

#
config.set()