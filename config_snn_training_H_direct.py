'''
    Configuration for SNN direct training

'''

# GPU setting
import os
#os.environ['NCCL_P2P_DISABLE']='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["NCCL_P2P_DISABLE"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]='3'
#
from config import config
conf = config.flags

#conf.debug_mode = True

# conf.debug_mode = True
conf.save_best_model_only = True
conf.save_models_max_to_keep = 1

######
conf.root_model_save = '/mnt/hdd1/kyccj/H_direct/base/1/'
# conf.name_model_load= '/home/ssparknas/240907_ms_inf/resnet/'
# conf.name_model_load= '/home/ssparknas/240907_ms_inf/ms/'
# conf.name_model_load= '/home/ssparknas/240907_ms_inf/ours_ms/'
# conf.name_model_load= '/home/ssparknas/240907_ms_inf/ours_resnet/'
#conf.name_model_load= '/home/ssparknas/test1'
# conf.optimizer = 'ADAM'
conf.optimizer = 'ADAMW'
# conf.data_aug_mix = 'mixup'
conf.lr_schedule = 'COS'
#conf.lr_schedule = 'COSR'
conf.tdbn= False

conf.learning_rate = 1E-3
conf.train_epoch = 310
conf.weight_decay_AdamW = 1E-6
conf.batch_size = 100
conf.lmb=0
conf.label_smoothing =0.1

# conf.mode='inference'
# conf.n_conv1_spike_count = True
# conf.all_layer_spike_count = True
# conf.time_step=6

# Method
# conf.rmp_en = 'True'
# conf.rmp_k = 0.0005
# conf.im_en = 'True'
# conf.im_k = 0.001
#
conf.SEL_en = 'base'
# conf.SEL_en = 'AT'
# conf.SEL_en = 'FD'
# conf.SEL_en = 'DFE'
# conf.SEL_en = 'AT+FD'
# conf.SEL_en = 'ours'

# conf.num_train_data = 100
# conf.SEL_model_dataset = 'V16_C10'
# conf.SEL_model_dataset = 'V16_C100'
# conf.SEL_model_dataset = 'V16_DVS'
# conf.SEL_model_dataset = 'R19_C10'
# conf.SEL_model_dataset = 'R19_C100'
# conf.SEL_model_dataset = 'R19_DVS'
# conf.SEL_model_dataset = 'R20_C10'
# conf.SEL_model_dataset = 'R20_C100'
# conf.SEL_model_dataset = 'R20_DVS'
# conf.SEL_model_dataset = 'MS34_ImageNet'
# conf.SEL_model_dataset = '34_ImageNet'
conf.SEL_model_dataset = 'Spik_C10'
# conf.SEL_model_dataset = 'Spik_C100'
# conf.SEL_model_dataset = 'Spik_Img'
# conf.SEL_model_dataset = 'Spik_DVS'


# conf.batch_size_inf=100
# conf.batch_size=
#
#test
# conf.low_test=True

# conf.SEL_noise_raw = True
# conf.SEL_noise_en_layer = True
# conf.SEL_noise_en_spike = True
# conf.SEL_noise_std = 0.7
# conf.SEL_noise_th = 1
# conf.name_model_load='/home/sspark/Share/240429_ResNet32_ImageNet_SNN'

#
# conf.all_layer_spike_count = True
# conf.all_layer_SC_dir = '/home/ssparknas/SC/VGG16_CIFAR100/normal_Spike_count1.xlsx'


#For ImageNet

# conf.num_train_data = 10000

if conf.SEL_model_dataset == 'V16_C10':
    conf.model='VGG16'
    conf.dataset = 'CIFAR10'
    if conf.im_en:
        conf.adaptive_dec_vth_scale = 0.8
        conf.reg_psp_SEL_const = 5e-6
        conf.reg_psp_SEL_BN_ratio_value = -1
        conf.reg_psp_SEL_BN_ratio_rate = 1e-4
    else:
        conf.adaptive_dec_vth_scale = 0.8
        conf.reg_psp_SEL_const = 5e-6
        conf.reg_psp_SEL_BN_ratio_value = -1
        conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'V16_C100':
    conf.model='VGG16'
    conf.dataset = 'CIFAR100'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -0.8
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'V16_DVS':
    conf.model='VGG16'
    conf.dataset = 'CIFAR10_DVS'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-5 # 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1 # -1
    conf.reg_psp_SEL_BN_ratio_rate = 1e-2 # 1e-3
elif conf.SEL_model_dataset == 'R19_C10':
    conf.model='ResNet19'
    conf.dataset = 'CIFAR10'
    conf.adaptive_dec_vth_scale = 0.8 # not fix
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1.5
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'R19_C100':
    conf.model='ResNet19'
    conf.dataset = 'CIFAR100'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 3e-3
    conf.reg_psp_SEL_BN_ratio_value = -0.4
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'R19_DVS':
    conf.model='ResNet19'
    conf.dataset = 'CIFAR10_DVS'
    conf.adaptive_dec_vth_scale = 0.8 #not fix
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1.5
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'R20_C10':
    conf.model='ResNet20'
    conf.dataset = 'CIFAR10'
    conf.adaptive_dec_vth_scale = 0.2
    conf.reg_psp_SEL_const = 3e-3
    conf.reg_psp_SEL_BN_ratio_value = -0.4
    conf.reg_psp_SEL_BN_ratio_rate = 1e-3
elif conf.SEL_model_dataset == 'R20_C100':
    conf.model='ResNet20'
    conf.dataset = 'CIFAR100'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 3e-3
    conf.reg_psp_SEL_BN_ratio_value = -0.3
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'R20_DVS':
    conf.model='ResNet20'
    conf.dataset = 'CIFAR10_DVS'
    conf.adaptive_dec_vth_scale = 0.8 #not fix
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1.5
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'MS34_ImageNet':
    conf.model = 'ResNet34_MS'
    conf.dataset = 'ImageNet'
    conf.adaptive_dec_vth_scale = 0.8  # not fix
    conf.reg_psp_SEL_const = 3e-3
    conf.reg_psp_SEL_BN_ratio_value = -1
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == '34_ImageNet':
    conf.model = 'ResNet34'
    conf.dataset = 'ImageNet'
    conf.adaptive_dec_vth_scale = 0.8  # not fix
    conf.reg_psp_SEL_const = 3e-3
    conf.reg_psp_SEL_BN_ratio_value = -1
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'Spik_C10':
    conf.model='Spikformer'
    conf.dataset = 'CIFAR10'
    conf.patch_size = 4
    conf.embed_dims = 384
    conf.num_heads = 12
    conf.depths = 4
    conf.sr_ratios = 8
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -0.8
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'Spik_C100':
    conf.model='Spikformer'
    conf.dataset = 'CIFAR100'
    conf.patch_size = 4
    conf.embed_dims = 384
    conf.num_heads = 12
    conf.depths = 4
    conf.sr_ratios = 8
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -0.8
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'Spik_Img':
    conf.model='Spikformer'
    conf.dataset = 'ImageNet'
    conf.patch_size = 16
    conf.embed_dims = 512
    conf.num_heads = 16
    conf.depths = 10
    conf.sr_ratios = 8
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -0.8
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'Spik_DVS':
    conf.model='Spikformer'
    conf.dataset = 'CIFAR10_DVS'
    conf.batch_size = 10
    conf.patch_size = 16
    conf.embed_dims = 256
    conf.num_heads = 16
    conf.depths = 2
    conf.sr_ratios = 8
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -0.8
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4

if conf.dataset == 'CIFAR10_DVS':
    conf.learning_rate = 0.1
    conf.time_step = 16
if conf.dataset == 'CIFAR100':
    conf.learning_rate = 0.001
    conf.lmb = 0.01
if conf.dataset == 'ImageNet':
    conf.batch_size = 90
    conf.train_epoch = 90
    conf.step_decay_epoch = 30
conf.pooling_vgg = 'avg'

# conf.nn_mode = 'SNN'
conf.nn_mode = 'ANN'

conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'


conf.vth_rand_static = False
conf.vrest = 0.0

conf.reg_psp_SEL_BN_ratio = True

#
if conf.SEL_en == 'base':
    conf.adaptive_vth_SEL = False
    conf.reg_psp_SEL= False
    conf.reg_psp_SEL_BN = False
elif conf.SEL_en == 'AT':
    conf.adaptive_vth_SEL = True
    conf.reg_psp_SEL=False
    conf.reg_psp_SEL_BN = False
elif conf.SEL_en == 'FD':
    conf.adaptive_vth_SEL = False
    conf.reg_psp_SEL=True
    conf.reg_psp_SEL_BN = False
elif conf.SEL_en == 'DFE':
    conf.adaptive_vth_SEL = False
    conf.reg_psp_SEL=False
    conf.reg_psp_SEL_BN = True
elif conf.SEL_en == 'AT+FD':
    conf.adaptive_vth_SEL = True
    conf.reg_psp_SEL=True
    conf.reg_psp_SEL_BN = False
elif conf.SEL_en == 'ours':
    conf.adaptive_vth_SEL = True
    conf.reg_psp_SEL=True
    conf.reg_psp_SEL_BN = True


conf.adaptive_inc_vth_scale = 1

#conf.n_init_vth = 0.3

conf.leak_const_init = 0.9
#conf.leak_const_train = True

#
#conf.en_stdp_pathway = True
conf.stdp_pathway_weight = 0.1

config.set()
