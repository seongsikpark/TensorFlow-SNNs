'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ['NCCL_P2P_DISABLE']='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#
from config import config
conf = config.flags

##### training setting #####
# conf.debug_mode = True
# conf.verbose_snn_train = True
conf.save_best_model_only = True
conf.save_models_max_to_keep = 1
##########

##### inference mode setting #####
#conf.mode='inference'
#conf.batch_size=400
#conf.name_model_load='./models/VGG16_AP_CIFAR100/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-z'
##########
##########

##### hyper-parameter setting #####
conf.optimizer = 'ADAMW'
conf.lr_schedule = 'COS'

conf.learning_rate_init = 1e-5
conf.learning_rate = 6e-3
conf.weight_decay_AdamW = 2e-2
##########

##### neural network type setting #####
conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'

conf.pooling_vgg = 'avg'
##########


##### augmentation setting #####
conf.label_smoothing=0.1
conf.debug_lr = True
conf.lmb=1E-3
conf.regularizer=None
#conf.data_aug_mix='mixup'

conf.mix_off_iter = 500*200
conf.mix_alpha = 0.5

# data augmentation
conf.randaug_en = True
conf.randaug_mag = 0.9
conf.randaug_mag_std = 0.4
conf.randaug_n = 1
conf.randaug_rate = 0.5

conf.rand_erase_en = True
##########

##### surrogate function setting #####
conf.fire_surro_grad_func = 'boxcar'
# conf.fire_surro_grad_func = 'asym'
# conf.fire_surro_grad_func = 'predictiveness_asy'
# conf.fire_surro_grad_func = 'predictiveness_asy_timestep'
# conf.fire_surro_grad_func = 'predictiveness_asy_all_timestep'
# conf.fire_surro_grad_func = 'predictiveness_asy_all_timestep_0418'

if conf.fire_surro_grad_func == 'predictiveness_asy':
    conf.surro_grad_beth = 1.0
    conf.find_beta_low = 1.0
    conf.find_beta_high = 10.0
    conf.similarity_alpha = 1.0
    conf.ema_lambda = 0.0
    conf.train_beta_candidate_number = 50
    conf.test_beta_candidate_number = 200
    conf.accumulate_gradient_iteration = 500
elif conf.fire_surro_grad_func == 'predictiveness_asy_timestep':
    conf.surro_grad_beth = 1.0
    conf.find_beta_low = 1.0
    conf.find_beta_high = 10.0
    conf.similarity_alpha = 1.0
    conf.train_beta_candidate_number = 30
    conf.test_beta_candidate_number = 100
elif conf.fire_surro_grad_func == 'predictiveness_asy_all_timestep':
    conf.surro_grad_beth = 1.0
    conf.find_beta_low = 1.0
    conf.find_beta_high = 10.0
    conf.similarity_alpha = 0.5
    conf.train_beta_candidate_number = 30
    conf.test_beta_candidate_number = 100
elif conf.fire_surro_grad_func == 'predictiveness_asy_all_timestep_0418':
    conf.surro_grad_beth = 1.0
    conf.find_beta_low = 1.0
    conf.find_beta_high = 10.0
    conf.similarity_alpha = 1.0
    conf.train_beta_candidate_number = 30
    conf.test_beta_candidate_number = 100
elif conf.fire_surro_grad_func == 'cpng_tri':
    conf.chi_limit = 0.2
    conf.find_beta_low = 1.0
    conf.find_beta_high = 10.0
else :
    conf.surro_grad_alpha = 1.6
    conf.accumulate_gradient_iteration = 500

conf.debug_grad = True
# conf.debug_surro_grad = True
# conf.debug_surro_grad_per_iter = 500

##### model save setting #####
# conf.root_model_save = f'./model_ckpt_1/{conf.fire_surro_grad_func}_sim_{conf.similarity_alpha}_{conf.surro_grad_beth}_accumulate_gradient_5epoch'
conf.root_model_save = f'./model_ckpt_1/{conf.fire_surro_grad_func}_{conf.surro_grad_alpha}_weight_size'
# conf.root_model_save = f'./model_ckpt/{conf.fire_surro_grad_func}_alpha={conf.surro_grad_alpha}/warmup={conf.learning_rate_init} to {conf.learning_rate}, weight_decay={conf.weight_decay_AdamW}'
# conf.root_model_save = f'./model_ckpt_2/warmup={conf.learning_rate_init} to {conf.learning_rate}, weight_decay={conf.weight_decay_AdamW}'
# conf.root_model_save = f'./model_ckpt_test/'
# conf.root_model_save = f'./model_ckpt_distribution/{conf.fire_surro_grad_func}_{conf.surro_grad_alpha}_peak_change/'
##########
# conf.exp_set_name = 'cosine_similarity_0414'
# conf.exp_set_name = '0421'
# conf.exp_set_name = 'predictiveness_asy_timestep'
# conf.exp_set_name = 'predictiveness_asy_all_timestep'
# conf.exp_set_name = 'NeurIPS_2025_predictiveness_asy'
conf.exp_set_name = 'NeurIPS_2025_weight_size'
# conf.exp_set_name='surro_grad_new'
# conf.exp_set_name='confirm_0402'
# conf.exp_set_name='CPNG'
# conf.exp_set_name='Predictiveness'
# conf.exp_set_name='Predictiveness_0324'
# conf.exp_set_name='Predictiveness_0402'
# conf.exp_set_name='test'
# conf.exp_set_name='Predictiveness'
# conf.exp_set_name='distribution'
# conf.exp_set_name='adaptive_boxcar'
# conf.exp_set_name='adaptive_asy'
# conf.exp_set_name = 'younguk_convergencerate'


##### Loss setting #####
# conf.rmp_en = 'True'
# conf.rmp_k = 0.0005
# conf.im_en = 'True'
# conf.im_k = 0.001
##########

##### Model setting #####
###### VGG16
conf.SEL_model_dataset = 'V16_C10'
# conf.SEL_model_dataset = 'V16_C100'
# conf.SEL_model_dataset = 'V16_DVS'

###### VGG11
# conf.SEL_model_dataset = 'V11_DVS'

###### VGGSNN
# conf.SEL_model_dataset = 'VSNN_DVS'

###### ResNet19
# conf.SEL_model_dataset = 'R19_C10'
# conf.SEL_model_dataset = 'R19_C100'
# conf.SEL_model_dataset = 'R19_DVS'

###### MS ResNet19
# conf.SEL_model_dataset = 'MS_R19_C10'

###### ResNet20
# conf.SEL_model_dataset = 'R20_C10'
# conf.SEL_model_dataset = 'R20_C100'
# conf.SEL_model_dataset = 'R20_DVS'


##### Spikformer
# conf.SEL_model_dataset = 'Spik_C10'
# conf.SEL_model_dataset = 'Spik_C100'
# conf.SEL_model_dataset = 'Spik_Img'
# conf.SEL_model_dataset = 'Spik_DVS'
##########

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
elif conf.SEL_model_dataset == 'V11_DVS':
    conf.model = 'VGG11'
    conf.dataset = 'CIFAR10_DVS'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-5  # 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1  # -1
    conf.reg_psp_SEL_BN_ratio_rate = 1e-2  # 1e-3
elif conf.SEL_model_dataset == 'VSNN_DVS':
    conf.model = 'VGGSNN'
    conf.dataset = 'CIFAR10_DVS'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-5  # 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1  # -1
    conf.reg_psp_SEL_BN_ratio_rate = 1e-2  # 1e-3
elif conf.SEL_model_dataset == 'R19_C10':
    conf.model='ResNet19'
    conf.dataset = 'CIFAR10'
    conf.adaptive_dec_vth_scale = 0.8 # not fix
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1.5
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'MS_R19_C10':
    conf.model='ResNet19_MS'
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

##### training parameter setting #####
if conf.dataset == 'CIFAR10':
    conf.batch_size = 100
    conf.train_epoch = 310
    conf.time_step = 4
elif conf.dataset == 'CIFAR10_DVS':
    conf.batch_size = 32
    conf.train_epoch = 200
    conf.time_step = 4
elif conf.dataset == 'ImageNet':
    conf.batch_size = 90
    conf.train_epoch = 90
    conf.step_decay_epoch = 30
##########

##### neuron setting #####
conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'

conf.n_init_vth = 1.0

conf.vth_rand_static = False
conf.vrest = 0.0

conf.leak_const_init = 0.9
##########

##### stdp setting #####
#conf.en_stdp_pathway = True
# conf.stdp_pathway_weight = 0.1
##########

config.set()
