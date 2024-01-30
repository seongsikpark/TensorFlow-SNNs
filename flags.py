

import sys
from absl import flags


#


'''
#flags = flags
#flags = flags

flags.DEFINE_string('date','','date')

flags.DEFINE_integer('epoch', 300, 'Number os epochs')
#flags.DEFINE_string('gpu_fraction', '1/3', 'define the gpu fraction used')
#flags.DEFINE_string('activation', 'ReLU', '')
#flags.DEFINE_string('optim_type', 'adam', '[exp_decay, adam]')

#flags.DEFINE_string('nn_mode', 'ANN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')

#flags.DEFINE_string('output_dir', './tensorboard', 'Directory to write TensorBoard summaries')
#flags.DEFINE_string('checkpoint_dir', './models_ckpt', 'Directory to save checkpoints')
#flags.DEFINE_string('checkpoint_load_dir', './models_ckpt', 'Directory to load checkpoints')
#flags.DEFINE_integer('ckpt_epoch', 0, 'saved ckpt epoch')
#flags.DEFINE_bool('en_load_model', False, 'Enable to load model')

#
#flags.DEFINE_boolean('en_train', False, 'enable training')


# exponetial decay
#flags.DEFINE_float('init_lr', 0.1, '')
#flags.DEFINE_float('decay_factor', 0.1, '')
#flags.DEFINE_integer('num_epoch_per_decay', 350, '')
# adam optimizer
#flags.DEFINE_float('init_lr', 1e-5, '')
# SGD
flags.DEFINE_float('lr', 0.1, 'learning rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
# ADAM
#flags.DEFINE_float('lr', 0.0001, '')


flags.DEFINE_float('lr_decay', 0.1, '')
flags.DEFINE_integer('lr_decay_step', 50, '')



# deprecated -> delete
flags.DEFINE_integer('size_test_batch', 1, 'size of test batch') # not used now
'''


#flags.DEFINE_string('pooling', 'max', 'max or avg, only for CNN')
#flags.DEFINE_integer('save_interval', 10, 'save interval of model')
#flags.DEFINE_bool('en_remove_output_dir', False, 'enable removing output dir')




#flags.DEFINE_string('model_name', 'snn_train_mlp_mnist', 'model name')
#flags.DEFINE_string('config_name', '', 'config name')


#
#flags.DEFINE_string('dataset', 'MNIST', 'dataset')
#flags.DEFINE_string('ann_model', 'MLP', 'neural network model')


#


#
#flags.DEFINE_bool('f_real_value_input_snn',False,'f_real_value_input_snn')
#flags.DEFINE_bool('f_spike_max_pool',False,'f_spike_max_pool')

# neural coding
flags.DEFINE_bool('f_ws',False,'wieghted synapse')
flags.DEFINE_float('p_ws',8,'period of wieghted synapse')

#flags.DEFINE_integer('num_class',1000,'number_of_class')


# Neuron - analysis
flags.DEFINE_bool('f_tot_psp',False,'accumulate total psp')
flags.DEFINE_bool('f_isi',False,'isi stat')
flags.DEFINE_bool('f_refractory',False,'refractory mode')
flags.DEFINE_bool('f_comp_act',False,'compare activation')
flags.DEFINE_bool('f_entropy',False,'entropy test')


# TODO: check
# data.py - imagenet data
flags.DEFINE_string('data_path_imagenet', './imagenet', 'data path imagenet')
flags.DEFINE_integer('k_pathces', 5, 'patches for test (random crop)')
flags.DEFINE_integer('input_size', 28, 'input image width / height')


#
flags.DEFINE_string('path_result_root','./result/', 'path result root')
flags.DEFINE_bool('f_save_result',True,'save result to xlsx file')


#
#flags.DEFINE_bool('f_data_std', True, 'data_standardization')


# pruning
#flags.DEFINE_bool('f_pruning_channel', False, 'purning - channel')



# temporal coding
#flags.DEFINE_float('tc',10.0,'time constant for temporal coding')
#flags.DEFINE_float('time_window',20.0,'time window of each layer for temporal coding')
#flags.DEFINE_float('time_fire_start',20.0,'time fire start (integration time before starting fire) for temporal coding')
#flags.DEFINE_float('time_fire_duration',20.0,'time fire duration for temporal coding')
flags.DEFINE_bool('f_s_dnn_tk_info',False,'info - surrogate DNN tk')
flags.DEFINE_integer('tc',10,'time constant for temporal coding')
flags.DEFINE_float('td',0.5,'time delay for temporal coding')
flags.DEFINE_integer('time_window',20,'time window of each layer for temporal coding')
#flags.DEFINE_integer('time_fire_start',20,'time fire start (integration time before starting fire) for temporal coding')
#flags.DEFINE_integer('time_fire_duration',20,'time fire duration for temporal coding')
flags.DEFINE_float('time_fire_start',20,'time fire start (integration time before starting fire) for temporal coding')
flags.DEFINE_float('time_fire_duration',20,'time fire duration for temporal coding')
# moved to new
#flags.DEFINE_bool('f_record_first_spike_time',False,'flag - recording first spike time of each neuron')
flags.DEFINE_bool('f_visual_record_first_spike_time',False,'flag - visual recording first spike time of each neuron')
flags.DEFINE_bool('f_train_tk',False,'flag - enable to train time constant for temporal coding')
flags.DEFINE_bool('f_train_tk_outlier',True,'flag - enable to outlier roubst train time constant for temporal coding')
flags.DEFINE_string('tk_file_name',None,'temporal kernel (tk) file name - prefix (tc and td)')
flags.DEFINE_bool('f_load_time_const',False,'flag - load time constant for temporal coding')
flags.DEFINE_integer('time_const_num_trained_data',0,'number of trained data - time constant')
flags.DEFINE_integer('time_const_save_interval',10000,'save interval - training time constant')
flags.DEFINE_integer('epoch_train_time_const',1,'epoch - training time constant')

flags.DEFINE_bool('f_tc_based',False,'flag - tau based')
flags.DEFINE_integer('n_tau_fire_start',4,'n tau - fire start')
flags.DEFINE_integer('n_tau_fire_duration',4,'n tau - fire duration')
flags.DEFINE_integer('n_tau_time_window',4,'n tau - time window')



# SNN trianing w/ TTFS coding
flags.DEFINE_integer("init_first_spike_time_n",-1,"init_first_spike_time = init_first_spike_n x time_windw")

# surrogate training model
flags.DEFINE_bool("f_surrogate_training_model", False, "flag - surrogate training model (DNN)")

#
flags.DEFINE_bool("f_overwrite_train_model", False, "overwrite trained model")

#
flags.DEFINE_bool("f_validation_snn", False, "validation on SNN")

#
flags.DEFINE_bool("en_tensorboard_write", False, "Tensorboard write")


#################################################################################
## Deep SNNs training w/ tepmoral information - surrogate DNN model
#################################################################################
#flags.DEFINE_integer('epoch_start_train_tk',100,'epoch start train tk')
#flags.DEFINE_float('w_train_tk',1,'weight train tk')
#flags.DEFINE_float('w_train_tk_reg',0,'weight train tk regularization, lambda')
#flags.DEFINE_string('t_train_tk_reg','L2-I','type train tk regularization, lambda')
#flags.DEFINE_bool('f_train_tk_reg',False,'flag for tk regularization')
#flags.DEFINE_string('train_tk_strategy','N','traing tk strategy')
#
#flags.DEFINE_integer('epoch_start_train_t_int',100,'epoch start train t_int')
#flags.DEFINE_integer('epoch_start_train_floor',100,'epoch start train floor')
#flags.DEFINE_integer('epoch_start_train_clip_tw',1,'epoch start train clip tw')
#flags.DEFINE_integer('epoch_start_loss_enc_spike',100,'epoch start encoded spike loss')
#
##
#flags.DEFINE_float('bypass_pr',1.0,'bypass probabilty')
#flags.DEFINE_integer('bypass_target_epoch',1,'bypass target epoch')
#
##
#flags.DEFINE_bool('f_loss_enc_spike',False,'flag for encoded spike loss')
#
##
#flags.DEFINE_float('w_loss_enc_spike',0.001,'weight of encoded spike loss')
## TODO: update below two parameters in run.sh (they are only denoted in run_enc_dec_tk.sh)
#flags.DEFINE_string('d_loss_enc_spike','b','target distribution of encoded spike loss')
#flags.DEFINE_string('ems_loss_enc_spike','n','encoded spike loss - encoding max spike: f (fixed), n (n x time window=nt)')
#
## coefficient of beta distribution
#flags.DEFINE_float('beta_dist_a',0.1,'coefficient of beta distribution - alpha')
#flags.DEFINE_float('beta_dist_b',0.1,'coefficient of beta distribution - beta')
#
##
#flags.DEFINE_integer('enc_st_n_tw',10,'target max encoded spike time - number of time window')
##flags.DEFINE_float('enc_st_n_tw',10,'target max encoded spike time - number of time window')
#
#flags.DEFINE_bool('f_td_training',True,'flag td training')
#
## quantization-aware vth - TTFS coding, surrogate training model
#flags.DEFINE_bool('f_qvth',False,'quantization-aware vth, rounding function')
#
##
#flags.DEFINE_string('tfboard_log_file_name',None,'tfboard log file name')
#


## noise
flags.DEFINE_bool('noise_en',False,'noise injection mode enable')
#flags.DEFINE_string('noise_type',None,'noise type - DEL ..')
#flags.DEFINE_float('noise_pr',0.1,'noise probability for DEL')
#flags.DEFINE_bool('noise_robust_en',False,'noise robust mode enable')
#flags.DEFINE_bool('noise_robust_comp_pr_en',False,'noise robust compenstation pr enable - only DEL')
#flags.DEFINE_integer('noise_robust_spike_num',0,'noise robust spike number')
#flags.DEFINE_integer('rep',-1,'repeat - noise experiments')


################################################################################
# new configurations
################################################################################

#
flags.DEFINE_boolean('tf32_mode',False, 'TF32 mode - Ampere GPU')


#
flags.DEFINE_boolean('verbose',False, 'verbose mode')
#flags.DEFINE_boolean('verbose',True, 'verbose mode')

flags.DEFINE_boolean('verbose_visual',False, 'verbose visual mode')
#flags.DEFINE_boolean('verbose_visual',True, 'verbose visual mode')

#
flags.DEFINE_boolean('verbose_snn_train',False, 'verbose mode')
#flags.DEFINE_boolean('verbose_snn_train',True, 'verbose mode')


################################
# Common
################################

# exp_set_name
flags.DEFINE_string('exp_set_name', 'experiment', 'exp set name')
#flags.DEFINE_string('exp_set_name', 'manual_test', 'exp set name')
#flags.DEFINE_string('exp_set_name', '221002_train_snn_VGG16_CIFAR-10', 'exp set name')

# mode
#flags.DEFINE_enum('mode', 'inference', ['train', 'load_and_train', 'inference'], 'run mode')
#flags.DEFINE_enum('mode', 'load_and_train', ['train', 'load_and_train', 'inference'], 'run mode')
flags.DEFINE_enum('mode', 'train', ['train', 'load_and_train', 'inference'], 'run mode')


#
flags.DEFINE_bool('save_model', True, 'save trained model')

#
flags.DEFINE_bool('hp_tune', False, 'hyperparameter tune mode')
#flags.DEFINE_bool('hp_tune', True, 'hyperparameter tune mode')


#
flags.DEFINE_bool('dnn_to_snn', False, 'dnn-to-snn conversion')
#flags.DEFINE_bool('dnn_to_snn', True, 'dnn-to-snn conversion')

# neural network mode
#flags.DEFINE_string('nn_mode', 'ANN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')
flags.DEFINE_string('nn_mode', 'SNN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')

# datasets
flags.DEFINE_string('dataset', 'CIFAR10', 'dataset')
#flags.DEFINE_string('dataset', 'CIFAR100', 'dataset')
#flags.DEFINE_string('dataset', 'ImageNet', 'dataset')



# models - CIFAR
flags.DEFINE_string('model', 'VGG16', 'model')
#flags.DEFINE_string('model', 'ResNet20', 'model')
#flags.DEFINE_string('model', 'ResNet32', 'model')
#flags.DEFINE_string('model', 'ResNet44', 'model')
#flags.DEFINE_string('model', 'ResNet56', 'model')
#flags.DEFINE_string('model', 'ResNet18', 'model')
#flags.DEFINE_string('model', 'ResNet34', 'model')


# models - ImageNet
#flags.DEFINE_string('model', 'VGG16', 'model')    # ImageNet
#flags.DEFINE_string('model', 'ResNet50', 'model')  # ImageNet
#flags.DEFINE_string('model', 'ResNet50V2', 'model')   # not supported yet
#flags.DEFINE_string('model', 'ResNet101', 'model')  # ImageNet
#flags.DEFINE_string('model', 'ResNet152', 'model')  # ImageNet
#flags.DEFINE_string('model', 'MobileNet', 'model')    # not supported yet
#flags.DEFINE_string('model', 'MobileNetV2', 'model')  # ImageNet
#flags.DEFINE_string('model', 'EfficientNetV2S', 'model')  #ImageNet
#flags.DEFINE_string('model', 'EfficientNetV2M', 'model')



#
#flags.DEFINE_bool('load_best_model', True, 'load best model (model, dataset)')
flags.DEFINE_bool('load_best_model', False, 'load best model (model, dataset)')

#
#flags.DEFINE_integer('batch_size', 1000, '')
#flags.DEFINE_integer('batch_size', 500, '')
#flags.DEFINE_integer('batch_size', 800, '')
#flags.DEFINE_integer('batch_size', 600, '')
#flags.DEFINE_integer('batch_size', 400, '')
#flags.DEFINE_integer('batch_size', 200, '')
flags.DEFINE_integer('batch_size', 100, '')
#flags.DEFINE_integer('batch_size', 10, '')
#flags.DEFINE_integer('batch_size', 2, '')
#flags.DEFINE_integer('batch_size', 1, '')

#flags.DEFINE_integer('batch_size_inf', 1000, '')
#flags.DEFINE_integer('batch_size_inf', 500, '')
#flags.DEFINE_integer('batch_size_inf', 400, '')
#flags.DEFINE_integer('batch_size_inf', 250, '')
#flags.DEFINE_integer('batch_size_inf', 200, '')
flags.DEFINE_integer('batch_size_inf', 100, '')
#flags.DEFINE_integer('batch_size_inf', 10, '')
#flags.DEFINE_integer('batch_size_inf', 1, '')

#
flags.DEFINE_integer('train_epoch', 300, 'train epoch')

#
flags.DEFINE_integer('step_decay_epoch', 100, 'learning rate schedule - step decy')

#
flags.DEFINE_enum('optimizer', 'SGD', ['SGD', 'ADAM'], 'optimizer')

#
flags.DEFINE_enum('lr_schedule', 'STEP', ['STEP', 'STEP_WUP', 'COS', 'COSR'], 'learning rate scheduler')

#
flags.DEFINE_enum('train_type', 'scratch', ['scratch', 'transfer', 'finetuing'], 'training_type')


#
# VGG
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')   # SNN
#flags.DEFINE_float('learning_rate', 0.02, 'learning rate')
#flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
#flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
#flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
# ResNet
#flags.DEFINE_float('learning_rate', 0.2, 'learning rate')

# regularizer
flags.DEFINE_string('regularizer', 'L2', 'L2 or L1 regularizer')
# VGG
#flags.DEFINE_float('lmb',1.0E-5, 'lambda')
#flags.DEFINE_float('lmb',5.0E-5, 'lambda')
flags.DEFINE_float('lmb',1.0E-4, 'lambda') # SNN
#flags.DEFINE_float('lmb',1.0E-3, 'lambda') # SNN
# ResNet
#flags.DEFINE_float('lmb',1.0E-4, 'lambda')

#
flags.DEFINE_float('grad_clipnorm',None,'gradient clip norm')
#flags.DEFINE_float('grad_clipnorm',1.0,'gradient clip norm')
#flags.DEFINE_float('grad_clipnorm',2.0,'gradient clip norm')
#flags.DEFINE_float('grad_clipnorm',5.0,'gradient clip norm')

# data preprocessing
# default - default (e.g., standardization) depending on each dataset
# max_norm - div by max value of dataset
# max_norm_d - div by max value of batch
# max_norm_d_c - div by max value of batch - channel-wise
flags.DEFINE_enum('data_prep', 'default', ['default', 'max_norm', 'max_norm_d', 'max_mord_d_c'], 'data preprocessing')
#flags.DEFINE_enum('data_prep', 'max_norm_d_c', ['default', 'max_norm', 'max_norm_d', 'max_norm_d_c'], 'data preprocessing')

# data augmentation
#flags.DEFINE_enum('data_aug_mix', 'None', ['mixup', 'cutmix', 'None'], 'data augmentation - mixup')
flags.DEFINE_enum('data_aug_mix', 'cutmix', ['mixup', 'cutmix', 'None'], 'data augmentation - mixup')


## data_format - DO NOT TOUCH
flags.DEFINE_string('data_format', 'channels_last', 'data format')


#
flags.DEFINE_boolean('use_bias', True, 'use bias')
#flags.DEFINE_boolean('use_bias', False, 'use bias')

#
flags.DEFINE_boolean('use_bn', True, 'use batchnorm')
#flags.DEFINE_boolean('use_bn', False, 'use batchnorm')

#
flags.DEFINE_boolean('tf_fused_bn', None, 'tf fused bn operation for computation efficiency - CNN')
#flags.DEFINE_boolean('tf_fused_bn', False, 'tf fused bn operation for computation efficiency - CNN')


flags.DEFINE_string('pooling_vgg', 'max', 'max or avg, only for VGG')
flags.DEFINE_string('pooling_resnet_imagenet_pre', 'max', 'max or avg, for ResNet imagenet pre')

################
# Directories
################
flags.DEFINE_bool('save_best_model_only', True, 'save best model only')
flags.DEFINE_integer('save_model_freq_epoch', -1, 'save model frequency in epoch when save_model_best_only==False')
flags.DEFINE_integer('save_models_max_to_keep', 1, 'save models max to keep')

flags.DEFINE_string('root_tensorboard', './tensorboard/', 'root - tensorboard')

flags.DEFINE_string('root_model_best', '/home/sspark/Models/CNN', 'root model best')
flags.DEFINE_string('root_model_save', './models_ckpt', 'root model save')
#flags.DEFINE_string('root_model_load', '/home/sspark/Projects/00_SNN/models', 'root model load')
flags.DEFINE_string('root_model_load', './models_ckpt', 'root model load')
#flags.DEFINE_string('root_model_load', '/home/sspark/Models/CNN', 'root model load')

#
flags.DEFINE_string('name_model_load','','default - root_model_load/model_dataset/conf')
#flags.DEFINE_string('name_model_load','/home/sspark/Projects/00_SNN/hp_tune/220411_manual_test/trial_0005','default - root_model_load/model_dataset/conf')
#flags.DEFINE_string('name_model_load','/home/sspark/Projects/00_SNN/hp_tune/220607_finetune_test/trial_0000','default - root_model_load/model_dataset/conf')


flags.DEFINE_string('name_model_save','','default - root_model_save/model_dataset/conf')

#
flags.DEFINE_boolean('overwrite_train_model', False, 'overwrite train model')

#
flags.DEFINE_boolean('overwrite_tensorboard', True, 'overwrite tensorboard')

#
flags.DEFINE_string('root_results', './results', 'root results')

################
# Debug
################
flags.DEFINE_bool('debug_mode', False, 'debug mode')
#flags.DEFINE_bool('debug_mode', True, 'debug mode')

#flags.DEFINE_bool('en_record_output', False, 'save intermediate layer output')
flags.DEFINE_bool('en_record_output', True, 'save intermediate layer output')

flags.DEFINE_integer('idx_train_data', 0, 'start index of train data')
flags.DEFINE_integer('num_train_data', -1, 'number of train data - default: -1 (full dataset)')
#flags.DEFINE_integer('num_train_data', 100, 'number of train data - default: -1 (full dataset)')
#flags.DEFINE_integer('num_train_data', 10000, 'number of train data - default: -1 (full dataset)')

flags.DEFINE_bool('full_test', True, 'full dataset test')
#flags.DEFINE_bool('full_test', False, 'full dataset test')

flags.DEFINE_integer('idx_test_data', 0, 'start index of test data')

flags.DEFINE_integer('num_test_data', 400, 'number of test data')



################################
# SNN
################################

# neuron type in SNN
#flags.DEFINE_string('n_type', 'IF', 'LIF or IF: neuron type')
flags.DEFINE_string('n_type', 'LIF', 'LIF or IF: neuron type')

flags.DEFINE_enum('n_reset_type','reset_by_sub', ['reset_by_sub', 'reset_to_zero'], 'neuron reset type')
#flags.DEFINE_enum('n_reset_type','reset_to_zero', ['reset_by_sub', 'reset_to_zero'], 'neuron reset type')


flags.DEFINE_bool('f_positive_vmem',False,'positive vmem')
#flags.DEFINE_bool('f_positive_vmem',True,'positive vmem')

flags.DEFINE_bool('f_neg_cap_vmem',False,'negative capped vmem')
#flags.DEFINE_bool('f_neg_cap_vmem',True,'negative capped vmem')

flags.DEFINE_string('input_spike_mode','REAL','input spike mode - REAL, POISSON, WEIGHTED_SPIKE, others...')
#flags.DEFINE_string('input_spike_mode','POISSON','input spike mode - REAL, POISSON, WEIGHTED_SPIKE, others...')
flags.DEFINE_string('neural_coding','RATE','neural coding - RATE, WEIGHTED_SPIKE, PROPOSED')

#
flags.DEFINE_bool('binary_spike', True, 'binary spike activation, if false - vth activation')
#flags.DEFINE_bool('binary_spike', False, 'binary spike activation, if false - vth activation')

#
#flags.DEFINE_float('n_init_vth', 20.0, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 10.0, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 8.0, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 6.0, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 4.0, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 3.0, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 2.0, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 1.1, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 1.0, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 0.9, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 0.7, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 0.6, 'initial value of vth')
flags.DEFINE_float('n_init_vth', 0.5, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 0.4, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 0.3, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 0.2, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 0.1, 'initial value of vth')
#flags.DEFINE_float('n_init_vth', 0.01, 'initial value of vth')

flags.DEFINE_float('n_in_init_vth', 0.1, 'initial value of vth of n_in')
flags.DEFINE_float('n_init_vinit', 0.0, 'initial value of vinit')
flags.DEFINE_float('n_init_vrest', 0.0, 'initial value of vrest')

#
flags.DEFINE_enum('snn_output_type',"VMEM", ["SPIKE", "VMEM", "FIRST_SPIKE_TIME"], "snn output type")
#flags.DEFINE_enum('snn_output_type',"SPIKE", ["SPIKE", "VMEM", "FIRST_SPIKE_TIME"], "snn output type")

#
#flags.DEFINE_integer('time_step', 1024, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 512, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 256, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 128, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 64, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 32, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 20, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 16, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 8, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 6, 'time steps per sample in SNN')
flags.DEFINE_integer('time_step', 4, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 2, 'time steps per sample in SNN')
#flags.DEFINE_integer('time_step', 1, 'time steps per sample in SNN')

#flags.DEFINE_integer('time_step_save_interval',10,'snn test save interval')
flags.DEFINE_integer('time_step_save_interval',8,'snn test save interval')
#flags.DEFINE_integer('time_step_save_interval',2,'snn test save interval')
#flags.DEFINE_integer('time_step_save_interval',1,'snn test save interval')

#
#flags.DEFINE_bool('f_record_first_spike_time',True,'flag - recording first spike time of each neuron')
flags.DEFINE_bool('f_record_first_spike_time',False,'flag - recording first spike time of each neuron')


################
# DNN-to-SNN conversion
################

# weight normalization
#
#flags.DEFINE_bool('f_fused_bn',True,'f_fused_bn')
flags.DEFINE_bool('f_fused_bn',False,'f_fused_bn')

#
#flags.DEFINE_bool('f_w_norm_data',True,'f_w_norm_data')
flags.DEFINE_bool('f_w_norm_data',False,'f_w_norm_data')

#
flags.DEFINE_string('norm_stat','max_999','data-based normalization stat (max, max_999, mean, etc.)')
#flags.DEFINE_string('norm_stat','max_997','data-based normalization stat (max, max_999, mean, etc.)')


flags.DEFINE_bool('f_write_stat',False,'write stat')
#flags.DEFINE_bool('f_write_stat',True,'write stat')

flags.DEFINE_bool('f_stat_train_mode',True,'stat with train data')
#flags.DEFINE_bool('f_stat_train_mode',False,'stat with train data')


#flags.DEFINE_string('path_stat_root','', 'path stat - root, empty->path_model_load')
flags.DEFINE_string('path_stat_root','/home/sspark/Models/CNN/VGG16_CIFAR10', 'path stat - root, empty->path_model_load')

flags.DEFINE_string('path_stat_dir','stat', 'path stat dir under path_stat_root')
flags.DEFINE_string('prefix_stat', '', 'prefix of stat file name')

#flags.DEFINE_string('act_save_mode','channel','activation save mode')


################
# calibration - DNN-to-SNN conversion
################
#flags.DEFINE_bool('bias_control',True,'bias control')
flags.DEFINE_bool('bias_control',False,'bias control')

#
flags.DEFINE_bool('calibration_vmem_ICLR',False,'calibration - vmem, init_vmem=0.5*vth')
#flags.DEFINE_bool('calibration_vmem_ICLR',True,'calibration - vmem, init_vmem=0.5*vth')

#
flags.DEFINE_bool('leak_off_after_bias_en',False,'leakage off in neuron after bias enable')
#flags.DEFINE_bool('leak_off_after_bias_en',True,'leakage off in neuron after bias enable')

# new
#flags.DEFINE_bool('vth_search',True,'vth search for DNN-to-SNN conversion')
flags.DEFINE_bool('vth_search',False,'vth search for DNN-to-SNN conversion')


#flags.DEFINE_bool('calibration_bias_new',True,'bias calibration')
flags.DEFINE_bool('calibration_bias_new',False,'bias calibration')


flags.DEFINE_bool('weight_comp_proposed',False,'calibration - bias, new')
#flags.DEFINE_bool('weight_comp_proposed',True,'calibration - bias, new')



################
# SNN training - supervised learning, surrogate gradient
################
#flags.DEFINE_bool('tdbn',False,'threshold-dependent batch normalization - AAAI21')
flags.DEFINE_bool('tdbn',True,'threshold-dependent batch normalization - AAAI21')


#
#flags.DEFINE_bool('snn_training_spatial_first',True,'SNN training spatial domain first')
flags.DEFINE_bool('snn_training_spatial_first',False,'SNN training spatial domain first')   # STBP


#
flags.DEFINE_bool('f_hold_temporal_tensor',False,'hold temporal tensor during SNN training')
#flags.DEFINE_bool('f_hold_temporal_tensor',True,'hold temporal tensor during SNN training')


#
flags.DEFINE_bool('leak_const_train',False,'leak const train')

#
flags.DEFINE_float('leak_const_init',0.9,'leak const initial value')


#
flags.DEFINE_bool('vth_rand_static',False,'vth random - N(vth,0.1)')
#flags.DEFINE_bool('vth_rand_static',True,'vth random - N(vth,0.1)')

#
flags.DEFINE_float('vrest',0.0,'vrest')

#
flags.DEFINE_bool('vrest_rand_static',False,'vth random - N(vrest,0.1)')
#flags.DEFINE_bool('vrest_rand_static',False,'vth random - N(vth,0.1)')

#
flags.DEFINE_bool('adaptive_vth',False,'adaptive vth')
#flags.DEFINE_bool('adaptive_vth',True,'adaptive vth')

flags.DEFINE_float('adaptive_vth_scale',1.1,'adaptive vth scale')


# sptr
flags.DEFINE_bool('sptr',False,'spatio-backprop, temporal-realtime learning')
#flags.DEFINE_bool('sptr',True,'spatio-backprop, temporal-realtime learning')

flags.DEFINE_float('sptr_decay',0.9,'sptr decay const')



#############
#### old ####
#############
#flags.DEFINE_bool('calibration_weight',True,'calibration - weight')
flags.DEFINE_bool('calibration_weight',False,'calibration - weight')

#flags.DEFINE_bool('calibration_weight_act_based',True,'calibration - weight')
flags.DEFINE_bool('calibration_weight_act_based',False,'calibration - weight')

#flags.DEFINE_bool('calibration_bias',True,'calibration - weight')
flags.DEFINE_bool('calibration_bias',False,'calibration - weight')

#flags.DEFINE_bool('calibration_vth',True,'calibration - vth')
flags.DEFINE_bool('calibration_vth',False,'calibration - vth')

#flags.DEFINE_bool('calibration_vmem',True,'calibration - vmem')
flags.DEFINE_bool('calibration_vmem',False,'calibration - vmem')

#flags.DEFINE_bool('vth_toggle',True,'vth toggle mode')
flags.DEFINE_bool('vth_toggle',False,'vth toggle mode')

#flags.DEFINE_float('vth_toggle_init',0.3,'vth toggle init - toggle between {init, 2-init}')
#flags.DEFINE_float('vth_toggle_init',0.5,'vth toggle init - toggle between {init, 2-init}')
#flags.DEFINE_float('vth_toggle_init',0.6,'vth toggle init - toggle between {init, 2-init}')
#flags.DEFINE_float('vth_toggle_init',0.7,'vth toggle init - toggle between {init, 2-init}')
#flags.DEFINE_float('vth_toggle_init',0.8,'vth toggle init - toggle between {init, 2-init}')
flags.DEFINE_float('vth_toggle_init',0.9,'vth toggle init - toggle between {init, 2-init}')
#flags.DEFINE_float('vth_toggle_init',0.95,'vth toggle init - toggle between {init, 2-init}')

#
flags.DEFINE_bool('vth_search_ig',False,'vth search - integrated gradient based')
#flags.DEFINE_bool('vth_search_ig',True,'vth search - integrated gradient based')

#
#flags.DEFINE_integer('vth_search_num_batch',150,'calibration num batch')
#flags.DEFINE_integer('vth_search_num_batch',32,'calibration num batch')
#flags.DEFINE_integer('vth_search_num_batch',8,'calibration num batch')
#flags.DEFINE_integer('vth_search_num_batch',6,'calibration num batch')
#flags.DEFINE_integer('vth_search_num_batch',5,'calibration num batch')
#flags.DEFINE_integer('vth_search_num_batch',4,'calibration num batch')
#flags.DEFINE_integer('vth_search_num_batch',3,'calibration num batch')
#flags.DEFINE_integer('vth_search_num_batch',2,'calibration num batch')
flags.DEFINE_integer('vth_search_num_batch',1,'calibration num batch')

#
#flags.DEFINE_integer('calibration_num_batch',100,'calibration num batch')
#flags.DEFINE_integer('calibration_num_batch',64,'calibration num batch')
#flags.DEFINE_integer('calibration_num_batch',32,'calibration num batch')
#flags.DEFINE_integer('calibration_num_batch',10,'calibration num batch')
#flags.DEFINE_integer('calibration_num_batch',8,'calibration num batch')
#flags.DEFINE_integer('calibration_num_batch',6,'calibration num batch')
#flags.DEFINE_integer('calibration_num_batch',5,'calibration num batch')
flags.DEFINE_integer('calibration_num_batch',4,'calibration num batch')
#flags.DEFINE_integer('calibration_num_batch',3,'calibration num batch')
#flags.DEFINE_integer('calibration_num_batch',2,'calibration num batch')
#flags.DEFINE_integer('calibration_num_batch',1,'calibration num batch')

#
flags.DEFINE_bool('calibration_bias_ICLR_21',False,'calibration - bias, ICML-21')
#flags.DEFINE_bool('calibration_bias_ICLR_21',True,'calibration - bias, ICML-21')

flags.DEFINE_bool('calibration_bias_ICML_21',False,'calibration - bias, ICML-21')
#flags.DEFINE_bool('calibration_bias_ICML_21',True,'calibration - bias, ICML-21')

flags.DEFINE_bool('calibration_vmem_ICML_21',False,'calibration - bias, ICML-21')
#flags.DEFINE_bool('calibration_vmem_ICML_21',True,'calibration - bias, ICML-21')


#
flags.DEFINE_bool('calibration_bias_up_prog',False,'calibration - bias, update progressive')
#flags.DEFINE_bool('calibration_bias_up_prog',True,'calibration - bias, update progressive')


flags.DEFINE_bool('idx_search_append',False,'idx search append mode')
#flags.DEFINE_bool('idx_search_append',True,'idx search append mode')

#
flags.DEFINE_bool('early_stop_search',False,'early stop - vth_search_idx_test and calibration_idx_test')
#flags.DEFINE_bool('early_stop_search',True,'early stop - vth_search_idx_test and calibration_idx_test')
flags.DEFINE_float('early_stop_search_acc',0.0,'early stop accuracy (0~1)')
#flags.DEFINE_float('early_stop_search_acc',0.9,'early stop accuracy (0~1)')

#
flags.DEFINE_bool('vth_search_idx_test',False,'vth_search_idx test')
#flags.DEFINE_bool('vth_search_idx_test',True,'vth_search_idx test')
flags.DEFINE_integer('vth_search_idx',0,'vth_search_idx')

#
flags.DEFINE_bool('calibration_idx_test',False,'calibration idx test')
#flags.DEFINE_bool('calibration_idx_test',True,'calibration idx test')
flags.DEFINE_integer('calibration_idx',0,'calibration idx')

#
flags.DEFINE_bool('dynamic_bn_test',False,'dynamic bn test')
#flags.DEFINE_bool('dynamic_bn_test',True,'dynamic bn test')
flags.DEFINE_float('dynamic_bn_dnn_act_scale',0.05,'dynamic bn dnn act scale')
#flags.DEFINE_float('dynamic_bn_test_const',0.01,'dynamic bn dnn act scale')
flags.DEFINE_float('dynamic_bn_test_const',0.7,'dynamic bn dnn act scale')

# TODO: move to near "verbose visual"
#flags.DEFINE_integer('verbose_visual_idx',99,'verbose visual index')
flags.DEFINE_integer('verbose_visual_idx',99,'verbose visual index')


# time dependent leakage
flags.DEFINE_bool('leak_time_dep',False,'time dependent leakage - LIF')
#flags.DEFINE_bool('leak_time_dep',True,'time dependent leakage - LIF')




#
flags.DEFINE_bool('ds_err_act_check',False,'dynamic/static activation check')
#flags.DEFINE_bool('ds_err_act_check',True,'dynamic/static activation check')

# finetune - quantization
flags.DEFINE_bool('fine_tune_quant',False,'fine tuning - quantization')
#flags.DEFINE_bool('fine_tune_quant',True,'fine tuning - quantization')


# regularization - spike
flags.DEFINE_bool('reg_spike_out',False,'regularization - spike output in neuron')
flags.DEFINE_float('reg_spike_out_const',7E-3,'regularization - spike count const')
flags.DEFINE_float('reg_spike_out_alpha',0,'regularization - spike count alpha')
flags.DEFINE_bool('reg_spike_out_norm',False,'regularization - spike output in neuron - norm based (L2,prev work)')
flags.DEFINE_bool('reg_spike_out_sc',False,'regularization - spike out * spike_count (norm)')
flags.DEFINE_bool('reg_spike_out_sc_wta',True,'regularization - spike out * spike_count (norm), winner-take-all')
flags.DEFINE_bool('reg_spike_out_sc_train',False,'regularization - spike out * spike_count (norm), coefficient train')
flags.DEFINE_bool('reg_spike_out_sc_sm',False,'regularization - spike out * spike_count (norm), softmax normalization')
flags.DEFINE_bool('reg_spike_out_sc_sq',False,'regularization - spike out * spike_count (norm), squre')
flags.DEFINE_string('trained_model_reg_spike','NORMAL','NORMAL, WTA_1, WTA_2, SIM_A, SIM_S')
flags.DEFINE_bool('reg_spike_vis_fmap_sc',False,'regularization - spike, visualization, fmap of integrated spi')



# regularization - postsynaptic potential (PSP)
flags.DEFINE_bool('reg_psp',False,'regularization - psp - entropy')
flags.DEFINE_float('reg_psp_const',1E-3,'regularization - psp - entropy')
flags.DEFINE_float('reg_psp_eps',1E-10,'regularization - psp - entropy - epsilon in backward')
flags.DEFINE_bool('reg_psp_min',False,'regularization - psp - entropy min / max')


# stdp - pathway
flags.DEFINE_bool('en_stdp_pathway',False,'enable - stdp pathway')
flags.DEFINE_float('stdp_pathway_weight',0.5,'stdp pathway weight - weight = (1-alpha)+alpha*spike_trace')
flags.DEFINE_bool('spike_trace_debug',False,'debug spike trace - record')

# DVS
flags.DEFINE_bool('input_data_time_dim',False,'input data has time dimension')
flags.DEFINE_integer('time_dim_size',1,'time dimension size')


#
conf=flags.FLAGS
conf(sys.argv)