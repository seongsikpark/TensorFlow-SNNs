

import tensorflow as tf

#
flags = tf.compat.v1.app.flags



#flags = tf.compat.v1.app.flags
flags = tf.compat.v1.app.flags
tf.compat.v1.app.flags.DEFINE_string('date','','date')

tf.compat.v1.app.flags.DEFINE_integer('epoch', 300, 'Number os epochs')
tf.compat.v1.app.flags.DEFINE_string('gpu_fraction', '1/3', 'define the gpu fraction used')
tf.compat.v1.app.flags.DEFINE_string('activation', 'ReLU', '')
tf.compat.v1.app.flags.DEFINE_string('optim_type', 'adam', '[exp_decay, adam]')

#tf.compat.v1.app.flags.DEFINE_string('nn_mode', 'ANN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')

tf.compat.v1.app.flags.DEFINE_string('output_dir', './tensorboard', 'Directory to write TensorBoard summaries')
tf.compat.v1.app.flags.DEFINE_string('checkpoint_dir', './models_ckpt', 'Directory to save checkpoints')
tf.compat.v1.app.flags.DEFINE_string('checkpoint_load_dir', './models_ckpt', 'Directory to load checkpoints')
tf.compat.v1.app.flags.DEFINE_integer('ckpt_epoch', 0, 'saved ckpt epoch')
tf.compat.v1.app.flags.DEFINE_bool('en_load_model', False, 'Enable to load model')

#
tf.compat.v1.app.flags.DEFINE_boolean('en_train', False, 'enable training')


# exponetial decay
'''
tf.compat.v1.app.flags.DEFINE_float('init_lr', 0.1, '')
tf.compat.v1.app.flags.DEFINE_float('decay_factor', 0.1, '')
tf.compat.v1.app.flags.DEFINE_integer('num_epoch_per_decay', 350, '')
'''
# adam optimizer
#tf.compat.v1.app.flags.DEFINE_float('init_lr', 1e-5, '')
# SGD
tf.compat.v1.app.flags.DEFINE_float('lr', 0.1, 'learning rate')
tf.compat.v1.app.flags.DEFINE_float('momentum', 0.9, 'momentum')
# ADAM
#tf.compat.v1.app.flags.DEFINE_float('lr', 0.0001, '')


tf.compat.v1.app.flags.DEFINE_float('lr_decay', 0.1, '')
tf.compat.v1.app.flags.DEFINE_integer('lr_decay_step', 50, '')




# deprecated -> delete
tf.compat.v1.app.flags.DEFINE_integer('size_test_batch', 1, 'size of test batch') # not used now



tf.compat.v1.app.flags.DEFINE_string('pooling', 'max', 'max or avg, only for CNN')

tf.compat.v1.app.flags.DEFINE_integer('save_interval', 10, 'save interval of model')

tf.compat.v1.app.flags.DEFINE_bool('en_remove_output_dir', False, 'enable removing output dir')




tf.compat.v1.app.flags.DEFINE_string('model_name', 'snn_train_mlp_mnist', 'model name')
tf.compat.v1.app.flags.DEFINE_string('config_name', '', 'config name')


#
#tf.compat.v1.app.flags.DEFINE_string('dataset', 'MNIST', 'dataset')
tf.compat.v1.app.flags.DEFINE_string('ann_model', 'MLP', 'neural network model')

#
tf.compat.v1.app.flags.DEFINE_boolean('verbose',True, 'verbose mode')
tf.compat.v1.app.flags.DEFINE_boolean('verbose_visual',True, 'verbose visual mode')

#


#
tf.compat.v1.app.flags.DEFINE_bool('f_real_value_input_snn',False,'f_real_value_input_snn')
tf.compat.v1.app.flags.DEFINE_bool('f_spike_max_pool',False,'f_spike_max_pool')

tf.compat.v1.app.flags.DEFINE_bool('f_ws',False,'wieghted synapse')
tf.compat.v1.app.flags.DEFINE_float('p_ws',8,'period of wieghted synapse')

#tf.compat.v1.app.flags.DEFINE_integer('num_class',1000,'number_of_class')


tf.compat.v1.app.flags.DEFINE_bool('f_positive_vmem',False,'positive vmem')
tf.compat.v1.app.flags.DEFINE_bool('f_tot_psp',False,'accumulate total psp')

tf.compat.v1.app.flags.DEFINE_bool('f_isi',False,'isi stat')
tf.compat.v1.app.flags.DEFINE_bool('f_refractory',False,'refractory mode')

tf.compat.v1.app.flags.DEFINE_bool('f_comp_act',False,'compare activation')
tf.compat.v1.app.flags.DEFINE_bool('f_entropy',False,'entropy test')


tf.compat.v1.app.flags.DEFINE_bool('f_save_result',True,'save result to xlsx file')

# data.py - imagenet data
tf.compat.v1.app.flags.DEFINE_string('data_path_imagenet', './imagenet', 'data path imagenet')
tf.compat.v1.app.flags.DEFINE_integer('k_pathces', 5, 'patches for test (random crop)')
tf.compat.v1.app.flags.DEFINE_integer('input_size', 28, 'input image width / height')


#


#
tf.compat.v1.app.flags.DEFINE_bool('f_data_std', True, 'data_standardization')


# pruning
#tf.compat.v1.app.flags.DEFINE_bool('f_pruning_channel', False, 'purning - channel')


tf.compat.v1.app.flags.DEFINE_string('path_result_root','./result/', 'path result root')

# temporal coding
#tf.compat.v1.app.flags.DEFINE_float('tc',10.0,'time constant for temporal coding')
#tf.compat.v1.app.flags.DEFINE_float('time_window',20.0,'time window of each layer for temporal coding')
#tf.compat.v1.app.flags.DEFINE_float('time_fire_start',20.0,'time fire start (integration time before starting fire) for temporal coding')
#tf.compat.v1.app.flags.DEFINE_float('time_fire_duration',20.0,'time fire duration for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_s_dnn_tk_info',False,'info - surrogate DNN tk')
tf.compat.v1.app.flags.DEFINE_integer('tc',10,'time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_float('td',0.5,'time delay for temporal coding')
tf.compat.v1.app.flags.DEFINE_integer('time_window',20,'time window of each layer for temporal coding')
#tf.compat.v1.app.flags.DEFINE_integer('time_fire_start',20,'time fire start (integration time before starting fire) for temporal coding')
#tf.compat.v1.app.flags.DEFINE_integer('time_fire_duration',20,'time fire duration for temporal coding')
tf.compat.v1.app.flags.DEFINE_float('time_fire_start',20,'time fire start (integration time before starting fire) for temporal coding')
tf.compat.v1.app.flags.DEFINE_float('time_fire_duration',20,'time fire duration for temporal coding')
# moved to new
#tf.compat.v1.app.flags.DEFINE_bool('f_record_first_spike_time',False,'flag - recording first spike time of each neuron')
tf.compat.v1.app.flags.DEFINE_bool('f_visual_record_first_spike_time',False,'flag - visual recording first spike time of each neuron')
tf.compat.v1.app.flags.DEFINE_bool('f_train_tk',False,'flag - enable to train time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_train_tk_outlier',True,'flag - enable to outlier roubst train time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_string('tk_file_name',None,'temporal kernel (tk) file name - prefix (tc and td)')
tf.compat.v1.app.flags.DEFINE_bool('f_load_time_const',False,'flag - load time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_integer('time_const_num_trained_data',0,'number of trained data - time constant')
tf.compat.v1.app.flags.DEFINE_integer('time_const_save_interval',10000,'save interval - training time constant')
tf.compat.v1.app.flags.DEFINE_integer('epoch_train_time_const',1,'epoch - training time constant')

tf.compat.v1.app.flags.DEFINE_bool('f_tc_based',False,'flag - tau based')
tf.compat.v1.app.flags.DEFINE_integer('n_tau_fire_start',4,'n tau - fire start')
tf.compat.v1.app.flags.DEFINE_integer('n_tau_fire_duration',4,'n tau - fire duration')
tf.compat.v1.app.flags.DEFINE_integer('n_tau_time_window',4,'n tau - time window')



# SNN trianing w/ TTFS coding
tf.compat.v1.app.flags.DEFINE_integer("init_first_spike_time_n",-1,"init_first_spike_time = init_first_spike_n x time_windw")

# surrogate training model
tf.compat.v1.app.flags.DEFINE_bool("f_surrogate_training_model", False, "flag - surrogate training model (DNN)")

#
tf.compat.v1.app.flags.DEFINE_bool("f_overwrite_train_model", False, "overwrite trained model")

#
tf.compat.v1.app.flags.DEFINE_bool("f_validation_snn", False, "validation on SNN")

#
tf.compat.v1.app.flags.DEFINE_bool("en_tensorboard_write", False, "Tensorboard write")


#################################################################################
## Deep SNNs training w/ tepmoral information - surrogate DNN model
#################################################################################
#tf.compat.v1.app.flags.DEFINE_integer('epoch_start_train_tk',100,'epoch start train tk')
#tf.compat.v1.app.flags.DEFINE_float('w_train_tk',1,'weight train tk')
#tf.compat.v1.app.flags.DEFINE_float('w_train_tk_reg',0,'weight train tk regularization, lambda')
#tf.compat.v1.app.flags.DEFINE_string('t_train_tk_reg','L2-I','type train tk regularization, lambda')
#tf.compat.v1.app.flags.DEFINE_bool('f_train_tk_reg',False,'flag for tk regularization')
#tf.compat.v1.app.flags.DEFINE_string('train_tk_strategy','N','traing tk strategy')
#
#tf.compat.v1.app.flags.DEFINE_integer('epoch_start_train_t_int',100,'epoch start train t_int')
#tf.compat.v1.app.flags.DEFINE_integer('epoch_start_train_floor',100,'epoch start train floor')
#tf.compat.v1.app.flags.DEFINE_integer('epoch_start_train_clip_tw',1,'epoch start train clip tw')
#tf.compat.v1.app.flags.DEFINE_integer('epoch_start_loss_enc_spike',100,'epoch start encoded spike loss')
#
##
#tf.compat.v1.app.flags.DEFINE_float('bypass_pr',1.0,'bypass probabilty')
#tf.compat.v1.app.flags.DEFINE_integer('bypass_target_epoch',1,'bypass target epoch')
#
##
#tf.compat.v1.app.flags.DEFINE_bool('f_loss_enc_spike',False,'flag for encoded spike loss')
#
##
#tf.compat.v1.app.flags.DEFINE_float('w_loss_enc_spike',0.001,'weight of encoded spike loss')
## TODO: update below two parameters in run.sh (they are only denoted in run_enc_dec_tk.sh)
#tf.compat.v1.app.flags.DEFINE_string('d_loss_enc_spike','b','target distribution of encoded spike loss')
#tf.compat.v1.app.flags.DEFINE_string('ems_loss_enc_spike','n','encoded spike loss - encoding max spike: f (fixed), n (n x time window=nt)')
#
## coefficient of beta distribution
#tf.compat.v1.app.flags.DEFINE_float('beta_dist_a',0.1,'coefficient of beta distribution - alpha')
#tf.compat.v1.app.flags.DEFINE_float('beta_dist_b',0.1,'coefficient of beta distribution - beta')
#
##
#tf.compat.v1.app.flags.DEFINE_integer('enc_st_n_tw',10,'target max encoded spike time - number of time window')
##tf.compat.v1.app.flags.DEFINE_float('enc_st_n_tw',10,'target max encoded spike time - number of time window')
#
#tf.compat.v1.app.flags.DEFINE_bool('f_td_training',True,'flag td training')
#
## quantization-aware vth - TTFS coding, surrogate training model
#tf.compat.v1.app.flags.DEFINE_bool('f_qvth',False,'quantization-aware vth, rounding function')
#
##
#tf.compat.v1.app.flags.DEFINE_string('tfboard_log_file_name',None,'tfboard log file name')
#


## noise
tf.compat.v1.app.flags.DEFINE_bool('noise_en',False,'noise injection mode enable')
#tf.compat.v1.app.flags.DEFINE_string('noise_type',None,'noise type - DEL ..')
#tf.compat.v1.app.flags.DEFINE_float('noise_pr',0.1,'noise probability for DEL')
#tf.compat.v1.app.flags.DEFINE_bool('noise_robust_en',False,'noise robust mode enable')
#tf.compat.v1.app.flags.DEFINE_bool('noise_robust_comp_pr_en',False,'noise robust compenstation pr enable - only DEL')
#tf.compat.v1.app.flags.DEFINE_integer('noise_robust_spike_num',0,'noise robust spike number')
#tf.compat.v1.app.flags.DEFINE_integer('rep',-1,'repeat - noise experiments')


################################################################################
# new configurations
################################################################################

################################
# Common
################################

# train mode
#tf.compat.v1.app.flags.DEFINE_bool('train', False, 'train mode')
tf.compat.v1.app.flags.DEFINE_bool('train', True, 'train mode')

#
#tf.compat.v1.app.flags.DEFINE_bool('dnn_to_snn', False, 'dnn-to-snn conversion')
tf.compat.v1.app.flags.DEFINE_bool('dnn_to_snn', True, 'dnn-to-snn conversion')

# neural network mode
tf.compat.v1.app.flags.DEFINE_string('nn_mode', 'ANN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')
#tf.compat.v1.app.flags.DEFINE_string('nn_mode', 'SNN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')

# models
#tf.compat.v1.app.flags.DEFINE_string('model', 'VGG16', 'model')
#tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet18', 'model')
#tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet20', 'model')
#tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet32', 'model')
#tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet34', 'model')
tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet44', 'model')
#tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet50', 'model')
#tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet18V2', 'model')
#tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet20V2', 'model')
#tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet32V2', 'model')
#tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet34V2', 'model')
#tf.compat.v1.app.flags.DEFINE_string('model', 'ResNet50V2', 'model')


# datasets
tf.compat.v1.app.flags.DEFINE_string('dataset', 'CIFAR10', 'dataset')
#tf.compat.v1.app.flags.DEFINE_string('dataset', 'CIFAR100', 'dataset')

#
#tf.compat.v1.app.flags.DEFINE_bool('load_best_model', True, 'load best model (model, dataset)')
tf.compat.v1.app.flags.DEFINE_bool('load_best_model', False, 'load best model (model, dataset)')

#
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 100, '')
#tf.compat.v1.app.flags.DEFINE_integer('batch_size_inf', 1000, '')
tf.compat.v1.app.flags.DEFINE_integer('batch_size_inf', 500, '')
#tf.compat.v1.app.flags.DEFINE_integer('batch_size_inf', 1, '')

#
# VGG
#tf.compat.v1.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
# ResNet
tf.compat.v1.app.flags.DEFINE_float('learning_rate', 0.2, 'learning rate')

# regularizer
tf.compat.v1.app.flags.DEFINE_string('regularizer', 'L2', 'L2 or L1 regularizer')
# VGG
#tf.compat.v1.app.flags.DEFINE_float('lmb',1.0E-5, 'lambda')
# ResNet
tf.compat.v1.app.flags.DEFINE_float('lmb',1.0E-4, 'lambda')


# data augmentation
tf.compat.v1.app.flags.DEFINE_string('data_aug_mix', 'cutmix', 'data augmentation - mixup or cutmix or None')




## data_format - DO NOT TOUCH
tf.compat.v1.app.flags.DEFINE_string('data_format', 'channels_last', 'data format')

#
tf.compat.v1.app.flags.DEFINE_boolean('use_bias', True, 'use bias')

################
# Directories
################
tf.compat.v1.app.flags.DEFINE_string('root_tensorboard', './tensorboard/', 'root - tensorboard')

tf.compat.v1.app.flags.DEFINE_string('root_model_best', './models_best', 'root model best')


#
tf.compat.v1.app.flags.DEFINE_string('root_results', './results', 'root results')

################
# Debug
################
tf.compat.v1.app.flags.DEFINE_bool('debug_mode', False, 'debug mode')
#tf.compat.v1.app.flags.DEFINE_bool('debug_mode', True, 'debug mode')

tf.compat.v1.app.flags.DEFINE_bool('full_test', True, 'full dataset test')
#tf.compat.v1.app.flags.DEFINE_bool('full_test', False, 'full dataset test')

tf.compat.v1.app.flags.DEFINE_integer('idx_test_data', 8, 'start index of test data')
#tf.compat.v1.app.flags.DEFINE_integer('idx_test_data', 9, 'start index of test data')
tf.compat.v1.app.flags.DEFINE_integer('num_test_data', 500, 'number of test data')
#tf.compat.v1.app.flags.DEFINE_integer('num_test_data', 100, 'number of test data')
#tf.compat.v1.app.flags.DEFINE_integer('num_test_data', 10, 'number of test data')
#tf.compat.v1.app.flags.DEFINE_integer('num_test_data', 3, 'number of test data')
#tf.compat.v1.app.flags.DEFINE_integer('num_test_data', 1, 'number of test data')

################################
# SNN
################################

# neuron type in SNN
tf.compat.v1.app.flags.DEFINE_string('n_type', 'IF', 'LIF or IF: neuron type')

tf.compat.v1.app.flags.DEFINE_string('input_spike_mode','REAL','input spike mode - REAL, POISSON, WEIGHTED_SPIKE, others...')
tf.compat.v1.app.flags.DEFINE_string('neural_coding','RATE','neural coding - RATE, WEIGHTED_SPIKE, PROPOSED')

#
tf.compat.v1.app.flags.DEFINE_bool('binary_spike', True, 'binary spike activation, if false - vth activation')
#tf.compat.v1.app.flags.DEFINE_bool('binary_spike', False, 'binary spike activation, if false - vth activation')

#
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 4.0, 'initial value of vth')
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 3.0, 'initial value of vth')
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 2.0, 'initial value of vth')
tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 1.0, 'initial value of vth')
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.9, 'initial value of vth')
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.7, 'initial value of vth')
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.5, 'initial value of vth')
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.4, 'initial value of vth')
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.3, 'initial value of vth')
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.2, 'initial value of vth')
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.1, 'initial value of vth')
#tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.01, 'initial value of vth')

tf.compat.v1.app.flags.DEFINE_float('n_in_init_vth', 0.1, 'initial value of vth of n_in')
tf.compat.v1.app.flags.DEFINE_float('n_init_vinit', 0.0, 'initial value of vinit')
tf.compat.v1.app.flags.DEFINE_float('n_init_vrest', 0.0, 'initial value of vrest')

#
tf.compat.v1.app.flags.DEFINE_enum('snn_output_type',"VMEM", ["SPIKE", "VMEM", "FIRST_SPIKE_TIME"], "snn output type")

#
#tf.compat.v1.app.flags.DEFINE_integer('time_step', 2048, 'time steps per sample in SNN')
#tf.compat.v1.app.flags.DEFINE_integer('time_step', 1024, 'time steps per sample in SNN')
#tf.compat.v1.app.flags.DEFINE_integer('time_step', 256, 'time steps per sample in SNN')
tf.compat.v1.app.flags.DEFINE_integer('time_step', 128, 'time steps per sample in SNN')
#tf.compat.v1.app.flags.DEFINE_integer('time_step', 64, 'time steps per sample in SNN')
#tf.compat.v1.app.flags.DEFINE_integer('time_step', 32, 'time steps per sample in SNN')
tf.compat.v1.app.flags.DEFINE_integer('time_step_save_interval',10,'snn test save interval')
#tf.compat.v1.app.flags.DEFINE_integer('time_step_save_interval',2,'snn test save interval')
#tf.compat.v1.app.flags.DEFINE_integer('time_step_save_interval',1,'snn test save interval')

#
tf.compat.v1.app.flags.DEFINE_bool('f_record_first_spike_time',True,'flag - recording first spike time of each neuron')
#tf.compat.v1.app.flags.DEFINE_bool('f_record_first_spike_time',False,'flag - recording first spike time of each neuron')


################
# DNN-to-SNN conversion
################

# weight normalization
#
#tf.compat.v1.app.flags.DEFINE_bool('f_fused_bn',True,'f_fused_bn')
tf.compat.v1.app.flags.DEFINE_bool('f_fused_bn',False,'f_fused_bn')

#
#tf.compat.v1.app.flags.DEFINE_bool('f_w_norm_data',True,'f_w_norm_data')
tf.compat.v1.app.flags.DEFINE_bool('f_w_norm_data',False,'f_w_norm_data')

tf.compat.v1.app.flags.DEFINE_bool('f_write_stat',False,'write stat')
#tf.compat.v1.app.flags.DEFINE_bool('f_write_stat',True,'write stat')

tf.compat.v1.app.flags.DEFINE_bool('f_stat_train_mode',True,'stat with train data')
#tf.compat.v1.app.flags.DEFINE_bool('f_stat_train_mode',False,'stat with train data')

tf.compat.v1.app.flags.DEFINE_string('path_stat','./stat/', 'path stat')
tf.compat.v1.app.flags.DEFINE_string('prefix_stat', '', 'prefix of stat file name')

#tf.compat.v1.app.flags.DEFINE_string('act_save_mode','channel','activation save mode')


################
# calibration - DNN-to-SNN conversion
################
#tf.compat.v1.app.flags.DEFINE_bool('calibration_weight',True,'calibration - weight')
tf.compat.v1.app.flags.DEFINE_bool('calibration_weight',False,'calibration - weight')

tf.compat.v1.app.flags.DEFINE_bool('calibration_weight_post',True,'calibration - weight')
#tf.compat.v1.app.flags.DEFINE_bool('calibration_weight_post',False,'calibration - weight')

#tf.compat.v1.app.flags.DEFINE_bool('calibration_bias',True,'calibration - weight')
tf.compat.v1.app.flags.DEFINE_bool('calibration_bias',False,'calibration - weight')

#tf.compat.v1.app.flags.DEFINE_bool('calibration_vth',True,'calibration - vth')
tf.compat.v1.app.flags.DEFINE_bool('calibration_vth',False,'calibration - vth')

#tf.compat.v1.app.flags.DEFINE_bool('calibration_vmem',True,'calibration - vmem')
tf.compat.v1.app.flags.DEFINE_bool('calibration_vmem',False,'calibration - vmem')

tf.compat.v1.app.flags.DEFINE_bool('bias_control',True,'bias control')
#tf.compat.v1.app.flags.DEFINE_bool('bias_control',False,'bias control')

#tf.compat.v1.app.flags.DEFINE_bool('vth_toggle',True,'vth toggle mode')
tf.compat.v1.app.flags.DEFINE_bool('vth_toggle',False,'vth toggle mode')

#tf.compat.v1.app.flags.DEFINE_float('vth_toggle_init',0.3,'vth toggle init - toggle between {init, 2-init}')
#tf.compat.v1.app.flags.DEFINE_float('vth_toggle_init',0.5,'vth toggle init - toggle between {init, 2-init}')
#tf.compat.v1.app.flags.DEFINE_float('vth_toggle_init',0.6,'vth toggle init - toggle between {init, 2-init}')
#tf.compat.v1.app.flags.DEFINE_float('vth_toggle_init',0.7,'vth toggle init - toggle between {init, 2-init}')
#tf.compat.v1.app.flags.DEFINE_float('vth_toggle_init',0.8,'vth toggle init - toggle between {init, 2-init}')
tf.compat.v1.app.flags.DEFINE_float('vth_toggle_init',0.9,'vth toggle init - toggle between {init, 2-init}')
#tf.compat.v1.app.flags.DEFINE_float('vth_toggle_init',0.95,'vth toggle init - toggle between {init, 2-init}')


#
tf.compat.v1.app.flags.DEFINE_bool('calibration_bias_ICLR_21',False,'calibration - bias, ICML-21')
#tf.compat.v1.app.flags.DEFINE_bool('calibration_bias_ICLR_21',True,'calibration - bias, ICML-21')

#tf.compat.v1.app.flags.DEFINE_bool('calibration_bias_ICML_21',False,'calibration - bias, ICML-21')
tf.compat.v1.app.flags.DEFINE_bool('calibration_bias_ICML_21',True,'calibration - bias, ICML-21')

tf.compat.v1.app.flags.DEFINE_bool('calibration_vmem_ICML_21',False,'calibration - bias, ICML-21')
#tf.compat.v1.app.flags.DEFINE_bool('calibration_vmem_ICML_21',True,'calibration - bias, ICML-21')


#
conf = flags.FLAGS
