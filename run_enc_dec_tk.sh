#!/bin/bash

###############################################################################
## Aguments
###############################################################################
# $1: # of epoch for training surrogate model
# $2:

#epoch=$1
#epoch_start_train_tk=$2
#epoch_start_train_t_int=$3
#epoch_start_train_floor=$4
#epoch_start_train_clip_tw=$5
#epoch_start_loss_enc_spike=$6
#
#bypass_pr=$7
#bypass_target_epoch=$8
#
#cp_mode=$9
#training_mode=$10



#
source ../05_SNN/venv/bin/activate
#source ./venv/bin/activate


###############################################################################
## Include
###############################################################################
# path
source ./scripts/path.conf

# weight normalization
source ./scripts/weight_norm.conf

# utils
source ./scripts/utils.conf

# model description
source ./scripts/models_descript.conf



###############################################################################
## Debug
###############################################################################

#verbose=True
verbose=False

#verbose_visual=True
verbose_visual=False

en_tensorboard_write=True
#en_tensorboard_write=False


###############################################################################
## Model & Dataset
###############################################################################
nn_mode='ANN'
#nn_mode='SNN'


#exp_case='CNN_MNIST'
exp_case='VGG16_CIFAR-10'
#exp_case='VGG16_CIFAR-100'
#exp_case='ResNet50_ImageNet'


###############################################################################
## Deep SNNs training w/ temporal information - surrogate DNN model
###############################################################################


epoch_start_train_tk=$2
epoch_start_train_t_int=$3
epoch_start_train_floor=$4
epoch_start_train_clip_tw=$5
epoch_start_loss_enc_spike=$6

#
bypass_pr=$7
bypass_target_epoch=$8


#
# encoded spike distribution loss
f_loss_enc_spike=False
#f_loss_enc_spike=True

# weight of loss
w_loss_enc_spike=10

# coefficient of beta distribution for KL loss
#beta_dist_a=0.1
#beta_dist_b=0.9

#beta_dist_a=0.9
#beta_dist_b=0.1

beta_dist_a=1
beta_dist_b=2

# target max encoded spike time - number of time window
enc_st_n_tw=2

###############################################################################
## Run
###############################################################################

#training_mode=True
#training_mode=False
training_mode=$10

#
# If this flag is False, then the trained model is overwritten
load_and_train=False
#load_and_train=True

#
f_validation_snn=False
#f_validation_snn=True

#
regularizer='L2'
#regularizer='L1'
#regularizer='L1_L2'

#
# TODO: depreicated - remove
#f_overwrite_train_model=True
f_overwrite_train_model=False


# full test
f_full_test=True
#f_full_test=False


#
#time_step=1100
#time_step=1000
#time_step=900
#time_step=700
#time_step=400
#time_step=300
#time_step=200
#time_step=30


#time_step_save_interval=50
#time_step_save_interval=100
#time_step_save_interval=20
#time_step_save_interval=1



# for MNIST, CNN
# DNN-to-SNN, inference
time_step=200
#time_step_save_interval=10
#time_step_save_interval=2



# for MNIST, CNN
# SNN training
#time_step=200
time_step_save_interval=100
#time_step_save_interval=40
#time_step_save_interval=20
#time_step_save_interval=10
#time_step_save_interval=5


###############################################################
# Batch size - small test
###############################################################

# for MNIST, CNN
#batch_size=250
batch_size=25
#batch_size=1

idx_test_dataset_s=0
num_test_dataset=25
#num_test_dataset=500
#num_test_dataset=50000
#num_test_dataset=10000
#num_test_dataset=250









###############################################################################
## Neural coding
###############################################################################

#
## input spike mode
#
input_spike_mode='REAL'
#input_spike_mode='POISSON'
#input_spike_mode='WEIGHTED_SPIKE'
#input_spike_mode='BURST'
#input_spike_mode='TEMPORAL'

#
## neural coding
#
#neural_coding='RATE'
#neural_coding='WEIGHTED_SPIKE'
#neural_coding='BURST'
neural_coding='TEMPORAL'
#neural_coding='NON_LINEAR'     # PF-Neuron


# TODO: it should be deprecated
# default: False
#f_real_value_input_snn=True
f_real_value_input_snn=False


# TODO: it should be deprecated
# weighted synapse
# should modify
#f_ws=True
f_ws=False


###############################################################################
## Neuron
###############################################################################

# default: IF
#n_type='LIF'
n_type='IF'

# positive vmem
#f_positive_vmem=True
f_positive_vmem=False

# refractory mode
#f_refractory=True
f_refractory=False


#vth=1.1
vth=1.0        # weight norm. default
#vth=0.9

vth_n_in=${vth}


###############################################################
# for weighted spike coding (phase coding)
###############################################################
p_ws=8

###############################################################
# for burst coding
###############################################################
if [ ${input_spike_mode} = 'BURST' ]
then
    vth_n_in=0.125
    #vth_n_in=0.0625
    #vth_n_in=0.0078125  # 1/256
fi


if [ ${neural_coding} = 'BURST' ]
then
    # decreasing
    #vth=1.0
    #vth=0.5        # 1/2
    #vth=0.25       # 1/4
    # increasing
    vth=0.125      # 1/8 # default
    #vth=0.0625     # 1/16
    #vth=0.03125    # 1/32
    #vth=0.015625   # 1/64
    #vth=0.015625   # 1/128
    #vth=0.0078125  # 1/256
fi



###############################################################################
## SNN output type
###############################################################################

#snn_output_type='SPIKE'
snn_output_type='VMEM'
#snn_output_type='FIRST_SPIKE_TIME'       # for TTFS coding


#if [ ${training_mode} = True ] && [ ${neural_coding} = "TEMPORAL" ]
#then
#    snn_output_type='FIRST_SPIKE_TIME'
#fi



###############################################################################
## configurations for TTFS coding (TEMPORAL)
###############################################################################



###############################################################
## Gradient-based optimization of tc and td
## only for the TTFS coding (TEMPORAL)
## "Deep Spiking Neural Networks with Time-to-first-spike Coding", DAC-20
###############################################################

batch_run_mode=False
#batch_run_mode=True

#
batch_run_train_tc=False
#batch_run_train_tc=True
###############################################################

f_visual_record_first_spike_time=False
#f_visual_record_first_spike_time=True

# train time cosntant for temporal coding
f_train_time_const=False
#f_train_time_const=True

#
f_train_time_const_outlier=True
#f_train_time_const_outlier=False


f_load_time_const=False
#f_load_time_const=True

#
time_const_init_file_name='./temporal_coding'

#
#time_const_num_trained_data=50000
time_const_num_trained_data=40000
#time_const_num_trained_data=30000
#time_const_num_trained_data=20000
#time_const_num_trained_data=10000
#time_const_num_trained_data=0

#
time_const_save_interval=1000

#
epoch_train_time_const=1



# full test
#time_step=1500
#time_step_save_interval=100

# detail
#time_step=3000
#time_step=1500
#time_step=600
#time_step=900
#time_step=700
#time_step=500
#time_step_save_interval=1

# small test
#time_step=1000
#time_step_save_interval=100

# comp act
#time_step=1000
#time_step_save_interval=1

# isi
#time_step=10
#time_step_save_interval=1

# entropy
#time_step=1000
#time_step_save_interval=100




# time constant for temporal coding
#tc=25
#time_window=100

#tc=25
#time_window=100


# TTFS - MNIST default setting
#tc=5
#time_fire_start=20    # integration duration - n x tc
#time_fire_duration=20   # time window - n x tc
#time_window=${time_fire_duration}

# TTFS - CIFAR-10 default setting
#tc=10
tc=20
#tc=16
time_fire_start=80    # integration duration - n x tc
time_fire_duration=80   # time window - n x tc
time_window=${time_fire_duration}


# TTFS - CIFAR-10
tc=10
time_fire_start=40    # integration duration - n x tc
time_fire_duration=40   # time window - n x tc
time_window=${time_fire_duration}


# TTFS - CIFAR-10
tc=8
time_fire_start=32    # integration duration - n x tc
time_fire_duration=32   # time window - n x tc
time_window=${time_fire_duration}


#f_tc_based=True
f_tc_based=False

n_tau_fire_start=3
n_tau_fire_duration=3
n_tau_time_window=${n_tau_fire_duration}
###############################################################################



###############################################################################
## configurations for SNN training w/ TTFS coding (TEMPORAL)
###############################################################################

#
# training SNN w/ surrogate model (DNN)
# actual training is performed in the surrogate DNN model
f_surrogate_training_model=True
#f_surrogate_training_model=False


#
# Initial value for the first spike time
# This value is used as as ground truth representing zero in one-hot label
# init_first_spike_time = ${init_first_spike_time_n} x ${time_window}
#init_first_spike_time_n=10
init_first_spike_time_n=1





#
# TODO: move these code below later to "DO NOT TOUCH"

if [ ${f_surrogate_training_model} = True ]
then
    f_w_norm_data=False
else
    f_validation_snn=False
fi





###############################################################################


#num_epoch=300
#num_epoch=500
num_epoch=1000
save_interval=50

lr=0.001
lr_decay=0.1
lr_decay_step=100
lamb=0.0001
#batch_size_training=1
batch_size_training=128



##############################

if [ ${f_isi} = True ]
then
    #time_step=1000
    #time_step_save_interval=100

    #batch_size=100
    #num_test_dataset=100


    # test
    time_step=1000
    time_step_save_interval=100

    #batch_size=100
    #num_test_dataset=100


    batch_size=2
    num_test_dataset=2

fi



###############################################################################
## TEST ONLY
###############################################################################

# channel pruning - only support in SNN mode
#f_pruning_channel=True
f_pruning_channel=False



###############################################################################
## DO NOT TOUCH
###############################################################################

if [ ${training_mode} = True ]
then
    _exp_case=TRAIN_${exp_case}
else
    if [ ${f_surrogate_training_model} = True ]
    then
        _exp_case=INFER_${exp_case}_SUR
    else
        _exp_case=INFER_${exp_case}
    fi
fi




if [ ${training_mode} = True ]
then

    if [ ${load_and_train} = True ]
    then
        en_load_model=True
    else
        en_load_model=False
    fi


    en_train=True

    #nn_mode='ANN'
    f_fused_bn=False
    f_stat_train_mode=False
    f_w_norm_data=False
    # TODO: training and inference bach size seperate (training / validation)
    batch_size=${batch_size_training}
    f_full_test=True

else
    en_train=False
    en_load_model=True
fi


case ${_exp_case} in
###############################################################
## Inference setup
###############################################################
INFER_CNN_MNIST)
    echo "Inference mode - "${nn_mode}", Model: CNN, Dataset: MNIST"
    dataset='MNIST'
    ann_model='CNN'
    # TODO: model name parameterize
    #model_name='cnn_mnist_ro_0'
    model_name='cnn_mnist_train_ANN'

    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        if [ ${f_tc_based} = True ]
        then
            time_step="$((5*${n_tau_fire_start}*${tc} + ${n_tau_fire_duration}*${tc}))"
        else
            time_step="$((4*${time_fire_start} + ${time_fire_duration}))"
        fi
    fi
    ;;


INFER_VGG16_CIFAR-10)
    echo "Inference mode - "${nn_mode}", Model: VGG16, Dataset: CIFAR-10"
    ann_model='CNN'         # CNN-CIFAR: VGG16
    dataset='CIFAR-10'
    model_name='vgg_cifar_ro_0'
    #model_name='vgg16_cifar10_train_ANN'

    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        #time_step="$((17 * ${time_window}))"
        #time_step="$((16*${time_fire_start}*${tc} + ${time_fire_duration}*${tc}))"

        if [ ${f_tc_based} = True ]
        then
            time_step="$((18*${n_tau_fire_start}*${tc} + ${n_tau_fire_duration}*${tc}))"
        else
            time_step="$((16*${time_fire_start} + ${time_fire_duration}))"
        fi
    fi
    ;;


INFER_VGG16_CIFAR-100)
    echo "Inference mode - "${nn_mode}", Model: VGG16, Dataset: CIFAR-100"
    ann_model='CNN'         # CNN-CIFAR: VGG16
    dataset='CIFAR-100'
    model_name='vgg_cifar100_ro_0'
    #model_name='vgg16_cifar100_train_ANN'


    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        #time_step="$((17 * ${time_window}))"
        #time_step="$((16*${time_fire_start}*${tc} + ${time_fire_duration}*${tc}))"

        if [ ${f_tc_based} = True ]
        then
            time_step="$((18*${n_tau_fire_start}*${tc} + ${n_tau_fire_duration}*${tc}))"
        else
            time_step="$((16*${time_fire_start} + ${time_fire_duration}))"
        fi
    fi
    ;;

INFER_ResNet50_ImageNet)
    echo "Inference mode - "${nn_mode}", Model: ResNet50, Dataset: ImageNet"
    ann_model='ResNet50'
    dataset='ImageNet'
    model_name='resnet50_imgnet'

    if [ ${f_full_test} = True ]
    then
        batch_size=250
        idx_test_dataset_s=0
        num_test_dataset=50000
    fi
    ;;


###############################################################
## Inference - surrogate DNN model
###############################################################
INFER_CNN_MNIST_SUR)
    echo "Inference mode - "${nn_mode}", Model: CNN (surrogate), Dataset: MNIST"
    dataset='MNIST'
    ann_model='CNN'

    # SNN, 5, 10, 20 - 99.31%
    # TensorBoard log - 20200504-1107
    #model_name='cnn_mnist_train_ANN_surrogate_0'


    # SNN, 5, 20, 20 - 99.3%, # spikes - 2506
    # TensorBoard log - 20200504-1427
    #model_name='cnn_mnist_train_ANN_surrogate_1'


    # SNN, 5, 20, 20 - 99.33%, # spikes - 1463, epoch - 608
    # TensorBoard log - 20200506-1445
    #model_name='cnn_mnist_train_ANN_surrogate_2'

    # SNN, 5, 20, 20 - 99.37%, # spikes - 1600, epoch - 635
    # TensorBoard log - 20200507-1042
    #model_name='cnn_mnist_train_ANN_surrogate_3'

    #
    # SNN, 5, 20, 20 - (val, best) 99.42%, (val) 13.16%  (test) ? # spikes - 160, epoch - 6306
    # TensorBoard log - 20200509-0055
    # regularization needed, tc increased as training progressed
    #model_name='cnn_mnist_train_ANN_surrogate_no_reg'

    #
    # SNN, 5, 20, 20 - 99.42%, # spikes - 1460, epoch - 924
    # regularizer - encoded spike time (KL loss, beta dist.(0.9,0.1))
    # TensorBoard log -
    #model_name='cnn_mnist_train_ANN_surrogate_reg-enc_0'

    #
    # SNN, 5, 20, 20 - 99.42%, # spikes - 1460, epoch - 890
    # regularizer - encoded spike time (KL loss, beta dist.(0.1,0.9))
    # TensorBoard log -
    #model_name='cnn_mnist_train_ANN_surrogate_reg-enc_3'


    model_name='cnn_mnist_train_ANN_surrogate'

    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        if [ ${f_tc_based} = True ]
        then
            time_step="$((5*${n_tau_fire_start}*${tc} + ${n_tau_fire_duration}*${tc}))"
        else
            time_step="$((4*${time_fire_start} + ${time_fire_duration}))"
        fi
    fi
    ;;


INFER_VGG16_CIFAR-10_SUR)
    echo "Inference mode - "${nn_mode}", Model: VGG16 (surrogate), Dataset: CIFAR-10"
    dataset='CIFAR-10'
    ann_model='VGG16'

    #
    # SNN, 20, 80, 80 - 88.79 %, # spikes - 9.091E+4
    # regularizer -
    # TensorBoard log -
    # model_name='vgg16_cifar10_train_ANN_surrogate_0'

    model_name='vgg16_cifar10_train_ANN_surrogate'
    #model_name='vgg16_cifar10_train_ANN_surrogate_mix_1000_65'

    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        if [ ${f_tc_based} = True ]
        then
            time_step="$((18*${n_tau_fire_start}*${tc} + ${n_tau_fire_duration}*${tc}))"
        else
            time_step="$((16*${time_fire_start} + ${time_fire_duration}))"
        fi
    fi
    ;;






###############################################################
## Training setup
###############################################################
TRAIN_CNN_MNIST)
    echo "Training mode - "${nn_mode}", Model: CNN, Dataset: MNIST"
    dataset='MNIST'
    ann_model='CNN'

    num_epoch=10000


    if [ ${f_surrogate_training_model} = True ]
    then
        model_name='cnn_mnist_train_'${nn_mode}_'surrogate'
    else
        model_name='cnn_mnist_train_'${nn_mode}
    fi

    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        if [ ${f_tc_based} = True ]
        then
            time_step="$((5*${n_tau_fire_start}*${tc} + ${n_tau_fire_duration}*${tc}))"
        else
            time_step="$((4*${time_fire_start} + ${time_fire_duration}))"
        fi
    fi
    ;;

TRAIN_VGG16_CIFAR-10)
    echo "Training mode - "${nn_mode}", Model: VGG16, Dataset: CIFAR-10"
    dataset='CIFAR-10'
    ann_model='VGG16'


    #
    #model_name='vgg16_cifar_train_ANN_surrogate_88_2'


    if [ ${f_surrogate_training_model} = True ]
    then
        #num_epoch=5000
        num_epoch=$1
        model_name='vgg16_cifar10_train_'${nn_mode}_'surrogate'
    else
        num_epoch=2000
        model_name='vgg16_cifar10_train_'${nn_mode}
    fi

    if [ ${f_full_test} = True ]
    then
        #batch_size=400
        batch_size=250
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        if [ ${f_tc_based} = True ]
        then
            time_step="$((18*${n_tau_fire_start}*${tc} + ${n_tau_fire_duration}*${tc}))"
        else
            time_step="$((16*${time_fire_start} + ${time_fire_duration}))"
        fi
    fi
    ;;

TRAIN_VGG16_CIFAR-100)
    echo "Training mode - "${nn_mode}", Model: VGG16, Dataset: CIFAR-100"
    dataset='CIFAR-100'
    ann_model='VGG16'

    #num_epoch=2000

    if [ ${f_surrogate_training_model} = True ]
    then
        num_epoch=$1
        model_name='vgg16_cifar100_train_'${nn_mode}_'surrogate'
    else
        num_epoch=4000
        model_name='vgg16_cifar100_train_'${nn_mode}
    fi

    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        if [ ${f_tc_based} = True ]
        then
            time_step="$((18*${n_tau_fire_start}*${tc} + ${n_tau_fire_duration}*${tc}))"
        else
            time_step="$((16*${time_fire_start} + ${time_fire_duration}))"
        fi
    fi
    ;;



*)
    echo "not supported experiment case:" ${_exp_case}
    exit 1
    ;;
esac




################################################
## batch run mode
################################################
if [ ${batch_run_mode} = True ]
then
    # for temporal coding
    tc=$1
    time_window=$2
    time_fire_start=$3
    time_fire_duration=$4
    time_step=$5
    time_step_save_interval=$6

fi
################################################


################################################
## batch run mode - SNN training (surrogate model)
################################################
if [ ${training_mode} = False ]
then
    ${model_name}=${model_name}_${log_file_name}
fi



###############################################
##
###############################################
# record first spike time of each neuron - it should be True for training time constant
# overwirte this flag as True if f_train_time_const is Ture
# it should be True when TTFS coding is used
if [ ${neural_coding} = "TEMPORAL" ]
then
    f_record_first_spike_time=True
    f_refractory=True
else
    f_record_first_spike_time=False
fi



###############################################3
## training time constant
###############################################3
if [ ${f_train_time_const} = True ]
then
    f_record_first_spike_time=True
fi
###############################################3

###############################################3
## visual spike time ditribution
###############################################3
if [ ${f_visual_record_first_spike_time} = True ]
then
    f_record_first_spike_time=True

    idx_test_dataset_s=0
    num_test_dataset=${batch_size}
fi
###############################################3




#
echo "time_step: " ${time_step}


case ${dataset} in
MNIST)
    echo "Dataset: MNIST"
    num_class=10
    input_size=28
    ;;
CIFAR-10)
    echo "Dataset: CIFAR-10"
    num_class=10
    input_size=28
    ;;
CIFAR-100)
    echo "Dataset: CIFAR-100"
    num_class=100
    input_size=28
    ;;
ImageNet)
    echo "Dataset: ImageNet"
    num_class=1000
    input_size=224
    ;;
*)
    echo "not supported dataset"
    num_class=0
    exit 1
    ;;
esac



if [ ${nn_mode} = 'ANN' ]
then
    echo "ANN mode"
    log_file_post_fix=''
else
    echo "SNN mode"
    log_file_post_fix=_${n_type}_time_step_${time_step}_vth_${vth}_c_infer
fi

#log_file=${path_log_root}/log_${model_name}_${nn_mode}_${log_file_post_fix}
#log_file


date=`date +%Y%m%d_%H%M`

path_result_root=${path_result_root}/${model_name}
time_const_root=${time_const_init_file_name}/${model_name}
path_log_root=${path_log_root}/${model_name}

#
mkdir -p ${path_log_root}
mkdir -p ${path_result_root}
mkdir -p ${time_const_root}

#log_file=${path_log_root}/log_${model_name}_${nn_mode}_${time_step}_${vth}_999_norm_${f_real_value_input_snn}_${date}
#log_file=${path_log_root}/${date}.log



#epoch=$1
#epoch_start_train_tk=$2
#epoch_start_train_t_int=$3
#epoch_start_train_floor=$4
#epoch_start_train_clip_tw=$5
#epoch_start_loss_enc_spike=$6
#
#bypass_pr=$7
#bypass_target_epoch=$8
log_file_name=ep-$1_tk-$2_int-$3_fl-$4_cl-$5_le-$6_bp-$7_bt-$8
log_file=${path_log_root}/${log_file_name}.log
tfboard_log_file_name=${log_file_name}

#{ unbuffer time kernprof -l main.py \
#{ CUDA_VISIBLE_DEVICES=0 unbuffer time python main.py \
#{ unbuffer time python -m line_profiler main.py \
{ unbuffer time python -u main.py \
    -date=${date}\
    -verbose=${verbose}\
    -verbose_visual=${verbose_visual}\
    -time_step_save_interval=${time_step_save_interval}\
    -en_load_model=${en_load_model}\
    -checkpoint_load_dir=${path_models_ckpt}\
	-checkpoint_dir=${path_models_ckpt}\
	-model_name=${model_name}\
   	-en_train=${en_train}\
	-save_interval=${save_interval}\
	-nn_mode=${nn_mode}\
	-use_bias=True\
	-regularizer=${regularizer}\
	-pooling=${pooling}\
	-epoch=${num_epoch}\
	-idx_test_dataset_s=${idx_test_dataset_s}\
	-num_test_dataset=${num_test_dataset}\
	-n_type=${n_type}\
	-time_step=${time_step}\
	-n_init_vth=${vth}\
	-n_in_init_vth=${vth_n_in}\
	-lr=${lr}\
    -lr_decay=${lr_decay}\
    -lr_decay_step=${lr_decay_step}\
    -lamb=${lamb}\
    -ann_model=${ann_model}\
    -num_class=${num_class}\
    -f_fused_bn=${f_fused_bn}\
    -f_stat_train_mode=${f_stat_train_mode}\
    -f_real_value_input_snn=${f_real_value_input_snn}\
    -f_vth_conp=${f_vth_conp}\
    -f_spike_max_pool=${f_spike_max_pool}\
    -f_w_norm_data=${f_w_norm_data}\
    -f_ws=${f_ws}\
    -p_ws=${p_ws}\
    -f_isi=${f_isi}\
    -f_refractory=${f_refractory}\
    -input_spike_mode=${input_spike_mode}\
    -neural_coding=${neural_coding}\
    -f_positive_vmem=${f_positive_vmem}\
    -f_tot_psp=${f_tot_psp}\
    -f_comp_act=${f_comp_act}\
    -f_entropy=${f_entropy}\
    -f_write_stat=${f_write_stat}\
    -act_save_mode=${act_save_mode}\
    -f_save_result=${f_save_result}\
    -path_stat=${path_stat}\
    -path_result_root=${path_result_root}\
    -prefix_stat=${prefix_stat}\
    -f_pruning_channel=${f_pruning_channel}\
    -tc=${tc}\
    -time_window=${time_window}\
    -time_fire_start=${time_fire_start}\
    -time_fire_duration=${time_fire_duration}\
    -f_record_first_spike_time=${f_record_first_spike_time}\
    -f_visual_record_first_spike_time=${f_visual_record_first_spike_time}\
    -f_train_time_const=${f_train_time_const}\
    -f_train_time_const_outlier=${f_train_time_const_outlier}\
    -f_load_time_const=${f_load_time_const}\
    -time_const_init_file_name=${time_const_init_file_name}\
    -time_const_num_trained_data=${time_const_num_trained_data}\
    -time_const_save_interval=${time_const_save_interval}\
    -f_tc_based=${f_tc_based}\
    -n_tau_fire_start=${n_tau_fire_start}\
    -n_tau_fire_duration=${n_tau_fire_duration}\
    -n_tau_time_window=${n_tau_time_window}\
    -epoch_train_time_const=${epoch_train_time_const}\
    -snn_output_type=${snn_output_type}\
    -dataset=${dataset}\
    -input_size=${input_size}\
    -batch_size=${batch_size}\
    -init_first_spike_time_n=${init_first_spike_time_n}\
    -f_surrogate_training_model=${f_surrogate_training_model}\
    -f_overwrite_train_model=${f_overwrite_train_model}\
    -f_validation_snn=${f_validation_snn}\
    -en_tensorboard_write=${en_tensorboard_write}\
    -epoch_start_train_tk=${epoch_start_train_tk}\
    -epoch_start_train_t_int=${epoch_start_train_t_int}\
    -epoch_start_train_floor=${epoch_start_train_floor}\
    -epoch_start_train_clip_tw=${epoch_start_train_clip_tw}\
    -epoch_start_loss_enc_spike=${epoch_start_loss_enc_spike}\
    -bypass_pr=${bypass_pr}\
    -bypass_target_epoch=${bypass_target_epoch}\
    -f_loss_enc_spike=${f_loss_enc_spike}\
    -w_loss_enc_spike=${w_loss_enc_spike}\
    -beta_dist_a=${beta_dist_a}\
    -beta_dist_b=${beta_dist_b}\
    -enc_st_n_tw=${enc_st_n_tw}\
    -tfboard_log_file_name=${tfboard_log_file_name}\
    ; } 2>&1 | tee ${log_file}

echo 'log_file: '${log_file}

#
cp_model=$9

if [ ${training_mode} = True ]
then
    if [ ${cp_model} = True ]
    then
        cp -r ${path_models_ckpt}/${model_name} ${path_models_ckpt}/${model_name}_${log_file_name}
    fi
fi
