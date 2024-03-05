#!/bin/bash


#model='ResNet19'
#dataset='CIFAR10'


#trained_models_reg_spike=(1E-7 3E-7 5E-7 7E-7 1E-6 3E-6 5E-6 7E-6 1E-5 3E-5 5E-5 7E-5 1E-4 3E-4 5E-4 7E-4 1E-3)
trained_models_reg_spike=('NORMAL' 'WTA-1' 'WTA-2' 'SIM-A' 'SIM-S')
#trained_models_reg_spike=('NORMAL' 'WTA-1')
#trained_models_reg_spike=('NORMAL' 'WTA-1')
#trained_models_reg_spike=('WTA-1' 'WTA-2' 'SIM-A' 'SIM-S')


for ((i_train_model=0;i_train_model<${#trained_models_reg_spike[@]};i_train_model++)) do
    trained_model_reg_spike=${trained_models_reg_spike[$i_train_model]}

    { time python -u main_snn_training_WTA_SNN.py \
        -nn_mode='SNN' \
        -pooling_vgg='avg'\
        -trained_model_reg_spike=$trained_model_reg_spike \
        ; }
done
