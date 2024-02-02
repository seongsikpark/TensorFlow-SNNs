#!/bin/bash


model='ResNet19'
dataset='CIFAR10'


reg_spike_out_consts=(1E-7 3E-7 5E-7 7E-7 1E-6 3E-6 5E-6 7E-6 1E-5 3E-5 5E-5 7E-5 1E-4 3E-4 5E-4 7E-4 1E-3)


for ((i_reg_sout_c=0;i_reg_sout_c<${#reg_spike_out_consts[@]};i_reg_sout_c++)) do
    reg_spike_out_const=${reg_spike_out_consts[$i_reg_sout_c]}

    { time python -u main_snn_training.py \
        -model=${model} \
        -dataset=${dataset} \
        -exp_set_name='spike_reg_resnet19' \
        -nn_mode='SNN' \
        -pooling_vgg='avg'\
        -reg_spike_out=True \
        -reg_spike_out_sc=True \
        -reg_spike_out_sc_sm=True \
        -reg_spike_out_alpha=4 \
        -reg_spike_out_const=${reg_spike_out_const} \
        ; }
done
