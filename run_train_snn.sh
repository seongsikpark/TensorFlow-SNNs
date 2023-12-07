#!/bin/bash

source ../00_SNN/venv/bin/activate

#models=('ResNet18' 'ResNet20' 'ResNet32' 'ResNet34' 'ResNet50' 'ResNet18V2' 'ResNet20V2' 'ResNet32V2' 'ResNet34V2' 'ResNet44V2' 'ResNet50V2' 'ResNet56V2')
#models=('ResNet18' 'ResNet34' 'ResNet18V2' 'ResNet34V2')
#models=('ResNet20V2' 'ResNet32V2' 'ResNet44V2' 'ResNet56V2')
#models=('ResNet20' 'ResNet32' 'ResNet44' 'ResNet56')
#models=('ResNet50' 'ResNet56' 'ResNet50V2' 'ResNet56V2')
models=('ResNet20'
datasets=('CIFAR10')


reg_spike_out_consts=(1E-3 3E-3 5E-3 7E-3 9E-3)

for ((i_reg_sout_c=0;i_model<${#reg_sout_c[@]};i_reg_sout_c++)) do
    reg_spike_out_const=${reg_spike_out_consts[$i_regsout_c]}
    { time python -u main_snn_training.py \
        -model=${model} \
        -dataset=${dataset} \
        -reg_spike_out=True \
        -reg_spike_out_sc=True \
        -reg_spike_out_sc_sm=True \
        -reg_spike_out_alpha=4 \
        -reg_spike_out_const=${reg_spike_out_const} \

        ; }
done


