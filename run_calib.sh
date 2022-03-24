#!/bin/bash

#
source ../00_SNN/venv/bin/activate

####

model=${1}
dataset=${2}

neuron_type=${3}

vth_search=${4}

calibration_bias=${5}

bias_control=${6}



#{ unbuffer time kernprof -l main.py \
#{ CUDA_VISIBLE_DEVICES=0 unbuffer time python main.py \
#{ unbuffer time python -m line_profiler main.py \
{ time python -u main_hp_tune.py \
    -model=${model} \
    -dataset=${dataset} \
    -n_type=${neuron_type}\
    -vth_search=${vth_search} \
    -calibration_bias_new=${calibration_bias}\
    -bias_control=${bias_control} \
    ; }

