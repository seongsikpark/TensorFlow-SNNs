#!/bin/bash

#
source ../00_SNN/venv/bin/activate
#source ./venv/bin/activate


#
output_log_name="run_real-input.log"

{ python main_hp_tune.py \
    -debug_mode=True\
    -verbose_snn_train=True\
    -input_spike_mode='REAL'\
  ; } 2>&1 | tee ${output_log_name}


output_log_name="run_poisson-input.log"
{ python main_hp_tune.py \
    -debug_mode=True\
    -verbose_snn_train=True\
    -input_spike_mode='POISSON'\
  ; } 2>&1 | tee ${output_log_name}


