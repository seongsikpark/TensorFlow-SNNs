#!/bin/basverbose_snn_trainh

#
#source ../00_SNN/venv/bin/activate

####

#model=${1}
#dataset=${2}
#vth_toggle_init=${1}
#mode_arr=('NORMAL' 'WTA-1' 'WTA-2' 'SIM-A' 'SIM-S')
#mode_arr=('WTA-1' 'WTA-2' 'SIM-A' 'SIM-S')
#mode_arr=('SIM-A' 'SIM-S')

#{ unbuffer time kernprof -l main.py \
#{ CUDA_VISIBLE_DEVICES=0 unbuffer time python main.py \
#{ unbuffer time python -m line_profiler main.py \
#{ time python -u main_hp_tune.py \
#    -vth_toggle_init=${vth_toggle_init} \
#  ; }


#for ((i=0;i<${#mode_arr[@]};i++)) do
#    mode=${mode_arr[$i]}
#    python main_snn_training_WTA_SNN.py \
#      -trained_model_reg_spike=${mode}
#done




for ((i=0;i<100;i++)) do
    python main_snn_training_WTA_SNN.py \
      -sm_batch_index=${i}
done



