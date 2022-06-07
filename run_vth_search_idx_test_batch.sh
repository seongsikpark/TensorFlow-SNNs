#!/bin/bash


#models=('ResNet18' 'ResNet20' 'ResNet32' 'ResNet34' 'ResNet50' 'ResNet18V2' 'ResNet20V2' 'ResNet32V2' 'ResNet34V2' 'ResNet44V2' 'ResNet50V2' 'ResNet56V2')
#models=('ResNet18' 'ResNet34' 'ResNet18V2' 'ResNet34V2')
#models=('ResNet20V2' 'ResNet32V2' 'ResNet44V2' 'ResNet56V2')
#models=('ResNet20' 'ResNet32' 'ResNet44' 'ResNet56')
#models=('ResNet50' 'ResNet56' 'ResNet50V2' 'ResNet56V2')
#models=('VGG16' 'ResNet56' 'ResNet50V2' 'ResNet56V2')
#datasets=('CIFAR10')



source ../00_SNN/venv/bin/activate


# based on
#-exp_set_name='220414_vth_search_idx_test_VGG16_CIFAR100_ts-128'\  -> best idx:34
#-exp_set_name='220416_search-bias-1st_VGG16_CIFAR100_ts-128'\      -> 55
#-exp_set_name='220417_search-vth-2nd_VGG16_CIFAR100_ts-128'\       -> 79 - 71.89

for ((i=0;i<125;i++))
do
  echo $i
  python main_hp_tune.py \
    -verbose=False\
  	-exp_set_name='220420_search-vth-3nd_VGG16_CIFAR100_ts-128'\
  	-model='VGG16'\
  	-dataset='CIFAR100'\
  	-time_step=128\
  	-early_stop_search=True\
  	-early_stop_search_acc=0.7\
    -vth_search_idx_test=True\
    -vth_search_idx=${i}\
    -calibration_idx_test=False\
    -calibration_idx=${i}\
    -calibration_bias_new=True
done
