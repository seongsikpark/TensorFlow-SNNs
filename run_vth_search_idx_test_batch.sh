#!/bin/bash


#models=('ResNet18' 'ResNet20' 'ResNet32' 'ResNet34' 'ResNet50' 'ResNet18V2' 'ResNet20V2' 'ResNet32V2' 'ResNet34V2' 'ResNet44V2' 'ResNet50V2' 'ResNet56V2')
#models=('ResNet18' 'ResNet34' 'ResNet18V2' 'ResNet34V2')
#models=('ResNet20V2' 'ResNet32V2' 'ResNet44V2' 'ResNet56V2')
#models=('ResNet20' 'ResNet32' 'ResNet44' 'ResNet56')
#models=('ResNet50' 'ResNet56' 'ResNet50V2' 'ResNet56V2')
#models=('VGG16' 'ResNet56' 'ResNet50V2' 'ResNet56V2')
#datasets=('CIFAR10')



source ../00_SNN/venv/bin/activate



for ((i=0;i<125;i++))
do
  echo $i
  python main_hp_tune.py \
  	-exp_set_name='220410_vth_search_idx_test_ts-128'\
  	-model='ResNet32'\
  	-dataset='CIFAR100'\
  	-time_step=128\
  	-early_stop_search=True\
  	-early_stop_search_acc=90.0\
    -vth_search_idx_test=True\
    -vth_search_idx=${i}
done
