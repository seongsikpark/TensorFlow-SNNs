#!/bin/bash


#models=('ResNet18' 'ResNet20' 'ResNet32' 'ResNet34' 'ResNet50' 'ResNet18V2' 'ResNet20V2' 'ResNet32V2' 'ResNet34V2' 'ResNet44V2' 'ResNet50V2' 'ResNet56V2')
#models=('ResNet18' 'ResNet34' 'ResNet18V2' 'ResNet34V2')
#models=('ResNet20V2' 'ResNet32V2' 'ResNet44V2' 'ResNet56V2')
#models=('ResNet20' 'ResNet32' 'ResNet44' 'ResNet56')
#models=('ResNet50' 'ResNet56' 'ResNet50V2' 'ResNet56V2')
#models=('VGG16' 'ResNet56' 'ResNet50V2' 'ResNet56V2')
#datasets=('CIFAR10')

#
#scale_arr=(1.0 0.5 0.3 0.1 0.05 0.01)
test_const_arr=(1.0 0.5 0.3 0.1 0.05 0.01)

source ../00_SNN/venv/bin/activate

#
#exp_set_name='220504_dynamic_bn_scale_search_VGG16_CIFAR10_ts-128'\

#
for ((i_scale=0;i_scale<${#test_const_arr[@]};i_scale++))
do
  #scale=${scale_arr[$i_scale]}
  #echo ${scale}
  test_const=${test_const_arr[$i_scale]}
  echo ${test_const}
  python main_hp_tune.py \
    -verbose=False\
  	-exp_set_name='220504_dynamic_bn_test_const_search_VGG16_CIFAR10_ts-128'\
  	-model='VGG16'\
  	-dataset='CIFAR10'\
  	-time_step=128\
  	-dynamic_bn_test=True\
  	-dynamic_bn_dnn_act_scale=0.05\
  	-dynamic_bn_test_const=${test_const}
done
