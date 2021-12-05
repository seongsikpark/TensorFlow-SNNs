#!/bin/bash


#models=('ResNet18' 'ResNet20' 'ResNet32' 'ResNet34' 'ResNet50' 'ResNet18V2' 'ResNet20V2' 'ResNet32V2' 'ResNet34V2' 'ResNet50V2')
#datasets=('CIFAR10')
vth_toggle_inits=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)




for ((i_vth_toggle_init=0;i_vth_toggle_init<${#vth_toggle_inits[@]};i_vth_toggle_init++)) do
  vth_toggle_init=${vth_toggle_inits[$i_vth_toggle_init]}

  echo 'vth_toggle_init- '${vth_toggle_init}

  ./run_vth_toggle.sh ${vth_toggle_init}
done



