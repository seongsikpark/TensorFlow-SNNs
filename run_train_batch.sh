#!/bin/bash


#models=('ResNet18' 'ResNet20' 'ResNet32' 'ResNet34' 'ResNet50' 'ResNet18V2' 'ResNet20V2' 'ResNet32V2' 'ResNet34V2' 'ResNet50V2')
models=('ResNet20' 'ResNet32' 'ResNet44' 'ResNet56')
#models=('ResNet44' 'ResNet56' 'ResNet44V2' 'ResNet56V2')
datasets=('CIFAR10')




for ((i_model=0;i_model<${#models[@]};i_model++)) do
    model=${models[$i_model]}
    for ((i_dataset=0;i_dataset<${#datasets[@]};i_dataset++)) do
        dataset=${datasets[$i_dataset]}

        echo 'train - '${model}', '${dataset}
        ./run_train.sh ${model} ${dataset}
    done
done



