#!/bin/bash


#models=('ResNet18' 'ResNet20' 'ResNet32' 'ResNet34' 'ResNet50' 'ResNet18V2' 'ResNet20V2' 'ResNet32V2' 'ResNet34V2' 'ResNet44V2' 'ResNet50V2' 'ResNet56V2')
#models=('ResNet18' 'ResNet34' 'ResNet18V2' 'ResNet34V2')
#models=('ResNet20V2' 'ResNet32V2' 'ResNet44V2' 'ResNet56V2')
#models=('ResNet20' 'ResNet32' 'ResNet44' 'ResNet56')
models=('VGG16' 'ResNet50')
datasets=('ImageNet')



echo 'Test ImageNet Inference'

for ((i_model=0;i_model<${#models[@]};i_model++)) do
    model=${models[$i_model]}
    for ((i_dataset=0;i_dataset<${#datasets[@]};i_dataset++)) do
        dataset=${datasets[$i_dataset]}

        echo ${model}', '${dataset}

        #python -u main_hp_tune.py \
        python -u main_make_imagenet_model_from_keras.py \
            -model=${model} \
            -dataset=${dataset} \
            -mode='inference'
    done
done



