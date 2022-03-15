#!/bin/bash


#models=('ResNet18' 'ResNet20' 'ResNet32' 'ResNet34' 'ResNet50' 'ResNet18V2' 'ResNet20V2' 'ResNet32V2' 'ResNet34V2' 'ResNet44V2' 'ResNet50V2' 'ResNet56V2')
#models=('ResNet18' 'ResNet34' 'ResNet18V2' 'ResNet34V2')
#models=('ResNet20V2' 'ResNet32V2' 'ResNet44V2' 'ResNet56V2')

models=('VGG16' 'ResNet20' 'ResNet32' 'ResNet44')
datasets=('CIFAR10')

arr_neuron_type=('IF' 'LIF')

arr_vth_search=(True False)

arr_calibration_bias=(True False)

arr_bias_control=(True False)



for ((i_model=0;i_model<${#models[@]};i_model++)) do
    model=${models[$i_model]}

    for ((i_dataset=0;i_dataset<${#datasets[@]};i_dataset++)) do
        dataset=${datasets[$i_dataset]}

        for ((i_neuron_type=0;i_neuron_type<${#arr_neuron_type[@]};i_neuron_type++)) do
            neuron_type=${arr_neuron_type[$i_neuron_type]}

			for ((i_vth_search=0;i_vth_search<${#arr_vth_search[@]};i_vth_search++)) do
				vth_search=${arr_vth_search[$i_vth_search]}

				for ((i_calibration_bias=0;i_calibration_bias<${#arr_calibration_bias[@]};i_calibration_bias++)) do
					calibration_bias=${arr_calibration_bias[$i_calibration_bias]}

					for ((i_bias_control=0;i_bias_control<${#arr_bias_control[@]};i_bias_control++)) do
						bias_control=${arr_bias_control[$i_bias_control]}

						#echo 'train - '${model}', '${dataset}

						./run_train.sh ${model} ${dataset} ${neuron_type} ${vth_search} ${calibration_bias} ${bias_control}

					done
				done
            done
        done
    done
done



