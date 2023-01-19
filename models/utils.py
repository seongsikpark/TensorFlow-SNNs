

import tensorflow as tf

import h5py

def transfrom_model_from_v1(model,weight_path,model_dataset_name):
    """
        transfrom model from v1 of this simulator
        v1 - integrated layer (synapse-bn-act)

    """

    #path_v1 = '/home/sspark/Models/DNN/CNN'

    #model_name = 'VGG16'

    #dataset_name = 'CIFAR10'





    #
    #model_dataset = model_name+'_'+dataset_name

    #weights_v1 = h5py.File(weight_path,'r')


    with h5py.File(weight_path,'r') as weight_v1:

        #print('Keys: %s'%weight_v1.keys())
        #a_group_key = list(weight_v1.keys())[0]
        #print(type(weight_v1[a_group_key]))
        #data = list(weight_v1[a_group_key])

        # optimizer weight - such as momentum
        #weight_v1['optimizer_weights']

        # model weight
        print(weight_v1['model_weights'].keys())


        #for key in weight_v1['model_weights'].keys():
        #    weight_v1['model_weights'][key]

        w = weight_v1['model_weights']

        # kernel
        model.get_layer('conv1').kernel.assign(w['conv1']['conv1']['kernel:0'][:])
        model.get_layer('conv1_1').kernel.assign(w['conv1_1']['conv1_1']['kernel:0'][:])

        model.get_layer('conv2').kernel.assign(w['conv2']['conv2']['kernel:0'][:])
        model.get_layer('conv2_1').kernel.assign(w['conv2_1']['conv2_1']['kernel:0'][:])

        model.get_layer('conv3').kernel.assign(w['conv3']['conv3']['kernel:0'][:])
        model.get_layer('conv3_1').kernel.assign(w['conv3_1']['conv3_1']['kernel:0'][:])
        model.get_layer('conv3_2').kernel.assign(w['conv3_2']['conv3_2']['kernel:0'][:])

        model.get_layer('conv4').kernel.assign(w['conv4']['conv4']['kernel:0'][:])
        model.get_layer('conv4_1').kernel.assign(w['conv4_1']['conv4_1']['kernel:0'][:])
        model.get_layer('conv4_2').kernel.assign(w['conv4_2']['conv4_2']['kernel:0'][:])

        model.get_layer('conv5').kernel.assign(w['conv5']['conv5']['kernel:0'][:])
        model.get_layer('conv5_1').kernel.assign(w['conv5_1']['conv5_1']['kernel:0'][:])
        model.get_layer('conv5_2').kernel.assign(w['conv5_2']['conv5_2']['kernel:0'][:])

        model.get_layer('fc1').kernel.assign(w['fc1']['fc1']['kernel:0'][:])
        model.get_layer('fc2').kernel.assign(w['fc2']['fc2']['kernel:0'][:])
        model.get_layer('predictions').kernel.assign(w['predictions']['predictions']['kernel:0'][:])

        # bias
        model.get_layer('conv1').bias.assign(w['conv1']['conv1']['bias:0'][:])
        model.get_layer('conv1_1').bias.assign(w['conv1_1']['conv1_1']['bias:0'][:])

        model.get_layer('conv2').bias.assign(w['conv2']['conv2']['bias:0'][:])
        model.get_layer('conv2_1').bias.assign(w['conv2_1']['conv2_1']['bias:0'][:])

        model.get_layer('conv3').bias.assign(w['conv3']['conv3']['bias:0'][:])
        model.get_layer('conv3_1').bias.assign(w['conv3_1']['conv3_1']['bias:0'][:])
        model.get_layer('conv3_2').bias.assign(w['conv3_2']['conv3_2']['bias:0'][:])

        model.get_layer('conv4').bias.assign(w['conv4']['conv4']['bias:0'][:])
        model.get_layer('conv4_1').bias.assign(w['conv4_1']['conv4_1']['bias:0'][:])
        model.get_layer('conv4_2').bias.assign(w['conv4_2']['conv4_2']['bias:0'][:])

        model.get_layer('conv5').bias.assign(w['conv5']['conv5']['bias:0'][:])
        model.get_layer('conv5_1').bias.assign(w['conv5_1']['conv5_1']['bias:0'][:])
        model.get_layer('conv5_2').bias.assign(w['conv5_2']['conv5_2']['bias:0'][:])

        model.get_layer('fc1').bias.assign(w['fc1']['fc1']['bias:0'][:])
        model.get_layer('fc2').bias.assign(w['fc2']['fc2']['bias:0'][:])
        model.get_layer('predictions').bias.assign(w['predictions']['predictions']['bias:0'][:])

        # bn
        model.get_layer('bn_conv1').beta.assign(w['conv1']['conv1']['conv1_bn']['beta:0'][:])
        model.get_layer('bn_conv1').gamma.assign(w['conv1']['conv1']['conv1_bn']['gamma:0'][:])
        model.get_layer('bn_conv1').moving_mean.assign(w['conv1']['conv1']['conv1_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv1').moving_variance.assign(w['conv1']['conv1']['conv1_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv1_1').beta.assign(w['conv1_1']['conv1_1']['conv1_1_bn']['beta:0'][:])
        model.get_layer('bn_conv1_1').gamma.assign(w['conv1_1']['conv1_1']['conv1_1_bn']['gamma:0'][:])
        model.get_layer('bn_conv1_1').moving_mean.assign(w['conv1_1']['conv1_1']['conv1_1_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv1_1').moving_variance.assign(w['conv1_1']['conv1_1']['conv1_1_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv2').beta.assign(w['conv2']['conv2']['conv2_bn']['beta:0'][:])
        model.get_layer('bn_conv2').gamma.assign(w['conv2']['conv2']['conv2_bn']['gamma:0'][:])
        model.get_layer('bn_conv2').moving_mean.assign(w['conv2']['conv2']['conv2_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv2').moving_variance.assign(w['conv2']['conv2']['conv2_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv2_1').beta.assign(w['conv2_1']['conv2_1']['conv2_1_bn']['beta:0'][:])
        model.get_layer('bn_conv2_1').gamma.assign(w['conv2_1']['conv2_1']['conv2_1_bn']['gamma:0'][:])
        model.get_layer('bn_conv2_1').moving_mean.assign(w['conv2_1']['conv2_1']['conv2_1_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv2_1').moving_variance.assign(w['conv2_1']['conv2_1']['conv2_1_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv3').beta.assign(w['conv3']['conv3']['conv3_bn']['beta:0'][:])
        model.get_layer('bn_conv3').gamma.assign(w['conv3']['conv3']['conv3_bn']['gamma:0'][:])
        model.get_layer('bn_conv3').moving_mean.assign(w['conv3']['conv3']['conv3_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv3').moving_variance.assign(w['conv3']['conv3']['conv3_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv3_1').beta.assign(w['conv3_1']['conv3_1']['conv3_1_bn']['beta:0'][:])
        model.get_layer('bn_conv3_1').gamma.assign(w['conv3_1']['conv3_1']['conv3_1_bn']['gamma:0'][:])
        model.get_layer('bn_conv3_1').moving_mean.assign(w['conv3_1']['conv3_1']['conv3_1_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv3_1').moving_variance.assign(w['conv3_1']['conv3_1']['conv3_1_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv3_2').beta.assign(w['conv3_2']['conv3_2']['conv3_2_bn']['beta:0'][:])
        model.get_layer('bn_conv3_2').gamma.assign(w['conv3_2']['conv3_2']['conv3_2_bn']['gamma:0'][:])
        model.get_layer('bn_conv3_2').moving_mean.assign(w['conv3_2']['conv3_2']['conv3_2_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv3_2').moving_variance.assign(w['conv3_2']['conv3_2']['conv3_2_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv4').beta.assign(w['conv4']['conv4']['conv4_bn']['beta:0'][:])
        model.get_layer('bn_conv4').gamma.assign(w['conv4']['conv4']['conv4_bn']['gamma:0'][:])
        model.get_layer('bn_conv4').moving_mean.assign(w['conv4']['conv4']['conv4_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv4').moving_variance.assign(w['conv4']['conv4']['conv4_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv4_1').beta.assign(w['conv4_1']['conv4_1']['conv4_1_bn']['beta:0'][:])
        model.get_layer('bn_conv4_1').gamma.assign(w['conv4_1']['conv4_1']['conv4_1_bn']['gamma:0'][:])
        model.get_layer('bn_conv4_1').moving_mean.assign(w['conv4_1']['conv4_1']['conv4_1_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv4_1').moving_variance.assign(w['conv4_1']['conv4_1']['conv4_1_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv4_2').beta.assign(w['conv4_2']['conv4_2']['conv4_2_bn']['beta:0'][:])
        model.get_layer('bn_conv4_2').gamma.assign(w['conv4_2']['conv4_2']['conv4_2_bn']['gamma:0'][:])
        model.get_layer('bn_conv4_2').moving_mean.assign(w['conv4_2']['conv4_2']['conv4_2_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv4_2').moving_variance.assign(w['conv4_2']['conv4_2']['conv4_2_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv5').beta.assign(w['conv5']['conv5']['conv5_bn']['beta:0'][:])
        model.get_layer('bn_conv5').gamma.assign(w['conv5']['conv5']['conv5_bn']['gamma:0'][:])
        model.get_layer('bn_conv5').moving_mean.assign(w['conv5']['conv5']['conv5_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv5').moving_variance.assign(w['conv5']['conv5']['conv5_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv5_1').beta.assign(w['conv5_1']['conv5_1']['conv5_1_bn']['beta:0'][:])
        model.get_layer('bn_conv5_1').gamma.assign(w['conv5_1']['conv5_1']['conv5_1_bn']['gamma:0'][:])
        model.get_layer('bn_conv5_1').moving_mean.assign(w['conv5_1']['conv5_1']['conv5_1_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv5_1').moving_variance.assign(w['conv5_1']['conv5_1']['conv5_1_bn']['moving_variance:0'][:])

        model.get_layer('bn_conv5_2').beta.assign(w['conv5_2']['conv5_2']['conv5_2_bn']['beta:0'][:])
        model.get_layer('bn_conv5_2').gamma.assign(w['conv5_2']['conv5_2']['conv5_2_bn']['gamma:0'][:])
        model.get_layer('bn_conv5_2').moving_mean.assign(w['conv5_2']['conv5_2']['conv5_2_bn']['moving_mean:0'][:])
        model.get_layer('bn_conv5_2').moving_variance.assign(w['conv5_2']['conv5_2']['conv5_2_bn']['moving_variance:0'][:])

        model.get_layer('bn_fc1').beta.assign(w['fc1']['fc1']['fc1_bn']['beta:0'][:])
        model.get_layer('bn_fc1').gamma.assign(w['fc1']['fc1']['fc1_bn']['gamma:0'][:])
        model.get_layer('bn_fc1').moving_mean.assign(w['fc1']['fc1']['fc1_bn']['moving_mean:0'][:])
        model.get_layer('bn_fc1').moving_variance.assign(w['fc1']['fc1']['fc1_bn']['moving_variance:0'][:])

        model.get_layer('bn_fc2').beta.assign(w['fc2']['fc2']['fc2_bn']['beta:0'][:])
        model.get_layer('bn_fc2').gamma.assign(w['fc2']['fc2']['fc2_bn']['gamma:0'][:])
        model.get_layer('bn_fc2').moving_mean.assign(w['fc2']['fc2']['fc2_bn']['moving_mean:0'][:])
        model.get_layer('bn_fc2').moving_variance.assign(w['fc2']['fc2']['fc2_bn']['moving_variance:0'][:])

        return model






