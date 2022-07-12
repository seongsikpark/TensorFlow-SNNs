


#
def load_resnet_block(model, weights, name_layer_model, name_layer_weight):
    name_layer_weight_conv = name_layer_weight+'_conv'
    name_layer_weight_bn = name_layer_weight+'_bn'

    model.get_layer(name_layer_model).kernel.assign(weights[name_layer_weight_conv][name_layer_weight_conv]['kernel:0'][:])
    model.get_layer(name_layer_model).bias.assign(weights[name_layer_weight_conv][name_layer_weight_conv]['bias:0'][:])
    model.get_layer(name_layer_model).bn.gamma.assign(weights[name_layer_weight_bn][name_layer_weight_bn]['gamma:0'][:])
    model.get_layer(name_layer_model).bn.beta.assign(weights[name_layer_weight_bn][name_layer_weight_bn]['beta:0'][:])
    model.get_layer(name_layer_model).bn.moving_mean.assign(weights[name_layer_weight_bn][name_layer_weight_bn]['moving_mean:0'][:])
    model.get_layer(name_layer_model).bn.moving_variance.assign(weights[name_layer_weight_bn][name_layer_weight_bn]['moving_variance:0'][:])

#
def load_resnet(model, weights, num_blocks):
    num_blocks=num_blocks

    model.get_layer('conv1_conv').kernel.assign(weights['conv1_conv']['conv1_conv']['kernel:0'][:])
    model.get_layer('conv1_conv').bias.assign(weights['conv1_conv']['conv1_conv']['bias:0'][:])
    model.get_layer('conv1_conv').bn.gamma.assign(weights['conv1_bn']['conv1_bn']['gamma:0'][:])
    model.get_layer('conv1_conv').bn.beta.assign(weights['conv1_bn']['conv1_bn']['beta:0'][:])
    model.get_layer('conv1_conv').bn.moving_mean.assign(weights['conv1_bn']['conv1_bn']['moving_mean:0'][:])
    model.get_layer('conv1_conv').bn.moving_variance.assign(weights['conv1_bn']['conv1_bn']['moving_variance:0'][:])

    for idx_stack, num_block in enumerate(num_blocks):
        _idx_stack = idx_stack+2
        name_conv = 'conv'+str(_idx_stack)

        for idx_block in range(num_block):
            _idx_block = idx_block+1
            name_block = 'block'+str(_idx_block)

            if _idx_block==1:
                name_layer_model = name_conv+'_'+name_block+'_conv0'
                name_layer_weight = name_conv+'_'+name_block+'_0'

                #name_layer_weight_conv = name_layer_weight+'_conv'
                #name_layer_weight_bn = name_layer_weight+'_bn'
                #model.get_layer(name_layer_model).kernel.assign(weights[name_layer_weight_conv][name_layer_weight_conv]['kernel:0'][:])
                #model.get_layer(name_layer_model).bias.assign(weights[name_layer_weight_conv][name_layer_weight_conv]['bias:0'][:])
                #model.get_layer(name_layer_model).bn.gamma.assign(weights[name_layer_weight_bn][name_layer_weight_bn]['gamma:0'][:])
                ##model.get_layer(name_layer_model).bn.beta.assign(weights[name_layer_weight_bn][name_layer_weight_bn]['beta:0'][:])
                #model.get_layer(name_layer_model).bn.moving_mean.assign(weights[name_layer_weight_bn][name_layer_weight_bn]['moving_mean:0'][:])
                #model.get_layer(name_layer_model).bn.moving_variance.assign(weights[name_layer_weight_bn][name_layer_weight_bn]['moving_variance:0'][:])

                load_resnet_block(model,weights,name_layer_model,name_layer_weight)

            name_layer_model = name_conv + '_' + name_block
            name_layer_weight = name_conv + '_' + name_block
            load_resnet_block(model, weights, name_layer_model+'_conv1', name_layer_weight+'_1')
            load_resnet_block(model, weights, name_layer_model+'_conv2', name_layer_weight+'_2')
            load_resnet_block(model, weights, name_layer_model+'_conv3', name_layer_weight+'_3')

    model.get_layer('predictions').kernel.assign(weights['probs']['probs']['kernel:0'][:])
    model.get_layer('predictions').bias.assign(weights['probs']['probs']['bias:0'][:])

    #assert False

    if False:
        # conv2_block1_0
        model.get_layer('conv2_block1_conv0').kernel.assign(weights['conv2_block1_0_conv']['conv2_block1_0_conv_W_1:0'][:])
        model.get_layer('conv2_block1_conv0').bias.assign(weights['conv2_block1_0_conv']['conv2_block1_0_conv_b_1:0'][:])
        model.get_layer('conv2_block1_conv0').bn.gamma.assign(weights['conv2_block1_0_bn']['conv2_block1_0_bn']['gamma:0'][:])
        model.get_layer('conv2_block1_conv0').bn.beta.assign(weights['conv2_block1_0_bn']['conv2_block1_0_bn']['beta:0'][:])
        model.get_layer('conv2_block1_conv0').bn.moving_mean.assign(weights['conv2_block1_0_bn']['conv2_block1_0_bn']['moving_mean:0'][:])
        model.get_layer('conv2_block1_conv0').bn.moving_variance.assign(weights['conv2_block1_0_bn']['conv2_block1_0_bn']['moving_variance:0'][:])

        # conv2_block1_1
        model.get_layer('conv2_block1_conv1').kernel.assign(weights['conv2_block1_1_conv']['conv2_block1_1_conv_W_1:0'][:])
        model.get_layer('conv2_block1_conv1').bias.assign(weights['conv2_block1_1_conv']['conv2_block1_1_conv_b_1:0'][:])
        model.get_layer('conv2_block1_conv1').bn.gamma.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['gamma:0'][:])
        model.get_layer('conv2_block1_conv1').bn.beta.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['beta:0'][:])
        model.get_layer('conv2_block1_conv1').bn.moving_mean.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['moving_mean:0'][:])
        model.get_layer('conv2_block1_conv1').bn.moving_variance.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['moving_variance:0'][:])

        # conv2_block1_2
        model.get_layer('conv2_block1_conv1').kernel.assign(weights['conv2_block1_1_conv']['conv2_block1_1_conv_W_1:0'][:])
        model.get_layer('conv2_block1_conv1').bias.assign(weights['conv2_block1_1_conv']['conv2_block1_1_conv_b_1:0'][:])
        model.get_layer('conv2_block1_conv1').bn.gamma.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['gamma:0'][:])
        model.get_layer('conv2_block1_conv1').bn.beta.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['beta:0'][:])
        model.get_layer('conv2_block1_conv1').bn.moving_mean.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['moving_mean:0'][:])
        model.get_layer('conv2_block1_conv1').bn.moving_variance.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['moving_variance:0'][:])



def load_resnet50(model, weights):
    num_blocks = [3,4,6,3]
    load_resnet(model, weights, num_blocks)

def load_resnet101(model,weights):
    num_blocks = [3,4,23,3]
    load_resnet(model, weights, num_blocks)

def load_resnet152(model,weights):
    num_blocks = [3,8,36,3]
    load_resnet(model, weights, num_blocks)





#
def load_resnet50v2(model, weights):

    assert False

    model.get_layer('conv1_conv').kernel.assign(weights['conv1_conv']['conv1_conv_W_1:0'][:])
    model.get_layer('conv1_conv').bias.assign(weights['conv1_conv']['conv1_conv_b_1:0'][:])

    model.get_layer('conv2_block1_conv0').kernel.assign(weights['conv2_block1_0_conv']['conv2_block1_0_conv_W_1:0'][:])
    model.get_layer('conv2_block1_conv0').bias.assign(weights['conv2_block1_0_conv']['conv2_block1_0_conv_b_1:0'][:])

    model.get_layer('conv2_block1_conv1').kernel.assign(weights['conv2_block1_1_conv']['conv2_block1_1_conv_W_1:0'][:])
    model.get_layer('conv2_block1_conv1').bias.assign(weights['conv2_block1_1_conv']['conv2_block1_1_conv_b_1:0'][:])
    model.get_layer('conv2_block1_conv1').bn.gamma.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['gamma:0'][:])
    model.get_layer('conv2_block1_conv1').bn.beta.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['beta:0'][:])
    model.get_layer('conv2_block1_conv1').bn.moving_mean.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['moving_mean:0'][:])
    model.get_layer('conv2_block1_conv1').bn.moving_variance.assign(weights['conv2_block1_1_bn']['conv2_block1_1_bn']['moving_variance:0'][:])

    model.get_layer('conv2_block1_conv2').kernel.assign(weights['conv2_block1_2_conv']['conv2_block1_2_conv_W_1:0'][:])
    model.get_layer('conv2_block1_conv2').bias.assign(weights['conv2_block1_2_conv']['conv2_block1_2_conv_b_1:0'][:])
    model.get_layer('conv2_block1_conv2').bn.gamma.assign(weights['conv2_block1_2_bn']['conv2_block1_2_bn']['gamma:0'][:])
    model.get_layer('conv2_block1_conv2').bn.beta.assign(weights['conv2_block1_2_bn']['conv2_block1_2_bn']['beta:0'][:])
    model.get_layer('conv2_block1_conv2').bn.moving_mean.assign(weights['conv2_block1_2_bn']['conv2_block1_2_bn']['moving_mean:0'][:])
    model.get_layer('conv2_block1_conv2').bn.moving_variance.assign(weights['conv2_block1_2_bn']['conv2_block1_2_bn']['moving_variance:0'][:])

    model.get_layer('conv2_block1_conv2').kernel.assign(weights['conv2_block1_2_conv']['conv2_block1_2_conv_W_1:0'][:])
    model.get_layer('conv2_block1_conv2').bias.assign(weights['conv2_block1_2_conv']['conv2_block1_2_conv_b_1:0'][:])
