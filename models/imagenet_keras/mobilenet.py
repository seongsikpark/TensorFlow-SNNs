
from keras import backend
from models.mobilenet_v2 import _make_divisible


def load_inverted_res_block(model, weights, expansion, stride, alpha, filters, block_id):

    prefix = 'block_{}_'.format(block_id)


    lname_cv_s = 'mobl'+str(block_id)+'_conv_'+str(block_id)+'_'
    lname_cv_e_s = lname_cv_s+'expand'
    lname_cv_d_s = lname_cv_s+'depthwise'
    lname_cv_p_s = lname_cv_s+'project'


    lname_bn_s = 'bn'+str(block_id)+'_conv_'+str(block_id)+'_bn_'
    lname_bn_e_s = lname_bn_s+'expand'
    lname_bn_d_s = lname_bn_s+'depthwise'
    lname_bn_p_s = lname_bn_s+'project'


    if block_id:
        lname_e_t = prefix + 'expand'
        lname_d_t = prefix + 'depthwise'
        lname_p_t = prefix + 'project'

        model.get_layer(lname_e_t).kernel.assign(weights[lname_cv_e_s][lname_cv_e_s]['kernel:0'])
        model.get_layer(lname_e_t).bn.beta.assign(weights[lname_bn_e_s][lname_bn_e_s]['beta:0'])
        model.get_layer(lname_e_t).bn.gamma.assign(weights[lname_bn_e_s][lname_bn_e_s]['gamma:0'])
        model.get_layer(lname_e_t).bn.moving_mean.assign(weights[lname_bn_e_s][lname_bn_e_s]['moving_mean:0'])
        model.get_layer(lname_e_t).bn.moving_variance.assign(weights[lname_bn_e_s][lname_bn_e_s]['moving_variance:0'])
    else:
        prefix = 'expanded_conv_'

        lname_e_t = prefix + 'expand'
        lname_d_t = prefix + 'depthwise'
        lname_p_t = prefix + 'project'

    # Depthwise 3x3 convolution.
    model.get_layer(lname_d_t).depthwise_kernel.assign(weights[lname_cv_d_s][lname_cv_d_s]['depthwise_kernel:0'])
    model.get_layer(lname_d_t).bn.beta.assign(weights[lname_bn_d_s][lname_bn_d_s]['beta:0'])
    model.get_layer(lname_d_t).bn.gamma.assign(weights[lname_bn_d_s][lname_bn_d_s]['gamma:0'])
    model.get_layer(lname_d_t).bn.moving_mean.assign(weights[lname_bn_d_s][lname_bn_d_s]['moving_mean:0'])
    model.get_layer(lname_d_t).bn.moving_variance.assign(weights[lname_bn_d_s][lname_bn_d_s]['moving_variance:0'])

    # Project with a pointwise 1x1 convolution.
    model.get_layer(lname_p_t).kernel.assign(weights[lname_cv_p_s][lname_cv_p_s]['kernel:0'])
    model.get_layer(lname_p_t).bn.beta.assign(weights[lname_bn_p_s][lname_bn_p_s]['beta:0'])
    model.get_layer(lname_p_t).bn.gamma.assign(weights[lname_bn_p_s][lname_bn_p_s]['gamma:0'])
    model.get_layer(lname_p_t).bn.moving_mean.assign(weights[lname_bn_p_s][lname_bn_p_s]['moving_mean:0'])
    model.get_layer(lname_p_t).bn.moving_variance.assign(weights[lname_bn_p_s][lname_bn_p_s]['moving_variance:0'])



def load_mobilenetv2(model, weights):

    model.get_layer('Conv1').kernel.assign(weights['Conv1']['Conv1']['kernel:0'][:])
    model.get_layer('Conv1').bn.beta.assign(weights['bn_Conv1']['bn_Conv1']['beta:0'][:])
    model.get_layer('Conv1').bn.gamma.assign(weights['bn_Conv1']['bn_Conv1']['gamma:0'][:])
    model.get_layer('Conv1').bn.moving_mean.assign(weights['bn_Conv1']['bn_Conv1']['moving_mean:0'][:])
    model.get_layer('Conv1').bn.moving_variance.assign(weights['bn_Conv1']['bn_Conv1']['moving_variance:0'][:])

    #
    alpha = 1.0
    print('W: load_mobilenetv2 - alpha: {:}'.format(alpha))

    load_inverted_res_block(model, weights, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)
    load_inverted_res_block(model, weights, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    load_inverted_res_block(model, weights, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    load_inverted_res_block(model, weights, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    load_inverted_res_block(model, weights, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    load_inverted_res_block(model, weights, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    load_inverted_res_block(model, weights, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    load_inverted_res_block(model, weights, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    load_inverted_res_block(model, weights, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    load_inverted_res_block(model, weights, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    load_inverted_res_block(model, weights, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    load_inverted_res_block(model, weights, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    load_inverted_res_block(model, weights, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    load_inverted_res_block(model, weights, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    load_inverted_res_block(model, weights, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    load_inverted_res_block(model, weights, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    load_inverted_res_block(model, weights, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    #
    model.get_layer('Conv_1').kernel.assign(weights['Conv_1']['Conv_1']['kernel:0'][:])
    model.get_layer('Conv_1').bn.beta.assign(weights['Conv_1_bn']['Conv_1_bn']['beta:0'][:])
    model.get_layer('Conv_1').bn.gamma.assign(weights['Conv_1_bn']['Conv_1_bn']['gamma:0'][:])
    model.get_layer('Conv_1').bn.moving_mean.assign(weights['Conv_1_bn']['Conv_1_bn']['moving_mean:0'][:])
    model.get_layer('Conv_1').bn.moving_variance.assign(weights['Conv_1_bn']['Conv_1_bn']['moving_variance:0'][:])

    #
    model.get_layer('predictions').kernel.assign(weights['Logits']['Logits']['kernel:0'][:])
    model.get_layer('predictions').bias.assign(weights['Logits']['Logits']['bias:0'][:])

