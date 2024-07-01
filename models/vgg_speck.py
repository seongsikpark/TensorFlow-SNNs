import tensorflow as tf

from tensorflow.python.keras.engine import training

import functools

#
#import lib_snn.layers
import lib_snn

from lib_snn.layers import tfn


#
def VGG_SPECK(
        # def __init__(self, input_shape, data_format, conf):
        batch_size,
        input_shape,
        conf,
        model_name,
        include_top=True,
        # weights='imagenet',
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        #nn_mode='ANN',
        #name='VGG16',
        dataset_name=None,
        **kwargs):


    data_format = conf.data_format


    #lib_snn.model.Model.__init__(self, input_shape, data_format, classes, conf)
    #lib_snn.model.Model.__init__(batch_size, input_shape, data_format, classes, conf)
    #Model = lib_snn.model.Model(batch_size, input_shape, data_format, classes, conf)


    #
    act_relu = 'relu'
    act_sm = 'softmax'

    #
    if conf.nn_mode=='ANN':
        dropout_conv_r = [0.2, 0.2, 0.0]      # DNN training
    elif conf.nn_mode=='SNN':
        #dropout_conv_r = [0.2, 0.2, 0.0]      # SNN training
        #dropout_conv_r = [0.25, 0.25, 0.25]      # SNN training
        dropout_conv_r = [0.0, 0.0, 0.0]      # SNN training
    else:
        assert False

    #
    initial_channels = kwargs.pop('initial_channels', None)
    #assert initial_channels is not None
    #if initial_channels is None:
    #    initial_channels = 64

    initial_channels = 32


    #
    use_bn_feat = conf.use_bn
    use_bn_cls = conf.use_bn

    #
    channels = initial_channels

    #
    n_dim_cls = 64

    #
    k_init = 'glorot_uniform'
    #k_init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0,seed=None)

    # pooling
    if conf.pooling_vgg=='max':
        pool = lib_snn.layers.MaxPool2D
    elif conf.pooling_vgg=='avg':
        pool = lib_snn.layers.AveragePooling2D
    else:
        assert False


    #
    if conf.nn_mode=='ANN':
        act_type = 'relu'
        act_type_out = 'softmax'
    else:
        act_type = conf.n_type
        act_type_out = conf.n_type


    #
    tdbn_first_layer = conf.nn_mode=='SNN' and conf.input_spike_mode=='POISSON' and conf.tdbn
    tdbn = conf.nn_mode=='SNN' and conf.tdbn

    #
    img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    # x = img_input
    input = lib_snn.layers.InputGenLayer(name='in')(img_input)
    if conf.nn_mode=='SNN':
        input = lib_snn.activations.Activation(act_type=act_type,loc='IN',name='n_in')(input)

    #
    syn_c1_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='conv1_1')(input)
    norm_c1_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv1_1')(syn_c1_1)
    a_c1_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv1_1')(norm_c1_1)

    #
    syn_c1_2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='conv1_2')(a_c1_1)
    norm_c1_2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv1_2')(syn_c1_2)
    a_c1_2 = lib_snn.activations.Activation(act_type=act_type,name='n_conv1_2')(norm_c1_2)

    #
    syn_c1_3 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='conv1_3')(a_c1_2)
    norm_c1_3 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv1_3')(syn_c1_3)
    a_c1_3 = lib_snn.activations.Activation(act_type=act_type,name='n_conv1_3')(norm_c1_3)
    a_p_c1 = pool((2, 2), (2, 2), name='conv1_p')(a_c1_3)

    #
    syn_c2_1 = lib_snn.layers.Conv2D(channels*2, 3, padding='SAME', kernel_initializer=k_init, name='conv2_1')(a_p_c1)
    norm_c2_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv2_1')(syn_c2_1)
    a_c2_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv2_1')(norm_c2_1)
    a_p_c2_1 = pool((2, 2), (2, 2), name='conv2_1_p')(a_c2_1)

    #
    syn_c2_2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv2_2')(a_p_c2_1)
    norm_c2_2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv2_2')(syn_c2_2)
    a_c2_2 = lib_snn.activations.Activation(act_type=act_type,name='n_conv2_2')(norm_c2_2)
    a_p_c2_2 = pool((2, 2), (2, 2), name='conv2_2_p')(a_c2_2)

    #
    syn_c3_1 = lib_snn.layers.Conv2D(channels*2, 3, padding='SAME', kernel_initializer=k_init, name='conv3_1')(a_p_c2_2)
    norm_c3_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3_1')(syn_c3_1)
    a_c3_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv3_1')(norm_c3_1)
    a_p_c3_1 = pool((2, 2), (2, 2), name='conv3_1_p') (a_c3_1)

    syn_c3_2 = lib_snn.layers.Conv2D(channels*2, 3, padding='SAME', kernel_initializer=k_init, name='conv3_2')(a_p_c3_1)
    norm_c3_2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3_2')(syn_c3_2)
    a_c3_2 = lib_snn.activations.Activation(act_type=act_type,name='n_conv3_2')(norm_c3_2)
    a_p_c3_2 = pool((2, 2), (2, 2), name='conv3_2_p') (a_c3_2)

    #
    a_p_c3_2_f = tf.keras.layers.Flatten(data_format=data_format,name='flatten')(a_p_c3_2)
    syn_fc1 = lib_snn.layers.Dense(n_dim_cls, kernel_initializer=k_init, name='fc1')(a_p_c3_2_f)
    norm_fc1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_fc1')(syn_fc1)
    a_fc1 = lib_snn.activations.Activation(act_type=act_type,name='n_fc1')(norm_fc1)

    #
    syn_p = lib_snn.layers.Dense(classes, last_layer=True, kernel_initializer=k_init, name='predictions')(a_fc1)
    a_p = lib_snn.activations.Activation(act_type=act_type_out,loc='OUT',name='n_predictions')(syn_p)
    if conf.nn_mode=='SNN':
        a_p = lib_snn.activations.Activation(act_type='softmax',name='a_predictions')(a_p)


    #
    model = lib_snn.model.Model(img_input, a_p, batch_size, input_shape,  classes, conf, name=model_name)


    return model
