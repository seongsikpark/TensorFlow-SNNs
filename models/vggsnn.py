import tensorflow as tf

from tensorflow.python.keras.engine import training

import functools

#
#import lib_snn.layers
import lib_snn

from lib_snn.layers import tfn


#
def VGGSNN(
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
    if initial_channels is None:
        initial_channels = 64


    #
    use_bn_feat = conf.use_bn
    use_bn_cls = conf.use_bn

    #
    channels = initial_channels

    #
    if dataset_name=='ImageNet':
        n_dim_cls = 4096
    elif 'CIFAR' or 'CALTECH' in dataset_name:
        n_dim_cls = 512
    else:
        assert False

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

    # Layer 1 [2, 48, 48] -> [64, 48, 48]
    syn_c1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='conv1')(input)
    if use_bn_feat:
        norm_c1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn_first_layer,name='bn_conv1')(syn_c1)
    else:
        norm_c1 = syn_c1
    a_c1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv1')(norm_c1)

    # Layer 1_1 [64, 48, 48] -> [128, 24, 24]
    channels = channels * 2
    syn_c1_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv1_1')(a_c1)
    if use_bn_feat:
        norm_c1_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv1_1')(syn_c1_1)
    else:
        norm_c1_1 = syn_c1_1
    a_c1_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv1_1')(norm_c1_1)
    a_p_c1_1 = pool((2, 2), (2, 2), name='conv1_1_p')(a_c1_1)

    # Layer 2 [128, 24, 24] -> [256, 24, 24]
    channels = channels * 2
    syn_c2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv2')(a_p_c1_1)
    if use_bn_feat:
        norm_c2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv2')(syn_c2)
    else:
        norm_c2 = syn_c2
    a_c2 = lib_snn.activations.Activation(act_type=act_type,name='n_conv2')(norm_c2)
    a_d_c2 = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv2_do')(a_c2)

    # Layer 2_1 [256, 24, 24] -> [256, 12, 12]
    syn_c2_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv2_1')(a_d_c2)
    if use_bn_feat:
        norm_c2_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv2_1')(syn_c2_1)
    else:
        norm_c2_1 = syn_c2_1
    a_c2_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv2_1')(norm_c2_1)
    a_p_c2_1 = pool((2, 2), (2, 2), name='conv2_1p') (a_c2_1)

    # Layer 3 [256, 12, 12] -> [512, 12, 12]
    channels = channels * 2
    syn_c3 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3')(a_p_c2_1)
    if use_bn_feat:
        norm_c3 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3')(syn_c3)
    else:
        norm_c3 = syn_c3
    a_c3 = lib_snn.activations.Activation(act_type=act_type,name='n_conv3')(norm_c3)
    a_d_c3 = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv3_do')(a_c3)

    # Layer 3_1 [512, 12, 12] -> [512, 6, 6]
    syn_c3_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3_1')(a_d_c3)
    if use_bn_feat:
        norm_c3_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3_1')(syn_c3_1)
    else:
        norm_c3_1 = syn_c3_1
    a_c3_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv3_1')(norm_c3_1)
    a_p_c3_1 = pool((2, 2), (2, 2), name='conv3_1p') (a_c3_1)

    # Layer 4 [512, 6, 6] -> [512, 6, 6]
    syn_c4 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4')(a_p_c3_1)
    if use_bn_feat:
        norm_c4 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4')(syn_c4)
    else:
        norm_c4 = syn_c4
    a_c4 = lib_snn.activations.Activation(act_type=act_type,name='n_conv4')(norm_c4)
    a_d_c4 = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv4_do')(a_c4)

    # Layer 4_1 [512, 6, 6] -> [512, 3, 3]
    syn_c4_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4_1')(a_d_c4)
    if use_bn_feat:
        norm_c4_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4_1')(syn_c4_1)
    else:
        norm_c4_1 = syn_c4_1
    a_c4_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv4_1')(norm_c4_1)
    a_p_c4_1 = pool((2, 2), (2, 2), name='conv4_1p') (a_c4_1)

    # FC [512, 3, 3] -> [4608] -> [10]
    # flatten [1024, 2, 2] -> [4608]
    a_p_c4_1_f = tf.keras.layers.Flatten(data_format=data_format, name='flatten')(a_p_c4_1)
    a_d_c4_1_f = tf.keras.layers.Dropout(dropout_conv_r[2], name='flatten_do')(a_p_c4_1_f)
    # Dense [4608] -> [10]
    syn_fc = lib_snn.layers.Dense(classes, last_layer=True, kernel_initializer=k_init, name='fc')(a_d_c4_1_f)
    a_p = lib_snn.activations.Activation(act_type=act_type_out, loc='OUT', name='n_fc')(syn_fc)
    if conf.nn_mode == 'SNN':
        a_p = lib_snn.activations.Activation(act_type='softmax', name='a_predictions')(a_p)


    #
    model = lib_snn.model.Model(img_input, a_p, batch_size, input_shape,  classes, conf, name=model_name)


    return model
