import tensorflow as tf

from tensorflow.python.keras.engine import training

import functools

#
#import lib_snn.layers
import lib_snn

from lib_snn.layers import tfn


#
# noinspection PyUnboundLocalVariable
# class VGG16_CIFAR(lib_snn.model.Model,tf.keras.layers.Layer):
# class VGG16(lib_snn.model.Model):
def VGG16(
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
        #dropout_conv_r = [0.5, 0.5, 0.0]      # SNN training
        #dropout_conv_r = [0.25, 0.25, 0.25]      # SNN training
        #dropout_conv_r = [0.3, 0.3, 0.3]      # SNN training
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
    elif 'CIFAR' in dataset_name:
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

    #
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, en_tdbn=tdbn_first_layer, name='conv1')(x)
    syn_c1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='conv1')(input)
    if use_bn_feat:
        norm_c1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn_first_layer,name='bn_conv1')(syn_c1)
    else:
        norm_c1 = syn_c1
    a_c1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv1')(norm_c1)
    a_d_c1 = tf.keras.layers.Dropout(dropout_conv_r[0], name='conv1_do')(a_c1)
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv1_1')(x)
    syn_c1_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv1_1')(a_d_c1)
    if use_bn_feat:
        norm_c1_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv1_1')(syn_c1_1)
    else:
        norm_c1_1 = syn_c1_1
    a_c1_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv1_1')(norm_c1_1)
    #x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv1_p')(x)
    #x = tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv1_p')(x)
    #x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv1_p', dynamic=True)(x)
    #x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv1_p')(x)
    a_p_c1_1 = pool((2, 2), (2, 2), name='conv1_p')(a_c1_1)

    #
    channels = channels * 2
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv2')(x)
    syn_c2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv2')(a_p_c1_1)
    if use_bn_feat:
        norm_c2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv2')(syn_c2)
    else:
        norm_c2 = syn_c2
    a_c2 = lib_snn.activations.Activation(act_type=act_type,name='n_conv2')(norm_c2)
    a_d_c2 = tf.keras.layers.Dropout(dropout_conv_r[0], name='conv2_do')(a_c2)
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv2_1')(x)
    syn_c2_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv2_1')(a_d_c2)
    if use_bn_feat:
        norm_c2_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv2_1')(syn_c2_1)
    else:
        norm_c2_1 = syn_c2_1
    a_c2_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv2_1')(norm_c2_1)
    #x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv2_p')(x)
    #x = tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv2_p')(x)
    #x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv2_p')(x)
    a_p_c2_1 = pool((2, 2), (2, 2), name='conv2_p')(a_c2_1)

    #
    channels = channels * 2
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv3')(x)
    syn_c3 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3')(a_p_c2_1)
    if use_bn_feat:
        norm_c3 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3')(syn_c3)
    else:
        norm_c3 = syn_c3
    a_c3 = lib_snn.activations.Activation(act_type=act_type,name='n_conv3')(norm_c3)
    a_d_c3 = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv3_do')(a_c3)
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv3_1')(x)
    syn_c3_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3_1')(a_d_c3)
    if use_bn_feat:
        norm_c3_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3_1')(syn_c3_1)
    else:
        norm_c3_1 = syn_c3_1
    a_c3_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv3_1')(norm_c3_1)
    a_d_c3_1 = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv3_1_do')(a_c3_1)
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv3_2')(x)
    syn_c3_2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3_2')(a_d_c3_1)
    if use_bn_feat:
        norm_c3_2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3_2')(syn_c3_2)
    else:
        norm_c3_2 = syn_c3_2
    a_c3_2 = lib_snn.activations.Activation(act_type=act_type,name='n_conv3_2')(norm_c3_2)
    #x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv3_p')(x)
    #x = tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv3_p')(x)
    #x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv3_p') (x)
    a_p_c3_2 = pool((2, 2), (2, 2), name='conv3_p') (a_c3_2)

    #
    channels = channels * 2
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv4')(x)
    syn_c4 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4')(a_p_c3_2)
    if use_bn_feat:
        norm_c4 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4')(syn_c4)
    else:
        norm_c4 = syn_c4
    a_c4 = lib_snn.activations.Activation(act_type=act_type,name='n_conv4')(norm_c4)
    a_d_c4 = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv4_do')(a_c4)
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv4_1')(x)
    syn_c4_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4_1')(a_d_c4)
    if use_bn_feat:
        norm_c4_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4_1')(syn_c4_1)
    else:
        norm_c4_1 = syn_c4_1
    a_c4_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv4_1')(norm_c4_1)
    a_d_c4_1 = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv4_1_do')(a_c4_1)
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv4_2')(x)
    syn_c4_2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4_2')(a_d_c4_1)
    if use_bn_feat:
        norm_c4_2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4_2')(syn_c4_2)
    else:
        norm_c4_2 = syn_c4_2
    a_c4_2 = lib_snn.activations.Activation(act_type=act_type,name='n_conv4_2')(norm_c4_2)
    #x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv4_p')(x)
    #x = tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv4_p')(x)
    #x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv4_p') (x)
    a_p_c4_2 = pool((2, 2), (2, 2), name='conv4_p') (a_c4_2)


    #
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv5')(x)
    syn_c5 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv5')(a_p_c4_2)
    if use_bn_feat:
        norm_c5 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv5')(syn_c5)
    else:
        norm_c5 = syn_c5
    a_c5 = lib_snn.activations.Activation(act_type=act_type,name='n_conv5')(norm_c5)
    a_d_c5 = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv5_do')(a_c5)
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv5_1')(x)
    syn_c5_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv5_1')(a_d_c5)
    if use_bn_feat:
        norm_c5_1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv5_1')(syn_c5_1)
    else:
        norm_c5_1 = syn_c5_1
    a_c5_1 = lib_snn.activations.Activation(act_type=act_type,name='n_conv5_1')(norm_c5_1)
    a_d_c5_1 = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv5_1_do')(a_c5_1)
    #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv5_2')(x)
    syn_c5_2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv5_2')(a_d_c5_1)
    if use_bn_feat:
        norm_c5_2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv5_2')(syn_c5_2)
    else:
        norm_c5_2 = syn_c5_2
    a_c5_2 = lib_snn.activations.Activation(act_type=act_type,name='n_conv5_2')(norm_c5_2)
    #x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv5_p')(x)
    #x = tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv5_p')(x)
    #x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv5_p') (x)
    a_p_c5_2 = pool((2, 2), (2, 2), name='conv5_p') (a_c5_2)


    #
    a_p_c5_2_f = tf.keras.layers.Flatten(data_format=data_format,name='flatten')(a_p_c5_2)
    a_d_c5_2_f = tf.keras.layers.Dropout(dropout_conv_r[2], name='flatten_do')(a_p_c5_2_f)
    #x = lib_snn.layers.Dense(n_dim_cls, activation=act_relu, use_bn=use_bn_cls, kernel_initializer=k_init, naimg_input, a_p, batch_size, input_shape,  classes, conf, name=model_nameme='fc1')(x)
    syn_fc1 = lib_snn.layers.Dense(n_dim_cls, kernel_initializer=k_init, name='fc1')(a_d_c5_2_f)
    if use_bn_cls:
        norm_fc1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_fc1')(syn_fc1)
    else:
        norm_fc1 = syn_fc1
    a_fc1 = lib_snn.activations.Activation(act_type=act_type,name='n_fc1')(norm_fc1)
    a_d_fc1 = tf.keras.layers.Dropout(dropout_conv_r[2], name='fc1_do')(a_fc1)
    #x = lib_snn.layers.Dense(n_dim_cls, activation=act_relu, use_bn=use_bn_cls, kernel_initializer=k_init, name='fc2')(x)
    syn_fc2 = lib_snn.layers.Dense(n_dim_cls, kernel_initializer=k_init, name='fc2')(a_d_fc1)
    if use_bn_cls:
        norm_fc2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_fc2')(syn_fc2)
    else:
        norm_fc2 = syn_fc2
    a_fc2 = lib_snn.activations.Activation(act_type=act_type,name='n_fc2')(norm_fc2)
    a_d_fc2 = tf.keras.layers.Dropout(dropout_conv_r[2], name='fc2_do')(a_fc2)
    #x = lib_snn.layers.Dense(classes, activation=act_sm, use_bn=False, last_layer=True, kernel_initializer=k_init, name='predictions')(x)
    syn_p = lib_snn.layers.Dense(classes, last_layer=True, kernel_initializer=k_init, name='predictions')(a_d_fc2)
    a_p = lib_snn.activations.Activation(act_type=act_type_out,loc='OUT',name='n_predictions')(syn_p)
    if conf.nn_mode=='SNN':
        a_p = lib_snn.activations.Activation(act_type='softmax',name='a_predictions')(a_p)




    #model = training.Model(img_input, x, name=name)
    #model = lib_snn.model.Model(img_input, x, batch_size, input_shape,  classes, conf, name=model_name)
    model = lib_snn.model.Model(img_input, a_p, batch_size, input_shape,  classes, conf, name=model_name)
    #model = lib_snn.model.Model(img_input, out, batch_size, input_shape,  classes, conf, name=model_name)
    #model = lib_snModel.init_graph(img_input, x, name=name)

    #model.add_loss(tf.reduce_sum(syn_c1))
    #model.add_loss(tf.reduce_sum(a_p))
    #model.get_layer('conv1')
    #model.add_loss(functools.partial(tf.reduce_mean,syn_c1))
    #model.add_loss(functools.partial(tf.reduce_mean,syn_c2))
    #model.add_loss(lambda: tf.reduce_mean(syn_c2))



    # return training.Model(img_input, x, name=self.name)

    #        self.out = x

    #    def build(self, input_shape):
    #        img_input = tf.keras.layers.Input(shape=self.in_shape, batch_size=self.batch_size)
    #        # create model
    #        self.model = training.Model(img_input, self.out, name=self.name)

    # self.model.load_weights(weights)

    # self.load_weights = weights

    # if weights is not None:
    # self.model.load_weights(weights)
    # self.set_weights(weights)
    # self.model.set_weights(weights)
    # print('load weights done')

    #model.summary()

    return model
