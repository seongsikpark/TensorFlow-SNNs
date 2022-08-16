import tensorflow as tf

from tensorflow.python.keras.engine import training

#
import lib_snn


#
# noinspection PyUnboundLocalVariable
# class VGG16_CIFAR(lib_snn.model.Model,tf.keras.layers.Layer):
# class VGG16(lib_snn.model.Model):
def VGG16(
        # def __init__(self, input_shape, data_format, conf):
        batch_size,
        input_shape,
        conf,
        include_top=True,
        # weights='imagenet',
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        #nn_mode='ANN',
        name='VGG16',
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
        dropout_conv_r = [0.0, 0.0, 0.0]      # SNN training
    else:
        assert False

    #
    initial_channels = kwargs.pop('initial_channels', None)
    assert initial_channels is not None

    #
    use_bn_feat = conf.use_bn
    use_bn_cls = conf.use_bn

    #
    channels = initial_channels

    img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    # x = img_input
    x = lib_snn.layers.InputGenLayer(name='in')(img_input)

    #
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv1')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[0], name='conv1_do')(x)
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv1_1')(x)
    x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv1_p')(x)

    #
    channels = channels * 2
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv2')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[0], name='conv2_do')(x)
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv2_1')(x)
    x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv2_p')(x)

    #
    channels = channels * 2
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv3')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv3_do')(x)
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv3_1')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv3_1_do')(x)
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv3_2')(x)
    x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv3_p')(x)

    #
    channels = channels * 2
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv4')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv4_do')(x)
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv4_1')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv4_1_do')(x)
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv4_2')(x)
    x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv4_p')(x)

    #
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv5')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv5_do')(x)
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv5_1')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv5_1_do')(x)
    x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, name='conv5_2')(x)
    x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv5_p')(x)

    #
    x = tf.keras.layers.Flatten(data_format=data_format)(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[2], name='flatten_do')(x)
    x = lib_snn.layers.Dense(512, activation=act_relu, use_bn=use_bn_cls, name='fc1')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[2], name='fc1_do')(x)
    x = lib_snn.layers.Dense(512, activation=act_relu, use_bn=use_bn_cls, name='fc2')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[2], name='fc2_do')(x)
    x = lib_snn.layers.Dense(classes, activation=act_sm, use_bn=False, last_layer=True, name='predictions')(x)

    #model = training.Model(img_input, x, name=name)
    model = lib_snn.model.Model(img_input, x, batch_size, input_shape,  data_format, classes, conf, name=name)
    #model = lib_snModel.init_graph(img_input, x, name=name)



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

#
# def build(self, input_shapes):
#
## build model
# lib_snn.model.Model.build(self, input_shapes)
#
#
##self.model.set_weights(self.load_weights)
#
