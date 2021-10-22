import tensorflow as tf

from tensorflow.python.keras.engine import training

#
import lib_snn


#
# noinspection PyUnboundLocalVariable
# class VGG16_CIFAR(lib_snn.model.Model,tf.keras.layers.Layer):
class VGG16(lib_snn.model.Model):
    # def __init__(self, input_shape, data_format, conf):
    def __init__(self,
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
                 model_name='VGG16',
                 **kwargs):
        self.batch_size = batch_size
        self.in_shape = input_shape

        data_format = conf.data_format

        lib_snn.model.Model.__init__(self, batch_size, input_shape, data_format, classes, conf)

        #
        act_relu = 'relu'
        act_sm = 'softmax'

        #
        self.dropout_conv_r = [0, 0, 0]

        #
        initial_channels = kwargs.pop('initial_channels', None)
        assert initial_channels is not None

        #
        use_bn_feat = True
        use_bn_cls = True

        #
        channels = initial_channels

        img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

        # x = img_input
        self.layer_in = lib_snn.layers.InputGenLayer(name='in')

        #
        self.conv1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                           name='conv1')
        self.conv1_do = tf.keras.layers.Dropout(self.dropout_conv_r[0], name='conv1_do')
        self.conv1_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                             name='conv1_1')
        self.conv1_p = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv1_p')

        #
        channels = channels * 2
        self.conv2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                           name='conv2')
        self.conv2_do = tf.keras.layers.Dropout(self.dropout_conv_r[0], name='conv2_do')
        self.conv2_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                             name='conv2_1')
        self.conv2_p = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv2_p')

        #
        channels = channels * 2
        self.conv3 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                           name='conv3')
        self.conv3_do = tf.keras.layers.Dropout(self.dropout_conv_r[1], name='conv3_do')
        self.conv3_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                             name='conv3_1')
        self.conv3_1_do = tf.keras.layers.Dropout(self.dropout_conv_r[1], name='conv3_1_do')
        self.conv3_2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                             name='conv3_2')
        self.conv3_p = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv3_p')

        #
        channels = channels * 2
        self.conv4 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                           name='conv4')
        self.conv4_do = tf.keras.layers.Dropout(self.dropout_conv_r[1], name='conv4_do')
        self.conv4_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                             name='conv4_1')
        self.conv4_1_do = tf.keras.layers.Dropout(self.dropout_conv_r[1], name='conv4_1_do')
        self.conv4_2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                             name='conv4_2')
        self.conv4_p = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv4_p')

        #
        self.conv5 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                           name='conv5')
        self.conv5_do = tf.keras.layers.Dropout(self.dropout_conv_r[1], name='conv5_do')
        self.conv5_1 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                             name='conv5_1')
        self.conv5_1_do = tf.keras.layers.Dropout(self.dropout_conv_r[1], name='conv5_1_do')
        self.conv5_2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat,
                                             name='conv5_2')
        self.conv5_p = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv5_p')

        #
        self.flat = tf.keras.layers.Flatten(data_format=data_format)
        self.flat_do = tf.keras.layers.Dropout(self.dropout_conv_r[2], name='flatten_do')
        self.fc1 = lib_snn.layers.Dense(512, activation=act_relu, use_bn=use_bn_cls, name='fc1')
        self.fc1_do = tf.keras.layers.Dropout(self.dropout_conv_r[2], name='fc1_do')
        self.fc2 = lib_snn.layers.Dense(512, activation=act_relu, use_bn=use_bn_cls, name='fc2')
        self.fc2_do = tf.keras.layers.Dropout(self.dropout_conv_r[2], name='fc2_do')
        self.predictions = lib_snn.layers.Dense(classes, activation=act_sm, use_bn=False, name='predictions')

        #


        # self.model = training.Model(img_input, x, name=self.name)
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

    # self.model.summary()
    # self.summary()
    # assert False

    #def call(self,inputs,training,**kwargs):
        #ret = self.call_ann(inputs,training,**kwargs)
        #return ret


    def call_ann(self, inputs, training, **kwargs):
        x = self.layer_in(inputs)

        x = self.conv1(x)
        x = self.conv1_do(x)
        x = self.conv1_1(x)
        x = self.conv1_p(x)

        x = self.conv2(x)
        x = self.conv2_do(x)
        x = self.conv2_1(x)
        x = self.conv2_p(x)

        x = self.conv3(x)
        x = self.conv3_do(x)
        x = self.conv3_1(x)
        x = self.conv3_1_do(x)
        x = self.conv3_2(x)
        x = self.conv3_p(x)

        x = self.conv4(x)
        x = self.conv4_do(x)
        x = self.conv4_1(x)
        x = self.conv4_1_do(x)
        x = self.conv4_2(x)
        x = self.conv4_p(x)

        x = self.conv5(x)
        x = self.conv5_do(x)
        x = self.conv5_1(x)
        x = self.conv5_1_do(x)
        x = self.conv5_2(x)
        x = self.conv5_p(x)

        x = self.flat(x)
        x = self.flat_do(x)
        x = self.fc1(x)
        x = self.fc1_do(x)
        x = self.fc2(x)
        x = self.fc2_do(x)
        x = self.predictions(x)

        return x
