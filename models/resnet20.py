import tensorflow as tf

#
import lib_snn


#
#class block1(tf.keras.layers):
class block1(tf.keras.Sequential):
    def __init__(self,
                 input_shape,
                 conf,
                 filters,

                 conv_shortcut,
                 **kwargs):
        bn_axis = 3 if conf.data_format == 'channels_last' else None

        if conv_shortcut:
            self.add(lib_snn.layers.Conv2D(4*filters,) )




    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)




def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
  """A residual block.

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  if conv_shortcut:
    shortcut = layers.Conv2D(
        4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = layers.Conv2D(
      filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Activation('relu', name=name + '_2_relu')(x)

  x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation('relu', name=name + '_out')(x)
  return

#
class ResNet20(lib_snn.model.Model):
    def __init__(self,
                 input_shape,
                 conf,
                 include_top=True,
                 #weights='imagenet',
                 weights=None,
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 classifier_activation='softmax',
                 **kwargs):
        #train=False,
        #add_top=False):

        data_format = conf.data_format


        lib_snn.model.Model.__init__(self, input_shape, data_format, classes, conf)
        #lib_snn.layers.Layer.__init__(self, input_shape, data_format, conf)
        #tf.keras.layers.Layer.__init__(self,name='')

        #
        self.kernel_size = 3

        padding = 'SAME'
        act_relu = 'relu'
        act_sm = 'softmax'


        #
        #self.dropout_conv = tf.keras.layers.Dropout(0.3)
        #self.dropout_conv2 = tf.keras.layers.Dropout(0.4)
        #self.dropout = tf.keras.layers.Dropout(0.5)
        #self.dropout_conv_r = [0.3,0.4,0.5]

        #
        n_dim_classifer = kwargs.pop('n_dim_classifier', None)

        if n_dim_classifer is None:
            n_dim_classifer = (4096, 4096)


        #
        self.model = tf.keras.Sequential()


        #






        assert False

        #use_bn_feat = False
        use_bn_feat = True
        use_bn_cls = True

        #if train:
        #self.model.add(tf.keras.layers.experimental.preprocessing.RandomZoom((-0.1, 0.1)))
        #self.model.add(tf.keras.layers.experimental.preprocessing.RandomRotation((-0.03, 0.03)))

        self.model.add(lib_snn.layers.InputLayer(input_shape=input_shape,batch_size=conf.batch_size,name='in'))
        self.model.add(lib_snn.layers.InputGenLayer())
        #self.model.add(lib_snn.layers.InputLayer(input_shape=input_shape,name='in'))
        self.model.add(lib_snn.layers.Conv2D(64,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[0],name='conv1_do'))
        self.model.add(lib_snn.layers.Conv2D(64,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv1_1'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),name='conv1_p'))

        self.model.add(lib_snn.layers.Conv2D(128,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv2'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[0],name='conv2_do'))
        self.model.add(lib_snn.layers.Conv2D(128,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv2_1'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),name='conv2_p'))

        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv3'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1],name='conv3_do'))
        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv3_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1],name='conv3_1_do'))
        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv3_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),name='conv3_p'))

        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv4'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1],name='conv4_do'))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv4_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1],name='conv4_1_do'))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv4_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),name='conv4_p'))

        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv5'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1],name='conv5_do'))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv5_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1],name='conv5_1_do'))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv5_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),name='conv5_p'))

        self.model.add(tf.keras.layers.Flatten(data_format=data_format))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2],name='flatten_do'))
        #self.model.add(lib_snn.layers.Dense(4096,activation=act_relu,use_bn=use_bn_cls,name='fc1'))
        #self.model.add(lib_snn.layers.Dense(512,activation=act_relu,use_bn=use_bn_cls,name='fc1'))
        self.model.add(lib_snn.layers.Dense(n_dim_classifer[0],activation=act_relu,use_bn=use_bn_cls,name='fc1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2],name='fc1_do'))
        #self.model.add(lib_snn.layers.Dense(4096,activation=act_relu,use_bn=use_bn_cls,name='fc2'))
        #self.model.add(lib_snn.layers.Dense(512,activation=act_relu,use_bn=use_bn_cls,name='fc2'))
        self.model.add(lib_snn.layers.Dense(n_dim_classifer[1],activation=act_relu,use_bn=use_bn_cls,name='fc2'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2],name='fc2_do'))
        self.model.add(lib_snn.layers.Dense(classes,activation=act_sm,use_bn=False,name='predictions'))

        #self.model.load_weights(weights)

        #self.load_weights = weights

        #if weights is not None:
        #self.model.load_weights(weights)
        #self.set_weights(weights)
        #self.model.set_weights(weights)
        #print('load weights done')

        self.model.summary()

    #
    #def build(self, input_shapes):
#
## build model
#lib_snn.model.Model.build(self, input_shapes)
#
#
##self.model.set_weights(self.load_weights)
#
