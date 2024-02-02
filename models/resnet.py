import tensorflow as tf

from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils

#
import lib_snn



#
#from config import conf
#from config_common import conf
from absl import flags
conf = flags.FLAGS

tdbn = conf.mode=='train' and conf.nn_mode=='SNN' and conf.tdbn


#
if conf.nn_mode=='ANN':
    act_type = 'relu'
    act_type_out = 'softmax'
else:
    act_type = conf.n_type
    act_type_out = conf.n_type


BASE_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/resnet/')
WEIGHTS_HASHES = {
    'resnet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
    'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21'),
    'resnet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
    'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'resnet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
    'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                  '62527c363bdd9ec598bed41947b379fc'),
    'resnext101':
        ('34fb605428fcc7aa4d62f44404c11509', '0f678c91647380debd923963594981b3')
}


#
def block_basic(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
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
    # bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    # bn_axis = 3  # 'channels_last' only

    if conv_shortcut:
        #shortcut = lib_snn.layers.Conv2D(filters, 1, strides=stride, use_bn=True, activation=None, name=name + '_conv0')(x)
        #shortcut = lib_snn.layers.Conv2D(filters, 1, strides=stride, name=name + '_conv0',kernel_initializer='zeros')(x)
        shortcut = lib_snn.layers.Conv2D(filters, 1, strides=stride, name=name + '_conv0')(x)
        shortcut = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name=name+'_conv0_bn')(shortcut)
    else:
        #shortcut = x
        shortcut = lib_snn.layers.Identity(name=name + '_conv0_i') (x)
        #shortcut = lib_snn.layers.Conv2D(1,1,strides=1,use_bn=False,activation=None,name=name+'_conv0_i',kernel_initializer='ones',trainable=False)(x)

    #x = lib_snn.layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME', use_bn=True, activation='relu',name=name + '_conv1')(x)
    x = lib_snn.layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME', name=name + '_conv1')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name=name+'_conv1_bn')(x)
    x = lib_snn.activations.Activation(act_type=act_type,name=name+'_conv1_n')(x)

    #x = lib_snn.layers.Conv2D(filters, kernel_size, padding='SAME', use_bn=True, activation='relu',name=name + '_conv2')(x)
    #x = lib_snn.layers.Conv2D(filters, kernel_size, padding='SAME', use_bn=True, activation=None,name=name + '_conv2')(x)
    x = lib_snn.layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_conv2')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name=name+'_conv2_bn')(x)

    #x = lib_snn.layers.Add(use_bn=False, activation='relu', name=name + '_out')([shortcut, x])
    #print([shortcut, x])
    x = lib_snn.layers.Add(name=name + '_out')([shortcut, x])
    x = lib_snn.activations.Activation(act_type=act_type,name=name+'_out_n')(x)

    return x


# keras - resnet.py based
def block_bottleneck(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
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
    # bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    # bn_axis = 3  # 'channels_last' only



    if conv_shortcut:
        #shortcut = lib_snn.layers.Conv2D(4 * filters, 1, strides=stride, use_bn=True, activation=None, name=name + '_conv0')(x)
        #shortcut = lib_snn.layers.Conv2D(4 * filters, 1, strides=stride, use_bn=True, activation=None, name=name + '_conv0',kernel_initializer='zeros')(x)
        shortcut = lib_snn.layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_conv0')(x)
        shortcut = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name=name+'_conv0_bn')(shortcut)
    else:
        #shortcut = x
        shortcut = lib_snn.layers.Identity(name=name + '_conv0_i') (x)
        #shortcut = lib_snn.layers.Conv2D(1,1,strides=1,use_bn=False,activation=None,name=name+'_conv0_i',kernel_initializer='ones',trainable=False) (x)

    #x = lib_snn.layers.Conv2D(filters, 1, strides=stride, use_bn=True, activation='relu', epsilon=1.001e-5, name=name + '_conv_1')(x)
    #x = lib_snn.layers.Conv2D(filters, 1, strides=stride, use_bn=True, activation='relu', name=name + '_conv1',kernel_initializer='zeros')(x)
    x = lib_snn.layers.Conv2D(filters, 1, strides=stride, name=name + '_conv1')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name=name+'_conv1_bn')(x)
    x = lib_snn.activations.Activation(act_type=act_type,name=name+'conv1_n')(x)

    #x = tf.keras.layers.Dropout(0.3,name=name+'_conv1_do')(x)
    # x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    # x = layers.Activation('relu', name=name + '_1_relu')(x)

    #x = lib_snn.layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME', use_bn=True, activation='relu',epsilon=1.001e-5, name=name + '_conv_2')(x)
    #x = lib_snn.layers.Conv2D(filters, kernel_size, padding='SAME', use_bn=True, activation='relu',name=name + '_conv2',kernel_initializer='zeros')(x)
    x = lib_snn.layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_conv2')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name=name+'_conv2_bn')(x)
    x = lib_snn.activations.Activation(act_type=act_type,name=name+'conv2_n')(x)


    #x = tf.keras.layers.Dropout(0.4,name=name+'_conv2_do')(x)
    # x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    # x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    # x = layers.Activation('relu', name=name + '_2_relu')(x)

    #x = lib_snn.layers.Conv2D(4*filters, 1, strides=stride, use_bn=True, activation=None,epsilon=1.001e-5, name=name + '_conv_3')(x)
    #x = lib_snn.layers.Conv2D(4*filters, 1, use_bn=True, activation=None, name=name + '_conv3',kernel_initializer='zeros')(x)
    x = lib_snn.layers.Conv2D(4*filters, 1, name=name + '_conv3')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name=name+'_conv3_bn')(x)

    #x = tf.keras.layers.Dropout(0.5,name=name+'_conv3_do')(x)
    # x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    # x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    #x = lib_snn.layers.Add(use_bn=False, activation='relu', name=name + '_out')([shortcut, x])
    x = lib_snn.layers.Add(name=name + '_out')([shortcut, x])
    x = lib_snn.activations.Activation(act_type=act_type,name=name+'_out_n')(x)

    # x = layers.Add(name=name + '_add')([shortcut, x])
    # x = layers.Activation('relu', name=name + '_out')(x)

    return x

#
#
#
#def block_basic(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
#    """A residual block.
#
#    Args:
#      x: input tensor.
#      filters: integer, filters of the bottleneck layer.
#      kernel_size: default 3, kernel size of the bottleneck layer.
#      stride: default 1, stride of the first layer.
#      conv_shortcut: default True, use convolution shortcut if True,
#          otherwise identity shortcut.
#      name: string, block label.
#
#    Returns:
#      Output tensor for the residual block.
#    """
#    # bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
#    # bn_axis = 3  # 'channels_last' only
#
#    if conv_shortcut:
#        shortcut = lib_snn.layers.Conv2D(filters, 1, strides=stride, name=name + '_conv0',kernel_initializer='zeros')(x)
#        shortcut = lib_snn.layers_new.BatchNormalization(en_tdbn=tdbn,name=name+'_conv0_bn')(shortcut)
#    else:
#        #shortcut = x
#        shortcut = lib_snn.layers.Identity(name=name + '_conv0_i') (x)
#        #shortcut = lib_snn.layers.Conv2D(1,1,strides=1,use_bn=False,activation=None,name=name+'_conv0_i',kernel_initializer='ones',trainable=False)(x)
#
#    #x = lib_snn.layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME', use_bn=True, activation='relu',name=name + '_conv1')(x)
#    x = lib_snn.layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME', name=name + '_conv1')(x)
#    x = lib_snn.layers_new.BatchNormalization(en_tdbn=tdbn,name=name+'_conv1_bn')(x)
#    x = lib_snn.activations.Activation(act_type=act_type,name=name+'conv1_n')(x)
#
#
#    #x = lib_snn.layers.Conv2D(filters, kernel_size, padding='SAME', use_bn=True, activation=None,name=name + '_conv2')(x)
#    x = lib_snn.layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_conv2')(x)
#    x = lib_snn.layers_new.BatchNormalization(en_tdbn=tdbn,name=name+'_conv2_bn')(x)
#
#
#    #x = lib_snn.layers.Add(use_bn=False, activation='relu', name=name + '_out')([shortcut, x])
#    #x = lib_snn.layers.Add(name=name + '_out')([shortcut, x])
#    x = lib_snn.activations.Activation(act_type=act_type,name=name+'_out_n')(x)
#
#    return x



# keras, resnet.py based
def stack1(x, filters, block, num_block, stride=2, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = block(x, filters, stride=stride, name=name + '_block1')
    for i in range(2, num_block + 1):
        x = block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


## keras, resnet.py based
#def ResNet50(include_top=True,
#             weights='imagenet',
#             input_tensor=None,
#             input_shape=None,
#             pooling=None,
#             classes=1000,
#             **kwargs):
#    """Instantiates the ResNet50 architecture."""
#
#    def stack_fn(x):
#        x = stack1(x, 64, 3, stride1=1, name='conv2')
#        x = stack1(x, 128, 4, name='conv3')
#        x = stack1(x, 256, 6, name='conv4')
#        return stack1(x, 512, 3, name='conv5')
#
#    return ResNet(stack_fn, False, True, 'resnet50', include_top, weights,
#                  input_tensor, input_shape, pooling, classes, **kwargs)

# TODO - class
"""Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

Args:
  stack_fn: a function that returns output tensor for the
    stacked residual blocks.
  preact: whether to use pre-activation or not
    (True for ResNetV2, False for ResNet and ResNeXt).
  use_bias: whether to use biases for convolutional layers or not
    (True for ResNet and ResNetV2, False for ResNeXt).
  name: string, model name.
  include_top: whether to include the fully-connected
    layer at the top of the network.
  weights: one of `None` (random initialization),
    'imagenet' (pre-training on ImageNet),
    or the path to the weights file to be loaded.
  input_tensor: optional Keras tensor
    (i.e. output of `layers.Input()`)
    to use as image input for the model.
  input_shape: optional shape tuple, only to be specified
    if `include_top` is False (otherwise the input shape
    has to be `(224, 224, 3)` (with `channels_last` data format)
    or `(3, 224, 224)` (with `channels_first` data format).
    It should have exactly 3 inputs channels.
  pooling: optional pooling mode for feature extraction
    when `include_top` is `False`.
    - `None` means that the output of the model will be
        the 4D tensor output of the
        last convolutional layer.
    - `avg` means that global average pooling
        will be applied to the output of the
        last convolutional layer, and thus
        the output of the model will be a 2D tensor.
    - `max` means that global max pooling will
        be applied.
  classes: optional number of classes to classify images
    into, only to be specified if `include_top` is True, and
    if no `weights` argument is specified.
  classifier_activation: A `str` or callable. The activation function to use
    on the "top" layer. Ignored unless `include_top=True`. Set
    `classifier_activation=None` to return the logits of the "top" layer.
    When loading pretrained weights, `classifier_activation` can only
    be `None` or `"softmax"`.
  **kwargs: For backwards compatibility only.

Returns:
  A `keras.Model` instance.
"""

# keras, resnet.py based
#class ResNet(lib_snn.model.Model):
#def __init__(self,
def ResNet(
    batch_size,
    input_shape,
    conf,
    model_name,
    block,
    initial_channels,
    num_blocks,
    preact=False,
    #use_bias=True,
    #name='ResNet',
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    **kwargs):

    data_format = conf.data_format

    dataset_name = kwargs.pop('dataset_name', None)
    if dataset_name == 'ImageNet':
        imagenet_pretrain = True
    else:
        imagenet_pretrain = False

    #
    cifar_stack = True if len(num_blocks) == 3 else False

    #lib_snn.model.Model.__init__(self, input_shape, data_format, classes, conf, **kwargs)

    bn_axis = 3


    img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    #img_input = tf.keras.layers.InputLayer(shape=input_shape, batch_size=batch_size)
    x = lib_snn.layers.InputGenLayer(name='in')(img_input)

    if conf.nn_mode=='SNN':
        x = lib_snn.activations.Activation(act_type=act_type,loc='IN',name='n_in')(x)

    #img_input = lib_snn.layers.InputLayer(input_shape=input_shape,batch_size=conf.batch_size,name='in')

    if imagenet_pretrain:
        # ImageNet pretrained model - tf.keras.applications
        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
    else:
        pass
        #x = img_input
        #x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_pad')(x)

    if not preact:
        preact_bn = True
        #preact_act = 'relu'
        preact_act = act_type
    else:
        preact_bn = False
        preact_act = None

    if imagenet_pretrain:
        #x = lib_snn.layers.Conv2D(initial_channels, 7, strides=2, use_bn=preact_bn, activation=preact_act, name='conv1_conv')(x)
        x = lib_snn.layers.Conv2D(initial_channels, 7, strides=2, name='conv1_conv')(x)
        if preact_bn:
            x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='conv1_conv_bn')(x)
        if not preact_act is None:
            #x = lib_snn.activations.Activation(act_type=act_type,name='conv1_conv_n')(x)
            x = lib_snn.activations.Activation(act_type=preact_act,name='conv1_conv_n')(x)

    else:
        #x = lib_snn.layers.Conv2D(initial_channels, 3, strides=1, padding='same', use_bn=preact_bn, activation=preact_act, name='conv1_conv')(x)
        x = lib_snn.layers.Conv2D(initial_channels, 3, strides=1, padding='same', name='conv1_conv')(x)
        if preact_bn:
            x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='conv1_conv_bn')(x)
        if not preact_act is None:
            x = lib_snn.activations.Activation(act_type=preact_act,name='conv1_conv_n')(x)

    if imagenet_pretrain:
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        #x = lib_snn.layers.MaxPool2D(3, strides=2, name='pool1_pool')(x)
        x = lib_snn.layers.AveragePooling2D(3, strides=2, name='pool1_pool')(x)
    else:
        pass


    if cifar_stack:
        x = stack1(x, initial_channels, block, num_blocks[0], stride=1, name='conv2')
        x = stack1(x, initial_channels*2, block, num_blocks[1], name='conv3')
        x = stack1(x, initial_channels*4, block, num_blocks[2], name='conv4')
    else:
        x = stack1(x, initial_channels, block, num_blocks[0], stride=1, name='conv2')
        x = stack1(x, initial_channels*2, block, num_blocks[1], name='conv3')
        x = stack1(x, initial_channels*4, block, num_blocks[2], name='conv4')
        x = stack1(x, initial_channels*8, block, num_blocks[3], name='conv5')


    if preact:
        x = tf.keras.layers.BatchNormalization(name='post_bn')(x)
        x = tf.keras.layers.Activation('relu', name='post_relu')(x)

    if include_top:
        #x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = lib_snn.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        #x = tf.keras.layers.Dense(classes, activation=classifier_activation, name='predictions')(x)
        #x = lib_snn.layers.Dense(classes, activation=classifier_activation, use_bn=False, last_layer=True, name='predictions')(x)
        x = lib_snn.layers.Dense(classes, last_layer=True, name='predictions')(x)
        x = lib_snn.activations.Activation(act_type=act_type_out,loc='OUT',name='predictions_n')(x)
        if conf.nn_mode=='SNN':
            x = lib_snn.activations.Activation(act_type='softmax',name='predictions_a')(x)

    else:
        assert False
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            assert False
            #x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)
            #x = lib_snn.layers.GlobalMaxPooling2D(name='max_pool')(x)


    # Create model.
    #self.model = training.Model(img_input, x, name=model_name)
    model = lib_snn.model.Model(img_input, x, batch_size, input_shape, classes, conf, name=model_name)

    # Load weights.
    if False:
        if (weights == 'imagenet') and (name in WEIGHTS_HASHES):
            if include_top:
                file_name = name + '_weights_tf_dim_ordering_tf_kernels.h5'
                file_hash = WEIGHTS_HASHES[name][0]
            else:
                file_name = name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
                file_hash = WEIGHTS_HASHES[name][1]
            weights_path = data_utils.get_file(
                file_name,
                BASE_WEIGHTS_PATH + file_name,
                cache_subdir='models',
                file_hash=file_hash)
            self.model.load_weights(weights_path)
        elif weights is not None:
            self.model.load_weights(weights)

    #
    #self.model.summary()

    return model

#
def ResNet18(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [2,2,2,2]
    return ResNet(input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
    #return ResNet(input_shape=input_shape, block=block_bottleneck, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
#
def ResNet34(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [3,4,6,3]
    return ResNet(input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
    #return ResNet(input_shape=input_shape, block=block_bottleneck, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
#
def ResNet50(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [3,4,6,3]
    return ResNet(input_shape=input_shape, block=block_bottleneck, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
#
def ResNet101(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [3,4,23,3]
    return ResNet(input_shape=input_shape, block=block_bottleneck, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
#
def ResNet152(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [3,8,36,3]
    return ResNet(input_shape=input_shape, block=block_bottleneck, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)


#
def ResNet19(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 128
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [3,3,2]
    return ResNet(input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)


#
def ResNet20(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [3,3,3]
    return ResNet(input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)

#
def ResNet32(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [5,5,5]
    return ResNet(input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)

#
def ResNet44(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [7,7,7]
    return ResNet(input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)

#
def ResNet56(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [9,9,9]
    return ResNet(input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)





#
def ResNet18V2(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [2,2,2,2]
    return ResNet(preact=True, input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
    #return ResNet(preact=True, input_shape=input_shape, block=block_bottleneck, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
#
def ResNet34V2(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [3,4,6,3]
    return ResNet(preact=True, input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
    #return ResNet(preact=True, input_shape=input_shape, block=block_bottleneck, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
#
def ResNet50V2(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [3,4,6,3]
    return ResNet(preact=True, input_shape=input_shape, block=block_bottleneck, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
#
def ResNet20V2(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [3,3,3]
    return ResNet(preact=True, input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
#
def ResNet32V2(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [5,5,5]
    return ResNet(preact=True, input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
#
def ResNet44V2(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [7,7,7]
    return ResNet(preact=True, input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
#
def ResNet56V2(input_shape, conf, include_top, weights, classes, **kwargs):
    _initial_channels = 64
    initial_channels = kwargs.pop('initial_channels', _initial_channels)
    num_blocks = [9,9,9]
    return ResNet(preact=True, input_shape=input_shape, block=block_basic, initial_channels=initial_channels,
                  num_blocks=num_blocks, conf=conf, include_top=include_top,
                  weights=weights, classes=classes, **kwargs)
