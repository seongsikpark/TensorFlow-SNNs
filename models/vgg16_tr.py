import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16

#
import lib_snn


from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils

# from keras applications.VGG16
WEIGHTS_PATH = ('https://storage.googleapis.com/tensorflow/keras-applications/'
                'vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                       'keras-applications/vgg16/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
#
# noinspection PyUnboundLocalVariable
#class VGG16_CIFAR(lib_snn.model.Model,tf.keras.layers.Layer):
#class VGG16():
def VGG16_TR(
        input_shape,
        conf,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'):


    data_format = conf.data_format


    #
    #dropout_conv = tf.keras.layers.Dropout(0.3)
    dropout_conv2 = tf.keras.layers.Dropout(0.4)
    dropout = tf.keras.layers.Dropout(0.5)
    dropout_conv_r = [0.3,0.4,0.5]

    use_bn=False

    #
    lmb=conf.lmb

    #
    model = tf.keras.Sequential()

    pretrained_model = VGG16(input_shape=input_shape, include_top=False, classes=classes, weights='imagenet')

    #
    pretrained_model.trainable = False

    # train = True
    # data augmentation
    #if train:
        ## model.add(tf.keras.layers.GaussianNoise(0.1))
        #model.add(tf.keras.layers.experimental.preprocessing.RandomZoom((-0.1, 0.1)))
        #model.add(tf.keras.layers.experimental.preprocessing.RandomRotation((-0.03, 0.03)))

    model.add(pretrained_model)
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation=None, kernel_regularizer=tf.keras.regularizers.L2(lmb),
                                    name='fc1'))
    model.add(tf.keras.layers.BatchNormalization(name='fc1_bn'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation=None, kernel_regularizer=tf.keras.regularizers.L2(lmb),
                                    name='fc2'))
    model.add(tf.keras.layers.BatchNormalization(name='fc2_bn'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(classes, activation='softmax', name='predictions'))


    model.summary()

    return model