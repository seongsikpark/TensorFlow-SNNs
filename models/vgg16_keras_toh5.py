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
class VGG16(lib_snn.model.Model):

    def __init__(self,
                 input_shape,
                 conf,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 # input_shape=None,
                 pooling=None,
                 classes=1000,
                 classifier_activation='softmax'):

        data_format = conf.data_format

        lib_snn.model.Model.__init__(self, input_shape, data_format, classes, conf)
        #lib_snn.layers.Layer.__init__(self, input_shape, data_format, conf)
        #tf.keras.layers.Layer.__init__(self,name='')

        #
        self.kernel_size = 3

        padding = 'SAME'
        padding_pool = 'valid'
        #padding_pool = 'SAME'
        #act_relu=tf.nn.relu
        act_relu=tf.keras.layers.ReLU()


        #
        #self.dropout_conv = tf.keras.layers.Dropout(0.3)
        self.dropout_conv2 = tf.keras.layers.Dropout(0.4)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dropout_conv_r = [0.3,0.4,0.5]

        use_bn=False

        #
        self.model = tf.keras.Sequential()

        self.model.add(lib_snn.layers.InputLayer(input_shape=input_shape,batch_size=conf.batch_size,name='in'))
        self.model.add(lib_snn.layers.Conv2D(64,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[0]))
        self.model.add(lib_snn.layers.Conv2D(64,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv1_1'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),padding_pool,name='conv1_p'))

        self.model.add(lib_snn.layers.Conv2D(128,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv2'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[0]))
        self.model.add(lib_snn.layers.Conv2D(128,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv2_1'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),padding_pool,name='conv2_p'))

        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv3'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv3_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv3_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),padding_pool,name='conv3_p'))

        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv4'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv4_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv4_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),padding_pool,name='conv4_p'))

        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv5'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv5_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=use_bn,name='conv5_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),padding_pool,name='conv5_p'))

        if include_top:
            self.model.add(tf.keras.layers.Flatten(data_format=data_format))
            self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
            #self.model.add(lib_snn.layers.Dense(512,activation=act_relu,use_bn=use_bn,name='fc1'))
            self.model.add(lib_snn.layers.Dense(4096,activation=act_relu,use_bn=use_bn,name='fc1'))
            self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
            #self.model.add(lib_snn.layers.Dense(512,activation=act_relu,use_bn=use_bn,name='fc2'))
            self.model.add(lib_snn.layers.Dense(4096,activation=act_relu,use_bn=use_bn,name='fc2'))
            self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
            self.model.add(lib_snn.layers.Dense(classes,activation=None,use_bn=use_bn,name='fc3'))
        #

        # Create model.
        #inputs = layer_utils.get_source_inputs(input_tensor)
        #self.model = training.Model(inputs, self.model.output, name='vgg16')

    #
    #def build(self, input_shapes):

        # build model
        #lib_snn.model.Model.build(self, input_shapes)

        #weights='imagenet'
        #include_top=True

        # Load weights.
        #if weights == 'imagenet':
        if True:
            if include_top:
                weights_path = data_utils.get_file(
                    'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                    WEIGHTS_PATH,
                    cache_subdir='models',
                    file_hash='64373286793e3c8b2b4e3219cbf3544b')
            else:
                weights_path = data_utils.get_file(
                    'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='6d6bbae143d832006294945121d1f1fc')
            self.model.load_weights(weights_path)
        elif weights is not None:
            self.model.load_weights(weights)


        use_bn=True
        self.model.add(tf.keras.layers.Flatten(data_format=data_format))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
        #self.model.add(lib_snn.layers.Dense(512,activation=act_relu,use_bn=use_bn,name='fc1'))
        self.model.add(lib_snn.layers.Dense(4096,activation=act_relu,use_bn=use_bn,name='fc1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
        #self.model.add(lib_snn.layers.Dense(512,activation=act_relu,use_bn=use_bn,name='fc2'))
        self.model.add(lib_snn.layers.Dense(4096,activation=act_relu,use_bn=use_bn,name='fc2'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
        self.model.add(lib_snn.layers.Dense(classes,activation=None,use_bn=use_bn,name='fc3'))

        assert False



        #return model


        #print(self.model.get_layer(name='fc3'))
        #print(self.model.get_layer(name='fc3').bias)
        #assert False

        self.model.summary()
