import tensorflow as tf

#
import lib_snn


#
# noinspection PyUnboundLocalVariable
#class VGG16_CIFAR(lib_snn.model.Model,tf.keras.layers.Layer):
class VGG16(lib_snn.model.Model):
    #def __init__(self, input_shape, data_format, conf):
    def __init__(self,
        input_shape,
        conf,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        #input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'):

        data_format = conf.data_format


        lib_snn.model.Model.__init__(self, input_shape, data_format, conf)
        #lib_snn.layers.Layer.__init__(self, input_shape, data_format, conf)
        #tf.keras.layers.Layer.__init__(self,name='')

        #
        self.kernel_size = 3

        padding = 'SAME'
        #act_relu=tf.nn.relu
        act_relu=tf.keras.layers.ReLU()


        #
        #self.dropout_conv = tf.keras.layers.Dropout(0.3)
        self.dropout_conv2 = tf.keras.layers.Dropout(0.4)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dropout_conv_r = [0.3,0.4,0.5]


        #
        self.model = tf.keras.Sequential()

        self.model.add(lib_snn.layers.InputLayer(input_shape=input_shape,batch_size=conf.batch_size,name='in'))
        self.model.add(lib_snn.layers.Conv2D(64,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[0]))
        self.model.add(lib_snn.layers.Conv2D(64,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv1_1'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),padding,name='conv1_p'))

        self.model.add(lib_snn.layers.Conv2D(128,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv2'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[0]))
        self.model.add(lib_snn.layers.Conv2D(128,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv2_1'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),padding,name='conv2_p'))

        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv3'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv3_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv3_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),padding,name='conv3_p'))

        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv4'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv4_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv4_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),padding,name='conv4_p'))

        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv5'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv5_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,activation=act_relu,use_bn=True,name='conv5_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),padding,name='conv5_p'))

        self.model.add(tf.keras.layers.Flatten(data_format=data_format))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
        self.model.add(lib_snn.layers.Dense(512,activation=act_relu,use_bn=True,name='fc1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
        self.model.add(lib_snn.layers.Dense(512,activation=act_relu,use_bn=True,name='fc2'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
        self.model.add(lib_snn.layers.Dense(self.num_class,activation=None,use_bn=True,name='fc3'))

    #
    def build(self, input_shapes):

        # build model
        lib_snn.model.Model.build(self, input_shapes)


