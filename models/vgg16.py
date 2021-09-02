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
        #weights='imagenet',
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'):
        #train=False,
        #add_top=False):

        data_format = conf.data_format


        lib_snn.model.Model.__init__(self, input_shape, data_format, classes, conf)
        #lib_snn.layers.Layer.__init__(self, input_shape, data_format, conf)
        #tf.keras.layers.Layer.__init__(self,name='')

        #
        self.kernel_size = 3

        padding = 'SAME'
        #act_relu=tf.nn.relu
        act_relu=tf.keras.layers.ReLU()
        act_sm = tf.keras.layers.Softmax()


        #
        #self.dropout_conv = tf.keras.layers.Dropout(0.3)
        self.dropout_conv2 = tf.keras.layers.Dropout(0.4)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dropout_conv_r = [0.3,0.4,0.5]


        #
        self.model = tf.keras.Sequential()

        use_bn_feat = False
        use_bn_clas = True

        #if train:
            #self.model.add(tf.keras.layers.experimental.preprocessing.RandomZoom((-0.1, 0.1)))
            #self.model.add(tf.keras.layers.experimental.preprocessing.RandomRotation((-0.03, 0.03)))

        self.model.add(lib_snn.layers.InputLayer(input_shape=input_shape,batch_size=conf.batch_size,name='in'))
        #self.model.add(lib_snn.layers.InputLayer(input_shape=input_shape,name='in'))
        self.model.add(lib_snn.layers.Conv2D(64,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[0]))
        self.model.add(lib_snn.layers.Conv2D(64,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv1_1'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),name='conv1_p'))

        self.model.add(lib_snn.layers.Conv2D(128,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv2'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[0]))
        self.model.add(lib_snn.layers.Conv2D(128,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv2_1'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),name='conv2_p'))

        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv3'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv3_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(256,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv3_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),name='conv3_p'))

        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv4'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv4_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv4_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),name='conv4_p'))

        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv5'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv5_1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[1]))
        self.model.add(lib_snn.layers.Conv2D(512,self.kernel_size,padding=padding,
                                             activation=act_relu,use_bn=use_bn_feat,name='conv5_2'))
        self.model.add(lib_snn.layers.MaxPool2D((2,2),(2,2),name='conv5_p'))

        self.model.add(tf.keras.layers.Flatten(data_format=data_format))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
        self.model.add(lib_snn.layers.Dense(4096,activation=act_relu,use_bn=use_bn_clas,name='fc1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
        self.model.add(lib_snn.layers.Dense(4096,activation=act_relu,use_bn=use_bn_clas,name='fc2'))
        self.model.add(tf.keras.layers.Dropout(self.dropout_conv_r[2]))
        self.model.add(lib_snn.layers.Dense(classes,activation=act_sm,use_bn=False,name='predictions'))

        #self.model.load_weights(weights)

        self.model.summary()

#    #
#    def build(self, input_shapes):
#
#        # build model
#        lib_snn.model.Model.build(self, input_shapes)


