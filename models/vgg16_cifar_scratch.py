import tensorflow as tf

#
import lib_snn


#
# noinspection PyUnboundLocalVariable
#class VGG16_CIFAR(lib_snn.model.Model,tf.keras.layers.Layer):
class VGG16_CIFAR(lib_snn.model.Model):
    def __init__(self, input_shape, data_format, conf):
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

        # model loading V2
        self.load_layer_ann_checkpoint = self.load_layer_ann_checkpoint_func()

    #
    def load_layer_ann_checkpoint_func(self):
        if self.conf.f_surrogate_training_model:
            load_layer_ann_checkpoint = tf.train.Checkpoint(
                conv1=self.list_layer['conv1'],
                conv1_bn=self.list_layer['conv1_bn'],
                conv1_1=self.list_layer['conv1_1'],
                conv1_1_bn=self.list_layer['conv1_1_bn'],
                conv2=self.list_layer['conv2'],
                conv2_bn=self.list_layer['conv2_bn'],
                conv2_1=self.list_layer['conv2_1'],
                conv2_1_bn=self.list_layer['conv2_1_bn'],
                conv3=self.list_layer['conv3'],
                conv3_bn=self.list_layer['conv3_bn'],
                conv3_1=self.list_layer['conv3_1'],
                conv3_1_bn=self.list_layer['conv3_1_bn'],
                conv3_2=self.list_layer['conv3_2'],
                conv3_2_bn=self.list_layer['conv3_2_bn'],
                conv4=self.list_layer['conv4'],
                conv4_bn=self.list_layer['conv4_bn'],
                conv4_1=self.list_layer['conv4_1'],
                conv4_1_bn=self.list_layer['conv4_1_bn'],
                conv4_2=self.list_layer['conv4_2'],
                conv4_2_bn=self.list_layer['conv4_2_bn'],
                conv5=self.list_layer['conv5'],
                conv5_bn=self.list_layer['conv5_bn'],
                conv5_1=self.list_layer['conv5_1'],
                conv5_1_bn=self.list_layer['conv5_1_bn'],
                conv5_2=self.list_layer['conv5_2'],
                conv5_2_bn=self.list_layer['conv5_2_bn'],
                fc1=self.list_layer['fc1'],
                fc1_bn=self.list_layer['fc1_bn'],
                fc2=self.list_layer['fc2'],
                fc2_bn=self.list_layer['fc2_bn'],
                fc3=self.list_layer['fc3'],
                fc3_bn=self.list_layer['fc3_bn'],
                list_tk=self.list_tk
            )
        else:
            load_layer_ann_checkpoint = tf.train.Checkpoint(
                conv1=self.model.get_layer('conv1'),
                conv1_bn=self.model.get_layer('conv1').bn,
                conv1_1=self.model.get_layer('conv1_1'),
                conv1_1_bn=self.model.get_layer('conv1_1').bn,
                conv2=self.model.get_layer('conv2'),
                conv2_bn=self.model.get_layer('conv2').bn,
                conv2_1=self.model.get_layer('conv2_1'),
                conv2_1_bn=self.model.get_layer('conv2_1').bn,
                conv3=self.model.get_layer('conv3'),
                conv3_bn=self.model.get_layer('conv3').bn,
                conv3_1=self.model.get_layer('conv3_1'),
                conv3_1_bn=self.model.get_layer('conv3_1').bn,
                conv3_2=self.model.get_layer('conv3_2'),
                conv3_2_bn=self.model.get_layer('conv3_2').bn,
                conv4=self.model.get_layer('conv4'),
                conv4_bn=self.model.get_layer('conv4').bn,
                conv4_1=self.model.get_layer('conv4_1'),
                conv4_1_bn=self.model.get_layer('conv4_1').bn,
                conv4_2=self.model.get_layer('conv4_2'),
                conv4_2_bn=self.model.get_layer('conv4_2').bn,
                conv5=self.model.get_layer('conv5'),
                conv5_bn=self.model.get_layer('conv5').bn,
                conv5_1=self.model.get_layer('conv5_1'),
                conv5_1_bn=self.model.get_layer('conv5_1').bn,
                conv5_2=self.model.get_layer('conv5_2'),
                conv5_2_bn=self.model.get_layer('conv5_2').bn,
                fc1=self.model.get_layer('fc1'),
                fc1_bn=self.model.get_layer('fc1').bn,
                fc2=self.model.get_layer('fc2'),
                fc2_bn=self.model.get_layer('fc2').bn,
                fc3=self.model.get_layer('fc3'),
                fc3_bn=self.model.get_layer('fc3').bn
            )

        return load_layer_ann_checkpoint




