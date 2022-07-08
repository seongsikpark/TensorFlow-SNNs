
import tensorflow as tf


Xception = tf.keras.applications.Xception
VGG16 = tf.keras.applications.VGG16
VGG19 = tf.keras.applications.VGG19
ResNet50 = tf.keras.applications.ResNet50
ResNet101 = tf.keras.applications.ResNet101
ResNet152 = tf.keras.applications.ResNet152
ResNet50V2 = tf.keras.applications.ResNet50V2
ResNet101V2 = tf.keras.applications.ResNet101V2
ResNet152V2 = tf.keras.applications.ResNet152V2
InceptionV3 = tf.keras.applications.InceptionV3
InceptionResNetV2 = tf.keras.applications.InceptionResNetV2
MobileNet = tf.keras.applications.MobileNet
MobileNetV2 = tf.keras.applications.MobileNetV2
MobileNetV3Small = tf.keras.applications.MobileNetV3Small
MobileNetV3Large = tf.keras.applications.MobileNetV3Large
DenseNet121 = tf.keras.applications.DenseNet121
DenseNet169 = tf.keras.applications.DenseNet169
DenseNet201 = tf.keras.applications.DenseNet201
NASNetMobile = tf.keras.applications.NASNetMobile
NASNetLarge = tf.keras.applications.NASNetLarge
EfficientNetB0 = tf.keras.applications.EfficientNetB0
EfficientNetB1 = tf.keras.applications.EfficientNetB1
EfficientNetB2 = tf.keras.applications.EfficientNetB2
EfficientNetB3 = tf.keras.applications.EfficientNetB3
EfficientNetB4 = tf.keras.applications.EfficientNetB4
EfficientNetB5 = tf.keras.applications.EfficientNetB5
EfficientNetB6 = tf.keras.applications.EfficientNetB6
EfficientNetB7 = tf.keras.applications.EfficientNetB7
EfficientNetV2S = tf.keras.applications.EfficientNetV2S
EfficientNetV2M = tf.keras.applications.EfficientNetV2M
EfficientNetV2L = tf.keras.applications.EfficientNetV2L



#from tensorflow.keras.applications.xception import Xception
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.applications.resnet import ResNet50
#from tensorflow.keras.applications.resnet import ResNet101
#from tensorflow.keras.applications.resnet import ResNet152
#from tensorflow.keras.applications.resnet_v2 import ResNet50V2
#from tensorflow.keras.applications.resnet_v2 import ResNet101V2
#from tensorflow.keras.applications.resnet_v2 import ResNet152V2
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
#from tensorflow.keras.applications.mobilenet import MobileNet
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#from tensorflow.keras.applications.densenet import DenseNet121
#from tensorflow.keras.applications.densenet import DenseNet169
#from tensorflow.keras.applications.densenet import DenseNet201
#from tensorflow.keras.applications.nasnet import NASNetMobile
#from tensorflow.keras.applications.nasnet import NASNetLarge
#from tensorflow.keras.applications.efficientnet import EfficientNetB0
#from tensorflow.keras.applications.efficientnet import EfficientNetB1
#from tensorflow.keras.applications.efficientnet import EfficientNetB2
#from tensorflow.keras.applications.efficientnet import EfficientNetB3
#from tensorflow.keras.applications.efficientnet import EfficientNetB4
#from tensorflow.keras.applications.efficientnet import EfficientNetB5
#from tensorflow.keras.applications.efficientnet import EfficientNetB6
#from tensorflow.keras.applications.efficientnet import EfficientNetB7
#from tensorflow.keras.applications import EfficientNetV2S






# models
models = {
    'Xception': Xception,
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'ResNet50V2': ResNet50V2,
    'ResNet101V2': ResNet101V2,
    'ResNet152V2': ResNet152V2,
    'InceptionV3': InceptionV3,
    'InceptionResNetV2': InceptionResNetV2,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'DenseNet201': DenseNet201,
    'NASNetMobile': NASNetMobile,
    'NASNetLarge': NASNetLarge,
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
    'EfficientNetB2': EfficientNetB2,
    'EfficientNetB3': EfficientNetB3,
    'EfficientNetB4': EfficientNetB4,
    'EfficientNetB5': EfficientNetB5,
    'EfficientNetB6': EfficientNetB6,
    'EfficientNetB7': EfficientNetB7,
}
