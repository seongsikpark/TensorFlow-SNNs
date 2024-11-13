
import tensorflow as tf


Xception_preprocess_input = tf.keras.applications.xception.preprocess_input
VGG16_preprocess_input = tf.keras.applications.vgg16.preprocess_input
VGG19_preprocess_input = tf.keras.applications.vgg19.preprocess_input
ResNet_preprocess_input = tf.keras.applications.resnet.preprocess_input
ResNetV2_preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
InceptionV3_preprocess_input = tf.keras.applications.inception_v3.preprocess_input
InceptionResNetV2_preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
MobileNet_preprocess_input = tf.keras.applications.mobilenet.preprocess_input
MobileNetV2_preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
DenseNet_preprocess_input = tf.keras.applications.densenet.preprocess_input
NASNet_preprocess_input = tf.keras.applications.nasnet.preprocess_input
EfficientNet_preprocess_input = tf.keras.applications.efficientnet.preprocess_input
EfficientNetV2_preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input





#image = tf.keras.preprocessing
#from tensorflow.keras.preprocessing import image


preprocessor_input_imagenet = {
    'Xception': Xception_preprocess_input,
    'VGG16': VGG16_preprocess_input,
    'VGG19': VGG19_preprocess_input,
    'ResNet18': ResNet_preprocess_input,
    'ResNet20': ResNet_preprocess_input,
    'ResNet32': ResNet_preprocess_input,
    'ResNet34': ResNet_preprocess_input,
    'ResNet50': ResNet_preprocess_input,
    'ResNet101': ResNet_preprocess_input,
    'ResNet152': ResNet_preprocess_input,
    'ResNet50V2': ResNetV2_preprocess_input,
    'ResNet101V2': ResNetV2_preprocess_input,
    'ResNet152V2': ResNetV2_preprocess_input,
    'InceptionV3': InceptionV3_preprocess_input,
    'InceptionResNetV2': InceptionResNetV2_preprocess_input,
    'MobileNet': MobileNet_preprocess_input,
    'MobileNetV2': MobileNetV2_preprocess_input,
    'DenseNet121': DenseNet_preprocess_input,
    'DenseNet169': DenseNet_preprocess_input,
    'DenseNet201': DenseNet_preprocess_input,
    'NASNetMobile': NASNet_preprocess_input,
    'NASNetLarge': NASNet_preprocess_input,
    'EfficientNetB0': EfficientNet_preprocess_input,
    'EfficientNetB1': EfficientNet_preprocess_input,
    'EfficientNetB2': EfficientNet_preprocess_input,
    'EfficientNetB3': EfficientNet_preprocess_input,
    'EfficientNetB4': EfficientNet_preprocess_input,
    'EfficientNetB5': EfficientNet_preprocess_input,
    'EfficientNetB6': EfficientNet_preprocess_input,
    'EfficientNetB7': EfficientNet_preprocess_input,
    'EfficientNetV2S': EfficientNetV2_preprocess_input,
    'EfficientNetV2M': EfficientNetV2_preprocess_input,
    'EfficientNetV2L': EfficientNetV2_preprocess_input,
}
