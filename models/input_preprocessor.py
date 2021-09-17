

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.xception import preprocess_input as Xception_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as VGG16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as VGG19_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as ResNet_preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as ResNetV2_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as InceptionV3_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as InceptionResNetV2_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as MobileNet_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as  MobileNetV2_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as DenseNet_preprocess_input
from tensorflow.keras.applications.nasnet import preprocess_input as NASNet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as EfficientNet_preprocess_input


preprocessor_input = {
    'Xception': Xception_preprocess_input,
    'VGG16': VGG16_preprocess_input,
    'VGG19': VGG19_preprocess_input,
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
}
