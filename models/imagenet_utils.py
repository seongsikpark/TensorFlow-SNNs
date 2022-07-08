
import h5py

from .imagenet_keras.vgg16 import load_vgg16
from .imagenet_keras.resnet import load_resnet50
from .imagenet_keras.mobilenet import load_mobilenetv2

################
#
################
load_weights_sel = {
    'VGG16': load_vgg16,
    #'ResNet20': load_resnet20,
    #'ResNet32': load_resnet32,
    #'ResNet44': load_resnet,
    #'ResNet56': load_resnet,
    'ResNet50': load_resnet50,
    'MobileNetV2': load_mobilenetv2,
}

#
def load_weights(model_name, model, weight_file):
    print('load_weights - weight_file: {:}'.format(weight_file))
    weights = h5py.File(weight_file,'r')

    load_weights_sel[model_name](model,weights)



    #load_vgg16(model,weights)



