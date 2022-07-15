
import collections
import h5py

from .imagenet_keras.vgg16 import load_vgg16
from .imagenet_keras.resnet import load_resnet50
from .imagenet_keras.resnet import load_resnet101
from .imagenet_keras.resnet import load_resnet152
from .imagenet_keras.mobilenet import load_mobilenetv2
from .imagenet_keras.efficientnet import load_efficientnetv2s


#
source_model_path_root = '/home/sspark/Models/refs/ImageNet/Keras'

#
source_model_file_dict = collections.OrderedDict()
source_model_file_dict['VGG16'] = 'VGG16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
source_model_file_dict['ResNet50'] = 'ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
source_model_file_dict['ResNet50V2'] = 'ResNet50V2/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5'
source_model_file_dict['ResNet101'] = 'ResNet101/resnet101_weights_tf_dim_ordering_tf_kernels.h5'
source_model_file_dict['ResNet152'] = 'ResNet152/resnet152_weights_tf_dim_ordering_tf_kernels.h5'
source_model_file_dict['MobileNet'] = 'MobileNet/mobilenet_1_0_224_tf.h5'
source_model_file_dict['MobileNetV2'] = 'MobileNetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'
source_model_file_dict['EfficientNetV2S'] = 'EfficientNetV2S/efficientnetv2-s.h5'

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
    'ResNet101': load_resnet101,
    'ResNet152': load_resnet152,
    'MobileNetV2': load_mobilenetv2,
    'EfficientNetV2S': load_efficientnetv2s,
}

#
def load_weights(model_name, model, weight_file=None):
    if weight_file == None:
        weight_file = source_model_file_dict[model_name]

    print('load_weights - weight_file: {:}'.format(weight_file))
    weights = h5py.File(weight_file,'r')

    load_weights_sel[model_name](model,weights)



    #load_vgg16(model,weights)



