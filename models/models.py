
from models.vgg16_tr import VGG16_TR
#from models.vgg16 import VGG16
from models.vgg11_func import VGG11
from models.vgg16_func import VGG16
from models.resnet import ResNet18
from models.resnet import ResNet19
from models.resnet import ResNet34
from models.resnet import ResNet50
from models.resnet import ResNet101
from models.resnet import ResNet152

from models.resnet import ResNet20
from models.resnet import ResNet32
from models.resnet import ResNet44
from models.resnet import ResNet56

from models.resnet import ResNet18V2
from models.resnet import ResNet34V2
from models.resnet import ResNet50V2

from models.resnet import ResNet20V2
from models.resnet import ResNet32V2
from models.resnet import ResNet44V2
from models.resnet import ResNet56V2


from models.mobilenet_v2 import MobileNetV2

from models.efficientnet_v2 import EfficientNetV2S
from models.efficientnet_v2 import EfficientNetV2M
from models.efficientnet_v2 import EfficientNetV2L

# model selector

# models
model_sel_tr = {
    'VGG16': VGG16_TR,
}

model_sel_sc = {
    'VGG11': VGG11,
    'VGG16': VGG16,
    'ResNet18': ResNet18,
    'ResNet19': ResNet19,
    'ResNet20': ResNet20,
    'ResNet32': ResNet32,
    'ResNet34': ResNet34,
    'ResNet44': ResNet44,
    'ResNet50': ResNet50,
    'ResNet56': ResNet56,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'ResNet18V2': ResNet18V2,
    'ResNet20V2': ResNet20V2,
    'ResNet32V2': ResNet32V2,
    'ResNet34V2': ResNet34V2,
    'ResNet44V2': ResNet44V2,
    'ResNet50V2': ResNet50V2,
    'ResNet56V2': ResNet56V2,
    'MobileNetV2': MobileNetV2,
    'EfficientNetV2S': EfficientNetV2S,
    'EfficientNetV2M': EfficientNetV2M,
    'EfficientNetV2L': EfficientNetV2L,
}


def model_sel(model_name, train_type):

    if train_type == 'transfer':
        model_top = model_sel_tr[model_name]
    elif train_type == 'scratch':
        model_top = model_sel_sc[model_name]
    else:
        assert False

    return model_top