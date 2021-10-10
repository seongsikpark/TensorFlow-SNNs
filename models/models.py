
from models.vgg16_tr import VGG16_TR
from models.vgg16 import VGG16
from models.resnet import ResNet18
from models.resnet import ResNet20
from models.resnet import ResNet32
from models.resnet import ResNet34
from models.resnet import ResNet50
from models.resnet import ResNet101
from models.resnet import ResNet152
from models.resnet import ResNet18V2
from models.resnet import ResNet20V2


# model selector

# models
model_sel_tr = {
    'VGG16': VGG16_TR,
}

model_sel_sc = {
    'VGG16': VGG16,
    'ResNet18': ResNet18,
    'ResNet20': ResNet20,
    'ResNet32': ResNet32,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'ResNet18V2': ResNet18V2,
    'ResNet20V2': ResNet20V2,
}


def model_sel(model_name, train_type):

    if train_type == 'transfer':
        model_top = model_sel_tr[model_name]
    elif train_type == 'scratch':
        model_top = model_sel_sc[model_name]
    else:
        assert False

    return model_top