
import tensorflow as tf
from tensorflow import keras

from autokeras_custom.blocks.basic import ConvBlock
from autokeras_custom.blocks.basic import DenseBlock
from autokeras_custom.blocks.basic import ResNetBlock
from autokeras_custom.blocks.basic import KerasApplicationBlock

from autokeras_custom.blocks.heads import ClassificationHead

from autokeras_custom.blocks.preprocessing import ImageAugmentation
from autokeras_custom.blocks.preprocessing import Normalization

from autokeras_custom.blocks.reduction import Flatten


def serialize(obj):
    return keras.utils.serialize_keras_object(obj)


def deserialize(config, custom_objects=None):
    return keras.utils.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="hypermodels",
    )