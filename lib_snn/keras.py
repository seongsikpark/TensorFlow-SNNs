
import types

from tensorflow.keras import utils

saving = types.SimpleNamespace()

saving.register_keras_serializable = utils.register_keras_serializable
