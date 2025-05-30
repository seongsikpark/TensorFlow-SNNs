# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import keras_cv
import tensorflow as tf

from keras_cv import core
from keras_cv.layers import preprocessing as cv_preprocessing
from keras_cv.layers.preprocessing.random_augmentation_pipeline import (
    RandomAugmentationPipeline,
)
from keras_cv.utils import preprocessing as preprocessing_utils

import keras_cv_local

import random
from keras_cv.utils import preprocessing
from keras import backend



_random_generator = backend.RandomGenerator()
    #seed, force_generator=force_generator, rng_type=rng_type)

#@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandAugment(RandomAugmentationPipeline):
    """RandAugment performs the Rand Augment operation on input images.

    This layer can be thought of as an all in one image augmentation layer.  The policy
    implemented by this layer has been benchmarked extensively and is effective on a
    wide variety of datasets.

    The policy operates as follows:

    For each augmentation in the range `[0, augmentations_per_image]`,
    the policy selects a random operation from a list of operations.
    It then samples a random number and if that number is less than
    `rate` applies it to the given image.

    References:
        - [RandAugment](https://arxiv.org/abs/1909.13719)

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
        augmentations_per_image: the number of layers to use in the rand augment policy.
            Defaults to `3`.
        magnitude: magnitude is the mean of the normal distribution used to sample the
            magnitude used for each data augmentation.  magnitude should
            be a float in the range `[0, 1]`.  A magnitude of `0` indicates that the
            augmentations are as weak as possible (not recommended), while a value of
            `1.0` implies use of the strongest possible augmentation.  All magnitudes
            are clipped to the range `[0, 1]` after sampling.  Defaults to `0.5`.
        magnitude_stddev: the standard deviation to use when drawing values
            for the perturbations.  Keep in mind magnitude will still be clipped to the
            range `[0, 1]` after samples are drawn from the normal distribution.
            Defaults to `0.15`.
        rate:  the rate at which to apply each augmentation.  This parameter is applied
            on a per-distortion layer, per image.  Should be in the range `[0, 1]`.
            To reproduce the original RandAugment paper results, set this to `10/11`.
            The original `RandAugment` paper includes an Identity transform.  By setting
            the rate to 10/11 in our implementation, the behavior is identical to
            sampling an Identity augmentation 10/11th of the time.
            Defaults to `1.0`.
        geometric: whether or not to include geometric augmentations.  This should be
            set to False when performing object detection.  Defaults to True.
    Usage:
    ```python
    (x_test, y_test), _ = tf.keras.datasets.cifar10.load_data()
    rand_augment = keras_cv.layers.RandAugment(
        value_range=(0, 255), augmentations_per_image=3, magnitude=0.5
    )
    x_test = rand_augment(x_test)
    ```
    """

    def __init__(
        self,
        value_range,
        augmentations_per_image=3,
        magnitude=0.5,
        magnitude_stddev=0.15,
        rate=10 / 11,
        geometric=True,
        seed=None,
        **kwargs,
    ):
        # As an optimization RandAugment makes all internal layers use (0, 255) while
        # and we handle range transformation at the _augment level.
        if magnitude < 0.0 or magnitude > 1:
            raise ValueError(
                f"`magnitude` must be in the range [0, 1], got `magnitude={magnitude}`"
            )
        if magnitude_stddev < 0.0 or magnitude_stddev > 1:
            raise ValueError(
                "`magnitude_stddev` must be in the range [0, 1], got "
                f"`magnitude_stddev={magnitude}`"
            )

        super().__init__(
            layers=RandAugment.get_standard_policy(
                (0, 255), magnitude, magnitude_stddev, geometric=geometric, seed=seed
            ),
            augmentations_per_image=augmentations_per_image,
            rate=rate,
            **kwargs,
            seed=seed,
        )
        self.magnitude = float(magnitude)
        self.value_range = value_range
        self.seed = seed
        self.geometric = geometric
        self.magnitude_stddev = float(magnitude_stddev)

        #


    def _augment(self, sample):
        sample["images"] = preprocessing_utils.transform_value_range(
            sample["images"], self.value_range, (0, 255)
        )
        result = super()._augment(sample)
        result["images"] = preprocessing_utils.transform_value_range(
            result["images"], (0, 255), self.value_range
        )
        result["images"]
        return result

    @staticmethod
    def get_standard_policy(
        value_range, magnitude, magnitude_stddev, geometric=True, seed=None
    ):
        policy = create_rand_augment_policy(magnitude, magnitude_stddev)

        auto_contrast = cv_preprocessing.AutoContrast(
            **policy["auto_contrast"], value_range=value_range, seed=seed
        )
        equalize = cv_preprocessing.Equalization(
            **policy["equalize"], value_range=value_range, seed=seed
        )

        solarize = cv_preprocessing.Solarization(
            **policy["solarize"], value_range=value_range, seed=seed
        )

        color = cv_preprocessing.RandomColorDegeneration(**policy["color"], seed=seed)
        contrast = cv_preprocessing.RandomContrast(**policy["contrast"], seed=seed)
        brightness = cv_preprocessing.RandomBrightness(
            **policy["brightness"], value_range=value_range, seed=seed
        )
        # sspark
        rotation = cv_preprocessing.RandomRotation(**policy["rotation"], seed=seed)
        sharpness = cv_preprocessing.RandomSharpness(**policy["sharpness"], value_range=value_range, seed=seed)
        posterization = cv_preprocessing.Posterization(value_range=value_range, **policy["posterization"], seed=seed)
        invert = keras_cv_local.layers.preprocessing.Invert(**policy["invert"])


        layers = [
            auto_contrast,
            equalize,
            solarize,
            color,
            contrast,
            brightness,
            rotation,
            sharpness,
            posterization,
            invert,
        ]

        if geometric:
            shear_x = cv_preprocessing.RandomShear(**policy["shear_x"], seed=seed)
            shear_y = cv_preprocessing.RandomShear(**policy["shear_y"], seed=seed)
            translate_x = cv_preprocessing.RandomTranslation(
                **policy["translate_x"], seed=seed
            )
            translate_y = cv_preprocessing.RandomTranslation(
                **policy["translate_y"], seed=seed
            )
            layers += [shear_x, shear_y, translate_x, translate_y]
        return layers

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "augmentations_per_image": self.augmentations_per_image,
                "magnitude": self.magnitude,
                "magnitude_stddev": self.magnitude_stddev,
                "rate": self.rate,
                "geometric": self.geometric,
                "seed": self.seed,
            }
        )
        # layers is recreated in the constructor
        del config["layers"]
        return config



# from timm
#def _enhance_level(magnitude, magnitude_stddev):
    ## range [0.1, 1.9]
    #return magnitude

def _randomly_negate(magnitude):
    # 50% - negate
    if random.uniform(0, 1) < 0.5:
        tf.multiply(magnitude, -1.0)
    return magnitude


def auto_contrast_policy(magnitude, magnitude_stddev):
    return {}


def equalize_policy(magnitude, magnitude_stddev):
    return {}


def solarize_policy(magnitude, magnitude_stddev):
    # We cap additions at 110, because if we add more than 110 we will be nearly
    # nullifying the information contained in the image, making the model train on noise
    maximum_addition_value = 110
    addition_factor = core.NormalFactorSampler(
        mean=magnitude * maximum_addition_value,
        stddev=magnitude_stddev * maximum_addition_value,
        min_value=0,
        max_value=maximum_addition_value,
    )
    threshold_factor = core.NormalFactorSampler(
        mean=(255 - (magnitude * 255)),
        stddev=(magnitude_stddev * 255),
        min_value=0,
        max_value=255,
    )
    return {"addition_factor": addition_factor, "threshold_factor": threshold_factor}


def color_policy(magnitude, magnitude_stddev):
    factor = core.NormalFactorSampler(
        mean=magnitude,
        stddev=magnitude_stddev,
        min_value=0,
        max_value=1,
    )
    return {"factor": factor}


def contrast_policy(magnitude, magnitude_stddev):
    # TODO(lukewood): should we integrate RandomContrast with `factor`?
    # RandomContrast layer errors when factor=0
    value_range = [0.0,1.0]
    factor = max(magnitude, 0.001)
    return {"value_range": value_range, "factor": factor}


def brightness_policy(magnitude, magnitude_stddev):
    # TODO(lukewood): should we integrate RandomBrightness with `factor`?
    return {"factor": magnitude}


def shear_x_policy(magnitude, magnitude_stddev):
    #factor = magnitude*0.3
    #factor *= preprocessing.random_inversion(_random_generator)
    #factor = (-factor, factor)

    scale = 0.3

    factor = core.NormalFactorSampler(
        mean=magnitude*scale,
        stddev=magnitude_stddev*scale,
        min_value=0,
        max_value=scale,
    )

    return {"x_factor": factor, "y_factor": 0}


def shear_y_policy(magnitude, magnitude_stddev):
    #factor = magnitude*0.3
    #factor = (-factor, factor)

    scale = 0.3

    factor = core.NormalFactorSampler(
        mean=magnitude*scale,
        stddev=magnitude_stddev*scale,
        min_value=0,
        max_value=scale,
    )

    return {"x_factor": 0, "y_factor": factor}


def translate_x_policy(magnitude, magnitude_stddev):
    # TODO(lukewood): should we integrate RandomTranslation with `factor`?
    #factor = magnitude*0.45
    #factor = (-factor, factor)

    scale = 0.45
    factor = _random_generator.random_normal([1,],magnitude,magnitude_stddev)
    factor *= scale

    return {"width_factor": factor, "height_factor": 0}


def translate_y_policy(magnitude, magnitude_stddev):
    # TODO(lukewood): should we integrate RandomTranslation with `factor`?
    #factor = magnitude*0.45
    #factor = (-factor, factor)
    #scale = 0.45

    scale = 0.45
    factor = _random_generator.random_normal([1,],magnitude,magnitude_stddev)
    factor *= scale


    return {"width_factor": 0, "height_factor": factor}


def rotation_policy(magnitude, magnitude_stddev):
    # TODO: random uniform -> gauss ?
    magnitude = 1/12*magnitude  # 1/12*2*pi = 30 -> -30 ~ 30

    return {"factor": magnitude}


def sharpness_policy(magnitude, magnitude_stddev):
    #factor = magnitude
    #factor = _randomly_negate(factor)

    factor = core.NormalFactorSampler(
        mean=magnitude,
        stddev=magnitude_stddev,
        min_value=0,
        max_value=1,
    )

    return {"factor": factor}





def posterization_policy(magnitude, magnitude_stddev):
    #magnitude = max(2, int(8*magnitude))
    #magnitude = max(1, int(4*magnitude))

    factor = tf.clip_by_value(
            tf.random.normal(
                shape=(1,),
                mean=magnitude,
                stddev=magnitude_stddev,
                #seed=self.seed,
                #dtype=dtype,
            ),
            0,
            1,
        )

    factor = max(1, int(4*factor))


    return {"bits": factor}

def invert_policy(magnitude, magnitude_stddev):
    return {"factor": magnitude}


POLICY_PAIRS = {
    "auto_contrast": auto_contrast_policy,
    "equalize": equalize_policy,
    "solarize": solarize_policy,
    "color": color_policy,
    "contrast": contrast_policy,
    "brightness": brightness_policy,
    "shear_x": shear_x_policy,
    "shear_y": shear_y_policy,
    "translate_x": translate_x_policy,
    "translate_y": translate_y_policy,
    "rotation": rotation_policy,
    "sharpness": sharpness_policy,
    "posterization": posterization_policy,
    "invert": invert_policy,
}



def create_rand_augment_policy(magnitude, magnitude_stddev):
    result = {}
    for name, policy_fn in POLICY_PAIRS.items():
        result[name] = policy_fn(magnitude, magnitude_stddev)
    return result
