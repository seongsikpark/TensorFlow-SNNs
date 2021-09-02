
# This file is based on imagenet_utils.py in TensorFlow, Keras

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for ImageNet data preprocessing & prediction decoding."""

import json
import warnings

import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.util.tf_export import keras_export



def _preprocess_symbolic_input(x, data_format, mode):
  """Preprocesses a tensor encoding a batch of images.

  Args:
    x: Input tensor, 3D or 4D.
    data_format: Data format of the image tensor.
    mode: One of "caffe", "tf" or "torch".
      - caffe: will convert the images from RGB to BGR,
          then will zero-center each color channel with
          respect to the ImageNet dataset,
          without scaling.
      - tf: will scale pixels between -1 and 1,
          sample-wise.
      - torch: will scale pixels between 0 and 1 and then
          will normalize each channel with respect to the
          ImageNet dataset.

  Returns:
      Preprocessed tensor.
  """
  if mode == 'tf':
    x /= 127.5
    x -= 1.
    return x
  elif mode == 'torch':
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  else:
    if data_format == 'channels_first':
      # 'RGB'->'BGR'
      if backend.ndim(x) == 3:
        x = x[::-1, ...]
      else:
        x = x[:, ::-1, ...]
    else:
      # 'RGB'->'BGR'
      x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None

  mean_tensor = backend.constant(-np.array(mean))

  # Zero-center by mean pixel
  if backend.dtype(x) != backend.dtype(mean_tensor):
    x = backend.bias_add(
        x, backend.cast(mean_tensor, backend.dtype(x)), data_format=data_format)
  else:
    x = backend.bias_add(x, mean_tensor, data_format)
  if std is not None:
    std_tensor = backend.constant(np.array(std))
    if data_format == 'channels_first':
      std_tensor = backend.reshape(std_tensor, (-1, 1, 1))
    x /= std_tensor
  return x