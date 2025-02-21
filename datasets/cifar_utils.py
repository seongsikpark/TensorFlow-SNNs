
# based on tf.keras.applications.imagenet_utils.py


import json
#import warnings

import numpy as np

#from keras import activations
from keras import backend
from keras.utils import data_utils

# isort: off
#from tensorflow.python.util.tf_export import keras_export


CLASS_INDEX = None
CLASS_INDEX_PATH = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "data/imagenet_class_index.json"
)


PREPROCESS_INPUT_DOC = """
Preprocesses a tensor or Numpy array encoding a batch of images.

Usage example with `applications.MobileNet`:

```python
i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
x = tf.cast(i, tf.float32)
x = tf.keras.applications.mobilenet.preprocess_input(x)
core = tf.keras.applications.MobileNet()
x = core(x)
model = tf.keras.Model(inputs=[i], outputs=[x])

image = tf.image.decode_png(tf.io.read_file('file.png'))
result = model(image)
```

Args:
x: A floating point `numpy.array` or a `tf.Tensor`, 3D or 4D with 3 color
channels, with values in the range [0, 255].
The preprocessed data are written over the input data
if the data types are compatible. To avoid this
behaviour, `numpy.copy(x)` can be used.
data_format: Optional data format of the image tensor/array. Defaults to
None, in which case the global setting
`tf.keras.backend.image_data_format()` is used (unless you changed it,
                                                                   it defaults to "channels_last").{mode}

Returns:
Preprocessed `numpy.array` or a `tf.Tensor` with type `float32`.
{ret}

Raises:
{error}
"""

PREPROCESS_INPUT_MODE_DOC = """
mode: One of "caffe", "tf" or "torch". Defaults to "caffe".
- caffe: will convert the images from RGB to BGR,
then will zero-center each color channel with
    respect to the ImageNet dataset,
without scaling.
- tf: will scale pixels between -1 and 1,
sample-wise.
- torch: will scale pixels between 0 and 1 and then
will normalize each channel with respect to the
ImageNet dataset.
"""

PREPROCESS_INPUT_DEFAULT_ERROR_DOC = """
    ValueError: In case of unknown `mode` or `data_format` argument."""


#def preprocess_input(x, data_format=None, mode="caffe"):
def preprocess_input(x, data_format=None, mode="torch"):
    """Preprocesses a tensor or Numpy array encoding a batch of images."""
    if mode not in {"caffe", "tf", "torch"}:
        raise ValueError(
            "Expected mode to be one of `caffe`, `tf` or `torch`. "
            f"Received: mode={mode}"
        )

    if data_format is None:
        data_format = backend.image_data_format()
    elif data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "Expected data_format to be one of `channels_first` or "
            f"`channels_last`. Received: data_format={data_format}"
        )

    if isinstance(x, np.ndarray):
        return _preprocess_numpy_input(x, data_format=data_format, mode=mode)
    else:
        return _preprocess_symbolic_input(x, data_format=data_format, mode=mode)


preprocess_input.__doc__ = PREPROCESS_INPUT_DOC.format(
    mode=PREPROCESS_INPUT_MODE_DOC,
    ret="",
    error=PREPROCESS_INPUT_DEFAULT_ERROR_DOC,
)


def decode_predictions(preds, top=5):
    """Decodes the prediction of an ImageNet model.

    Args:
      preds: Numpy array encoding a batch of predictions.
      top: Integer, how many top-guesses to return. Defaults to 5.

    Returns:
      A list of lists of top class prediction tuples
      `(class_name, class_description, score)`.
      One list of tuples per sample in batch input.

    Raises:
      ValueError: In case of invalid shape of the `pred` array
        (must be 2D).
    """
    global CLASS_INDEX

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError(
            "`decode_predictions` expects "
            "a batch of predictions "
            "(i.e. a 2D array of shape (samples, 1000)). "
            "Found array with shape: " + str(preds.shape)
        )
    if CLASS_INDEX is None:
        fpath = data_utils.get_file(
            "imagenet_class_index.json",
            CLASS_INDEX_PATH,
            cache_subdir="models",
            file_hash="c2c37ea517e94d9795004a39431a14cb",
        )
        with open(fpath) as f:
            CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def _preprocess_numpy_input(x, data_format, mode):
    """Preprocesses a Numpy array encoding a batch of images.

    Args:
      x: Input array, 3D or 4D.
      data_format: Data format of the image array.
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
        Preprocessed Numpy array.
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(backend.floatx(), copy=False)

    if mode == "tf":
        x /= 127.5
        x -= 1.0
        return x
    elif mode == "torch":
        x /= 255.0
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:
        if data_format == "channels_first":
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        assert False
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == "channels_first":
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x


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
    if mode == "tf":
        x /= 127.5
        x -= 1.0
        return x
    elif mode == "torch":
        x /= 255.0
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:
        if data_format == "channels_first":
            # 'RGB'->'BGR'
            if backend.ndim(x) == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        assert False
        mean = [103.939, 116.779, 123.68]
        std = None

    mean_tensor = backend.constant(-np.array(mean))

    # Zero-center by mean pixel
    if backend.dtype(x) != backend.dtype(mean_tensor):
        x = backend.bias_add(
            x,
            backend.cast(mean_tensor, backend.dtype(x)),
            data_format=data_format,
        )
    else:
        x = backend.bias_add(x, mean_tensor, data_format)
    if std is not None:
        std_tensor = backend.constant(np.array(std), dtype=backend.dtype(x))
        if data_format == "channels_first":
            std_tensor = backend.reshape(std_tensor, (-1, 1, 1))
        x /= std_tensor
    return x
