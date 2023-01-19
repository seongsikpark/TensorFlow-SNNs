
#
import tensorflow as tf
import numpy as np
from keras import backend



#
from config import conf

#
def preprocessing_input_img(x, data_format=None, mode='max_norm'):

    if mode not in {'max_norm', 'max_norm_d', 'max_norm_d_c'}:
        raise ValueError('Expected mode to be one of `max_norm` or `mox_norm_d`h`.'
                         f'Received: mode={mode}')

    if data_format is None:
        data_format = backend.image_data_format()
    elif data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Expected data_format to be one of `channels_first` or '
                         f'`channels_last`. Received: data_format={data_format}')

    if isinstance(x, np.ndarray):
        assert False, 'only support symbolic input'

    if mode=='max_norm':
        x /= 255.0
    elif mode=='max_norm_d':
        m = tf.reduce_max(x)
        x /= m
    elif mode=='max_norm_d_c':
        if data_format == 'channels_last':
            axis = [0,1]
        else:
            assert False, 'only support - channels_last'
        m = tf.reduce_max(x,axis=axis)
        x /= m

    return x
