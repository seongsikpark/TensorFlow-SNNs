from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils


# from tensorflow/.../convolutional.py
def cal_output_shape_Conv2D(data_format,input_shape,filters,kernel_size,strides):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    kernel_size = utils.normalize_tuple(kernel_size,2,'kernel_size')
    strides = utils.normalize_tuple(strides,2,'strides')
    dilation_rate = utils.normalize_tuple(1,2,'dilation_rate')

    if data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                #padding='valid',
                padding='same',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space + [filters])
    else:
        space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                #padding='valid',
                padding='same',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0], filters] + new_space)

def cal_output_shape_Conv2D_pad_val(data_format,input_shape,filters,kernel_size,strides):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    kernel_size = utils.normalize_tuple(kernel_size,2,'kernel_size')
    strides = utils.normalize_tuple(strides,2,'strides')
    dilation_rate = utils.normalize_tuple(1,2,'dilation_rate')

    if data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                padding='valid',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space + [filters])
    else:
        space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                padding='valid',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0], filters] + new_space)




#
def cal_output_shape_Pooling2D(data_format,input_shape,pool_size,strides):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()

    pool_size = utils.normalize_tuple(pool_size, 2, 'pool_size')
    strides = utils.normalize_tuple(strides, 2, 'strides')
    padding = 'same'

    if data_format == 'channels_first':
        rows = input_shape[2]
        cols = input_shape[3]
    else:
        rows = input_shape[1]
        cols = input_shape[2]

    rows = utils.conv_output_length(
            rows,
            pool_size[0],
            padding,
            strides[0]
        )

    cols = utils.conv_output_length(
            cols,
            pool_size[1],
            padding,
            strides[1]
        )

    if data_format == 'channels_first':
        return tensor_shape.TensorShape([input_shape[0], input_shape[1], rows, cols])
    else:
        return tensor_shape.TensorShape([input_shape[0], rows, cols, input_shape[3]])



