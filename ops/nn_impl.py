# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Implementation of Neural Net (NN) functions."""

import math

from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
# from tensorflow.python.framework import dtypes
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import candidate_sampling_ops
# from tensorflow.python.ops import check_ops
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import custom_gradient
# from tensorflow.python.ops import embedding_ops
# from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
# from tensorflow.python.ops import gen_nn_ops
#from lib_snn.ops import gen_nn_ops
# from tensorflow.python.ops import gen_sparse_ops
# from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import nn_ops
# from tensorflow.python.ops import variables
# from tensorflow.python.ops.losses import util as losses_util
# from tensorflow.python.platform import device_context
from tensorflow.python.util import dispatch
# from tensorflow.python.util.deprecation import deprecated_args
# from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

import tensorflow as tf

import matplotlib.pyplot as plt

import math

#import lib_snn
#from lib_snn.sim import glb_plot_gradient_bn

#from config import conf
#from config_common import conf
from absl import flags
conf = flags.FLAGS


# @tf_export("nn.batch_normalization")
@tf.custom_gradient
@dispatch.add_dispatch_support
def batch_normalization(x,
                        mean,
                        variance,
                        offset,
                        scale,
                        variance_epsilon,
                        name):
    r"""Batch normalization.

    Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
    `scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):

    \\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)

    `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
    shapes:

      * In all generality, they can have the same number of dimensions as the
        input `x`, with identical sizes as `x` for the dimensions that are not
        normalized over (the 'depth' dimension(s)), and dimension 1 for the
        others which are being normalized over.
        `mean` and `variance` in this case would typically be the outputs of
        `tf.nn.moments(..., keepdims=True)` during training, or running averages
        thereof during inference.
      * In the common case where the 'depth' dimension is the last dimension in
        the input tensor `x`, they may be one dimensional tensors of the same
        size as the 'depth' dimension.
        This is the case for example for the common `[batch, depth]` layout of
        fully-connected layers, and `[batch, height, width, depth]` for
        convolutions.
        `mean` and `variance` in this case would typically be the outputs of
        `tf.nn.moments(..., keepdims=False)` during training, or running averages
        thereof during inference.

    See equation 11 in Algorithm 2 of source:
    [Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift; S. Ioffe, C. Szegedy]
    (http://arxiv.org/abs/1502.03167).

    Args:
      x: Input `Tensor` of arbitrary dimensionality.
      mean: A mean `Tensor`.
      variance: A variance `Tensor`.
      offset: An offset `Tensor`, often denoted \\(\beta\\) in equations, or
        None. If present, will be added to the normalized tensor.
      scale: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
        `None`. If present, the scale is applied to the normalized tensor.
      variance_epsilon: A small float number to avoid dividing by 0.
      name: A name for this operation (optional).

    Returns:
      the normalized, scaled, offset tensor.

    References:
      Batch Normalization - Accelerating Deep Network Training by Reducing
      Internal Covariate Shift:
        [Ioffe et al., 2015](http://arxiv.org/abs/1502.03167)
        ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
    """

    # func input: x, mean, variance, scale, offset, variance_epsilon
    def grad(upstream):

        #
        mm = mean
        mv = variance
        epsilon = variance_epsilon
        y_backprop = upstream


        #if len(x.shape.as_list())==4:
        if x.shape.rank==4:
            axis=[0,1,2]
        else:
            axis=[0]
        #axis=[0]

        #inv = math_ops.rsqrt(variance + variance_epsilon)
        inv = tf.math.rsqrt(variance + variance_epsilon)
        #inv = 1/(variance + variance_epsilon)
        #inv = tf.math.sqrt(variance + variance_epsilon)

        # gradient - dx
        #dx = scale * tf.math.sqrt(mv*mv+epsilon)*(y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis) / (1/mv*1/mv+epsilon))
        #dx = scale * tf.math.rsqrt(1/mv+epsilon)*(y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis) / (mv+epsilon))
        #dx = scale * tf.math.rsqrt(mv+epsilon)*(y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis) / (mv+epsilon))

        dx = scale * inv * (y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis)/(mv+epsilon))
        #dx = scale * tf.math.rsqrt(mv+epsilon) * (y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis)/(mv+epsilon))

        #dx = tf.multiply(dx,2*(x+epsilon))
        #dx = tf.multiply(dx,0.5*tf.math.rsqrt(x+epsilon))

        # gradient - scale
        # moving variance = 1/reserve_space_2
        # due to numerical precision? rsqrt(mv*mv+eps)->sqrt(1/mv*1/mv+eps)
        #dscale = tf.reduce_sum(y_backprop*(x-mm)*tf.math.sqrt(mv*mv+epsilon),axis=axis)
        #dscale = tf.reduce_sum(y_backprop*(x-mm)*tf.math.rsqrt(1/mv+epsilon),axis=axis)
        #dscale = tf.reduce_sum(y_backprop*(x-mm)*tf.math.rsqrt(mv+epsilon),axis=axis)
        dscale = tf.reduce_sum(y_backprop*(x-mm)*inv,axis=axis)
        #dscale = tf.reduce_sum(y_backprop*(x-mm)*tf.math.rsqrt(mv+epsilon),axis=axis)


        # gradient - offset
        doffset = tf.reduce_sum(y_backprop, axis=axis)

        #
        dstop=tf.stop_gradient

        #if False:
        #if True:
        if conf.verbose_snn_train:
            print(name)

            var = x
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}, non_zero {:.3g}'
                  .format('x',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var),tf.math.count_nonzero(var,dtype=tf.int32)/tf.math.reduce_prod(var.shape)))

            var = y_backprop
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('y_backprop',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = dx
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('dx',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            if conf.verbose_visual:
                import lib_snn
                from lib_snn.sim import glb_plot_gradient_bn
                lib_snn.util.plot_hist(glb_plot_gradient_bn,dx,1000,norm_fit=True)

            var = scale
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('scale',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = inv
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('inv',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = (y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis)/(mv+epsilon))
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('last',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = (y_backprop-tf.reduce_mean(y_backprop,axis=axis))
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('last_1',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = ((x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis)/(mv+epsilon))
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('last_2',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = (1/(mv+epsilon))
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('last_2-1',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = (mv)
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('variance',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))




            #var = (x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis)/(mv+epsilon)
            #print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
            #      .format('last_last',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))




            print('doffset')
            print(tf.reduce_max(doffset))

            print('dscale')
            print(tf.reduce_max(dscale))

            print('')



        #return dx, dstop(mean), dstop(variance), doffset, dscale, dstop(variance_epsilon)
        return dx, None, None, doffset, dscale, None, None


    #with ops.name_scope(name, "batchnorm", [x, mean, variance, scale, offset]):
    with ops.name_scope(None, "batchnorm", [x, mean, variance, scale, offset]):
        #inv = math_ops.rsqrt(variance + variance_epsilon)
        #dev = math_ops.rsqrt(variance + variance_epsilon)
        #if scale is not None:
        #    dev*= scale
        # Note: tensorflow/contrib/quantize/python/fold_batch_norms.py depends on
        # the precise order of ops that are generated by the expression below.

        # statistic and deviation
        # standard
        sta = mean
        dev = math_ops.sqrt(variance + variance_epsilon)

        input_shape = x.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims-1)]

        # mean absolute
        #sta = mean
        #dev = tf.reduce_mean(tf.math.abs(x-mean),axis=reduction_axes)

        # right-semi
        #input_shape = x.shape
        #ndims = len(input_shape)
        #reduction_axes = [i for i in range(ndims-1)]
        #sta = mean
        #dev = tf.math.maximum(tf.reduce_mean(x-mean,axis=reduction_axes),0)

        # range-based
        #sta = (tf.reduce_max(x)+tf.reduce_min(x))*0.5
        #dev = tf.reduce_max(x) - tf.reduce_min(x)
        #max=tf.reduce_max(x,axis=reduction_axes)
        #min=tf.reduce_min(x,axis=reduction_axes)
        #sta = (max+min)*0.5
        #dev = max-min

        #dev = tf.where(dev==0,tf.ones(shape=dev.shape),dev)
        #dev = tf.where(dev<0.1,tf.ones(shape=dev.shape),dev)

        # worst-case
        #sta = tf.reduce_max(x)
        #dev = tf.reduce_max(x) - tf.reduce_mean(x)

        # test - range dev
        #sta = mean
        #dev = variance


        #if False:   # original
        #x_norm = (x-mean)/dev
        x_norm = (x-sta)/(dev)
        #x_norm = (x-sta-0.1)/(dev)
        ret = x_norm*scale+offset

        #return ret
        return ret, grad
        #return (x-mean)*inv*scale+offset, grad
        #return (x-mean)*inv*scale+offset

        #return x * math_ops.cast(inv, x.dtype) \
        #       + math_ops.cast(offset - mean * inv if offset is not None else -mean * inv, x.dtype), grad





# @tf_export(v1=["nn.fused_batch_norm"])
@dispatch.add_dispatch_support
def fused_batch_norm(
        x,
        scale,
        offset,  # pylint: disable=invalid-name
        mean=None,
        variance=None,
        epsilon=0.001,
        data_format="NHWC",
        is_training=True,
        name=None,
        exponential_avg_factor=1.0):
    r"""Batch normalization.


    See Source: [Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift; S. Ioffe, C. Szegedy]
    (http://arxiv.org/abs/1502.03167).

    Args:
      x: Input `Tensor` of 4 or 5 dimensions.
      scale: A `Tensor` of 1 dimension for scaling.
      offset: A `Tensor` of 1 dimension for bias.
      mean: A `Tensor` of 1 dimension for population mean. The shape and meaning
            of this argument depends on the value of is_training and
            exponential_avg_factor as follows:
            is_training==False (inference):
              Mean must be a `Tensor` of the same shape as scale containing the
              estimated population mean computed during training.
            is_training==True and exponential_avg_factor == 1.0:
              Mean must be None.
            is_training==True and exponential_avg_factor != 1.0:
              Mean must be a `Tensor` of the same shape as scale containing the
              exponential running mean.
      variance: A `Tensor` of 1 dimension for population variance. The shape and
            meaning of this argument depends on the value of is_training and
            exponential_avg_factor as follows:
            is_training==False (inference):
              Variance must be a `Tensor` of the same shape as scale containing
              the estimated population variance computed during training.
            is_training==True and exponential_avg_factor == 1.0:
              Variance must be None.
            is_training==True and exponential_avg_factor != 1.0:
              Variance must be a `Tensor` of the same shape as scale containing
              the exponential running variance.
      epsilon: A small float number added to the variance of x.
      data_format: The data format for x. Support "NHWC" (default) or "NCHW" for
                   4D tenors and "NDHWC" or "NCDHW" for 5D tensors.
      is_training: A bool value to specify if the operation is used for
                   training or inference.
      name: A name for this operation (optional).
      exponential_avg_factor: A float number (usually between 0 and 1) used
                              for controlling the decay of the running
                              population average of mean and variance.
                              If set to 1.0, the current batch average is
                              returned.

    Returns:
      y: A 4D or 5D Tensor for the normalized, scaled, offsetted x.
      running_mean: A 1D Tensor for the exponential running mean of x.
                    The output value is (1 - exponential_avg_factor) * mean +
                    exponential_avg_factor * batch_mean), where batch_mean
                    is the mean of the current batch in x.
      running_var: A 1D Tensor for the exponential running variance
                   The output value is (1 - exponential_avg_factor) * variance +
                   exponential_avg_factor * batch_variance), where batch_variance
                   is the variance of the current batch in x.

    References:
      Batch Normalization - Accelerating Deep Network Training by Reducing
      Internal Covariate Shift:
        [Ioffe et al., 2015](http://proceedings.mlr.press/v37/ioffe15.html)
        ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
    """
    if (not is_training or exponential_avg_factor != 1.0) and (
            (mean is None) or (variance is None)):
        raise ValueError("Both `mean` and `variance` must be a 1D tensor when "
                         "`is_training` is False or `exponential_avg_factor` != "
                         f"1.0. Received: `mean` {mean!r} and `variance` "
                         f"{variance!r}")
    x = ops.convert_to_tensor(x, name="input")
    scale = ops.convert_to_tensor(scale, name="scale")
    offset = ops.convert_to_tensor(offset, name="offset")
    if mean is None:
        mean = constant_op.constant([])
    if variance is None:
        variance = constant_op.constant([])

    # Set a minimum epsilon to 1.001e-5, which is a requirement by CUDNN to
    # prevent exception (see cudnn.h).
    min_epsilon = 1.001e-5
    epsilon = epsilon if epsilon > min_epsilon else min_epsilon

    y, running_mean, running_var, _, _, _ = gen_nn_ops.fused_batch_norm_v3(
    #y, running_mean, running_var, _, _, _ = gen_nn_ops.lib_snn_fused_batch_norm_v3(
        x,
        scale,
        offset,
        mean,
        variance,
        epsilon=epsilon,
        exponential_avg_factor=exponential_avg_factor,
        data_format=data_format,
        is_training=is_training,
        name=name)
    return y, running_mean, running_var



# @tf_export("nn.batch_normalization")
@tf.custom_gradient
@dispatch.add_dispatch_support
def batch_normalization_new(x,
                        mean,
                        variance,
                        offset,
                        scale,
                        variance_epsilon,
                        name=None):
    r"""Batch normalization.

    Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
    `scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):

    \\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)

    `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
    shapes:

      * In all generality, they can have the same number of dimensions as the
        input `x`, with identical sizes as `x` for the dimensions that are not
        normalized over (the 'depth' dimension(s)), and dimension 1 for the
        others which are being normalized over.
        `mean` and `variance` in this case would typically be the outputs of
        `tf.nn.moments(..., keepdims=True)` during training, or running averages
        thereof during inference.
      * In the common case where the 'depth' dimension is the last dimension in
        the input tensor `x`, they may be one dimensional tensors of the same
        size as the 'depth' dimension.
        This is the case for example for the common `[batch, depth]` layout of
        fully-connected layers, and `[batch, height, width, depth]` for
        convolutions.
        `mean` and `variance` in this case would typically be the outputs of
        `tf.nn.moments(..., keepdims=False)` during training, or running averages
        thereof during inference.

    See equation 11 in Algorithm 2 of source:
    [Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift; S. Ioffe, C. Szegedy]
    (http://arxiv.org/abs/1502.03167).

    Args:
      x: Input `Tensor` of arbitrary dimensionality.
      mean: A mean `Tensor`.
      variance: A variance `Tensor`.
      offset: An offset `Tensor`, often denoted \\(\beta\\) in equations, or
        None. If present, will be added to the normalized tensor.
      scale: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
        `None`. If present, the scale is applied to the normalized tensor.
      variance_epsilon: A small float number to avoid dividing by 0.
      name: A name for this operation (optional).

    Returns:
      the normalized, scaled, offset tensor.

    References:
      Batch Normalization - Accelerating Deep Network Training by Reducing
      Internal Covariate Shift:
        [Ioffe et al., 2015](http://arxiv.org/abs/1502.03167)
        ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
    """

    # func input: x, mean, variance, scale, offset, variance_epsilon
    def grad(upstream):

        #
        mm = mean
        mv = variance
        epsilon = variance_epsilon
        y_backprop = upstream


        #if len(x.shape.as_list())==4:
        if x.shape.rank==4:
            axis=[0,1,2]
        else:
            axis=[0]
        #axis=[0]

        #inv = math_ops.rsqrt(variance + variance_epsilon)
        inv = tf.math.rsqrt(variance + variance_epsilon)
        #inv = 1/(variance + variance_epsilon)
        #inv = tf.math.sqrt(variance + variance_epsilon)

        # gradient - dx
        #dx = scale * tf.math.sqrt(mv*mv+epsilon)*(y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis) / (1/mv*1/mv+epsilon))
        #dx = scale * tf.math.rsqrt(1/mv+epsilon)*(y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis) / (mv+epsilon))
        #dx = scale * tf.math.rsqrt(mv+epsilon)*(y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis) / (mv+epsilon))

        dx = scale * inv * (y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis)/(mv+epsilon))

        #dx = y_backprop * scale * (-tf.math.exp(-x))
        #dx = 1/tf.math.sqrt(math.pi)*tf.math.exp(-x)*(-1/2*tf.math.pow(x,-3/2)-tf.math.pow(x,-1/2))
        #dx = 1/2*tf.math.exp(-x)*(1-x)

        #dx = tf.multiply(dx,2*(x+epsilon))
        #dx = tf.multiply(dx,0.5*tf.math.rsqrt(x+epsilon))

        # gradient - scale
        # moving variance = 1/reserve_space_2
        # due to numerical precision? rsqrt(mv*mv+eps)->sqrt(1/mv*1/mv+eps)
        #dscale = tf.reduce_sum(y_backprop*(x-mm)*inv,axis=axis) # original
        #dscale = tf.reduce_sum(y_backprop*tf.math.exp(-x),axis=axis)
        dscale = tf.reduce_sum(y_backprop*x_norm,axis=axis)


        # gradient - offset
        doffset = tf.reduce_sum(y_backprop, axis=axis)

        #
        dstop=tf.stop_gradient

        #if False:
        #if True:
        if conf.verbose_snn_train:
            var = y_backprop
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('y_backprop',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = dx
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('dx',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            if conf.verbose_visual:
                import lib_snn
                from lib_snn.sim import glb_plot_gradient_bn
                lib_snn.util.plot_hist(glb_plot_gradient_bn,dx,1000,norm_fit=True)

            var = scale
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('scale',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = inv
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('inv',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = (y_backprop-tf.reduce_mean(y_backprop,axis=axis)-(x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis)/(mv+epsilon))
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('last',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = (y_backprop-tf.reduce_mean(y_backprop,axis=axis))
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('last_1',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = ((x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis)/(mv+epsilon))
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('last_2',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = (1/(mv+epsilon))
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('last_2-1',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))

            var = (mv)
            print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
                  .format('variance',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))




            #var = (x-mm)*tf.reduce_mean(y_backprop*(x-mm),axis=axis)/(mv+epsilon)
            #print('{:} - max {:.3g}, min {:.3g}, mean {:.3g}, std {:.3g}'
            #      .format('last_last',tf.reduce_max(var),tf.reduce_min(var),tf.reduce_mean(var),tf.math.reduce_std(var)))




            print('doffset')
            print(tf.reduce_max(doffset))

            print('dscale')
            print(tf.reduce_max(dscale))

            print('')



        #return dx, dstop(mean), dstop(variance), doffset, dscale, dstop(variance_epsilon)
        return dx, None, None, doffset, dscale, None


    with ops.name_scope(name, "batchnorm", [x, mean, variance, scale, offset]):
        #inv = math_ops.rsqrt(variance + variance_epsilon)
        #dev = math_ops.rsqrt(variance + variance_epsilon)
        #if scale is not None:
        #    dev*= scale
        # Note: tensorflow/contrib/quantize/python/fold_batch_norms.py depends on
        # the precise order of ops that are generated by the expression below.

        # statistic and deviation
        # standard
        sta = mean
        dev = math_ops.sqrt(variance + variance_epsilon)

        input_shape = x.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims-1)]

        # mean absolute
        #sta = mean
        #dev = tf.reduce_mean(tf.math.abs(x-mean),axis=reduction_axes)

        # right-semi
        #input_shape = x.shape
        #ndims = len(input_shape)
        #reduction_axes = [i for i in range(ndims-1)]
        #sta = mean
        #dev = tf.math.maximum(tf.reduce_mean(x-mean,axis=reduction_axes),0)

        # range-based
        #sta = (tf.reduce_max(x)+tf.reduce_min(x))*0.5
        #dev = tf.reduce_max(x) - tf.reduce_min(x)
        #max=tf.reduce_max(x,axis=reduction_axes)
        #min=tf.reduce_min(x,axis=reduction_axes)
        #sta = (max+min)*0.5
        #dev = max-min

        #dev = tf.where(dev==0,tf.ones(shape=dev.shape),dev)
        #dev = tf.where(dev<0.1,tf.ones(shape=dev.shape),dev)

        # worst-case
        #sta = tf.reduce_max(x)
        #dev = tf.reduce_max(x) - tf.reduce_mean(x)

        # test - range dev
        #sta = mean
        #dev = variance


        #if False:   # original
        x_norm = (x-mean)/dev
        #x_norm = (x-sta)/(dev)

        # exp distribution
        #x_norm = tf.math.exp(-x)
        #x_norm = 1/tf.math.sqrt(math.pi) * tf.math.pow(x,-1/2) * tf.math.exp(-x)
        #x_norm =  1/2 * x * tf.math.exp(-x)



        ret = x_norm*scale+offset

        #return ret
        return ret, grad
        #return (x-mean)*inv*scale+offset, grad
        #return (x-mean)*inv*scale+offset

        #return x * math_ops.cast(inv, x.dtype) \
        #       + math_ops.cast(offset - mean * inv if offset is not None else -mean * inv, x.dtype), grad

