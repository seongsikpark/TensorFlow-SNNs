"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: nn_ops.cc
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

from typing import TypeVar


_FusedBatchNormOutput = collections.namedtuple(
    "FusedBatchNorm",
    ["y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"])


def _fused_batch_norm(x, scale, offset, mean, variance, epsilon=0.0001, exponential_avg_factor=1, data_format="NHWC", is_training=True, name=None):
  r"""Batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    exponential_avg_factor: An optional `float`. Defaults to `1`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      The data format for x and y. Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `x`.
    batch_variance: A `Tensor`. Has the same type as `x`.
    reserve_space_1: A `Tensor`. Has the same type as `x`.
    reserve_space_2: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedBatchNorm", name, x, scale, offset, mean, variance,
        "epsilon", epsilon, "exponential_avg_factor", exponential_avg_factor,
        "data_format", data_format, "is_training", is_training)
      _result = _FusedBatchNormOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _fused_batch_norm_eager_fallback(
          x, scale, offset, mean, variance, epsilon=epsilon,
          exponential_avg_factor=exponential_avg_factor,
          data_format=data_format, is_training=is_training, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if exponential_avg_factor is None:
    exponential_avg_factor = 1
  exponential_avg_factor = _execute.make_float(exponential_avg_factor, "exponential_avg_factor")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedBatchNorm", x=x, scale=scale, offset=offset, mean=mean,
                          variance=variance, epsilon=epsilon,
                          exponential_avg_factor=exponential_avg_factor,
                          data_format=data_format, is_training=is_training,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "epsilon",
              _op.get_attr("epsilon"), "exponential_avg_factor",
              _op.get_attr("exponential_avg_factor"), "data_format",
              _op.get_attr("data_format"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedBatchNorm", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormOutput._make(_result)
  return _result

FusedBatchNorm = tf_export("raw_ops.FusedBatchNorm")(_ops.to_raw_op(_fused_batch_norm))


def _fused_batch_norm_eager_fallback(x, scale, offset, mean, variance, epsilon, exponential_avg_factor, data_format, is_training, name, ctx):
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if exponential_avg_factor is None:
    exponential_avg_factor = 1
  exponential_avg_factor = _execute.make_float(exponential_avg_factor, "exponential_avg_factor")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, scale, offset, mean, variance], ctx, [_dtypes.float32, ])
  (x, scale, offset, mean, variance) = _inputs_T
  _inputs_flat = [x, scale, offset, mean, variance]
  _attrs = ("T", _attr_T, "epsilon", epsilon, "exponential_avg_factor",
  exponential_avg_factor, "data_format", data_format, "is_training",
  is_training)
  _result = _execute.execute(b"FusedBatchNorm", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FusedBatchNorm", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormOutput._make(_result)
  return _result

_FusedBatchNormGradOutput = collections.namedtuple(
    "FusedBatchNormGrad",
    ["x_backprop", "scale_backprop", "offset_backprop", "reserve_space_3", "reserve_space_4"])


def fused_batch_norm_grad(y_backprop, x, scale, reserve_space_1, reserve_space_2, epsilon=0.0001, data_format="NHWC", is_training=True, name=None):
  r"""Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `y_backprop`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      The data format for y_backprop, x, x_backprop.
      Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `y_backprop`.
    offset_backprop: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_3: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_4: A `Tensor`. Has the same type as `y_backprop`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedBatchNormGrad", name, y_backprop, x, scale,
        reserve_space_1, reserve_space_2, "epsilon", epsilon, "data_format",
        data_format, "is_training", is_training)
      _result = _FusedBatchNormGradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fused_batch_norm_grad_eager_fallback(
          y_backprop, x, scale, reserve_space_1, reserve_space_2,
          epsilon=epsilon, data_format=data_format, is_training=is_training,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedBatchNormGrad", y_backprop=y_backprop, x=x, scale=scale,
                              reserve_space_1=reserve_space_1,
                              reserve_space_2=reserve_space_2,
                              epsilon=epsilon, data_format=data_format,
                              is_training=is_training, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "epsilon",
              _op.get_attr("epsilon"), "data_format",
              _op.get_attr("data_format"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedBatchNormGrad", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormGradOutput._make(_result)
  return _result



_FusedBatchNormGradV3Output = collections.namedtuple(
    "FusedBatchNormGradV3",
    ["x_backprop", "scale_backprop", "offset_backprop", "reserve_space_4", "reserve_space_5"])


def fused_batch_norm_grad_v3(y_backprop, x, scale, reserve_space_1, reserve_space_2, reserve_space_3, epsilon=0.0001, data_format="NHWC", is_training=True, name=None):
  r"""Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor` of type `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must be one of the following types: `float32`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    reserve_space_3: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for some intermediate results to be reused
      in gradient computation. When is_training is False, a dummy empty Tensor will be
      created.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `"NHWC", "NCHW", "NDHWC", "NCDHW"`. Defaults to `"NHWC"`.
      The data format for y_backprop, x, x_backprop.
      Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_4, reserve_space_5).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    offset_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_4: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_5: A `Tensor`. Has the same type as `reserve_space_1`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedBatchNormGradV3", name, y_backprop, x, scale,
        reserve_space_1, reserve_space_2, reserve_space_3, "epsilon", epsilon,
        "data_format", data_format, "is_training", is_training)
      _result = _FusedBatchNormGradV3Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fused_batch_norm_grad_v3_eager_fallback(
          y_backprop, x, scale, reserve_space_1, reserve_space_2,
          reserve_space_3, epsilon=epsilon, data_format=data_format,
          is_training=is_training, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedBatchNormGradV3", y_backprop=y_backprop, x=x, scale=scale,
                                reserve_space_1=reserve_space_1,
                                reserve_space_2=reserve_space_2,
                                reserve_space_3=reserve_space_3,
                                epsilon=epsilon, data_format=data_format,
                                is_training=is_training, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"),
              "epsilon", _op.get_attr("epsilon"), "data_format",
              _op.get_attr("data_format"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedBatchNormGradV3", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormGradV3Output._make(_result)
  return _result


_FusedBatchNormV3Output = collections.namedtuple(
    "FusedBatchNormV3",
    ["y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2", "reserve_space_3"])


def lib_snn_fused_batch_norm_v3(x, scale, offset, mean, variance, epsilon=0.0001, exponential_avg_factor=1, data_format="NHWC", is_training=True, name=None):
  r"""Batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    exponential_avg_factor: An optional `float`. Defaults to `1`.
    data_format: An optional `string` from: `"NHWC", "NCHW", "NDHWC", "NCDHW"`. Defaults to `"NHWC"`.
      The data format for x and y. Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2, reserve_space_3).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `scale`.
    batch_variance: A `Tensor`. Has the same type as `scale`.
    reserve_space_1: A `Tensor`. Has the same type as `scale`.
    reserve_space_2: A `Tensor`. Has the same type as `scale`.
    reserve_space_3: A `Tensor`. Has the same type as `scale`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedBatchNormV3", name, x, scale, offset, mean, variance,
        "epsilon", epsilon, "exponential_avg_factor", exponential_avg_factor,
        "data_format", data_format, "is_training", is_training)
      _result = _FusedBatchNormV3Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fused_batch_norm_v3_eager_fallback(
          x, scale, offset, mean, variance, epsilon=epsilon,
          exponential_avg_factor=exponential_avg_factor,
          data_format=data_format, is_training=is_training, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if exponential_avg_factor is None:
    exponential_avg_factor = 1
  exponential_avg_factor = _execute.make_float(exponential_avg_factor, "exponential_avg_factor")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedBatchNormV3", x=x, scale=scale, offset=offset, mean=mean,
                            variance=variance, epsilon=epsilon,
                            exponential_avg_factor=exponential_avg_factor,
                            data_format=data_format, is_training=is_training,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"),
              "epsilon", _op.get_attr("epsilon"), "exponential_avg_factor",
              _op.get_attr("exponential_avg_factor"), "data_format",
              _op.get_attr("data_format"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedBatchNormV3", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormV3Output._make(_result)
  return _result


#
#def fused_batch_norm_kernel(op_name, name, x, scale, offset, mean, std, eps, exp_avg_factor, data_format, is_training):
    #result
    #return result
