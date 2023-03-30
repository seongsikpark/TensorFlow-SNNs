
'''
 This file is created based on TF 2 gen_math_ops.py file


'''
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

from tensorflow.python.ops import math_grad
from tensorflow.python.ops import gen_math_ops

from typing import TypeVar

import tensorflow as tf

from absl import flags
conf = flags.FLAGS


# only for layers_new.dense.Dense
#def mat_mul_dense(a, b, transpose_a=False, transpose_b=False, name=None):
@tf.custom_gradient
#def mat_mul_dense(a=0, b=0, input_accum=None, transpose_a=False, transpose_b=False, name=None):
#def mat_mul_dense(a, b, input_accum, transpose_a, transpose_b, name):
#def mat_mul_dense(a, b, input_accum, transpose_a=False, transpose_b=False, name=None):
def mat_mul_dense(a, b, input_accum, decay):

    r"""Multiply the matrix "a" by the matrix "b".

    The inputs must be two-dimensional matrices and the inner dimension of
    "a" (after being transposed if transpose_a is true) must match the
    outer dimension of "b" (after being transposed if transposed_b is
    true).

    *Note*: The default kernel implementation for MatMul on GPUs uses
    cublas.

    Args:
      a: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
      b: A `Tensor`. Must have the same type as `a`.
      transpose_a: An optional `bool`. Defaults to `False`.
        If true, "a" is transposed before multiplication.
      transpose_b: An optional `bool`. Defaults to `False`.
        If true, "b" is transposed before multiplication.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `a`.
    """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data

    transpose_a = False
    transpose_b = False
    name = None

    #input_accum = input_accum * decay + a

    if tld.is_eager:
    #if False:
        #try:
        #    _result = pywrap_tfe.TFE_Py_FastPathExecute(
        #        _ctx, "MatMul", name, a, b, "transpose_a", transpose_a, "transpose_b",
        #        transpose_b)
        #    return _result
        #except _core._NotOkStatusException as e:
        #    _ops.raise_from_not_ok_status(e, name)
        #except _core._FallbackException:
        #    pass
        #try:
            #return mat_mul_dense_eager_fallback(
                #a, b, input_accum=input_accum, transpose_a=False, transpose_b=False, name=None,
                #ctx=_ctx)
        #except _core._SymbolicException:
            #pass  # Add nodes to the TensorFlow graph.

        _result = mat_mul_dense_eager_fallback(
                a, b, input_accum=input_accum, transpose_a=transpose_a, transpose_b=transpose_b, name=name,
                ctx=_ctx)

    else:
        # Add nodes to the TensorFlow graph.

        transpose_a = _execute.make_bool(transpose_a, "transpose_a")
        transpose_b = _execute.make_bool(transpose_b, "transpose_b")
        _, _, _op, _outputs = _op_def_library._apply_op_helper(
            "MatMul", a=a, b=b, transpose_a=transpose_a, transpose_b=transpose_b,
            name=name)
        _result = _outputs[:]
        _result, = _result

    def grad(upstream):
        '''
        from _MatMulGrad in math_grad.py
        '''
        #op = _op
        grad = upstream
        #try:
        #    skip_input_indices = op.skip_input_indices
        #    if skip_input_indices is not None:
        #        if 1 in skip_input_indices:
        #            return math_grad._MatMulGradAgainstFirstOnly(op, grad)
        #        elif 0 in skip_input_indices:
        #            return math_grad._MatMulGradAgainstSecondOnly(op, grad)
        #except AttributeError:
        #    # No gradient skipping, so do the full gradient computation
        #    pass

        #t_a = op.get_attr("transpose_a")
        #t_b = op.get_attr("transpose_b")
        t_a = transpose_a
        t_b = transpose_b
        #a = math_ops.conj(op.inputs[0])
        #b = math_ops.conj(op.inputs[1])
        if not t_a and not t_b:
            grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
            #grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True)
            grad_b = gen_math_ops.mat_mul(input_accum, grad, transpose_a=True)


        else:
            assert False

        #return grad_a, grad_b, tf.zeros(input_accum.shape)
        #return grad_a, grad_b, grad_a
        #return grad_a, grad_b, tf.zeros(input_accum.shape), grad_a
        return grad_a, grad_b, tf.zeros(input_accum.shape), tf.reduce_sum(grad_a,axis=[0])


    #if _execute.must_record_gradient():
    if False:
        _attrs = ("transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b",
                  _op._get_attr_bool("transpose_b"), "T", _op._get_attr_type("T"))

        #print(type(_op.inputs))
        #print(_op.inputs[0])
        #print(_op.inputs)
        #assert False

        _input = _op.inputs[0]
        #input_accum = input_accum + input
        #input_accum = input
        _weight = _op.inputs[1]

        #_inputs_flat = _op.inputs
        #_inputs_flat = (_op.inputs[0], _op.inputs[1])
        #_inputs_flat = (tf.zeros(_op.inputs[0].shape), _op.inputs[1])
        #_inputs_flat = (_op.inputs[0]*2.0, _op.inputs[1])

        _input_accum = tf.convert_to_tensor(input_accum)
        #_input_accum = input_accum
        _a=tf.convert_to_tensor(a)
        _b=tf.convert_to_tensor(b)

        #print(_input.id)
        #print(_a.id)
        if conf.sptr:
            #_inputs_flat_grad = (_input_accum, _b)
            #_inputs_flat_grad = (_input, _b)
            #_inputs_flat_grad = (_a*0.0, _b)
            _inputs_flat_grad = (_input, _weight)
        else:
            _inputs_flat_grad = (_input, _weight)

        #_inputs_flat = (_input, _weight)
        #_inputs_flat = _op.inputs

        #_execute.record_gradient("MatMul", _inputs_flat_grad, _attrs, _result)
        _execute.record_gradient("MatMulA", _inputs_flat_grad, _attrs, _result)

    #return _result
    return _result, grad

#MatMulDense = tf_export("raw_ops.MatMulDense")(_ops.to_raw_op(mat_mul_dense))


def mat_mul_dense_eager_fallback(a, b, input_accum, transpose_a, transpose_b, name, ctx):
    if transpose_a is None:
        transpose_a = False
    transpose_a = _execute.make_bool(transpose_a, "transpose_a")
    if transpose_b is None:
        transpose_b = False
    transpose_b = _execute.make_bool(transpose_b, "transpose_b")
    _attr_T, _inputs_T = _execute.args_to_matching_eager([a, b], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
    (a, b) = _inputs_T
    _inputs_flat = [a, b]
    _attrs = ("transpose_a", transpose_a, "transpose_b", transpose_b, "T",
              _attr_T)
    _result = _execute.execute(b"MatMul", 1, inputs=_inputs_flat, attrs=_attrs,
                               ctx=ctx, name=name)

    if False:
    #if _execute.must_record_gradient():

        if conf.sptr:
            _inputs_flat_grad = [input_accum,b]
        else:
            _inputs_flat_grad = [a,b]

        _execute.record_gradient(
            "MatMul", _inputs_flat_grad, _attrs, _result)
    _result, = _result
    return _result


