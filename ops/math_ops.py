'''
 This file is created based on ops.math_ops.py in TF2

'''

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
# ==============================================================================
"""Math Operations.

Note: Functions taking `Tensor` arguments can also take anything accepted by
`tf.convert_to_tensor`.

Note: Elementwise binary operations in TensorFlow follow [numpy-style
broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

TensorFlow provides a variety of math functions including:

* Basic arithmetic operators and trigonometric functions.
* Special math functions (like: `tf.math.igamma` and `tf.math.zeta`)
* Complex number functions (like: `tf.math.imag` and `tf.math.angle`)
* Reductions and scans (like: `tf.math.reduce_mean` and `tf.math.cumsum`)
* Segment functions (like: `tf.math.segment_sum`)

See: `tf.linalg` for matrix and tensor functions.

<a id=Segmentation></a>

## About Segmentation

TensorFlow provides several operations that you can use to perform common
math computations on tensor segments.
Here a segmentation is a partitioning of a tensor along
the first dimension, i.e. it  defines a mapping from the first dimension onto
`segment_ids`. The `segment_ids` tensor should be the size of
the first dimension, `d0`, with consecutive IDs in the range `0` to `k`,
where `k<d0`.
In particular, a segmentation of a matrix tensor is a mapping of rows to
segments.

For example:

```python
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
tf.math.segment_sum(c, tf.constant([0, 0, 1]))
#  ==>  [[0 0 0 0]
#        [5 6 7 8]]
```

The standard `segment_*` functions assert that the segment indices are sorted.
If you have unsorted indices use the equivalent `unsorted_segment_` function.
These functions take an additional argument `num_segments` so that the output
tensor can be efficiently allocated.

``` python
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
tf.math.unsorted_segment_sum(c, tf.constant([0, 1, 0]), num_segments=2)
# ==> [[ 6,  8, 10, 12],
#       [-1, -2, -3, -4]]
```

"""
import numbers
import numpy as np
import builtins

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_math_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export


np_dtypes = LazyLoader(
    "np_dtypes", globals(),
    "tensorflow.python.ops.numpy_ops.np_dtypes")


# Aliases for some automatically-generated names.
nextafter = gen_math_ops.next_after


# This is set by resource_variable_ops.py. It is included in this way since
# there is a circular dependency between math_ops and resource_variable_ops
_resource_variable_type = None


def _set_doc(doc):

    def _decorator(func):
        func.__doc__ = doc
        return func

    return _decorator

@tf_export("matmuldense")
@dispatch.add_dispatch_support
def matmuldense(a,
           b,
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           a_is_sparse=False,
           b_is_sparse=False,
           output_type=None,
           name=None):
    """Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

    The inputs must, following any transpositions, be tensors of rank >= 2
    where the inner 2 dimensions specify valid matrix multiplication dimensions,
    and any further outer dimensions specify matching batch size.

    Both matrices must be of the same type. The supported types are:
    `bfloat16`, `float16`, `float32`, `float64`, `int32`, `int64`,
    `complex64`, `complex128`.

    Either matrix can be transposed or adjointed (conjugated and transposed) on
    the fly by setting one of the corresponding flag to `True`. These are `False`
    by default.

    If one or both of the matrices contain a lot of zeros, a more efficient
    multiplication algorithm can be used by setting the corresponding
    `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
    This optimization is only available for plain matrices (rank-2 tensors) with
    datatypes `bfloat16` or `float32`.

    A simple 2-D tensor matrix multiplication:

    >>> a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
    >>> a  # 2-D tensor
    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)>
    >>> b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
    >>> b  # 2-D tensor
    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[ 7,  8],
           [ 9, 10],
           [11, 12]], dtype=int32)>
    >>> c = tf.matmul(a, b)
    >>> c  # `a` * `b`
    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[ 58,  64],
           [139, 154]], dtype=int32)>

    A batch matrix multiplication with batch shape [2]:

    >>> a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
    >>> a  # 3-D tensor
    <tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=
    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
           [[ 7,  8,  9],
            [10, 11, 12]]], dtype=int32)>
    >>> b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])
    >>> b  # 3-D tensor
    <tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=
    array([[[13, 14],
            [15, 16],
            [17, 18]],
           [[19, 20],
            [21, 22],
            [23, 24]]], dtype=int32)>
    >>> c = tf.matmul(a, b)
    >>> c  # `a` * `b`
    <tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
    array([[[ 94, 100],
            [229, 244]],
           [[508, 532],
            [697, 730]]], dtype=int32)>

    Since python >= 3.5 the @ operator is supported
    (see [PEP 465](https://www.python.org/dev/peps/pep-0465/)). In TensorFlow,
    it simply calls the `tf.matmul()` function, so the following lines are
    equivalent:

    >>> d = a @ b @ [[10], [11]]
    >>> d = tf.matmul(tf.matmul(a, b), [[10], [11]])

    Args:
      a: `tf.Tensor` of type `float16`, `float32`, `float64`, `int32`,
        `complex64`, `complex128` and rank > 1.
      b: `tf.Tensor` with same type and rank as `a`.
      transpose_a: If `True`, `a` is transposed before multiplication.
      transpose_b: If `True`, `b` is transposed before multiplication.
      adjoint_a: If `True`, `a` is conjugated and transposed before
        multiplication.
      adjoint_b: If `True`, `b` is conjugated and transposed before
        multiplication.
      a_is_sparse: If `True`, `a` is treated as a sparse matrix. Notice, this
        **does not support `tf.sparse.SparseTensor`**, it just makes optimizations
        that assume most values in `a` are zero.
        See `tf.sparse.sparse_dense_matmul`
        for some support for `tf.sparse.SparseTensor` multiplication.
      b_is_sparse: If `True`, `b` is treated as a sparse matrix. Notice, this
        **does not support `tf.sparse.SparseTensor`**, it just makes optimizations
        that assume most values in `a` are zero.
        See `tf.sparse.sparse_dense_matmul`
        for some support for `tf.sparse.SparseTensor` multiplication.
      output_type: The output datatype if needed. Defaults to None in which case
        the output_type is the same as input type. Currently only works when input
        tensors are type (u)int8 and output_type can be int32.
      name: Name for the operation (optional).

    Returns:
      A `tf.Tensor` of the same type as `a` and `b` where each inner-most matrix
      is the product of the corresponding matrices in `a` and `b`, e.g. if all
      transpose or adjoint attributes are `False`:

      `output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`,
      for all indices `i`, `j`.

      Note: This is matrix product, not element-wise product.


    Raises:
      ValueError: If `transpose_a` and `adjoint_a`, or `transpose_b` and
        `adjoint_b` are both set to `True`.
      TypeError: If output_type is specified but the types of `a`, `b` and
        `output_type` is not (u)int8, (u)int8 and int32.
    """

    with ops.name_scope(name, "MatMulDense", [a, b]) as name:
        if transpose_a and adjoint_a:
            raise ValueError(
                f"Only one of `transpose_a` and `adjoint_a` can be True. "
                f"Received `transpose_a`={transpose_a}, "
                f"`adjoint_a`={adjoint_a}.")
        if transpose_b and adjoint_b:
            raise ValueError(
                f"Only one of `transpose_b` and `adjoint_b` can be True. "
                f"Received `transpose_b`={transpose_b}, "
                f"`adjoint_b`={adjoint_b}.")

        if context.executing_eagerly():
            if not isinstance(a, (ops.EagerTensor, _resource_variable_type)):
                a = ops.convert_to_tensor(a, name="a")
            if not isinstance(b, (ops.EagerTensor, _resource_variable_type)):
                b = ops.convert_to_tensor(b, dtype_hint=a.dtype.base_dtype, name="b")
        else:
            a = ops.convert_to_tensor(a, name="a")
            b = ops.convert_to_tensor(b, dtype_hint=a.dtype.base_dtype, name="b")

        # TODO(apassos) remove _shape_tuple here when it is not needed.
        a_shape = a._shape_tuple()  # pylint: disable=protected-access
        b_shape = b._shape_tuple()  # pylint: disable=protected-access

        output_may_have_non_empty_batch_shape = (
                (a_shape is None or len(a_shape) > 2) or
                (b_shape is None or len(b_shape) > 2))

        # TODO(b/178749687): remove this boolean and all related branches once the
        # bridges are ready.
        # batch_matmul_v3 is for when input type is different from output type.
        use_batch_matmul_v3 = False
        if output_type and (output_type != a.dtype or output_type != b.dtype):
            use_batch_matmul_v3 = True

        if (not a_is_sparse and
            not b_is_sparse) and output_may_have_non_empty_batch_shape:
            # BatchMatmul does not support transpose, so we conjugate the matrix and
            # use adjoint instead. Conj() is a noop for real matrices.
            if transpose_a:
                a = conj(a)
                adjoint_a = True
            if transpose_b:
                b = conj(b)
                adjoint_b = True
            if use_batch_matmul_v3:
                return gen_math_ops.batch_mat_mul_v3(
                    a, b, adj_x=adjoint_a, adj_y=adjoint_b, Tout=output_type, name=name)
            else:
                return gen_math_ops.batch_mat_mul_v2(
                    a, b, adj_x=adjoint_a, adj_y=adjoint_b, name=name)

        # Neither matmul nor sparse_matmul support adjoint, so we conjugate
        # the matrix and use transpose instead. Conj() is a noop for real
        # matrices.
        if adjoint_a:
            a = conj(a)
            transpose_a = True
        if adjoint_b:
            b = conj(b)
            transpose_b = True

        use_sparse_matmul = False
        if a_is_sparse or b_is_sparse:
            sparse_matmul_types = [dtypes.bfloat16, dtypes.float32]
            use_sparse_matmul = (
                    a.dtype in sparse_matmul_types and b.dtype in sparse_matmul_types)
        if (((a.dtype == dtypes.bfloat16 and
              b.dtype not in (dtypes.int8, dtypes.uint8)) or
             (b.dtype == dtypes.bfloat16 and
              a.dtype not in (dtypes.int8, dtypes.uint8))) and a.dtype != b.dtype):
            # matmul currently doesn't handle mixed-precision inputs other than
            # fp16 * int8 which is supported in BatchMatMulV3.
            use_sparse_matmul = True
        if use_sparse_matmul:
            ret = sparse_matmul(
                a,
                b,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                a_is_sparse=a_is_sparse,
                b_is_sparse=b_is_sparse,
                name=name)
            # sparse_matmul always returns float32, even with
            # bfloat16 inputs. This prevents us from configuring bfloat16 training.
            # casting to bfloat16 also matches non-sparse matmul behavior better.
            if a.dtype == dtypes.bfloat16 and b.dtype == dtypes.bfloat16:
                ret = cast(ret, dtypes.bfloat16)
            return ret
        else:
            if use_batch_matmul_v3:
                adjoint_a = adjoint_a or transpose_a
                adjoint_b = adjoint_b or transpose_b
                return gen_math_ops.batch_mat_mul_v3(
                    a, b, adj_x=adjoint_a, adj_y=adjoint_b, Tout=output_type, name=name)
            else:
                return gen_math_ops.mat_mul(
                    a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
