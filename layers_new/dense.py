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
"""Contains the Dense layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

# based on keras.layers.core.dense.py


from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.dtensor import utils
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
import tensorflow.compat.v2 as tf



from tensorflow.python.eager import context as _context
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.ops import gen_math_ops


from tensorflow.python.util.tf_export import keras_export

#from ops import mat_mul_dense

from absl import flags
conf = flags.FLAGS


#@keras_export('layers_new.Dense')
class Dense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`). These are all attributes of
    `Dense`.

    Note: If the input to the layer has a rank greater than 2, then `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`,
    then we create a `kernel` with shape `(d1, units)`, and the `kernel` operates
    along axis 2 of the `input`, on every sub-tensor of shape `(1, 1, d1)`
    (there are `batch_size * d0` such sub-tensors).
    The output in this case will have shape `(batch_size, d0, units)`.

    Besides, layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).
    When a popular kwarg `input_shape` is passed, then keras will create
    an input layer to insert before the current layer. This can be treated
    equivalent to explicitly defining an `InputLayer`.

    Example:

    >>> # Create a `Sequential` model and add a Dense layer as the first layer.
    >>> model = tf.keras.models.Sequential()
    >>> model.add(tf.keras.Input(shape=(16,)))
    >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
    >>> # Now the model will take as input arrays of shape (None, 16)
    >>> # and output arrays of shape (None, 32).
    >>> # Note that after the first layer, you don't need to specify
    >>> # the size of the input anymore:
    >>> model.add(tf.keras.layers.Dense(32))
    >>> model.output_shape
    (None, 32)

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.

    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
    """

    @utils.allow_initializer_layout
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Dense, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(f'Received an invalid value for `units`, expected '
                             f'a positive integer. Received: units={units}')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('A Dense layer can only be built with a floating-point '
                            f'dtype. Received: dtype={dtype}')

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to a Dense layer '
                             'should be defined. Found None. '
                             f'Full input shape received: {input_shape}')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units,],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None

        #
        if conf.sptr and conf.nn_mode=='SNN':
            #self.input_accum = tf.zeros(input_shape)
            #self.input_accum = tf.Variable(tf.zeros(input_shape),trainable=False)
            self.input_accum = tf.Variable(tf.zeros(input_shape),trainable=False,name='input_accum')
            #self.sptr_decay = tf.Variable(tf.constant(conf.sptr_decay,shape=input_shape),trainable=False,name='sptr_decay')
            self.sptr_decay = tf.Variable(tf.constant(conf.sptr_decay,shape=input_shape[1]),name='sptr_decay')

        self.built = True

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            # In case we encounter a RaggedTensor with a fixed last dimension (last
            # dimension not ragged), we can flatten the input and restore the ragged
            # dimensions at the end.
            if tf.compat.dimension_value(inputs.shape[-1]) is None:
                raise ValueError('Dense layer only supports RaggedTensors when the '
                                 'innermost dimension is non-ragged. Received: '
                                 f'inputs.shape={inputs.shape}.')
            original_inputs = inputs
            if inputs.flat_values.shape.rank > 1:
                inputs = inputs.flat_values
            else:
                # Innermost partition is encoded using uniform_row_length.
                # (This is unusual, but we can handle it.)
                if inputs.shape.rank == 2:
                    inputs = inputs.to_tensor()
                    is_ragged = False
                else:
                    for _ in range(original_inputs.ragged_rank - 1):
                        inputs = inputs.values
                    inputs = inputs.to_tensor()
                    original_inputs = tf.RaggedTensor.from_nested_row_splits(
                        inputs, original_inputs.nested_row_splits[:-1])

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul operation for
            # large sparse input tensors. The op will result in a sparse gradient, as
            # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, tf.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id per row.
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding lookup as
                # a matrix multiply. We split our input matrix into separate ids and
                # weights tensors. The values of the ids tensor should be the column
                # indices of our input matrix and the values of the weights tensor
                # can continue to the actual matrix weights.
                # The column arrangement of ids and weights
                # will be summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
                # of the inputs to both ops.
                ids = tf.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape)
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(
                    self.kernel, ids, weights, combiner='sum')
            else:
                #outputs = tf.matmul(a=inputs, b=self.kernel)

                if conf.sptr and conf.nn_mode == 'SNN':
                    #outputs = mat_mul_dense(a=inputs, b=self.kernel, input_accum=self.input_accum)
                    #outputs = mat_mul_dense(a=inputs, b=self.kernel, input_accum=self.input_accum)
                    #outputs = mat_mul_dense(a=inputs, b=self.kernel, input_accum=inputs)
                    #outputs = mat_mul_dense(a=inputs, b=self.kernel)
                    #self.input_accum = self.input_accum + inputs
                    #print(self.input_accum)
                    #outputs = mat_mul_dense(a=inputs, b=self.kernel, input_accum=self.input_accum,
                    #                        transpose_a=False,transpose_b=False,name=None)

                    #self.input_accum.assign(self.input_accum*conf.sptr_decay+inputs)
                    self.input_accum.assign(self.input_accum*self.sptr_decay+inputs)
                    #outputs = mat_mul_dense(inputs,self.kernel,self.input_accum, self.sptr_decay)
                    outputs = self.mat_mul_dense_sptr(inputs,self.kernel,self.input_accum, self.sptr_decay)

                    #print('input_accum: '+self.name)
                    #print(self.input_accum)

                    #self.input_accum.assign(self.input_accum+inputs)

                else:
                    #outputs = mat_mul_dense(a=inputs, b=self.kernel)
                    #outputs = tf.matmul(a=inputs, b=self.kernel)
                    outputs = self.mat_mul_dense(inputs,self.kernel)



        # Broadcast kernel to inputs.
        else:
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the input shape of a Dense layer '
                             'should be defined. Found None. '
                             f'Received: input_shape={input_shape}')
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })
        return config



    @tf.custom_gradient
    #def mat_mul_dense(a=0, b=0, input_accum=None, transpose_a=False, transpose_b=False, name=None):
    #def mat_mul_dense(a, b, input_accum, transpose_a, transpose_b, name):
    #def mat_mul_dense(a, b, input_accum, transpose_a=False, transpose_b=False, name=None):
    def mat_mul_dense(self, a, b, input_accum=None, decay=None):

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

            _result = self.mat_mul_dense_eager_fallback(
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
                grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True)    # b.shape: (dim_in,dim_out)

                if conf.en_stdp_pathway and hasattr(self,'n_pre'):

                    #pre = self.n_pre.act.spike_trace
                    #post = self.n_post.act.spike_trace

                    pre = self.n_pre.act.spike_trace.read(conf.time_step-1)
                    post = self.n_pre.act.spike_trace.read(conf.time_step-1)

                    spike_trace = gen_math_ops.mat_mul(pre,post,transpose_a=True)

                    norm_spike_trace = spike_trace / tf.reduce_max(spike_trace)
                    alpha = conf.stdp_pathway_weight
                    grad_b_weight = (1-alpha)+alpha*norm_spike_trace
                    grad_b = grad_b * grad_b_weight

            else:
                assert False

            #return grad_a, grad_b, tf.zeros(input_accum.shape)
            #return grad_a, grad_b, grad_a
            #return grad_a, grad_b, tf.zeros(input_accum.shape), grad_a
            #return grad_a, grad_b, tf.zeros(input_accum.shape), tf.reduce_sum(grad_a,axis=[0])
            #return grad_a, grad_b, tf.zeros(a.shape), tf.zeros(a[0].shape)
            return grad_a, grad_b


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



    @tf.custom_gradient
    def mat_mul_dense_sptr(self, a, b, input_accum, decay):

        _ctx = _context._context or _context.context()
        tld = _ctx._thread_local_data

        transpose_a = False
        transpose_b = False
        name = None

        if tld.is_eager:
            _result = self.mat_mul_dense_eager_fallback(
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
            grad = upstream
            t_a = transpose_a
            t_b = transpose_b

            if not t_a and not t_b:
                grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
                grad_b = gen_math_ops.mat_mul(input_accum, grad, transpose_a=True)

            else:
                assert False

            #return grad_a, grad_b, tf.zeros(input_accum.shape), tf.reduce_sum(grad_a,axis=[0])
            return grad_a, grad_b, tf.zeros(input_accum.shape), tf.zeros(grad_a[0].shape)


        return _result, grad


    def mat_mul_dense_eager_fallback(self, a, b, input_accum, transpose_a, transpose_b, name, ctx):
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
