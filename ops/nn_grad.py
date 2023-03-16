#from tensorflow.python.ops import gen_nn_ops
#from tensorflow.python.ops import math_ops
#from tensorflow.python.ops import nn_ops

#
#@ops.RegisterGradient("FusedBatchNormV3")
#def _FusedBatchNormV3Grad(op, *grad):
#    return _BaseFusedBatchNormGrad(op, 2, *grad)
#
#
#def _BaseFusedBatchNormGrad(op, version, *grad):
#    """Return the gradients for the 3 inputs of BatchNorm.
#
#    Args:
#      op: The BatchNormOp for which we need to compute gradients.
#      version: Integer indicating which version to use of the fused batch
#        norm gradient.
#      *grad: An argument list for tensors of gradients wrt the outputs
#        with grad[0] as grad_y.
#
#    Returns:
#      grad_x: gradient for x, which is scale * rsqrt(variance + epsilon) *
#              [grad_y - mean(grad_y) - (x - mean(x)) *
#              mean(grad_y * (x - mean(x))) / (variance + epsilon)]
#              in training mode; grad_y * scale * rsqrt(pop_variance + epsilon)
#              in freeze mode.
#
#      grad_scale: gradient for scale, which is sum(grad_y * (x - mean(x)) *
#                  rsqrt(variance + epsilon)) in training mode;
#                  sum(grad_y * (x - pop_mean) * rsqrt(pop_variance + epsilon))
#                  in freeze mode.
#
#      grad_offset: gradient for offset, which is sum(grad_y) in training mode;
#                   sum(grad_y) in freeze mode.
#    """
#    x = op.inputs[0]
#    grad_y = grad[0]
#    scale = op.inputs[1]
#    epsilon = op.get_attr("epsilon")
#    data_format = op.get_attr("data_format")
#    is_training = op.get_attr("is_training")
#    if version == 2:
#        # grad_fun = gen_nn_ops.fused_batch_norm_grad_v3
#        pass
#    elif version == 1:
#        # grad_fun = gen_nn_ops.fused_batch_norm_grad_v2
#        assert False, 'only support version 2 (grad_v3)'
#    else:
#        # grad_fun = gen_nn_ops.fused_batch_norm_grad
#        assert False, 'only support version 2 (grad_v3)'
#
#    grad_fun = gen_nn_ops.fused_batch_norm_grad_v3
#    if is_training:
#        args = {
#            "y_backprop": grad_y,
#            "x": x,
#            "scale": scale,
#            "reserve_space_1": op.outputs[3],
#            "reserve_space_2": op.outputs[4],
#            "epsilon": epsilon,
#            "data_format": data_format,
#            "is_training": is_training
#        }
#        if version == 2:
#            args["reserve_space_3"] = op.outputs[5]
#        dx, dscale, doffset, _, _ = grad_fun(**args)
#    else:
#        pop_mean = op.inputs[3]
#        pop_var = op.inputs[4]
#        if data_format == b"NCHW":
#            x = array_ops.transpose(x, [0, 2, 3, 1])
#            grad_y = array_ops.transpose(grad_y, [0, 2, 3, 1])
#        elif data_format == b"NCDHW":
#            x = array_ops.transpose(x, [0, 2, 3, 4, 1])
#            grad_y = array_ops.transpose(grad_y, [0, 2, 3, 4, 1])
#        target_data_format = ("NHWC" if data_format in (b"NCHW",
#                                                        b"NHWC") else "NDHWC")
#        args = {
#            "y_backprop": grad_y,
#            "x": x,
#            "scale": scale,
#            "reserve_space_1": pop_mean,
#            "reserve_space_2": pop_var,
#            "epsilon": epsilon,
#            "data_format": target_data_format,
#            "is_training": is_training
#        }
#        if version == 2:
#            args["reserve_space_3"] = op.outputs[5]
#        dx, dscale, doffset, _, _ = grad_fun(**args)
#        if data_format == b"NCHW":
#            dx = array_ops.transpose(dx, [0, 3, 1, 2])
#        elif data_format == b"NCDHW":
#            dx = array_ops.transpose(dx, [0, 4, 1, 2, 3])
#    return dx, dscale, doffset, None, None
#