#import keras_core.src.saving
#import keras_nlp.src.backend.keras.saving
import tensorflow as tf

#import keras
#from keras_nlp.src.backend import keras
#from tensorflow.keras import utils
from lib_snn import keras

import math


#@tf.function
#@keras_core.saving.register_keras_serializable(package="LibSNN")
@keras.saving.register_keras_serializable(package="LibSNN")
#@utils.register_keras_serializable(package="LibSNN")
class LRSchedule_step(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, decay_step, decay_factor):
        self.initial_learning_rate = initial_learning_rate
        #self.learning_rate = tf.Tensor(self.initial_learning_rate,shape=(),dtype=tf.float32)
        #self.learning_rate = tf.Tensor(self.initial_learning_rate,value_index=(),dtype=tf.float32)
        #self.learning_rate = self.initial_learning_rate
        #self.laeraning_rate = tf.constant(self.initial_learning_rate)
        self.decay_step = int(decay_step)
        self.decay_factor = decay_factor
        self.learning_rate = initial_learning_rate

    def __call__(self,step):

        #print(step)
        #mod = tf.math.floormod(step,self.decay_step)
        factor_n = tf.cast(tf.math.floordiv(step,self.decay_step),tf.float32)
        #cond = tf.math.equal(mod,0)

        factor = tf.math.pow(self.decay_factor,factor_n)
        #factor = tf.where(cond,tf.math.pow(self.decay_factor,factor_n),1.0)
        self.learning_rate = self.initial_learning_rate*factor
        learning_rate = self.learning_rate

        #print(step)
        #print(factor_n)
        #print(factor)
        #print(learning_rate)

        return learning_rate


    def get_config(self):
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_step': self.decay_step,
            'decay_factor': self.decay_factor,
        }

        return config



# step, warm up
#@tf.function
class LRSchedule_step_wup(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, decay_step, decay_factor, warmup_step):
        self.initial_learning_rate = initial_learning_rate
        #self.learning_rate = tf.Tensor(self.initial_learning_rate,shape=(),dtype=tf.float32)
        #self.learning_rate = tf.Tensor(self.initial_learning_rate,value_index=(),dtype=tf.float32)
        #self.learning_rate = self.initial_learning_rate
        #self.laeraning_rate = tf.constant(self.initial_learning_rate)
        self.decay_step = decay_step
        self.decay_factor = decay_factor
        self.warmup_step = warmup_step

    def __call__(self,step):

        step_float = tf.cast(step,tf.float32)
        warmup_step_float = tf.cast(self.warmup_step,tf.float32)


        factor_n = tf.cast(tf.math.floordiv(step,self.decay_step),tf.float32)
        factor = tf.math.pow(self.decay_factor,factor_n)

        learning_rate = tf.cond(step_float<warmup_step_float,
                                lambda: self.initial_learning_rate*(step_float/warmup_step_float),
                                lambda: self.initial_learning_rate*factor)


        return learning_rate


    def get_config(self):
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_step': self.decay_step,
            'decay_fact-r': self.decay_factor,
        }

        return config



# from tf2.16.1 source code
# modified
#@keras_export("keras.optimizers.schedules.CosineDecay")
class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A `LearningRateSchedule` that uses a cosine decay with optional warmup.

    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.

    For the idea of a linear warmup of our learning rate,
    see [Goyal et al.](https://arxiv.org/pdf/1706.02677.pdf).

    When we begin training a model, we often want an initial increase in our
    learning rate followed by a decay. If `warmup_target` is an int, this
    schedule applies a linear increase per optimizer step to our learning rate
    from `initial_learning_rate` to `warmup_target` for a duration of
    `warmup_steps`. Afterwards, it applies a cosine decay function taking our
    learning rate from `warmup_target` to `alpha` for a duration of
    `decay_steps`. If `warmup_target` is None we skip warmup and our decay
    will take our learning rate from `initial_learning_rate` to `alpha`.
    It requires a `step` value to  compute the learning rate. You can
    just pass a backend variable that you increment at each training step.

    The schedule is a 1-arg callable that produces a warmup followed by a
    decayed learning rate when passed the current optimizer step. This can be
    useful for changing the learning rate value across different invocations of
    optimizer functions.

    Our warmup is computed as:

    ```python
    def warmup_learning_rate(step):
        completed_fraction = step / warmup_steps
        total_delta = target_warmup - initial_learning_rate
        return completed_fraction * total_delta
    ```

    And our decay is computed as:

    ```python
    if warmup_target is None:
        initial_decay_lr = initial_learning_rate
    else:
        initial_decay_lr = warmup_target

    def decayed_learning_rate(step):
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return initial_decay_lr * decayed
    ```

    Example usage without warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0.1
    lr_decayed_fn = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)
    ```

    Example usage with warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0
    warmup_steps = 1000
    target_learning_rate = 0.1
    lr_warmup_decayed_fn = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
        warmup_steps=warmup_steps
    )
    ```

    You can pass this schedule directly into a `keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `keras.optimizers.schedules.serialize` and
    `keras.optimizers.schedules.deserialize`.

    Args:
        initial_learning_rate: A Python float. The initial learning rate.
        decay_steps: A Python int. Number of steps to decay over.
        alpha: A Python float. Minimum learning rate value for decay as a
            fraction of `initial_learning_rate`.
        name: String. Optional name of the operation.  Defaults to
            `"CosineDecay"`.
        warmup_target: A Python float. The target learning rate for our
            warmup phase. Will cast to the `initial_learning_rate` datatype.
            Setting to `None` will skip warmup and begins decay phase from
            `initial_learning_rate`. Otherwise scheduler will warmup from
            `initial_learning_rate` to `warmup_target`.
        warmup_steps: A Python int. Number of steps to warmup over.

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar tensor of the
        same type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        alpha=0.0,
        name="CosineDecay",
        warmup_target=None,
        warmup_steps=0,
        lr_min=0,
    ):
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name
        self.warmup_steps = warmup_steps
        self.warmup_target = warmup_target
        self.lr_min = lr_min

        if self.decay_steps <= 0:
            raise ValueError(
                "Argument `decay_steps` must be > 0. "
                f"Received: decay_steps={self.decay_steps}"
            )

    def _decay_function(self, step, decay_steps, decay_from_lr, dtype):
        with tf.name_scope(self.name):
            completed_fraction = step / decay_steps
            pi = tf.constant(math.pi, dtype=dtype)
            cosine_decayed = 0.5 * (1.0 + tf.cos(pi * completed_fraction))
            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            ret = tf.multiply(decay_from_lr, decayed)
            ret = tf.where(ret>self.lr_min, ret, self.lr_min)
            return ret

    def _warmup_function(
        self, step, warmup_steps, warmup_target, initial_learning_rate
    ):
        with tf.name_scope(self.name):
            completed_fraction = step / warmup_steps
            total_step_delta = warmup_target - initial_learning_rate
            return total_step_delta * completed_fraction + initial_learning_rate

    def __call__(self, step):
        with tf.name_scope(self.name):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            global_step_recomp = tf.cast(step, dtype)

            if self.warmup_target is None:
                global_step_recomp = tf.minimum(
                    global_step_recomp, decay_steps
                )
                return self._decay_function(
                    global_step_recomp,
                    decay_steps,
                    initial_learning_rate,
                    dtype,
                )

            warmup_target = tf.cast(self.warmup_target, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)

            global_step_recomp = tf.minimum(
                global_step_recomp, decay_steps + warmup_steps
            )

            return tf.cond(
                global_step_recomp < warmup_steps,
                lambda: self._warmup_function(
                    global_step_recomp,
                    warmup_steps,
                    warmup_target,
                    initial_learning_rate,
                ),
                lambda: self._decay_function(
                    global_step_recomp - warmup_steps,
                    decay_steps,
                    warmup_target,
                    dtype,
                ),
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps,
        }


#class AdamW(adam.Adam):
class AdamW(tf.keras.optimizers.Adam):
    """Optimizer that implements the AdamW algorithm.

    AdamW optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments with an added
    method to decay weights per the techniques discussed in the paper,
    'Decoupled Weight Decay Regularization' by
    [Loshchilov, Hutter et al., 2019](https://arxiv.org/abs/1711.05101).

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the underlying Adam method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates.
            Defaults to `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates.
            Defaults to `0.999`.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just
            before Section 2.1), not the epsilon in Algorithm 1 of the paper.
            Defaults to 1e-7.
        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm
            from the paper "On the Convergence of Adam and beyond".
            Defaults to `False`.
        {{base_optimizer_keyword_args}}

    References:

    - [Loshchilov et al., 2019](https://arxiv.org/abs/1711.05101)
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980) for `adam`
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.
    """

    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamw",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )

        if self.weight_decay is None:
            raise ValueError(
                "Argument `weight_decay` must be a float. Received: "
                "weight_decay=None"
            )
