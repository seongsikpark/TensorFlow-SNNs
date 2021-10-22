
#
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import def_function

from tensorflow.python.keras.engine import training as keras_training
#import tensorflow.python.keras.engine.training as keras_training

########
# make test function
# overridding - keras.engine.training.make_test_function
# based on original make_test_function
########
def make_test_function(self):
    """Creates a function that executes one step of evaluation.

    This method can be overridden to support custom evaluation logic.
    This method is called by `Model.evaluate` and `Model.test_on_batch`.

    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual evaluation
    logic to `Model.test_step`.

    This function is cached the first time `Model.evaluate` or
    `Model.test_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return a `dict` containing values that will
      be passed to `tf.keras.Callbacks.on_test_batch_end`.
    """
    if self.test_function is not None:
        return self.test_function

    def step_function(model, iterator):
        """Runs a single evaluation step."""

        def run_step(data):
            outputs = model.test_step(data)
            # Ensure counter is updated only if `test_step` succeeds.
            with ops.control_dependencies(keras_training._minimum_control_deps(outputs)):
                model._test_counter.assign_add(1)  # pylint: disable=protected-access
            return outputs

        data = next(iterator)
        outputs = model.distribute_strategy.run(run_step, args=(data,))
        outputs = keras_training.reduce_per_replica(
            outputs, self.distribute_strategy, reduction='first')
        return outputs

    if self._steps_per_execution.numpy().item() == 1:

        def test_function(iterator):
            """Runs an evaluation execution with one step."""
            return step_function(self, iterator)

    else:

        def test_function(iterator):
            """Runs an evaluation execution with multiple steps."""
            for _ in math_ops.range(self._steps_per_execution):
                outputs = step_function(self, iterator)
            return outputs

    if not self.run_eagerly:
        test_function = def_function.function(
            test_function, experimental_relax_shapes=True)

    self.test_function = test_function
    return self.test_function
