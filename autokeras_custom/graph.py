
from autokeras import graph
from tensorflow import keras
from autokeras import keras_layers
from keras.optimizers import Adam, SGD
from keras.optimizers.optimizer_experimental.adamw import AdamW
import lib_snn
from keras.optimizers.schedules.learning_rate_schedule import CosineDecay, ExponentialDecay
from tensorflow import nest
from config import config

conf = config.flags
# batch_size = conf.batch_size
batch_size = None
learning_rate = conf.lr
input_shape = None
num_class = 10


class Graph(graph.Graph):

    def build(self, hp):
        """Build the HyperModel into a Keras Model."""
        self.compile()
        keras_nodes = {}
        keras_input_nodes = []
        for node in self.inputs:
            node_id = self._node_to_id[node]
            input_node = node.build_node(hp)
            output_node = node.build(hp, input_node)
            keras_input_nodes.append(input_node)
            keras_nodes[node_id] = output_node
        for block in self.blocks:
            temp_inputs = [
                keras_nodes[self._node_to_id[input_node]]
                for input_node in block.inputs
            ]
            outputs = block.build(hp, inputs=temp_inputs)
            outputs = nest.flatten(outputs)
            for output_node, real_output_node in zip(block.outputs, outputs):
                keras_nodes[self._node_to_id[output_node]] = real_output_node

        # Ryu
        inputs = keras_input_nodes
        outputs = [keras_nodes[self._node_to_id[output_node]] for output_node in self.outputs]
        train_mode = config.flags.nn_mode
        if train_mode == 'ANN':
            model = keras.Model(inputs, outputs)
        else:
            model = lib_snn.model.Model(inputs, outputs, batch_size, input_shape, num_class, None)

        return self._compile_keras_model(hp, model)

    def _compile_keras_model(self, hp, model):
        optimizer_name = "adam"
        # optimizer_name = "sgd"

        if optimizer_name == "adam":
            steps_per_epoch = int(self.num_samples / self.batch_size)
            print(steps_per_epoch, self.epochs, self.num_samples,
                  "Step per epoch, epochs, samples @@@@@@")
            # num_train_steps = steps_per_epoch * self.epochs
            optimizer = Adam(
                # learning_rate=lr_schedule,
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                name='Adam',
            )
        elif optimizer_name == "sgd":
            steps_per_epoch = int(self.num_samples / self.batch_size)
            print(steps_per_epoch, self.epochs, self.num_samples,
                  "Step per epoch, epochs, samples @@@@@@")
            # num_train_steps = steps_per_epoch * self.epochs
            optimizer = SGD(
                # learning_rate=lr_schedule,
                learning_rate=learning_rate,
                momentum=0.9,
                name='SGD',
            )

        # eager_mode=True
        # eager_mode = config.eager_mode
        eager_mode = False

        model.compile(
            optimizer=optimizer, metrics=self._get_metrics(), loss=self._get_loss(), run_eagerly=eager_mode
        )

        return model

