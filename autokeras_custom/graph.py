
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

import tensorflow as tf


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

        # sspark
        if conf.optimizer == None:
            # Specify hyperparameters from compile(...)
            optimizer_name = hp.Choice(
                "optimizer",
                ["adam", "sgd", "adam_weight_decay"],
                default="adam",
            )
            # TODO: add adadelta optimizer when it can optimize embedding layer on GPU.
            learning_rate = hp.Choice( "learning_rate", [1e-1, 1e-2, 1e-3, 1e-4, 2e-5, 1e-5], default=1e-3 )
        else:
            optimizer_name = conf.optimizer
            learning_rate = conf.learning_rate

        # TODO: move, parameterize
        if conf.num_train_data < 0:
            num_train_data = 50000
        else:
            num_train_data = conf.num_train_data
        train_steps_per_epoch = num_train_data / conf.batch_size
        train_epoch = conf.train_epoch
        lr_schedule_first_decay_step = train_steps_per_epoch * 10  # in iteration
        step_decay_epoch = conf.step_decay_epoch

        lr_schedule = conf.lr_schedule

        if lr_schedule == 'COS':
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(learning_rate, train_steps_per_epoch * train_epoch)
        elif lr_schedule == 'COSR':
            learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(learning_rate, lr_schedule_first_decay_step)
        elif lr_schedule == 'STEP':
            learning_rate = lib_snn.optimizers.LRSchedule_step(learning_rate, train_steps_per_epoch * step_decay_epoch, 0.1)
        elif lr_schedule == 'STEP_WUP':
            learning_rate = lib_snn.optimizers.LRSchedule_step_wup(learning_rate, train_steps_per_epoch * 100, 0.1,
                                                                   train_steps_per_epoch * 30)
        else:
            pass

        optimizer_name = optimizer_name.lower()
        if optimizer_name == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == "adam_weight_decay":
            steps_per_epoch = int(self.num_samples / self.batch_size)
            num_train_steps = steps_per_epoch * self.epochs
            warmup_steps = int(
                self.epochs * self.num_samples * 0.1 / self.batch_size
            )

            lr_schedule = keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=num_train_steps,
                end_learning_rate=0.0,
            )
            if warmup_steps:
                lr_schedule = keras_layers.WarmUp(
                    initial_learning_rate=learning_rate,
                    decay_schedule_fn=lr_schedule,
                    warmup_steps=warmup_steps,
                )

            optimizer = keras.optimizers.experimental.AdamW(
                learning_rate=lr_schedule,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
            )

        model.compile(
            optimizer=optimizer, metrics=self._get_metrics(), loss=self._get_loss()
        )

        return model


    def _compile_keras_model_old(self, hp, model):
        optimizer_name = "adam"
        #optimizer_name = "sgd"


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


        # sspark
        # metric
        metric_accuracy = tf.keras.metrics.categorical_accuracy
        metric_accuracy_top5 = tf.keras.metrics.top_k_categorical_accuracy

        #metric_name_acc = 'acc'
        #metric_name_acc_top5 = 'acc-5'
        #monitor_cri = 'val_' + metric_name_acc
        metric_name_acc = config.metric_name_acc
        metric_name_acc_top5 = config.metric_name_acc_top5

        metric_accuracy.name = metric_name_acc
        metric_accuracy_top5.name = metric_name_acc_top5


        model.compile(
            optimizer=optimizer, metrics=self._get_metrics(), loss=self._get_loss(), run_eagerly=eager_mode
        )

        return model

