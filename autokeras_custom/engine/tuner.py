# Copyright 2020 The AutoKeras Authors.
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

import collections
import copy
import os

import keras_tuner
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.layers.experimental import preprocessing

from autokeras import pipeline as pipeline_module
from autokeras.utils import data_utils
from autokeras.utils import utils

#
import tensorflow as tf
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner_utils
import warnings
import numpy as np

#
from autokeras_custom.engine import utils as tuner_utils_custom

#
from absl import flags
conf = flags.FLAGS


class AutoTuner(keras_tuner.engine.tuner.Tuner):
    """A Tuner class based on KerasTuner for AutoKeras.

    Different from KerasTuner's Tuner class. AutoTuner's not only tunes the
    Hypermodel which can be directly built into a Keras model, but also the
    preprocessors. Therefore, a HyperGraph stores the overall search space containing
    both the Preprocessors and Hypermodel. For every trial, the HyperGraph builds the
    PreprocessGraph and KerasGraph with the provided HyperParameters.

    The AutoTuner uses EarlyStopping for acceleration during the search and fully
    trains the model with full epochs and with both training and validation data.
    The fully trained model is the best model to be used by AutoModel.

    # Arguments
        oracle: keras_tuner Oracle.
        hypermodel: keras_tuner HyperModel.
        **kwargs: The args supported by KerasTuner.
    """

    def __init__(self, oracle, hypermodel, **kwargs):
        # sspark
        self.max_model_size_new = kwargs.pop('max_model_size_new',None)
        self.over_max_model = False
        #
        self.results_1st = None

        # Initialize before super() for reload to work.
        self._finished = False
        super().__init__(oracle, hypermodel, **kwargs)
        # Save or load the HyperModel.
        self.hypermodel.save(os.path.join(self.project_dir, "graph"))
        self.hyper_pipeline = None

        #
        self._display = tuner_utils_custom.Display(oracle=self.oracle)

    def _populate_initial_space(self):
        # Override the function to prevent building the model during initialization.
        return

    def get_best_model(self):
        with keras_tuner.engine.tuner.maybe_distribute(self.distribution_strategy):
            model = keras.models.load_model(self.best_model_path)
        return model

    def get_best_pipeline(self):
        return pipeline_module.load_pipeline(self.best_pipeline_path)

    def _pipeline_path(self, trial_id):
        return os.path.join(self.get_trial_dir(trial_id), "pipeline")

    def _prepare_model_build(self, hp, **kwargs):
        """Prepare for building the Keras model.

        It builds the Pipeline from HyperPipeline, transforms the dataset to set
        the input shapes and output shapes of the HyperModel.
        """
        dataset = kwargs["x"]
        pipeline = self.hyper_pipeline.build(hp, dataset)
        pipeline.fit(dataset)
        dataset = pipeline.transform(dataset)
        self.hypermodel.set_io_shapes(data_utils.dataset_shape(dataset))

        if "validation_data" in kwargs:
            validation_data = pipeline.transform(kwargs["validation_data"])
        else:
            validation_data = None
        return pipeline, dataset, validation_data

    def _build_and_fit_model(self, trial, *args, **kwargs):
        model = self._try_build(trial.hyperparameters)
        (
            pipeline,
            kwargs["x"],
            kwargs["validation_data"],
        ) = self._prepare_model_build(trial.hyperparameters, **kwargs)
        pipeline.save(self._pipeline_path(trial.trial_id))

        self.adapt(model, kwargs["x"])

        _, history = utils.fit_with_adaptive_batch_size(
            model, self.hypermodel.batch_size, **kwargs
        )
        return history

    @staticmethod
    def adapt(model, dataset):
        """Adapt the preprocessing layers in the model."""
        # Currently, only support using the original dataset to adapt all the
        # preprocessing layers before the first non-preprocessing layer.
        # TODO: Use PreprocessingStage for preprocessing layers adapt.
        # TODO: Use Keras Tuner for preprocessing layers adapt.
        x = dataset.map(lambda x, y: x)

        def get_output_layers(tensor):
            output_layers = []
            tensor = nest.flatten(tensor)[0]
            for layer in model.layers:
                if isinstance(layer, keras.layers.InputLayer):
                    continue
                input_node = nest.flatten(layer.input)[0]
                if input_node is tensor:
                    if isinstance(layer, preprocessing.PreprocessingLayer):
                        output_layers.append(layer)
            return output_layers

        dq = collections.deque()

        for index, input_node in enumerate(nest.flatten(model.input)):
            in_x = x.map(lambda *args: nest.flatten(args)[index])
            for layer in get_output_layers(input_node):
                dq.append((layer, in_x))

        while len(dq):
            layer, in_x = dq.popleft()
            layer.adapt(in_x)
            out_x = in_x.map(layer)
            for next_layer in get_output_layers(layer.output):
                dq.append((next_layer, out_x))

        return model

    def search(
            self,
            epochs=None,
            callbacks=None,
            validation_split=0,
            verbose=1,
            **fit_kwargs
    ):
        """Search for the best HyperParameters.

        If there is not early-stopping in the callbacks, the early-stopping callback
        is injected to accelerate the search process. At the end of the search, the
        best model will be fully trained with the specified number of epochs.

        # Arguments
            callbacks: A list of callback functions. Defaults to None.
            validation_split: Float.
        """
        if self._finished:
            return

        if callbacks is None:
            callbacks = []

        self.hypermodel.set_fit_args(validation_split, epochs=epochs)

        # Insert early-stopping for adaptive number of epochs.
        epochs_provided = True
        if epochs is None:
            epochs_provided = False
            epochs = 1000
            if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
                callbacks.append(
                    tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
                )

        # Insert early-stopping for acceleration.
        # TODO: parameterize
        early_stopping_inserted = False
        new_callbacks = self._deepcopy_callbacks(callbacks)
        if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
            early_stopping_inserted = True
            # TODO
            new_callbacks.append(
                tf_callbacks.EarlyStopping(monitor="val_loss", patience=conf.step_decay_epoch*0.8, min_delta=1e-4, verbose=1)
            )

        # Populate initial search space.
        hp = self.oracle.get_space()
        self._prepare_model_build(hp, **fit_kwargs)
        self._try_build(hp)
        self.oracle.update_space(hp)
        # sspark
        #super().search(epochs=epochs, callbacks=new_callbacks, verbose=verbose, **fit_kwargs)
        self._search(epochs=epochs, callbacks=new_callbacks, verbose=verbose, **fit_kwargs)

        # Train the best model use validation data.
        # Train the best model with enough number of epochs.
        if validation_split > 0 or early_stopping_inserted:
            copied_fit_kwargs = copy.copy(fit_kwargs)

            # Remove early-stopping since no validation data.
            # Remove early-stopping since it is inserted.
            copied_fit_kwargs["callbacks"] = self._remove_early_stopping(callbacks)

            # Decide the number of epochs.
            # sspark
            #epochs_final=200
            #epochs_final=120
            epochs_final = conf.train_epoch
            #copied_fit_kwargs["epochs"] = epochs
            copied_fit_kwargs["epochs"] = epochs_final
            if not epochs_provided:
                copied_fit_kwargs["epochs"] = self._get_best_trial_epochs()

            # Concatenate training and validation data.
            if validation_split > 0:
                copied_fit_kwargs["x"] = copied_fit_kwargs["x"].concatenate(
                    fit_kwargs["validation_data"]
                )
                copied_fit_kwargs.pop("validation_data")

            self.hypermodel.set_fit_args(0, epochs=copied_fit_kwargs["epochs"])
            copied_fit_kwargs["verbose"] = verbose
            pipeline, model, history = self.final_fit(**copied_fit_kwargs)
        else:

            ''' original - old
            # TODO: Add return history functionality in Keras Tuner
            model = self.get_best_models()[0]
            history = None
            pipeline = pipeline_module.load_pipeline(
                self._pipeline_path(self.oracle.get_best_trials(1)[0].trial_id)
            )
            '''
            copied_fit_kwargs = copy.copy(fit_kwargs) # train, validation data
            copied_fit_kwargs["callbacks"] = self._remove_early_stopping(callbacks)
            # TODO: parameterized
            copied_fit_kwargs["epochs"] = 200

            pipeline, model, history = self.final_fit(**copied_fit_kwargs)
            # sspark
            #pipeline, model, history = self.final_fit(**copied_fit_kwargs)
            #print('here')
            #pass


        model.save(self.best_model_path)
        pipeline.save(self.best_pipeline_path)
        self._finished = True
        return history


    # from keras_tuner.engine.base_tuner
    def _search(self, *fit_args, **fit_kwargs):
        """Performs a search for best hyperparameter configuations.

        Args:
            *fit_args: Positional arguments that should be passed to
              `run_trial`, for example the training and validation data.
            **fit_kwargs: Keyword arguments that should be passed to
              `run_trial`, for example the training and validation data.
        """
        if "verbose" in fit_kwargs:
            self._display.verbose = fit_kwargs.get("verbose")
        self.on_search_begin()
        while True:
            # sspark
            #trial = self.oracle.create_trial(self.tuner_id)
            #trial = self.create_trial()
            trial = self.create_trial_max_model_flag()

            #
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                tf.get_logger().info("Oracle triggered exit")
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue

            self.on_trial_begin(trial)

            # sspark
            if self.results_1st==None:
                results = self.run_trial(trial, *fit_args, **fit_kwargs)
                self.results_1st = copy.deepcopy(results)
            else:
                if self.over_max_model:
                    results = copy.deepcopy(self.results_1st)
                    print(results[0].history)
                    results[0].history['val_acc'] = [0.1]*len(results[0].history['val_acc'])
                    #results[0].history['val_acc'] = [0.8]*len(results[0].history['val_acc'])
                else:
                    results = self.run_trial(trial, *fit_args, **fit_kwargs)


            # `results` is None indicates user updated oracle in `run_trial()`.
            if results is None:
                warnings.warn(
                    "`Tuner.run_trial()` returned None. It should return one of "
                    "float, dict, keras.callbacks.History, or a list of one "
                    "of these types. The use case of calling "
                    "`Tuner.oracle.update_trial()` in `Tuner.run_trial()` is "
                    "deprecated, and will be removed in the future.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                self.oracle.update_trial(
                    trial.trial_id,
                    # Convert to dictionary before calling `update_trial()`
                    # to pass it from gRPC.
                    tuner_utils.convert_to_metrics_dict(
                        results, self.oracle.objective, "Tuner.run_trial()"
                    ),
                )
            self.on_trial_end(trial)
        self.on_search_end()

        print('search end')

    def create_trial(self):

        while True:
            #
            trial = self.oracle.create_trial(self.tuner_id)

            #
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                #tf.get_logger().info("Oracle triggered exit")
                return trial

            self._display.on_trial_begin(trial)

            #
            if self.max_model_size_new:

                hp = trial.hyperparameters
                #self._try_build(hp)
                model = self._build_hypermodel(hp)

                params = [keras.backend.count_params(p) for p in model.trainable_weights]
                model_size = int(np.sum(params))

                print('model size {:.3e}, max_model_size {:.3e}'.format(model_size,self.max_model_size_new))

                if model_size <= self.max_model_size_new:
                    #print('return trial')
                    return trial
                else:
                    print('model size {:.3e} > max_model_size {:.3e}'.format(model_size,self.max_model_size_new))
                    print('remove and retry trial')
                    trial_id = self.oracle.start_order.pop()
                    self.oracle.trials.pop(trial_id)
                    self.oracle.ongoing_trials.pop(self.tuner_id)

            else:
                print('return trial')
                return trial


        return trial



    def create_trial_max_model_flag(self):

        #
        self.over_max_model = False

        #
        trial = self.oracle.create_trial(self.tuner_id)


        #
        if trial.status == trial_module.TrialStatus.STOPPED:
            # Oracle triggered exit.
            #tf.get_logger().info("Oracle triggered exit")
            return trial


        ##
        #if trial.status == trial_module.TrialStatus.STOPPED:
            ## Oracle triggered exit.
            ##tf.get_logger().info("Oracle triggered exit")
            #return trial

        #self._display.on_trial_begin(trial)

        #
        if self.max_model_size_new:

            hp = trial.hyperparameters
            #self._try_build(hp)
            model = self._build_hypermodel(hp)

            params = [keras.backend.count_params(p) for p in model.trainable_weights]
            model_size = int(np.sum(params))

            print('model size {:.3e}, max_model_size {:.3e}'.format(model_size,self.max_model_size_new))

            #if model_size <= self.max_model_size:
                #print('return trial')
                #return trial
                #self.over_max_model=True
            #else:
            if model_size > self.max_model_size_new:
                #print('model size {:.3e} > max_model_size {:.3e}'.format(model_size,self.max_model_size))
                #print('remove and retry trial')
                #trial_id = self.oracle.start_order.pop()
                #self.oracle.trials.pop(trial_id)
                #self.oracle.ongoing_trials.pop(self.tuner_id)

                print('oversized model > over_max_model - flag on')
                self.over_max_model=True


        return trial

    # sspark
    # based on keras_tuner.engine.base_tuner.on_trial_end
    def on_trial_end(self,trial):
        """Called at the end of a trial.

        Args:
            trial: A `Trial` instance.
        """
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        self.oracle.end_trial(trial.trial_id, trial_module.TrialStatus.COMPLETED)
        self.oracle.update_space(trial.hyperparameters)
        # Display needs the updated trial scored by the Oracle.
        self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
        self.save()

    def get_state(self):
        state = super().get_state()
        state.update({"finished": self._finished})
        return state

    def set_state(self, state):
        super().set_state(state)
        self._finished = state.get("finished")

    @staticmethod
    def _remove_early_stopping(callbacks):
        return [
            copy.deepcopy(callbacks)
            for callback in callbacks
            #if not isinstance(callback, tf_callbacks.EarlyStopping)
            if not isinstance(callback, tf_callbacks.EarlyStopping)
            #tf.keras.callbacks.EarlyStopping
        ]

    def _get_best_trial_epochs(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        # steps counts from 0, so epochs = step + 1.
        return self.oracle.get_trial(best_trial.trial_id).best_step + 1

    def _build_best_model(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        return self._try_build(best_hp)

    def final_fit(self, **kwargs):
        best_trial = self.oracle.get_best_trials(1)[0]
        # sspark
        print('best trial')
        self._display.on_trial_begin(best_trial)
        best_hp = best_trial.hyperparameters
        self._display.on_trial_begin(best_trial)
        pipeline, kwargs["x"], kwargs["validation_data"] = self._prepare_model_build(
            best_hp, **kwargs
        )

        model = self._build_best_model()

        # sspark
        model.summary()

        self.adapt(model, kwargs["x"])
        model, history = utils.fit_with_adaptive_batch_size(
            model, self.hypermodel.batch_size, **kwargs
        )
        return pipeline, model, history

    @property
    def best_model_path(self):
        return os.path.join(self.project_dir, "best_model")

    @property
    def best_pipeline_path(self):
        return os.path.join(self.project_dir, "best_pipeline")

    @property
    def objective(self):
        return self.oracle.objective

    @property
    def max_trials(self):
        return self.oracle.max_trials
