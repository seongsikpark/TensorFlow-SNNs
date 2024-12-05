
import autokeras as ak
from autokeras.utils import data_utils

# from autokeras.graph import Graph
from autokeras_custom import graph as graph_module
from autokeras.engine import head as head_module
from autokeras.engine import node as node_module
import tensorflow as tf
from autokeras.utils import utils

import numpy as np
from tensorflow import keras
from tensorflow import nest
from pathlib import Path
from typing import List
from typing import Type
from typing import Union

from autokeras_custom import tuners
from autokeras_custom.engine import tuner
from autokeras.nodes import Input


import keras_tuner as kt
import random
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_lib
from typing import Optional
from autokeras.engine import tuner as tuner_module
from keras_tuner.engine import tuner as k_tuner_module
from config import config

# Ryu
conf = config.flags
max_trial = conf.max_trial
n_initial_points = 2
num_population = 200
num_candidate = 50
seeds = None
##################################################################
##################################################################
class EvolutionaryOracle(oracle_module.Oracle):
    """Evolutionary search oracle.

        It uses aging evluation algorithm following: https://arxiv.org/pdf/1802.01548.pdf.
        # Arguments
            objective: String or `kerastuner.Objective`. If a string,
              the direction of the optimization (min or max) will be
              inferred.
            max_trials: Int. Total number of trials
                (model configurations) to test at most.
                Note that the oracle may interrupt the search
                before `max_trial` models have been tested if the search space has been
                exhausted.
            num_initial_points: (Optional) Int. The number of randomly generated samples
                as initial training data for Evolutionary search. If not specified,
                a value of 3 times the dimensionality of the hyperparameter space is
                used.
            population_size: (Optional) Int. The number of trials to form the populations.
    candidate_size: (Optional) Int. The number of candidate trials in the tournament
    selection.
            seed: Int. Random seed.
            hyperparameters: HyperParameters class instance.
                Can be used to override (or register in advance)
                hyperparamters in the search space.
    """

    def __init__(
            self,
            objective=None,
            max_trials=max_trial,
            num_initial_points=None,
            population_size=None,
            candidate_size=None,
            seed=None,
            hyperparameters=None,
            allow_new_entries=True,
            tune_new_entries=True,
    ):
        super(EvolutionaryOracle, self).__init__(
            objective=objective,
            max_trials=max_trials,
            hyperparameters=hyperparameters,
            seed=seed,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        self.population_size = population_size or 20
        self.candidate_size = candidate_size or 5
        self.num_initial_points = num_initial_points or self.population_size
        self.num_initial_points = max(self.num_initial_points, population_size)
        self.population_trial_ids = []
        self.seed = seed or random.randint(1, int(1e4))
        self._seed_state = self.seed
        self._tried_so_far = set()
        self._max_collisions = 1000
        self._random_state = np.random.RandomState(self.seed)

    def _random_populate_space(self):
        values = self._random_values()
        if values is None:
            return {"status": trial_lib.TrialStatus.STOPPED, "values": None}
        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def _num_completed_trials(self):
        return len([t for t in self.trials.values() if t.status == "COMPLETED"])

    def populate_space(self, trial_id):

        if self._num_completed_trials() < self.num_initial_points:
            return self._random_populate_space()

        self.population_trial_ids = self.end_order[-self.population_size :]

        # candidate trial selection
        candidate_indices = self._random_state.choice(
            self.population_size, self.candidate_size, replace=False
        )
        self.candidate_indices = candidate_indices
        candidate_trial_ids = list(
            map(self.population_trial_ids.__getitem__, candidate_indices)
        )

        # get the best candidate based on the performance
        candidate_scores = [
            self.trials[trial_id].score for trial_id in candidate_trial_ids
        ]
        # Ryu, argmin -> argmax
        # best_candidate_trial_id = candidate_trial_ids[np.argmin(candidate_scores)]
        best_candidate_trial_id = candidate_trial_ids[np.argmax(candidate_scores)]
        # print(candidate_scores, "score @@@@")

        best_candidate_trial = self.trials[best_candidate_trial_id]

        # mutate the hps of the candidate
        values = self._mutate(best_candidate_trial)

        if values is None:
            return {"status": trial_lib.TrialStatus.STOPPED, "values": None}

        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def _mutate(self, best_trial):

        best_hps = best_trial.hyperparameters

        # get non-fixed and active hyperparameters in the trial to be mutated
        nonfixed_active_hps = [
            hp
            for hp in self.hyperparameters.space
            if not isinstance(hp, hp_module.Fixed) and best_hps.is_active(hp)
        ]

        # random select a hyperparameter to mutate
        hp_to_mutate = self._random_state.choice(nonfixed_active_hps, 1)[0]

        collisions = 0
        while True:
            hps = hp_module.HyperParameters()
            # Generate a set of random values.
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                # if not active, do nothing.
                # if active, check if selected to be changed.
                if hps.is_active(hp):
                    # if was active and not selected, do nothing.
                    if best_hps.is_active(hp.name) and hp.name != hp_to_mutate.name:
                        hps.values[hp.name] = best_hps.values[hp.name]
                        continue
                    # if was not active or selected, sample.
                    hps.values[hp.name] = hp.random_sample(self._seed_state)
                    self._seed_state += 1
            values = hps.values

            # Make sure the new hyperparameters has not been evaluated before
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions <= self._max_collisions:
                    continue
                return None
            self._tried_so_far.add(values_hash)
            break
            # continue
        return values

    def get_state(self):
        state = super(EvolutionaryOracle, self).get_state()
        state.update(
            {
                "num_initial_points": self.num_initial_points,
                "population_size": self.population_size,
                "candidate_size": self.candidate_size,
                "seed": self.seed,
                "_max_collisions": self._max_collisions,
            }
        )
        return state

    def set_state(self, state):
        super(EvolutionaryOracle, self).set_state(state)
        self.num_initial_points = state["num_initial_points"]
        self.population_size = state["population_size"]
        self.candidate_size = state["candidate_size"]
        self.population_trial_ids = self.end_order[-self.population_size :]
        self.seed = state["seed"]
        self._random_state = np.random.RandomState(self.seed)
        self._seed_state = self.seed
        self._max_collisions = state["max_collisions"]
##################################################################
##################################################################
class Evolution(k_tuner_module.Tuner):
    def __init__(
        self,
        # hypermodel: kt.HyperModel,
        # hyperparameters: Optional[kt.HyperParameters] = None,
        hypermodel: None,
        objective: str = kt.Objective("val_acc", "max"),
        max_trials: int = max_trial,
        num_initial_points=n_initial_points,
        population_size=num_population,
        candidate_size=num_candidate,
        seed: Optional[int] = seeds,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        **kwargs
    ):
        self.seed = seed
        oracle = EvolutionaryOracle(
            objective=objective,
            max_trials=max_trials,
            num_initial_points=num_initial_points,
            population_size=population_size,
            candidate_size=candidate_size,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        super(
            Evolution,
            self,
        ).__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
        # tuner_module.AutoTuner.__init__(self=self, oracle=oracle, hypermodel=hypermodel, **kwargs)
##################################################################
##################################################################
class Evolutionary(Evolution, tuner_module.AutoTuner):
    pass
##################################################################
##################################################################


TUNER_CLASSES = {
    "bayesian": tuners.BayesianOptimization,
    "random": tuners.RandomSearch,
    "hyperband": tuners.Hyperband,
    "greedy": tuners.Greedy,
    "evolution": Evolutionary,
}


def get_tuner_class(tuner):
    if isinstance(tuner, str) and tuner in TUNER_CLASSES:
        return TUNER_CLASSES.get(tuner)
    else:
        raise ValueError(
            'Expected the tuner argument to be one of "greedy", '
            '"random", "hyperband", or "bayesian", '
            "but got {tuner}".format(tuner=tuner)
        )


class AutoModel(ak.AutoModel):
    def __init__(
        self,
        inputs: Union[Input, List[Input]],
        outputs: Union[head_module.Head, node_module.Node, list],
        project_name: str = "auto_model",
        max_trials: int = 100,
        directory: Union[str, Path, None] = None,
        objective: str = "val_loss",
        tuner: Union[str, Type[tuner.AutoTuner]] = "bayesian",
        overwrite: bool = False,
        seed: Optional[int] = None,
        max_model_size: Optional[int] = None,
        **kwargs
    ):
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        self.seed = seed
        if seed:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        # TODO: Support passing a tuner instance.
        # Initialize the hyper_graph.
        graph = self._build_graph()
        if isinstance(tuner, str):
            tuner = get_tuner_class(tuner)
        self.tuner = tuner(
            hypermodel=graph,
            overwrite=overwrite,
            objective=objective,
            max_trials=max_trials,
            directory=directory,
            seed=self.seed,
            project_name=project_name,
            max_model_size=max_model_size,
            **kwargs
        )
        self.overwrite = overwrite
        self._heads = [output_node.in_blocks[0] for output_node in self.outputs]

    @property
    def objective(self):
        return self.tuner.objective

    def _build_graph(self):
        # Using functional API.
        if all([isinstance(output, node_module.Node) for output in self.outputs]):
            # graph = graph_module.Graph(inputs=self.inputs, outputs=self.outputs)
            graph = graph_module.Graph(inputs=self.inputs, outputs=self.outputs)
        # Using input/output API.
        elif all([isinstance(output, head_module.Head) for output in self.outputs]):
            # Clear session to reset get_uid(). The names of the blocks will
            # start to count from 1 for new blocks in a new AutoModel afterwards.
            # When initializing multiple AutoModel with Task API, if not
            # counting from 1 for each of the AutoModel, the predefined hp
            # values in task specifiec tuners would not match the names.
            keras.backend.clear_session()
            graph = self._assemble()
            self.outputs = graph.outputs
            keras.backend.clear_session()

        return graph

    def fit(
            self,
            train_data=None,
            validation_data=None,
            x=None,
            y=None,
            batch_size=32,
            epochs=None,
            callbacks=None,
            validation_split=0.2,
            verbose=1,
            **kwargs
    ):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x.
            y: numpy.ndarray or tensorflow.Dataset. Training data y.
            batch_size: Int. Number of samples per gradient update. Defaults to 32.
            epochs: Int. The number of epochs to train each model during the search.
                If unspecified, by default we train for a maximum of 1000 epochs,
                but we stop training if the validation loss stops improving for 10
                epochs (unless you specified an EarlyStopping callback as part of
                the callbacks argument, in which case the EarlyStopping callback you
                specified will determine early stopping).
            callbacks: List of Keras callbacks to apply during training and
                validation.
            validation_split: Float between 0 and 1. Defaults to 0.2.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset.
                The best model found would be fit on the entire dataset including the
                validation data.
            validation_data: Data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                `validation_data` will override `validation_split`. The type of the
                validation data should be the same as the training data.
                The best model found would be fit on the training dataset without the
                validation data.
            verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar,
                2 = one line per epoch. Note that the progress bar is not
                particularly useful when logged to a file, so verbose=2 is
                recommended when not running interactively (eg, in a production
                environment). Controls the verbosity of both KerasTuner search and
                [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)
            **kwargs: Any arguments supported by
                [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        # Returns
            history: A Keras History object corresponding to the best model.
                Its History.history attribute is a record of training
                loss values and metrics values at successive epochs, as well as
                validation loss values and validation metrics values (if applicable).
        """
        # Check validation information.
        if not validation_data and not validation_split:
            raise ValueError(
                "Either validation_data or a non-zero validation_split "
                "should be provided."
            )

        if validation_data:
            validation_split = 0

        if train_data is None:
            dataset, validation_data = self._convert_to_dataset(x=x, y=y, batch_size=batch_size, validation_data=validation_data)
        else:
            dataset = train_data

        # dataset, validation_data = self._convert_to_dataset(
        #    x=x, y=y, validation_data=validation_data, batch_size=batch_size )

        self._analyze_data(dataset)
        self._build_hyper_pipeline(dataset)

        # Split the data with validation_split.
        if validation_data is None and validation_split:
            dataset, validation_data = data_utils.split_dataset(
                dataset, validation_split
            )

        history = self.tuner.search(
            x=dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_split=validation_split,
            verbose=verbose,
            **kwargs
        )

        return history

    def evaluate(self, x=None, test_data=None, y=None, batch_size=32, verbose=1, **kwargs):
        """Evaluate the best model for the given data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            y: Any allowed types according to the head. Testing targets.
                Defaults to None.
            batch_size: Number of samples per batch.
                If unspecified, batch_size will default to 32.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
                Controls the verbosity of
                [keras.Model.evaluate](http://tensorflow.org/api_docs/python/tf/keras/Model#evaluate)
            **kwargs: Any arguments supported by keras.Model.evaluate.

        # Returns
            Scalar test loss (if the model has a single output and no metrics) or
            list of scalars (if the model has multiple outputs and/or metrics).
            The attribute model.metrics_names will give you the display labels for
            the scalar outputs.
        """
        if x is not None:
            self._check_data_format((x, y))
            if isinstance(x, tf.data.Dataset):
                dataset = x
                x = dataset.map(lambda x, y: x)
                y = dataset.map(lambda x, y: y)
            x = self._adapt(x, self.inputs, batch_size)
            y = self._adapt(y, self._heads, batch_size)
            dataset = tf.data.Dataset.zip((x, y))

        # Ryu
        if test_data is None:
            print("no test_data")
        elif x is None and test_data is None:
            print("evaluate has no datasets")
            assert False
        else:
            dataset = test_data

        pipeline = self.tuner.get_best_pipeline()
        dataset = pipeline.transform(dataset)
        model = self.tuner.get_best_model()
        return utils.evaluate_with_adaptive_batch_size(
            model=model, batch_size=batch_size, x=dataset, verbose=verbose, **kwargs
        )

    def export_model(self):
        """Export the best Keras Model.

        # Returns
            keras.Model instance. The best model found during the search, loaded
            with trained weights.
        """
        return self.tuner.get_best_model()
