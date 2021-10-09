# based on randomsearch.py in tensorflow keras

# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"Basic random search tuner."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from keras_tuner.engine import multi_execution_tuner
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_lib



from keras_tuner.engine import hyperparameters as hp_module

class GridSearchOracle(oracle_module.Oracle):
    """Random search oracle.

    Args:
        objective: A string or `keras_tuner.Objective` instance. If a string,
            the direction of the optimization (min or max) will be inferred.
        max_trials: Integer, the total number of trials (model configurations)
            to test at most. Note that the oracle may interrupt the search
            before `max_trial` models have been tested if the search space has
            been exhausted.
        seed: Optional integer, the random seed.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
    """

    def __init__(
        self,
        objective,
        #max_trials,
        #seed=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
    ):
        super(GridSearchOracle, self).__init__(
            objective=objective,
            #max_trials=max_trials,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            #seed=seed,
        )

        self.num_try = 0

        self.hp_idx = collections.OrderedDict()
        self.hp_len = collections.OrderedDict()

    def populate_space(self, trial_id):
        """Fill the hyperparameter space with values.

        Args:
            trial_id: A string, the ID for this Trial.

        Returns:
            A dictionary with keys "values" and "status", where "values" is
            a mapping of parameter names to suggested values, and "status"
            is the TrialStatus that should be returned for this trial (one
            of "RUNNING", "IDLE", or "STOPPED").
        """
        values = self.sample_values()

        #print(trial_id)
        #print(values)
        #assert False
        if values is None:
            return {"status": trial_lib.TrialStatus.STOPPED, "values": None}
        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def sample_values(self):
        """Fills the hyperparameter space - grid search

        Returns:
            A dictionary mapping parameter names to suggested values.
        """

        # for the first call
        if self.num_try == 0:
            for hp in self.hyperparameters.space:
                self.hp_idx[hp.name] = 0
                self.hp_len[hp.name] = len(hp.values)
            carry = 0
        else:
            carry = 1
        #
        hps = hp_module.HyperParameters()

        for hp in self.hyperparameters.space:

            idx = self.hp_idx[hp.name]+carry
            if idx < self.hp_len[hp.name]:
                hp_idx = idx
                carry = 0
            else:
                hp_idx = 0
                carry = 1

            self.hp_idx[hp.name] = hp_idx
            hps.values[hp.name] = hp.values[hp_idx]

        if carry == 1:
            return None

        values = hps.values

        values_hash = self._compute_values_hash(values)
        self._tried_so_far.add(values_hash)

        self.num_try += 1

        return values



class GridSearch(multi_execution_tuner.MultiExecutionTuner):
    """Random search tuner.

    Args:
        hypermodel: A `HyperModel` instance (or callable that takes
            hyperparameters and returns a Model instance).
        objective: A string or `keras_tuner.Objective` instance. If a string,
            the direction of the optimization (min or max) will be inferred.
        max_trials: Integer, the total number of trials (model configurations)
            to test at most. Note that the oracle may interrupt the search
            before `max_trial` models have been tested if the search space has
            been exhausted.
        seed: Optional integer, the random seed.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(
        self,
        hypermodel,
        objective,
        #max_trials,
        #seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        **kwargs
    ):
        #self.seed = seed
        oracle = GridSearchOracle(
            objective=objective,
            #max_trials=max_trials,
            #seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        super(GridSearch, self).__init__(oracle, hypermodel, **kwargs)
