# based on randomsearch.py in tensorflow keras

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from keras_tuner.engine import multi_execution_tuner
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_lib



from keras_tuner.engine import hyperparameters as hp_module

class GridSearchOracle(oracle_module.Oracle):
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
        values = self.sample_values()

        #print(trial_id)
        #print(values)
        #assert False
        if values is None:
            return {"status": trial_lib.TrialStatus.STOPPED, "values": None}
        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def sample_values(self):
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
