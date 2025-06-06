

import keras
from keras_tuner import utils
import time

from config import config
conf = config.flags


import lib_snn
from lib_snn.sim import glb_t

import collections

import tensorflow as tf

# based on keras_tuner.engine.tuner_utils.py -> Display class
#
class Display(object):
    def __init__(self, oracle, verbose=1):
        self.verbose = verbose
        self.oracle = oracle
        self.col_width = 18

        # Start time for the overall search
        self.search_start = None

        # Start time of the latest trial
        self.trial_start = None

    def on_trial_begin(self, trial):
        if self.verbose >= 1:

            self.trial_number = int(trial.trial_id) + 1
            print()
            print("Search: Running Trial #{}".format(self.trial_number))
            print()

            self.trial_start = time.time()
            if self.search_start is None:
                self.search_start = time.time()

            self.show_hyperparameter_table(trial)
            print()

    def on_trial_end(self, trial):
        if self.verbose >= 1:
            utils.try_clear()

            time_taken_str = self.format_time(time.time() - self.trial_start)
            print("Trial {} Complete [{}]".format(self.trial_number, time_taken_str))

            if trial.score is not None:
                print("{}: {}".format(self.oracle.objective.name, trial.score))

            print()
            best_trials = self.oracle.get_best_trials()
            if len(best_trials) > 0:
                best_score = best_trials[0].score
            else:
                best_score = None
            print(
                "Best {} So Far: {}".format(self.oracle.objective.name, best_score)
            )

            time_elapsed_str = self.format_time(time.time() - self.search_start)
            print("Total elapsed time: {}".format(time_elapsed_str))

    def show_hyperparameter_table(self, trial):
        template = "{{0:{0}}}|{{1:{0}}}|{{2}}".format(self.col_width)
        best_trials = self.oracle.get_best_trials()
        if len(best_trials) > 0:
            best_trial = best_trials[0]
        else:
            best_trial = None
        if trial.hyperparameters.values:
            print(template.format("Value", "Best Value So Far", "Hyperparameter"))
            for hp, value in trial.hyperparameters.values.items():
                if best_trial:
                    best_value = best_trial.hyperparameters.values.get(hp)
                else:
                    best_value = "?"
                print(
                    template.format(
                        self.format_value(value),
                        self.format_value(best_value),
                        hp,
                    )
                )
        else:
            print("default configuration")

    def format_value(self, val):
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return "{:.5g}".format(val)
        else:
            val_str = str(val)
            if len(val_str) > self.col_width:
                val_str = val_str[: self.col_width - 3] + "..."
            return val_str

    def format_time(self, t):   # t in seconds
        #return time.strftime("%Y-%m-%d %Hh %Mm %Ss", time.gmtime(t))

        day = t // (24*3600)
        time = t % (24*3600)
        hour = time // 3600
        time = time % 3600
        min = time // 60
        time = time % 60
        sec = time

        ret = "{:.0f}d {:.0f}h {:.0f}m {:.0f}s".format(day,hour,min,sec)
        return ret



# customized tuner callback
# based on tuner_utils.py
class TunerCallback(keras.callbacks.Callback):
    def __init__(self, tuner, trial):
        super(TunerCallback, self).__init__()
        self.tuner = tuner
        self.trial = trial


        # sspark
        # spike count
        self.list_spike_count = collections.OrderedDict()
        self.spike_count_total = 0
        self.spike_count_total_best = 0

        # tensorboard log dir
        self.log_dir = config.path_tensorboard
        self.writer = tf.summary.create_file_writer(self.log_dir)



    def on_train_begin(self, logs):
        #self.tuner.on_train_begin(self.trial, self.model, epoch, logs=logs)

        if conf.nn_mode=='SNN':
            self.model.init()
            glb_t.reset()
            lib_snn.proc.reset(self)

    def on_epoch_begin(self, epoch, logs):
        pass
        #self.tuner.on_epoch_begin(self.trial, self.model, epoch, logs=logs)

        if conf.nn_mode=='SNN':
            lib_snn.proc.spike_count_epoch_init(self)
            lib_snn.proc.reset(self)

    def on_batch_begin(self, batch, logs):
        #self.tuner.on_batch_begin(self.trial, self.model, batch, logs)
        if conf.nn_mode=='SNN':
            lib_snn.proc.reset_batch_snn(self)

    def on_batch_end(self, batch, logs):
        pass
        #self.tuner.on_batch_end(self.trial, self.model, batch, logs)

        #if conf.nn_mode=='SNN':
        #    lib_snn.proc.spike_count_batch_end_one_device(self)

    def on_epoch_end(self, epoch, logs):
        #self.tuner.on_epoch_end(self.trial, self.model, epoch, logs=logs)

        if conf.nn_mode=='SNN':
            lib_snn.proc.spike_count_epoch_end(self,epoch,logs,config.test_ds_num)

    def on_test_begin(self, logs):
       # self.tuner.on_test_begin(self.trial, self.model, epoch, logs=logs)

        if conf.nn_mode=='SNN':
            self.model.init()
            glb_t.reset()
            lib_snn.proc.reset(self)
            lib_snn.proc.spike_count_epoch_init(self)

    def on_test_batch_begin(self, batch, logs):
        if conf.nn_mode=='SNN':
            lib_snn.proc.reset_batch_snn(self)

    def on_test_batch_end(self, batch, logs):
        if conf.nn_mode=='SNN':
            lib_snn.proc.spike_count_batch_end_one_device(self)

    def on_test_end(self, logs):
        if conf.nn_mode=='SNN' and conf.mode=='inference':
            lib_snn.proc.spike_count_batch_end_one_device(self)
            lib_snn.proc.spike_count_epoch_end(self,0,logs,config.test_ds_num)
