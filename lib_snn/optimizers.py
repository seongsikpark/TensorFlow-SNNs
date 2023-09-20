#import keras_core.src.saving
#import keras_nlp.src.backend.keras.saving
import tensorflow as tf

#import keras
#from keras_nlp.src.backend import keras
#from tensorflow.keras import utils
from lib_snn import keras


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
        self.decay_step = decay_step
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
