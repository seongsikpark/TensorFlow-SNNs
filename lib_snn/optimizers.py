
import tensorflow as tf

#@tf.function
class LRSchedule_step(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, decay_step, decay_factor):
        self.initial_learning_rate = initial_learning_rate
        #self.learning_rate = tf.Tensor(self.initial_learning_rate,shape=(),dtype=tf.float32)
        #self.learning_rate = tf.Tensor(self.initial_learning_rate,value_index=(),dtype=tf.float32)
        #self.learning_rate = self.initial_learning_rate
        #self.laeraning_rate = tf.constant(self.initial_learning_rate)
        self.decay_step = decay_step
        self.decay_factor = decay_factor

    def __call__(self,step):
        mod = tf.math.floormod(step,self.decay_step)
        factor_n = tf.math.divide(step,self.decay_step)
        cond = tf.math.equal(mod,0)

        factor = tf.where(cond,tf.math.pow(self.decay_factor,factor_n),1.0)
        learning_rate = self.initial_learning_rate*factor

        return learning_rate


    def get_config(self):
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_step': self.decay_step,
            'decay_fact-r': self.decay_factor,
        }

        return config