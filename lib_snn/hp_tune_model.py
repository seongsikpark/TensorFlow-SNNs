

#
import tensorflow as tf

from keras_tuner import HyperModel

#
import lib_snn

#
#from main_hp_tune import dataset_name
#from main_hp_tune import model_name
##from main_hp_tune import opt
#from main_hp_tune import lr_schedule
#from main_hp_tune import train_epoch
#from main_hp_tune import model_top_glb
#from main_hp_tune import image_shape
#from main_hp_tune import conf
#from main_hp_tune import include_top
#from main_hp_tune import load_weight
#from main_hp_tune import num_class
#from main_hp_tune import train_steps_per_epoch
##from main_hp_tune import step_decay_epoch
#from main_hp_tune import metric_accuracy
#from main_hp_tune import metric_accuracy_top5





########
# HP Tune
########
class CustomHyperModel(HyperModel):
    def __init__(self, args, hps):
        self.args = args
        self.hps = hps

        HyperModel.__init__(self)

    def build(self, hp):
        return model_builder(hp, self.args, self.hps)



def model_builder(hp, args, hps):
    model = model_builder_default(hp, args, hps)

    return model


def model_builder_default(hp, args, hps):

    # args
    model_top = args['model_top']
    batch_size = args['batch_size']
    image_shape = args['image_shape']
    conf = args['conf']
    include_top = args['include_top']
    load_weight = args['load_weight']
    num_class = args['num_class']
    train_steps_per_epoch = args['train_steps_per_epoch']
    metric_accuracy = args['metric_accuracy']
    metric_accuracy_top5 = args['metric_accuracy_top_5']

    #
    hp_dataset = hp.Choice('dataset', values=hps['dataset'])
    hp_model = hp.Choice('model', values=hps['model'])
    hp_optimizer = hp.Choice('optimizer', values=hps['opt'])
    hp_lr_schedule = hp.Choice('lr_schedule', values=hps['lr_schedule'])
    hp_train_epoch = hp.Choice('train_epoch', values=hps['train_epoch'])
    hp_step_decay_epoch = hp.Choice('step_decay_epoch', values=hps['step_decay_epoch'])

    # hp_lmb = hp.Choice('lmb', values = [5e-4, 1e-4, 5e-5])
    hp_lmb = hp.Choice('lmb', values=[1e-4, 5e-5, 1e-5])
    #hp_lmb = hp.Choice('lmb', values=[1e-4, 5e-5])

    # hp_learning_rate = hp.Choice('learning_rate', values = [0.1, 0.2])
    # hp_learning_rate = hp.Choice('learning_rate', values = [0.01, 0.015, 0.02])
    #hp_learning_rate = hp.Choice('learning_rate', values=[0.005])
    hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.005, 0.01, 0.05, 0.1])
    #hp_learning_rate = hp.Choice('learning_rate', values=[0.001])

    hp_initial_channels = hp.Choice('initial_channel', values=[64])


    model = lib_snn.model_builder.model_builder(
        False, model_top, batch_size,  image_shape, conf, include_top, load_weight, num_class, hp_model, hp_lmb, hp_initial_channels,
        hp_train_epoch, train_steps_per_epoch,
        hp_optimizer, hp_learning_rate,
        hp_lr_schedule, hp_step_decay_epoch,
        metric_accuracy, metric_accuracy_top5)

    if not load_weight is None:
        model.load_weights(load_weight)

    return model