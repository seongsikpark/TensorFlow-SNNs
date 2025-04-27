import os

import matplotlib
import glob


# configuration
from config_snn_nas import config


import autokeras_custom.auto_model

# import lib_snn.callbacks

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import autokeras as ak
import autokeras_custom as akc

#import autokeras_custom as akc
#import autokeras_custom.a as akc
#from autokeras_custom.auto_model import AutoModel as akc

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow import keras
import tensorboard
import datasets.datasets as datasets
import datasets.augmentation_cifar as augmentation_path
import gc
#from keras.callbacks import MaxMetric
from keras.optimizers import Adam, SGD
from keras.optimizers.schedules.learning_rate_schedule import CosineDecay
from autokeras import keras_layers
# from lib_snn.callbacks import SNNLIB
# import config
import keras_tuner

import callbacks

from absl import flags
conf = flags.FLAGS


from keras_tuner.engine import hyperparameters

from keras.utils import layer_utils

#
import pandas as pd

#
import lib_snn

from lib_snn.sim import glb_t


#
max_trials = 100
batch_size = 100
epoch = conf.train_epoch_search
#epoch = 1
#learning_rate = 1e-1
learning_rate = conf.learning_rate


#
#max_model_size=1.5E6
#max_model_size=1.0E7
max_model_size=None

#
#model_path = "am/m-1.5e6_t-100_e-10"
#model_path = "am/test"
#model_path = "am/231006_0_Bay_VGG" # BEST
#model_path = "am/231012_0_RAND_VGG"
#model_path = "am/231014_0_Bay_VGG"
#model_path = "am/231020_0_Evo_VGG"
#model_path = "am/231020_0_Hyp_VGG"
#model_path = "am/231023_0_Rand_VGG"
#model_path = "am/231028_0_Rand_VNN"
#model_path = "am/241028_SNN_Bay_VGG"
#model_path = "am/2412_SNN_test"
#model_path = "am/241206_VGG_SNN_SGD_Rand_SG"
#model_path = "am/250305_test"
#model_path = "am/250328_test"
#model_path = "am/250410_inhibitory"
#model_path = "am/250411_inhibitory"

#model_path = "am/250414_inhibitory_tr20"
#model_path = "am/250417_inhibitory_vgg16_tr10"
#model_path = "am/250418_inhibitory_vgg16_tr10"



#model_path = "../02_SNN_training_1/am/250418_rand_inhibitory_fr_tr100"

#model_path = "../02_SNN_training_2/am/250418_rand_inhibitory_fr_tr50"
#model_path = "../02_SNN_training_3/am/250418_rand_inhibitory_fr_tr30"

model_path = "am/testtest"


train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch = datasets.load()
# train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class = \
#     datasets.datasets.load(model_name=None, dataset_name='CIFAR10', input_size=(32, 32, 3), train_type='scratch', train=True, conf=conf, num_parallel_call=15, batch_size=batch_size)
print(num_class)
print(train_ds, train_ds_num)
print(valid_ds, valid_ds_num)

config.train_ds_num = train_ds_num
config.valid_ds_num = valid_ds_num
config.test_ds_num = test_ds_num



# def lr_schedule(epoch):
#     lr = learning_rate
#     if epoch > 90:
#         lr *= 1e-3
#     elif epoch > 60:
#         lr *= 1e-2
#     elif epoch > 30:
#         lr *= 1e-1
#     print('Learning rate: ', lr)
#     return lr
#

#Adam/SGD
# optimizer = Adam(learning_rate=lr_schedule(0))
# optimizer = SGD(learning_rate=lr_schedule(0))


lr_schedule = CosineDecay(initial_learning_rate=learning_rate,
                          decay_steps=epoch,  # 0.1 to 0 in epochs
                          )

# warmup_steps = 5  # warmup 5 epoch
#
# if warmup_steps:
#     lr_schedule = keras_layers.WarmUp(
#         initial_learning_rate=learning_rate,
#         decay_schedule_fn=lr_schedule,
#         warmup_steps=warmup_steps,
#     )

# lr_schedule = lib_snn.optimizers.LRSchedule_step(initial_learning_rate=learning_rate,
#                                                  decay_step=70,
#                                                  decay_factor=0.1)

# lr_schedule = lib_snn.optimizers.LRSchedule_step_wup(initial_learning_rate=learning_rate,
#                                                      decay_step=70,
#                                                      decay_factor=0.1,
#                                                      warmup_step=20)

#lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
#lr_scheduler = lib_snn.optimizers.LRSchedule_step(learning_rate, train_steps_per_epoch * 3, 0.1)
#optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=0.9, name='SGD')

'''
lr_reducer = ReduceLROnPlateau(factor=0.1,
                               cooldown=0,
                               patience=5,
                               min_lr=1e-7,
                               monitor='val_acc')
'''


#callbacks = [ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True), ]


#callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1),
             #lr_reducer,
             #lr_scheduler,
             #tf.keras.callbacks.TensorBoard(log_dir=model_path, write_graph=True, histogram_freq=1), # foldername: 0,1,2 ~~
             #MaxMetric(metrics=['acc'])
#callbacks.callbacks_snn_train(model,train_ds_num,valid_ds,test_ds_num)


# TODO: move
def scheduler(epoch, lr):

    initial_learning_rate = conf.learning_rate
    decay_step = conf.step_decay_epoch
    decay_factor = 0.1

    factor_n = tf.cast(tf.math.floordiv(epoch,decay_step),tf.float32)
    factor = tf.math.pow(decay_factor,factor_n)
    learning_rate = initial_learning_rate*factor

    return learning_rate


callback_lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
#callback_lr_tracker = lib_snn.callbacks.LearningRateTracker()

#callbacks.append(callback_lr_schedule)
#callbacks.append(callback_lr_tracker)

acc = tf.keras.metrics.categorical_accuracy
acc_top5 = tf.keras.metrics.top_k_categorical_accuracy
acc.name = 'acc'
acc_top5.name = 'acc-5'
#metrics = [acc, acc_top5]
metrics = [acc, acc_top5]

loss = tf.keras.losses.CategoricalCrossentropy()

tuner = 'random'
#tuner = 'bayesian'
#tuner = 'greedy'
#tuner = 'evolution'
#tuner = 'hyperband'

filters_64 = hyperparameters.Choice("filters_64", [64], default=64)
filters_128 = hyperparameters.Choice("filters_128", [64, 128], default=128)
filters_256 = hyperparameters.Choice("filters_256", [64, 128, 256], default=256)
filters_512 = hyperparameters.Choice("filters_512", [64, 128, 256, 512], default=512)
#filters = hyperparameters.Choice("filters", [64, 128, 256], default=128)
#filters = [64, 128, 256, 512, 512]
kernel_size= hyperparameters.Choice("kernel_size", [3,5,7], default=3)
#num_layers= hyperparameters.Choice("num_layers", [1,2,3,4,5], default=2)
num_layers= hyperparameters.Choice("num_layers", [1,2,3], default=2)

num_units= hyperparameters.Choice("num_units", [64, 128, 256, 512], default=512)
mask_inh_r=hyperparameters.Choice("mask_inh_r", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], default=0.0)

transfer_learning=True
#transfer_learning=False

#Train_mode = "DNN"
Train_mode = "SNN"


# wrapper
#class (tf.keras.layers.Layer):
class FeatureExtractor(ak.Block):
    #def __init__(self, **kwargs):
        #super().__init__(**kwargs)
    def build(self, hp, inputs):

        if False:
            model = lib_snn.model_builder.model_builder(num_class, train_steps_per_epoch, valid_ds)
            model.load_weights("/home/sspark/Models/SNN/CNN_randaug/VGG16_AP_CIFAR10/ep-0304.hdf5")
            dummy_input = tf.zeros(model.inputs[0].shape)
            model(dummy_input)
            # layer = model.get_layer('flatten')
            feature_extractor = lib_snn.model.Model(model.inputs, model.get_layer("flatten").output, conf.batch_size,
                                                    model.inputs[0].shape)
            feature_extractor.trainable=False
            #feature_extractor.trainable=True

            output = feature_extractor(inputs)
            #feature_extractor.inputs = inputs
            #output = feature_extractor.output

        model = lib_snn.model_builder.model_builder(num_class, train_steps_per_epoch, valid_ds)
        model.load_weights("/home/sspark/Models/SNN/CNN_randaug/VGG16_AP_CIFAR10/ep-0304.hdf5")
        dummy_input = tf.zeros(model.inputs[0].shape)
        model(dummy_input)
        model.trainable = False

        # layer = model.get_layer('flatten')
        #feature_extractor = lib_snn.model.Model(model.inputs, model.get_layer("flatten").output, conf.batch_size, model.inputs[0].shape)
        #feature_extractor.trainable = False
        # feature_extractor.trainable=True

        #output = feature_extractor(inputs)
        # feature_extractor.inputs = inputs
        # output = feature_extractor.output
        #return output

        last_layer_name = "flatten"
        flat_layers = []
        for layer in model.layers:
            flat_layers.append(layer)
            if layer.name == last_layer_name:
                break


        #
        glb_t.reset()
        x = flat_layers[0](inputs[0])
        for layer in flat_layers[1:]:
            x = layer(x)

        output = x

        return output




if transfer_learning:
    if Train_mode == "SNN":

        #
        # transfer learning
        if False:
            model = lib_snn.model_builder.model_builder(num_class, train_steps_per_epoch, valid_ds)
            model.load_weights("/home/sspark/Models/SNN/CNN_randaug/VGG16_AP_CIFAR10/ep-0304.hdf5")
            dummy_input = tf.zeros(model.inputs[0].shape)
            model(dummy_input)
            # layer = model.get_layer('flatten')
            feature_extractor = lib_snn.model.Model(model.inputs, model.get_layer("flatten").output, conf.batch_size,
                                                    model.inputs[0].shape)
            feature_extractor(dummy_input)

            x = feature_extractor.output
            x = akc.DenseBlock(num_units=num_class, dropout=0, num_layers=1, use_batchnorm=True, tunable=True)(x)
            output_node = akc.ClassificationHead(dropout=0, loss=loss, metrics=metrics, tunable=True)(x)
            input_node = model.inputs

        # before 250428
        if False:
            input_node = akc.ImageInput()
            features = FeatureExtractor()(input_node)
            x = features

        #
        input_node = akc.ImageInput()
        x = FeatureExtractor()(input_node)


        pass


        x = akc.DenseBlock(num_units=num_units, mask_inh_r=mask_inh_r, dropout=0, num_layers=1, use_batchnorm=True, tunable=True, name='fc1')(x)
        x = akc.DenseBlock(num_units=num_units, mask_inh_r=mask_inh_r, dropout=0, num_layers=1, use_batchnorm=True, tunable=True, name='fc2')(x)
        output_node = akc.ClassificationHead(dropout=0, loss=loss, metrics=metrics, tunable=True)(x)


    else:
        assert False


else:
    # DNN_Mode
    if Train_mode == "DNN":
        input_node = akc.ImageInput()
        # print(input_node)
        # assert False
        # input_shape = (32,32,3)
        # input_node = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
        # output_node = lib_snn.layers.InputGenLayer(name='in')(input_node)

        ''' VGG16 model'''
        if False:
            output_node = input_node
            output_node = akc.ConvBlock(dropout=0.2, filters=64, kernel_size=kernel_size, num_blocks=1, num_layers=2, separable=False, max_pooling=True, use_batchnorm=True, tunable=True)(output_node)
            output_node = akc.ConvBlock(dropout=0.2, filters=128, kernel_size=kernel_size, num_blocks=1, num_layers=2, separable=False, max_pooling=True, use_batchnorm=True, tunable=True)(output_node)
            output_node = akc.ConvBlock(dropout=0.2, filters=256,  kernel_size=kernel_size, num_blocks=1, num_layers=3, separable=False, max_pooling=True, use_batchnorm=True, tunable=True)(output_node)
            output_node = akc.ConvBlock(dropout=0.2, filters=512, kernel_size=kernel_size, num_blocks=1, num_layers=3, separable=False, max_pooling=True, use_batchnorm=True, tunable=True)(output_node)
            output_node = akc.ConvBlock(dropout=0.2, filters=512, kernel_size=kernel_size, num_blocks=1, num_layers=3, separable=False, max_pooling=True, use_batchnorm=True, tunable=True)(output_node)
            # output_node = ak.ResNetBlock(pretrained=False, tunable=True)(output_node)
            output_node = akc.Flatten()(output_node)
            output_node = akc.DenseBlock(num_units=512, dropout=0.0, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
            output_node = akc.DenseBlock(num_units=512, dropout=0.0, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
            #output_node = akc.DenseBlock(num_units=num_class, dropout=0, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
            output_node = akc.ClassificationHead(dropout=0, loss=loss, metrics=metrics, tunable=True)(output_node)

        output_node = input_node
        output_node = akc.ConvBlock(dropout=0.2, filters=filters, kernel_size=kernel_size, num_blocks=1, num_layers=num_layers, separable=False, max_pooling=True, use_batchnorm=True, tunable=True)(output_node)
        output_node = akc.ConvBlock(dropout=0.2, filters=filters, kernel_size=kernel_size, num_blocks=1, num_layers=num_layers, separable=False, max_pooling=True, use_batchnorm=True, tunable=True)(output_node)
        output_node = akc.ConvBlock(dropout=0.2, filters=filters, kernel_size=kernel_size, num_blocks=1, num_layers=num_layers, separable=False, max_pooling=True, use_batchnorm=True, tunable=True)(output_node)
        output_node = akc.ConvBlock(dropout=0.2, filters=filters, kernel_size=kernel_size, num_blocks=1, num_layers=num_layers, separable=False, max_pooling=True, use_batchnorm=True, tunable=True)(output_node)
        #output_node = akc.ConvBlock(dropout=0.2, filters=filters[4], kernel_size=kernel_size, num_blocks=1, num_layers=3, separable=False, max_pooling=True, use_batchnorm=True, tunable=True)(output_node)
        # output_node = ak.ResNetBlock(pretrained=False, tunable=True)(output_node)
        output_node = akc.Flatten()(output_node)
        output_node = akc.DenseBlock(num_units=512, dropout=0.0, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
        output_node = akc.DenseBlock(num_units=512, dropout=0.0, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
        #output_node = akc.DenseBlock(num_units=num_class, dropout=0, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
        output_node = akc.ClassificationHead(dropout=0, loss=loss, metrics=metrics, tunable=True)(output_node)



        if False:
            output_node = input_node
            output_node = akc.ConvBlock(use_batchnorm=True, max_pooling=True, separable=False, tunable=True)(output_node)
            # output_node = ak.ResNetBlock(pretrained=False, tunable=True)(output_node)
            output_node = akc.Flatten()(output_node)
            output_node = akc.DenseBlock(use_batchnorm=True, tunable=True)(output_node)
            output_node = akc.ClassificationHead(dropout=0, loss=loss, metrics=metrics, tunable=True)(output_node)


    # SNN_Mode
    if Train_mode == "SNN":

        if False:
            pooling=conf.pooling_vgg
            # output_node = akc.ResNetBlock(pretrained=False, tunable=True)(output_node)

            input_node = akc.ImageInput()
            output_node = input_node
            output_node = akc.ConvBlock(dropout=0, filters=filters, num_blocks=1, num_layers=1, separable=False, pooling=pooling, tunable=True)(output_node)
            #output_node = akc.ConvBlock(dropout=0.5, filters=filters, num_blocks=1, num_layers=1, separable=False, pooling=pooling, tunable=True)(output_node)
            #output_node = akc.ConvBlock(dropout=0.1, filters=filters, num_blocks=1, num_layers=3, separable=False, pooling=pooling, tunable=True)(output_node)
            #output_node = akc.ConvBlock(dropout=0.1, filters=filters, num_blocks=1, num_layers=3, separable=False, pooling=pooling, tunable=True)(output_node)
            #output_node = akc.ConvBlock(dropout=0.1, filters=filters, num_blocks=1, num_layers=1, separable=False, pooling=pooling, tunable=True)(output_node)
            #output_node = akc.Flatten()(output_node)
            #output_node = akc.DenseBlock(num_units=512, dropout=0.1, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
            #output_node = akc.DenseBlock(num_units=512, dropout=0.5, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
            output_node = akc.DenseBlock(num_units=num_class, dropout=0, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
            output_node = akc.ClassificationHead(dropout=0, loss=loss, metrics=metrics, tunable=True)(output_node)

        # VGG16 like
        pooling = conf.pooling_vgg
        input_node = akc.ImageInput()
        output_node = input_node
        output_node = akc.ConvBlock(dropout=0, filters=filters_64, num_blocks=1, num_layers=2, separable=False, pooling=pooling, tunable=True, name='conv1')(output_node)
        output_node = akc.ConvBlock(dropout=0, filters=filters_128, num_blocks=1, num_layers=2, separable=False, pooling=pooling, tunable=True, name='conv2')(output_node)
        output_node = akc.ConvBlock(dropout=0, filters=filters_256, num_blocks=1, num_layers=3, separable=False, pooling=pooling, tunable=True, name='conv3')(output_node)
        output_node = akc.ConvBlock(dropout=0, filters=filters_512, num_blocks=1, num_layers=3, separable=False, pooling=pooling, tunable=True, name='conv4')(output_node)
        output_node = akc.ConvBlock(dropout=0, filters=filters_512, num_blocks=1, num_layers=3, separable=False, pooling=pooling, tunable=True, name='conv5')(output_node)
        output_node = akc.Flatten()(output_node)
        output_node = akc.DenseBlock(num_units=num_units, dropout=0, num_layers=1, use_batchnorm=True, tunable=True, name='fc1')(output_node)
        output_node = akc.DenseBlock(num_units=num_units, dropout=0, num_layers=1, use_batchnorm=True, tunable=True, name='fc2')(output_node)
        output_node = akc.ClassificationHead(dropout=0, loss=loss, metrics=metrics, tunable=True, name='head')(output_node)

clf = akc.auto_model.AutoModel(inputs=input_node, outputs=output_node, overwrite=False,
#clf = ak.auto_model.AutoModel(inputs=input_node, outputs=output_node, overwrite=True,
                               tuner=tuner, max_trials=max_trials, project_name=model_path, objective='val_acc', max_model_size_new=max_model_size)



#
monitor_cri = config.monitor_cri
filepath_save = config.filepath_save
path_tensorboard = config.path_tensorboard

best=0.0

# model checkpoint save and resume
cb_model_checkpoint = lib_snn.callbacks.ModelCheckpointResume(
    filepath=filepath_save + '/ep-{epoch:04d}.hdf5',
    save_weight_only=True,
    save_best_only=True,
    monitor=monitor_cri,
    verbose=1,
    best=best,
    log_dir=path_tensorboard,
    # tensorboard_writer=cb_tensorboard._writers['train']
)

cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=path_tensorboard, update_freq='epoch')

#callbacks_train, callbacks_test = callbacks.callbacks_snn_train(model,train_ds_num,valid_ds,test_ds_num)
#callbacks = callbacks_train


callbacks=[]
callbacks.append(cb_model_checkpoint)
callbacks.append(cb_tensorboard)


#
clf.tuner.metrics = metrics
clf.tuner.loss = loss
# clf.tuner.optimizer = optimizer

# conf = config.flags
# snn_cb = lib_snn.callbacks.SNNLIB(conf, model_path, train_ds_num, valid_ds_num)
# callbacks.append(snn_cb)

#cb_libsnn = lib_snn.callbacks.SNNLIB(config.flags,config.filepath_load,train_ds_num,test_ds_num)
#callbacks.append(cb_libsnn)

#
f_search=True
#f_search=False
if f_search:
    # batch_size already in dataset
    hist = clf.fit(train_data=train_ds, validation_data=valid_ds, epochs=epoch, callbacks=callbacks)

else:
    dataset = train_ds
    validation_split = 0
    epochs = None

    # input pipeline setting
    #self._analyze_data(dataset)
    #self._build_hyper_pipeline(dataset)
    clf._analyze_data(dataset)
    clf._build_hyper_pipeline(dataset)


    # load and analysis
    tuner = clf.tuner

    trials_dict = tuner.oracle.trials
    trials_dict_sorted = sorted(trials_dict.items())

    list_num_para = []
    list_acc = []
    list_hp = []

    for key, trial in trials_dict_sorted:
        hp = trial.hyperparameters
        score = trial.score


        tuner.hypermodel.set_fit_args(validation_split, epochs=epochs)

        build_for_trial=False
        if build_for_trial:
            tuner._prepare_model_build(hp, x=dataset)
            model = tuner._build_hypermodel(hp)
            num_parameters = layer_utils.count_params(model.trainable_weights)
        else:
            num_parameters=0

        list_acc.append(score)
        list_num_para.append(num_parameters)
        list_hp.append(hp.values)

        best_s_count = trial.metrics.metrics['best_s_count']._observations[0].value[0]
        print(hp.values)
        print("{:} - # of para: {:.3e}, acc: {:.2f}, best_s_count: {:.0f}".format(key, num_parameters, score * 100,best_s_count))
        print("")


    #df = pd.DataFrame()
    df = pd.DataFrame
    df = pd.DataFrame(list_acc)
    df.to_excel('text_out.xlsx')





if False:
    ## cant export because Activation name conflict
    model = clf.export_model()
    # print(model.summary())

    best_epoch = clf.tuner._get_best_trial_epochs()
    print(best_epoch, "best_trial_epochs@@@@@")
    trials = clf.tuner.oracle.get_best_trials(num_trials=max_trials)
    print(trials, "Best_Trials@@@@@")
    print(clf.tuner.get_best_pipeline(), "get best pipeline @@@@@")

    try:
        keras.models.Model.save(self=model, filepath=model_path, save_format="tf")
        print("@@DONE1@@")
    except Exception:
        keras.models.Model.save(self=model, filepath=model_path+'.h5')
        print("@@DONE1_h5@@")


    # load saved model
    try:
        loaded_Model = keras.models.load_model(filepath=model_path, custom_objects=ak.CUSTOM_OBJECTS)
        print("@@DONE2@@")
    except Exception:
        loaded_Model = keras.models.load_model(filepath=model_path+'.h5', custom_objects=ak.CUSTOM_OBJECTS)
        print("@@DONE2_h5@@")
    loaded_Model.summary()
    print(loaded_Model.evaluate(valid_ds, verbose=1), "loaded_model")

    # add warmup for hyperband tuner
    warmup_steps = 5  # warmup 5 epoch
    if warmup_steps:
        lr_schedule = keras_layers.WarmUp(
            initial_learning_rate=learning_rate,
            decay_schedule_fn=lr_schedule,
            warmup_steps=warmup_steps,
        )
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    callbacks[3] = lr_scheduler

    loaded_Model.fit(train_ds, validation_data=valid_ds, epochs=epoch, callbacks=callbacks, verbose=1)


    print("@@DONE3@@")
