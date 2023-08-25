import os

import matplotlib
import glob

#
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
##GPU_NUMBER=1
#os.environ["CUDA_VISIBLE_DEVICES"]="5"


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

#
import lib_snn

max_trials = 100
batch_size = 100
epoch = 10
model_path = "am/t-100_e-10"
learning_rate = 1e-1


train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch = datasets.load()
# train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class = \
#     datasets.datasets.load(model_name=None, dataset_name='CIFAR10', input_size=(32, 32, 3), train_type='scratch', train=True, conf=conf, num_parallel_call=15, batch_size=batch_size)
print(num_class)
print(train_ds, train_ds_num)
print(valid_ds, valid_ds_num)


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

#callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1),
callbacks = [ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True),
             #lr_reducer,
             #lr_scheduler,
             #tf.keras.callbacks.TensorBoard(log_dir=model_path, write_graph=True, histogram_freq=1), # foldername: 0,1,2 ~~
             #MaxMetric(metrics=['acc'])
             ]

acc = tf.keras.metrics.categorical_accuracy
acc_top5 = tf.keras.metrics.top_k_categorical_accuracy
acc.name = 'acc'
acc_top5.name = 'acc-5'
metrics = [acc, acc_top5]

loss = tf.keras.losses.CategoricalCrossentropy()

#tuner = 'random'
tuner = 'bayesian'

Train_mode = "DNN"
#Train_mode = "SNN"

# DNN_Mode
if Train_mode == "DNN":
    input_node = akc.ImageInput()
    # print(input_node)
    # assert False
    # input_shape = (32,32,3)
    # input_node = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    # output_node = lib_snn.layers.InputGenLayer(name='in')(input_node)

    ''' VGG like model
    output_node = input_node
    output_node = ak.ConvBlock(dropout=0, num_blocks=1, num_layers=1, separable=False, max_pooling=False, tunable=True)(output_node)
    output_node = ak.ConvBlock(dropout=0.5, num_blocks=1, num_layers=1, separable=False, max_pooling=False, tunable=True)(output_node)
    output_node = ak.ConvBlock(dropout=0.1, num_blocks=1, num_layers=3, separable=False, max_pooling=True, tunable=True)(output_node)
    output_node = ak.ConvBlock(dropout=0.1, num_blocks=1, num_layers=3, separable=False, max_pooling=False, tunable=True)(output_node)
    output_node = ak.ConvBlock(dropout=0.1, num_blocks=1, num_layers=1, separable=False, max_pooling=True, tunable=True)(output_node)
    # output_node = ak.ResNetBlock(pretrained=False, tunable=True)(output_node)
    output_node = ak.Flatten()(output_node)
    output_node = ak.DenseBlock(num_units=512, dropout=0.1, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
    output_node = ak.DenseBlock(num_units=512, dropout=0.5, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
    output_node = ak.DenseBlock(num_units=num_class, dropout=0, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
    output_node = ak.ClassificationHead(dropout=0, loss=loss, metrics=metrics, tunable=True)(output_node)
    '''

    output_node = input_node
    output_node = akc.ConvBlock(use_batchnorm=True, max_pooling=True, separable=False, tunable=True)(output_node)
    # output_node = ak.ResNetBlock(pretrained=False, tunable=True)(output_node)
    output_node = akc.Flatten()(output_node)
    output_node = akc.DenseBlock(use_batchnorm=True, tunable=True)(output_node)
    output_node = akc.ClassificationHead(dropout=0, loss=loss, metrics=metrics, tunable=True)(output_node)


# SNN_Mode
if Train_mode == "SNN":
    input_node = akc.ImageInput()
    output_node = input_node
    output_node = akc.ConvBlock(dropout=0, num_blocks=1, num_layers=1, separable=False, max_pooling=False, tunable=True)(output_node)
    output_node = akc.ConvBlock(dropout=0.5, num_blocks=1, num_layers=1, separable=False, max_pooling=False, tunable=True)(output_node)
    output_node = akc.ConvBlock(dropout=0.1, num_blocks=1, num_layers=3, separable=False, max_pooling=True, tunable=True)(output_node)
    output_node = akc.ConvBlock(dropout=0.1, num_blocks=1, num_layers=3, separable=False, max_pooling=False, tunable=True)(output_node)
    output_node = akc.ConvBlock(dropout=0.1, num_blocks=1, num_layers=1, separable=False, max_pooling=True, tunable=True)(output_node)
    # output_node = akc.ResNetBlock(pretrained=False, tunable=True)(output_node)
    output_node = akc.Flatten()(output_node)
    output_node = akc.DenseBlock(num_units=512, dropout=0.1, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
    output_node = akc.DenseBlock(num_units=512, dropout=0.5, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
    output_node = akc.DenseBlock(num_units=num_class, dropout=0, num_layers=1, use_batchnorm=True, tunable=True)(output_node)
    output_node = akc.ClassificationHead(dropout=0, loss=loss, metrics=metrics, tunable=True)(output_node)


clf = akc.auto_model.AutoModel(inputs=input_node, outputs=output_node, overwrite=True,
#clf = ak.auto_model.AutoModel(inputs=input_node, outputs=output_node, overwrite=True,
                               tuner=tuner, max_trials=max_trials, project_name=model_path, objective='val_acc', max_model_size_new=1e6)

clf.tuner.metrics = metrics
clf.tuner.loss = loss
# clf.tuner.optimizer = optimizer

# conf = config.flags
# snn_cb = lib_snn.callbacks.SNNLIB(conf, model_path, train_ds_num, valid_ds_num)
# callbacks.append(snn_cb)

# batch_size already in dataset
hist = clf.fit(train_data=train_ds, validation_data=valid_ds, epochs=epoch, callbacks=callbacks)

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
