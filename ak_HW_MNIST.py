import os
from tensorflow.python.keras.callbacks import TensorBoard
import config_snn_nas
from config import config
# from config_snn_nas import config
import tensorboard
from tensorflow import keras
import datasets.datasets as datasets
import datasets.mnist as mnist
import tensorflow as tf
import keras
import autokeras as ak
from autokeras import keras_layers
import autokeras_custom as akc
import autokeras_custom.auto_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers.schedules.learning_rate_schedule import CosineDecay
import lib_snn
import time
from keras.callbacks import MaxMetric
from lib_snn.optimizers import LRSchedule_step, LRSchedule_step_wup
import matplotlib.pyplot as plt

#
import tensorflow_datasets as tfds

os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'

max_trial = config.flags.max_trial
learning_rate = config.flags.lr
model_path = config.flags.model_path
epoch = 1

# train_ds, valid_ds = tf.keras.datasets.mnist.load_data()
# train_ds = train_ds[:1000]
# valid_ds = valid_ds[:1000]
# print(train_ds)
# print(valid_ds)

#train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num = mnist.load(conf=config.flags)
train_ds, valid_ds, test_ds = tfds.load('mnist',split=['train,valid,test'])
# (train_images, train_labels), valid_ds = tf.keras.datasets.mnist.load_data()

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# train_labels = train_labels[:1000]
# test_labels = test_labels[:1000]
#
# train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
# test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# set metrics
metric_name_acc = 'acc'
monitor_cri = 'val_' + metric_name_acc

acc = tf.keras.metrics.categorical_accuracy
acc_top5 = tf.keras.metrics.top_k_categorical_accuracy
acc.name = 'acc'
acc_top5.name = 'acc-5'
metrics = [acc, acc_top5]
loss = tf.keras.losses.CategoricalCrossentropy()
# metric_name_acc = 'acc'
# monitor_cri = 'val_' + metric_name_acc
#
# lr_schedule = LRSchedule_step(initial_learning_rate=0.1, decay_step=200, decay_factor=0.1)
# lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
# # lr_reducer = ReduceLROnPlateau(factor=0.1,
# #                                cooldown=0,
# #                                patience=5,
# #                                min_lr=1e-7,
# #                                monitor='val_acc')
#
# cb_cp = ModelCheckpoint(filepath=model_path, monitor=monitor_cri, verbose=1,
#                         # save_best_only=True,
#                         # save_weights_only=True,
#                         )
# cb_es = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=1000, verbose=1)
# cb_tb = keras.callbacks.TensorBoard(log_dir=model_path+'/tensorboard', write_graph=True, histogram_freq=1)  # file_name: 0,1,2 ~~
# callbacks = [cb_cp,
#              # cb_tb,
#              # cb_es,
#              # lr_reducer,
#              # lr_scheduler,
#              # MaxMetric,
#              ]
#
# acc = tf.keras.metrics.SparseCategoricalAccuracy()
# metrics = [acc]
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
#
def create_model():
    # cp = tf.train.latest_checkpoint("am/auto_model_38")
    # print(cp)
    model = tf.keras.Sequential([
        # tf.keras.layers.InputLayer(input_shape=(784,)),
        # keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        # lib_snn.layers.Dense(512, activation='relu', input_shape=(784,), last_layer=True),
        # keras.layers.Dropout(0.2),
        # keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=metrics)
    return model


# save_path = "am/auto_model_38"
load_model = create_model()

# load_model.save_weights(save_path.format(epoch=2))
load_model.summary()
clf = load_model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels),
                     callbacks=callbacks, verbose=1)
model = clf.export_model()
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
# epoch = 1
# retrain_epoch = 1

# load dataset
# train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch = datasets.load()
# train_ds_x = train_ds.element_spec[0]
# valid_ds_x = valid_ds.element_spec[0]
# print(train_ds_x.shape, valid_ds_x.shape, " = x_train, val shape")


# make AutoModel
###############################################################################################
###############################################################################################
# input_node = akc.ImageInput()
# output_node = input_node
# output_node = akc.ConvBlock(dropout=0, num_blocks=1, num_layers=1, separable=False, max_pooling=False, tunable=True)(output_node)
# output_node = akc.ConvBlock(dropout=0.5, num_blocks=1, num_layers=1, separable=False, max_pooling=False, tunable=True)(output_node)
# output_node = akc.ConvBlock(dropout=0.1, num_blocks=1, num_layers=3, separable=False, max_pooling=True, tunable=True)(output_node)
# output_node = akc.ConvBlock(dropout=0.1, num_blocks=1, num_layers=3, separable=False, max_pooling=False, tunable=True)(output_node)
# output_node = akc.ConvBlock(dropout=0.1, num_blocks=1, num_layers=1, separable=False, max_pooling=True, tunable=True)(output_node)
# output_node = akc.DenseBlock(num_layers=1, use_batchnorm=True, tunable=True)(output_node)
# output_node = akc.DenseBlock(num_layers=1, use_batchnorm=True, tunable=True)(output_node)
# output_node = akc.ClassificationHead(num_classes=10, dropout=0, loss=loss, metrics=metrics, tunable=True)(output_node)
#
# clf = akc.auto_model.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, tuner="bayesian",
#                                max_trials=max_trial, project_name=model_path, objective="val_acc")
# dont erase
# clf.tuner.metrics = metrics
# clf.tuner.loss = loss
###############################################################################################
###############################################################################################


# set callbacks
###############################################################################################
###############################################################################################
# lr_schedule = CosineDecay(initial_learning_rate=learning_rate,
#                           decay_steps=epoch,  # 0.1 to 0 in epochs
#                           )
# warmup_steps = 5  # warmup 5 epoch
# if warmup_steps:
#     lr_schedule = keras_layers.WarmUp(
#         initial_learning_rate=learning_rate,
#         decay_schedule_fn=lr_schedule,
#         warmup_steps=warmup_steps,
#     )
# lr_schedule = LRSchedule_step(initial_learning_rate=0.1, decay_step=200, decay_factor=0.1)
# lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
# lr_reducer = ReduceLROnPlateau(factor=0.1,
#                                cooldown=0,
#                                patience=5,
#                                min_lr=1e-7,
#                                monitor='val_acc')

# cb_lib_snn = lib_snn.callbacks.SNNLIB(config.flags, config.filepath_load, 60000, 10000)
# cb_lib_snn = lib_snn.callbacks.SNNLIB(config.flags, config.filepath_load, train_ds_num, valid_ds_num)
# cb_lib_snn = lib_snn.callbacks.SNNLIB(config.flags, model_path, train_ds_num, valid_ds_num)
# cb_cp = ModelCheckpoint(filepath=model_path, monitor=monitor_cri, verbose=1, save_best_only=True,
#                         save_weights_only=True
#                         )
# cb_search_es = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1)
# cb_es = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=1000, verbose=1)
# cb_tb = keras.callbacks.TensorBoard(log_dir=model_path+'/tensorboard', write_graph=True, histogram_freq=1)  # file_name: 0,1,2 ~~
# callbacks_search = [cb_cp,
#                     cb_tb,
#                     cb_search_es,
#                     lr_scheduler,
#                     ]
# callbacks = [cb_cp,
#              cb_tb,
#              cb_es,
#              # lr_reducer,
#              lr_scheduler,
#              # MaxMetric,
#              ]
# if config.flags.nn_mode == 'SNN':
#     callbacks.append(cb_lib_snn)
###############################################################################################
###############################################################################################


# check start time
###############################################################################################
now = time.localtime()
start_time = time.time()
start = time.strftime('%Y/%m/%d %I:%M:%S %p', now)
print("Start: ", start)
###############################################################################################


# train AutoModel
# train_model = clf.fit(train_data=train_ds, validation_data=valid_ds, epochs=epoch, callbacks=callbacks_search)
# train_model = clf.fit(x=train_images, y=train_labels, batch_size=100, validation_data=valid_ds, epochs=epoch, callbacks=callbacks_search)
# model = clf.export_model()


# check trial time
###############################################################################################
now = time.localtime()
trial_time = time.time()
trial_end = time.strftime('%Y/%m/%d %I:%M:%S %p', now)
print("Trial_End: ", trial_end)
time_take = time.gmtime(trial_time-start_time)
day = time_take.tm_mday-1
hour = time_take.tm_hour
minute = time_take.tm_min
sec = time_take.tm_sec
print("Trial time: ", day, "Day", hour, "Hour", minute, "Min", sec, "Sec")
###############################################################################################


# get best model and retrain
###############################################################################################
###############################################################################################
# best_model = clf.tuner.get_best_model()
# hist = best_model.fit(train_ds, validation_data=valid_ds, epochs=retrain_epoch, callbacks=callbacks)
# hist = best_model.fit(x=train_images, y=train_labels, batch_size=100,validation_data=valid_ds, epochs=retrain_epoch, callbacks=callbacks)
###############################################################################################
###############################################################################################


# check total time
###############################################################################################
now = time.localtime()
end_time = time.time()
end = time.strftime('%Y/%m/%d %I:%M:%S %p', now)
print("Total End: ", end)
time_take = time.gmtime(end_time-start_time)
day = time_take.tm_mday-1
hour = time_take.tm_hour
minute = time_take.tm_min
sec = time_take.tm_sec
print("Total time: ", day, "Day", hour, "Hour", minute, "Min", sec, "Sec")
###############################################################################################


# show plt
###############################################################################################
def vis(history, name):
    plt.title(f'{name.upper()}')
    plt.xlabel('epochs')
    plt.ylabel(f'{name.lower()}')
    value = history.history.get(name)
    val_value = history.history.get(f'val_{name}', None)
    epochs = range(1, len(value)+1)
    plt.plot(epochs, value, 'b-', label=f'training {name}')
    if val_value is not None:
        plt.plot(epochs, val_value, 'r-', label=f'validation {name}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.05, 1.2), fontsize=10, ncol=1)


def plot_history(history):
    key_value = list(set([i.split('val_')[-1] for i in list(history.history.keys())]))
    plt.figure(figsize=(12, 4))
    for idx, key in enumerate(key_value):
        plt.subplot(1, len(key_value), idx+1)
        vis(history, key)
    plt.tight_layout()
    plt.show()


plot_history(hist)
###############################################################################################

best = clf.tuner.oracle.get_best_trials(max_trial)
best[0].summary()
print("-> Best Architecture")
print("Best Trial Num: ", best[0].trial_id)

# save model
###############################################################################################
if config.flags.nn_mode == 'ANN':
    tf.keras.models.Model.save(self=model, filepath=model_path+'/model.h5', save_format="h5")
    print("@@DONE1_DNN@@")
else:
    tf.keras.models.Model.save(self=model, filepath=model_path+'/model.h5', save_format='h5')
    print("@@DONE1_SNN@@")
model.summary()
#############################################################################################

# load model
# ##############################################################################################
# try:
#     loaded_Model = tf.keras.models.load_model(filepath=model_path)
#     print("@@DONE2@@")
# except Exception:
#     loaded_Model = tf.keras.models.load_model(filepath=model_path + '.h5')
#     print("@@DONE2_h5@@")
# loaded_Model.summary()
# print(loaded_Model.evaluate(valid_ds), "loaded_model")
# ##############################################################################################

