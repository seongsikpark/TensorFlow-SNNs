


import os

## GPU setting
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
##os.environ["CUDA_VISIBLE_DEVICES"]="0,4"
#os.environ["CUDA_VISIBLE_DEVICES"]="6"

import tensorflow as tf
import tensorflow_datasets as tfds

import keras
from keras import layers, callbacks

import config_snn_training
from absl import flags
conf = flags.FLAGS

import config_loss_vis
#conf.train_epoch = 10

import datasets
import lib_snn
from lib_snn import loss_vis

import matplotlib.pyplot as plt

from config import config

from datasets.augmentation_cifar import resize_with_crop
#

#
#input_shape=(28,28,1)
#batch_size = 100
#train_ds, valid_ds = tfds.load('mnist',split=['train','test'])

#train_ds = train_ds.map(lambda image, label: resize_with_crop(image,label,input_shape,1,10,'torch'))
#train_ds = train_ds.batch(batch_size)
#train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
#valid_ds = valid_ds.batch(batch_size)
#valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)





# model
model_name = conf.model

# dataset
dataset_name = conf.dataset

train_type = conf.train_type

NUM_PARALLEL_CALL=16

train= (conf.mode=='train') or (conf.mode=='load_and_train')

image_shape = lib_snn.utils_vis.image_shape_vis(model_name, dataset_name)

# TODO: batch size calculation unification
#batch_size_inference = batch_size_inference_sel.get(model_name,256)
batch_size_train = conf.batch_size
if train:
    batch_size_inference = batch_size_train
else:
    if conf.full_test:
        batch_size_inference = conf.batch_size_inf
    else:
        if conf.batch_size_inf > conf.num_test_data:
            batch_size_inference = conf.num_test_data
        else:
            batch_size_inference = conf.batch_size_inf
    #batch_size_train = batch_size_train_sel.get(model_name,256)

if not conf.full_test:
    assert (conf.num_test_data%batch_size_inference)==0

#
if train:
    batch_size = batch_size_train
else:
    batch_size = batch_size_inference

train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch =\
    datasets.datasets.load(model_name, dataset_name,batch_size,train_type,train,conf,NUM_PARALLEL_CALL)


#label = tf.one_hot(label, num_class)
f_test_model=False
#f_test_model=True
if f_test_model:
    model = keras.Sequential(
        [
            keras.Input(shape=image_shape),
            layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
            layers.MaxPool2D(pool_size=(2,2)),
            layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
            layers.MaxPool2D(pool_size=(2,2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10,activation='softmax')
        ]
    )

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
else:
    model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch)


training_hist = [model.get_weights()]

collect_weights = callbacks.LambdaCallback(
    on_epoch_end=lambda batch, logs: training_hist.append(model.get_weights()) if (batch%10)==0 else (None)
)


filepath_save='./models_tmp'
monitor_cri=config.monitor_cri
best=None
path_tensorboard='./tensorboard'


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
cb_manage_saved_model = lib_snn.callbacks.ManageSavedModels(filepath=filepath_save)
cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=path_tensorboard, update_freq='epoch')



cb_libsnn = lib_snn.callbacks.SNNLIB(conf,config.filepath_load,test_ds_num,None)
callbacks_train = []
#if save_model:
#callbacks_train.append(cb_model_checkpoint)
#callbacks_train.append(cb_manage_saved_model)
callbacks_train.append(cb_libsnn)
callbacks_train.append(cb_model_checkpoint)
#callbacks_train.append(cb_tensorboard)
callbacks_train.append(collect_weights)

#history = model.fit(train_ds, validation_data=valid_ds, epochs=10, callbacks=[collect_weights])
history = model.fit(train_ds, validation_data=valid_ds, epochs=conf.train_epoch, callbacks=callbacks_train)
#history = model.fit(train_ds, validation_data=valid_ds, epochs=10)


#
train_batch, = train_ds.take(1)
x = train_batch[0]
y = train_batch[1]

#
pcoords = loss_vis.PCACoordinates(training_hist)
loss_surface = loss_vis.LossSurface(model,x,y)
#loss_surface = loss_vis.LossSurface(model,train_ds)
#loss_surface.compile(points=30,coords=pcoords,range=0.4)
loss_surface.compile(points=10,coords=pcoords,range=0.1)
#
ax = loss_surface.plot(dpi=150)
loss_vis.plot_training_path(pcoords, training_hist, ax)

plt.show()