import os
from config_snn_nas import config
import autokeras as ak
import datasets.datasets as datasets
from tensorflow import keras
from lib_snn.optimizers import LRSchedule_step, LRSchedule_step_wup
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import matplotlib.pyplot as plt

# model_path = config.flags.model_path
model_path = "evo_p20c5/5/best_model"

train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch = datasets.load()

# print(num_class)
# print(train_ds, train_ds_num)
# print(valid_ds, valid_ds_num, "\n")
#
# train_ds_x = train_ds.element_spec[0]
# train_ds_y = train_ds.element_spec[1]
# valid_ds_x = valid_ds.element_spec[0]
# valid_ds_y = valid_ds.element_spec[1]
# print(train_ds_x, valid_ds_x, " = train_ds_x, valid_ds_x")
# print(train_ds_y, valid_ds_y, " = train_ds_y, valid_ds_y\n")
# print(train_ds_x.shape, train_ds_y.shape, "train_ds_x shape, train_ds_y shape")
# print(valid_ds_x.shape, valid_ds_y.shape, "valid_ds_x shape, valid_ds_y shape\n")
#
# # ev_algo_og_dataset
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_california_housing
# house_dataset = fetch_california_housing()
# data = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)
# target = pd.Series(house_dataset.target, name="MEDV")
#
# X_train, X_test, y_train, y_test = train_test_split(
#     data, target, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.2, shuffle=False)
#
# print(X_train.shape, X_test.shape, X_val.shape, "x shape, train test val")
# print(y_train.shape, y_test.shape, y_val.shape, "y shape, train test val")
# print(X_train, "\n\n", y_train)

# load saved model
try:
    loaded_Model = keras.models.load_model(filepath=model_path, custom_objects=ak.CUSTOM_OBJECTS)
    # loaded_Model = keras.models.load_model(filepath=model_path)
    print("@@DONE@@")
except Exception:
    loaded_Model = keras.models.load_model(filepath=model_path, custom_objects=ak.CUSTOM_OBJECTS)
    print("@@DONE_h5@@")
# loaded_Model.summary()
print(loaded_Model.evaluate(valid_ds), "loaded_model")


###############################################################################################
lr_schedule = LRSchedule_step(initial_learning_rate=0.1, decay_step=200, decay_factor=0.1)
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
# lr_reducer = ReduceLROnPlateau(factor=0.1,
#                                cooldown=0,
#                                patience=5,
#                                min_lr=1e-7,
#                                monitor='val_acc')

cb_cp = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=True,
                        save_weights_only=True
                        )
cb_es = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=1000, verbose=1)
# cb_tb = keras.callbacks.TensorBoard(log_dir=model_path+'/tensorboard', write_graph=True, histogram_freq=1)  # file_name: 0,1,2 ~~
callbacks = [cb_cp,
             # cb_tb,
             cb_es,
             # lr_reducer,
             lr_scheduler,
             ]

hist = loaded_Model.fit(train_ds, validation_data=valid_ds, epochs=600, callbacks=callbacks)
print(loaded_Model.evaluate(valid_ds), "loaded_model_val")
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
