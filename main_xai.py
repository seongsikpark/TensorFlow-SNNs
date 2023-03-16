



import re
import datetime
import shutil

from functools import partial

import os

# TF logging setup
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras_tuner
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.engine import data_adapter

#
from tensorflow.python.keras.utils import data_utils

from keras.utils.vis_utils import plot_model

#
from absl import app
from absl import flags

# HP tune
import keras_tuner as kt
#import tensorboard.plugins.hparams import api as hp


#
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#
#import tqdm

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np
#np.set_printoptions(precision=4)
#np.set_printoptions(linewidth=np.inf)
#import tensorflow.experimental.numpy as tnp
#tnp.set_printoptions(linewidth=np.inf)

#import argparse
#import cv2

# configuration
import config
#from config import conf
from config_xai import conf

#config.conf = conf



# snn library
import lib_snn

#
import datasets
#global input_size
#global input_size_pre_crop_ratio
import collections


# TODO: check use
global model_name


#
from lib_snn import config_glb

# ImageNet utils
from models import imagenet_utils

#
#conf = flags.FLAGS


from models.models import model_sel

########################################
# configuration
########################################

# logging - ignore warning
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# GPU setting
#
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#GPU_NUMBER=1
#os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_NUMBER)
#os.environ["CUDA_VISIBLE_DEVICES"]="1,7,0,6"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,4"
#os.environ["CUDA_VISIBLE_DEVICES"]="1,3,5,7"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,4,5,7"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,4"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


exp_set_name = conf.exp_set_name

#
train= (conf.mode=='train') or (conf.mode=='load_and_train')

#
load_model = (conf.mode=='inference') or (conf.mode=='load_and_train')

#
#save_model = False
save_model = True

#
#overwrite_train_model =True
overwrite_train_model=False

#
overwrite_tensorboard = True

#
tf.config.experimental.enable_tensor_float_32_execution(conf.tf32_mode)

#epoch = 20000
#epoch = 20472
train_epoch = 300
#train_epoch = 500
#train_epoch = 1500
#train_epoch = 1000
#train_epoch = 100
#train_epoch = 10
#train_epoch = 1


# learning rate schedule - step_decay
step_decay_epoch = 100
#step_decay_epoch = 200


# TODO: move to config
#
root_hp_tune = './hp_tune'

#
# model
model_name = conf.model

# dataset
dataset_name = conf.dataset

#
learning_rate = conf.learning_rate

#
opt='SGD'

#
#lr_schedule = 'COS'     # COSine
#lr_schedule = 'COSR'    # COSine with Restart
lr_schedule = 'STEP'    # STEP wise
#lr_schedule = 'STEP_WUP'    # STEP wise, warmup


#
root_tensorboard = conf.root_tensorboard


#
#from lib_snn.hp_tune_model import model_builder



# training types
#train_type='finetuning' # not supported yet
#train_type='transfer'
train_type='scratch'


#
assert conf.data_format == 'channels_last', 'not support "{}", only support channels_last'.format(conf.data_format)

########################################
# DO NOT TOUCH
########################################
# l2-norm
lmb = conf.lmb

#
input_size_default = {
    'ImageNet': 224,
    'CIFAR10': 32,
    'CIFAR100': 32,
}

#
input_sizes_imagenet = {
    'Xception': 299,
    'InceptionV3': 299,
    'InceptionResNetV2': 299,
    'NASNetLarge': 331,
    'EfficientNetB1': 240,
    'EfficientNetB2': 260,
    'EfficientNetB4': 380,
    'EfficientNetB5': 456,
    'EfficientNetB6': 528,
    'EfficientNetB7': 600,
}

input_sizes_cifar = {
    'VGG16': 32,
}

#
input_size_sel ={
    'ImageNet': input_sizes_imagenet,
    'CIFAR10': input_sizes_cifar,
    'CIFAR100': input_sizes_cifar,
}

# TODO: integrate input size selector
input_size = input_size_sel[dataset_name].get(model_name,input_size_default[dataset_name])
#input_size = input_sizes.get(model_name,224)


#
initial_channels_sel= {
    'VGG16': 64,
}
initial_channels = initial_channels_sel.get(model_name,64)


# TODO: batch size calulation unification
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
#batch_size = batch_size_train
if train:
    batch_size = batch_size_train
else:
    batch_size = batch_size_inference


#
image_shape = (input_size, input_size, 3)




# model compile
metric_accuracy = tf.keras.metrics.categorical_accuracy
metric_accuracy_top5 = tf.keras.metrics.top_k_categorical_accuracy

# TODO: move to configuration

metric_name_acc = 'acc'
metric_name_acc_top5 = 'acc-5'
monitor_cri = 'val_' + metric_name_acc

metric_accuracy.name = metric_name_acc
metric_accuracy_top5.name = metric_name_acc_top5

#model.compile(optimizer='adam',
              #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              ## metrics=['accuracy'])
              #metrics=[metric_accuracy, metric_accuracy_top5])

################
# name set
################

#
if conf.load_best_model:
    root_model_load = conf.root_model_best
else:
    root_model_load = conf.root_model_load

root_model_save = conf.root_model_save


# TODO: configuration & file naming
#exp_set_name = model_name + '_' + dataset_name
model_dataset_name = model_name + '_' + dataset_name

# path_model = './'+exp_set_name
#path_model = os.path.join(root_model, exp_set_name)
#path_model = os.path.join(root_model, model_dataset_name)
#path_model_load = os.path.join(root_model_load, model_dataset_name)
#path_model_save = os.path.join(root_model_save, model_dataset_name)

#config_glb.path_model_load = path_model_load
#config_glb.path_stat = conf.path_stat


#
use_bn_dict = collections.OrderedDict()
use_bn_dict['VGG16_ImageNet'] = False

#
try:
    conf.use_bn = use_bn_dict[model_dataset_name]
except KeyError:
    pass



# hyperparameter tune name
#hp_tune_name = exp_set_name+'_'+model_dataset_name+'_ep-'+str(train_epoch)
hp_tune_name = exp_set_name

# TODO: functionalize
# file_name='checkpoint-epoch-{}-batch-{}.h5'.format(epoch,batch_size)
# config_name='ep-{epoch:04d}_bat-{}_lmb-{:.1E}'.format(batch_size,lmb)
# config_name='bat-{}_lmb-{:.1E}'.format(batch_size,lmb)

#config_name = 'bat-{}_opt-{}_lr-{:.0E}_lmb-{:.0E}'.format(batch_size,opt,learning_rate,lmb)
config_name = 'ep-{}_bat-{}_opt-{}_lr-{}-{:.0E}_lmb-{:.0E}'.format(train_epoch,batch_size_train,opt,lr_schedule,learning_rate,lmb)

#config_name = 'bat-{}_lmb-{:.0E}'.format(batch_size, lmb)
#config_name = 'bat-512_lmb-{:.1E}'.format(lmb)

if train_type=='transfer':
    config_name += '_tr'
elif train_type=='scratch':
    config_name += '_sc'
    #if n_dim_classifier is not None:
        #if model_name == 'VGG16':
            #config_name = config_name+'-'+str(n_dim_classifier[0])+'-'+str(n_dim_classifier[1])
else:
    assert False

if conf.data_aug_mix == 'mixup':
    en_mixup = True
    en_cutmix = False
elif conf.data_aug_mix == 'cutmix':
    en_mixup = False
    en_cutmix = True
else:
    en_mixup = False
    en_cutmix = False

if en_mixup:
    config_name += '_mu'
elif en_cutmix:
    config_name += '_cm'

#

#if train:
#    filepath = os.path.join(path_model, config_name)
#else:
#    if conf.load_best_model:
#        filepath = path_model
#    else:
#        filepath = os.path.join(path_model, config_name)


# TODO: configuration & file naming
#exp_set_name = model_name + '_' + dataset_name
#model_dataset_name = model_name + '_' + dataset_name

if 'VGG' in model_name:
    if conf.pooling_vgg=='max':
        conf_pool = '_MP'
    elif conf.pooling_vgg=='avg':
        conf_pool = '_AP'
    else:
        assert False
    model_dataset_name = model_name + conf_pool + '_' + dataset_name
else:
    model_dataset_name = model_name + '_' + dataset_name

if conf.name_model_load=='':
    path_model_load = os.path.join(root_model_load, model_dataset_name)
    if conf.load_best_model:
        filepath_load = path_model_load
    else:
        filepath_load = os.path.join(path_model_load, config_name)
else:
    path_model_load = conf.name_model_load
    filepath_load = path_model_load

if conf.name_model_save=='':
    path_model_save = os.path.join(root_model_save, model_dataset_name)
    filepath_save = os.path.join(path_model_save, config_name)
else:
    path_model_save = conf.name_model_save
    filepath_save = path_model_save


####
# glb config set
if conf.path_stat_root=='':
    path_stat_root = path_model_load
else:
    path_stat_root = conf.path_stat_root
#config_glb.path_stat = conf.path_stat
config_glb.path_stat = os.path.join(path_stat_root,conf.path_stat_dir)
config_glb.path_model_load = path_model_load
#config_glb.path_stat = conf.path_stat
config_glb.model_name = model_name
config_glb.dataset_name = dataset_name

#if conf.load_best_model:
#    filepath_load = path_model_load
#else:
#    filepath_load = os.path.join(path_model_load, config_name)
#filepath_save = os.path.join(path_model_save, config_name)



########################################
# load dataset
########################################
# dataset load
NUM_PARALLEL_CALL=-1
#dataset = dataset_sel[dataset_name]
#train_ds, valid_ds, test_ds = dataset.load(dataset_name,input_size,input_size_pre_crop_ratio,num_class,train,NUM_PARALLEL_CALL,conf,input_prec_mode)
train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class = \
    datasets.datasets.load(model_name, dataset_name,batch_size,input_size,train_type,train,conf,NUM_PARALLEL_CALL)


# data-based weight normalization (DNN-to-SNN conversion)
if conf.f_write_stat and conf.f_stat_train_mode:
    test_ds = train_ds

#assert False
train_steps_per_epoch = train_ds.cardinality().numpy()

########################################
# load model
########################################
model_top = model_sel(model_name,train_type)

# TODO: integration - ImageNet
if load_model:
    #if conf.dataset == 'ImageNet':
    if False:
        # ImageNet pretrained model
        load_weight = 'imagenet'
        include_top = True
        add_top = False
    else:
        # get latest saved model
        #latest_model = lib_snn.util.get_latest_saved_model(filepath)

        latest_model = lib_snn.util.get_latest_saved_model(filepath_load)
        load_weight = os.path.join(filepath_load, latest_model)
        #print('load weight: '+load_weight)
        #pre_model = tf.keras.models.load_model(load_weight)

        #latest_model = lib_snn.util.get_latest_saved_model(filepath)
        #load_weight = os.path.join(filepath, latest_model)


        if not latest_model.startswith('ep-'):
            #assert False, 'the name of latest model should start with ''ep-'''
            print('the name of latest model should start with ep-')

            load_weight = tf.train.latest_checkpoint(filepath_load)

            # TODO:
            init_epoch = 0
        else:
            print('load weight: '+load_weight)

            if conf.mode=='inference':
                init_epoch = int(re.split('-|\.',latest_model)[1])
            elif conf.mode=='load_and_train':
                init_epoch = 0
            else:
                assert False


        include_top = True
        add_top = False

        #if train_type == 'transfer':
            #model_top = model_sel_tr[model_name]
        #elif train_type == 'scratch':
            #model_top = model_sel_sc[model_name]
        #else:
            #assert False
else:
    if train_type == 'transfer':
        load_weight = 'imagenet'
        include_top = False
        add_top = True

        #model_top = model_sel_tr[model_name]

    elif train_type == 'scratch':
        load_weight = None
        include_top = True
        add_top = False

        #model_top = model_sel_sc[model_name]
    else:
        assert False

    init_epoch = 0



# TODO: move to parameter
# eager mode
if train:
    eager_mode=False
else:
    if conf.f_write_stat:
        eager_mode=True
        #eager_mode=False
    else:
        eager_mode=False

if conf.debug_mode:
    # TODO: parameterize - debug mode
    eager_mode=True


# for HP tune
model_top_glb = model_top

#
# model builder
model = lib_snn.model_builder.model_builder(
    eager_mode, model_top, batch_size, image_shape, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
    train_epoch, train_steps_per_epoch,
    opt, learning_rate,
    lr_schedule, step_decay_epoch,
    metric_accuracy, metric_accuracy_top5, dataset_name)





########################################
# load model
########################################
model_ann=None
if conf.nn_mode=='SNN' and conf.dnn_to_snn:
    print('DNN-to-SNN mode')
    nn_mode_ori = conf.nn_mode
    conf.nn_mode='ANN'
    model_ann = lib_snn.model_builder.model_builder(
        eager_mode, model_top, batch_size, image_shape, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
        train_epoch, train_steps_per_epoch,
        opt, learning_rate,
        lr_schedule, step_decay_epoch,
        metric_accuracy, metric_accuracy_top5, dataset_name)
    conf.nn_mode=nn_mode_ori

    model_ann.nn_mode = 'ANN'

    #model_ann.set_en_snn('ANN')

    if dataset_name=='ImageNet':
        #imagenet_utils.load_weights(model_name,model_ann)
        model_ann.load_weights(load_weight)
    else:
        model_ann.load_weights(load_weight)
        #model_ann.load_weights(load_weight,by_name=True)

    print('-- model_ann - load done')
    model.load_weights_dnn_to_snn(model_ann)
    #model.load_weights_dnn_to_snn(model_ann,by_name=True)
else:
    #model.load_weights(load_weight)

    #if False:
    import h5py

    with h5py.File(load_weight,'r') as weight:
        w = weight['model_weights']

        for layer in model.layers:
            if isinstance(layer,lib_snn.layers.Conv2D):
                layer.kernel.assign(w[layer.name][layer.name]['kernel:0'])
                layer.bias.assign(w[layer.name][layer.name]['bias:0'])

            if isinstance(layer,lib_snn.layers.Dense):
                layer.kernel.assign(w[layer.name][layer.name]['kernel:0'])
                layer.bias.assign(w[layer.name][layer.name]['bias:0'])

            if isinstance(layer,lib_snn.layers.BatchNormalization):
                layer.beta.assign(w[layer.name][layer.name]['beta:0'])
                layer.gamma.assign(w[layer.name][layer.name]['gamma:0'])
                layer.moving_mean.assign(w[layer.name][layer.name]['moving_mean:0'])
                layer.moving_variance.assign(w[layer.name][layer.name]['moving_variance:0'])


if conf.nn_mode=='ANN' or (conf.nn_mode=='SNN' and train):
    model_ann=None

# remove dir - train model
if not load_model:
    if overwrite_train_model:
        if os.path.isdir(filepath_save):
            shutil.rmtree(filepath_save)

path_tensorboard = os.path.join(root_tensorboard, exp_set_name)
path_tensorboard = os.path.join(path_tensorboard, model_dataset_name)
path_tensorboard = os.path.join(path_tensorboard, config_name)

if not overwrite_tensorboard:
    if os.path.isdir(path_tensorboard):
        date_cur = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
        path_dest_tensorboard = path_tensorboard + '_' + date_cur
        print('tensorboard data already exists')
        print('move {} to {}'.format(path_tensorboard, path_dest_tensorboard))

        shutil.move(path_tensorboard, path_dest_tensorboard)



########
# Callbacks
########

#
if train and load_model:
    print('Evaluate pretrained model')
    assert monitor_cri == 'val_acc', 'currently only consider monitor criterion - val_acc'
    result = model.evaluate(valid_ds)
    idx_monitor_cri = model.metrics_names.index('acc')
    best = result[idx_monitor_cri]
    print('previous best result - {}'.format(best))
else:
    best = None

# model checkpoint save and resume
cb_model_checkpoint = lib_snn.callbacks.ModelCheckpointResume(
    # filepath=filepath + '/ep-{epoch:04d}',
    # filepath=filepath + '/ep-{epoch:04d}.ckpt',
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

#cb_dnntosnn = lib_snn.callbacks.DNNtoSNN()
cb_libsnn = lib_snn.callbacks.SNNLIB(conf,path_model_load,test_ds_num,model_ann)
cb_libsnn_ann = lib_snn.callbacks.SNNLIB(conf,path_model_load,test_ds_num)

#
#callbacks_train = [cb_tensorboard]
callbacks_train = []
if save_model:
    callbacks_train.append(cb_model_checkpoint)
    callbacks_train.append(cb_manage_saved_model)
callbacks_train.append(cb_libsnn)
callbacks_train.append(cb_tensorboard)

callbacks_test = []
# TODO: move to parameters
#dnn_to_snn = True
#if dnn_to_snn:
    #callbacks_test.append(cb_dnntosnn)

callbacks_test = [cb_libsnn]
callbacks_test_ann = [cb_libsnn_ann]

#
if train:
    print('Train mode')

    model.summary()

    train_histories = model.fit(train_ds, epochs=train_epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                                    callbacks=callbacks_train)
else:
    print('Test mode')

    #
    #result = model.evaluate(test_ds, callbacks=callbacks_test)




[imgs, labels], = test_ds.take(1)

sample_idx=6   # horse -> good
sample_idx=10   # horse -> good example
sample_idx=30   # ? -> good
sample_idx=40   # -> good
sample_idx=50   # -> good


img = imgs[sample_idx]
label = labels[sample_idx]

baseline = tf.random.uniform(shape=img.shape,minval=0,maxval=1)


#m_steps = 99
m_steps = 50
#label_decoded=386
label_decoded = tf.argmax(label)

#image_processed = tf.expand_dims(img,axis=0)
img_exp = tf.expand_dims(img,axis=0)
ig_attribution = lib_snn.xai.integrated_gradients(model=model,
                                                  baseline=baseline,
                                                  images=img,
                                                  target_class_idxs=label_decoded,
                                                  m_steps=m_steps)

#_ = lib_snn.xai.plot_image_attributions(baseline,img_processed,ig_attribution)


# 5. Get the gradients of the last layer for the predicted label
grads = lib_snn.integrated_gradients.get_gradients(model,img_exp,top_pred_idx=label_decoded)

#
vis = lib_snn.integrated_gradients.GradVisualizer()

vis.visualize(
    image=img,
    gradients=grads[0].numpy(),
    integrated_gradients=ig_attribution.numpy(),
    clip_above_percentile=99,
    clip_below_percentile=0
)


vis.visualize(
    image=img,
    gradients=grads[0].numpy(),
    integrated_gradients=ig_attribution.numpy(),
    clip_above_percentile=95,
    clip_below_percentile=28,
    morphological_cleanup=True,
    outlines=True
)

plt.show()