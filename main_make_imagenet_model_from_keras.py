
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import re
import datetime
import shutil

from functools import partial

import os

# TF logging setup
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.engine import data_adapter

#
from tensorflow.python.keras.utils import data_utils

#
from absl import app
from absl import flags

# HP tune
#import kerastuner as kt
import keras_tuner as kt
#import tensorboard.plugins.hparams import api as hp


#
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#
#import tqdm

import matplotlib
matplotlib.use('TkAgg')
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
from config import conf

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
from lib_snn.sim import glb_plot
from lib_snn.sim import glb_plot_1
from lib_snn.sim import glb_plot_2

from lib_snn.sim import glb_ig_attributions
from lib_snn.sim import glb_rand_vth
from lib_snn.sim import glb_vth_search_err
from lib_snn.sim import glb_vth_init
from lib_snn.sim import glb_bias_comp

#
from lib_snn import config_glb

#
#conf = flags.FLAGS


import h5py

from models import imagenet_utils



########################################
# configuration
########################################

# logging - ignore warning
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# GPU setting
#
# GPU_NUMBER=2
GPU_NUMBER=5

GPU_PARALLEL_RUN = 1
#GPU_PARALLEL_RUN = 2
#GPU_PARALLEL_RUN = 3

# RTX3090
if GPU_PARALLEL_RUN == 1:
    gpu_mem = -1
    NUM_PARALLEL_CALL = 15
elif GPU_PARALLEL_RUN == 2:
    gpu_mem = 10240
    NUM_PARALLEL_CALL = 7
elif GPU_PARALLEL_RUN == 3:
    gpu_mem = 6144
    NUM_PARALLEL_CALL = 5
else:
    assert False



#
use_bn_dict = collections.OrderedDict()
use_bn_dict['VGG16_ImageNet'] = False



#
model_name = conf.model
dataset_name = conf.dataset
model_dataset_name = model_name + '_' + dataset_name




#
try:
    conf.use_bn = use_bn_dict[model_dataset_name]
except KeyError:
    pass


#
input_size_default = {
    #'ImageNet': 244,
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
    'EfficientNetV2S': 384,
    'EfficientNetV2M': 480,
}

# TODO: integrate input size selector
input_size = input_sizes_imagenet.get(model_name,input_size_default[dataset_name])


#
initial_channels_sel= {
    'VGG16': 64,
}
initial_channels = initial_channels_sel.get(model_name,64)



##########



#if not(conf.exp_set_name is None):
exp_set_name = conf.exp_set_name
#else:
#    exp_set_name = _exp_set_name

# hyperparamter tune mode
hp_tune = True
#hp_tune = False


#
#train=True
#train=False

train= (conf.mode=='train') or (conf.mode=='load_and_train')

# TODO: parameterize
load_model=True
#load_model=False

#
#save_model = False
save_model = True

#
#overwrite_train_model =True
overwrite_train_model=False

#
overwrite_tensorboard = True

#
#tf.config.experimental.enable_tensor_float_32_execution(conf.tf32_mode)
tf.config.experimental.enable_tensor_float_32_execution(False)

#epoch = 20000
#epoch = 20472
#train_epoch = 300
train_epoch = 1000
#train_epoch = 3000
#train_epoch =560
#train_epoch = 1


# learning rate schedule - step_decay
#step_decay_epoch = 100
step_decay_epoch = 200


# TODO: move to config
#
root_hp_tune = './hp_tune'

#
#root_model = './models_trained'
root_model = './models_trained_resnet_relu_debug'
#root_model = conf.root_model

# model
#model_name = 'VGG16'
#model_name = 'ResNet18'
#model_name = 'ResNet20'
#model_name = 'ResNet32'
#model_name = 'ResNet34'
#model_name = 'ResNet50'
#model_name = 'ResNet18V2'
#model_name = 'ResNet20V2'

# dataset
#dataset_name = 'CIFAR10'
#dataset_name = 'CIFAR100'
#dataset_name='ImageNet'

#
#learning_rate = 0.2
#learning_rate = 0.01
learning_rate = conf.learning_rate

#
opt='SGD'

#
#lr_schedule = 'COS'     # COSine
#lr_schedule = 'COSR'    # COSine with Restart
lr_schedule = 'STEP'    # STEP wise
#lr_schedule = 'STEP_WUP'    # STEP wise, warmup


#
#root_tensorboard = './tensorboard/'
root_tensorboard = conf.root_tensorboard





# lr schedule



# models
#from models.vgg16 import VGG16
from models.vgg16_keras_toh5 import VGG16 as VGG16_KERAS

#from models.vgg16_tr import VGG16_TR
#from models.vgg16 import VGG16
#from models.resnet import ResNet18
#from models.resnet import ResNet20
#from models.resnet import ResNet32
#from models.resnet import ResNet34
#from models.resnet import ResNet50
#from models.resnet import ResNet101
#from models.resnet import ResNet152
#from models.resnet import ResNet18V2
#from models.resnet import ResNet20V2

from models.models import model_sel


os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_NUMBER)

# TODO: gpu mem usage - parameterize
# GPU mem usage
#if False:
#if False:
#if True:
if gpu_mem != -1:
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if gpu:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpu[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem)])
        except RuntimeError as e:
            print(e)


# training types
#train_type='finetuning' # not supported yet
#train_type='transfer'
train_type='scratch'


#
#model_name='Xception'
#model_name='VGG16'
#model_name='VGG19'
#model_name='ResNet50'
#model_name='ResNet101'
#model_name='ResNet152'
#model_name='ResNet50V2'
#model_name='ResNet101V2'
#model_name='ResNet152V2'
#model_name='InceptionV3'
#model_name='InceptionResNetV2'
#model_name='MobileNet'
#model_name='MobileNetV2'
#model_name='DenseNet121'
#model_name='DenseNet169'
#model_name='DenseNet201'
#model_name='NASNetMobile'
#model_name='NASNetLarge'
#model_name='EfficientNetB0'
#model_name='EfficientNetB1'
#model_name='EfficientNetB2'
#model_name='EfficientNetB3'
#model_name='EfficientNetB4'
#model_name='EfficientNetB5'
#model_name='EfficientNetB6'
#model_name='EfficientNetB7'



#
assert conf.data_format == 'channels_last', 'not support "{}", only support channels_last'.format(conf.data_format)

########################################
# DO NOT TOUCH
########################################

#
f_hp_tune = train and hp_tune

# data augmentation - mix

# l2-norm
lmb = conf.lmb




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

NUM_PARALLEL_CALL = 15





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


#
if False:
    from tensorflow.keras.applications.vgg16 import VGG16
    m = VGG16(weights=source_model_file)
    m.compile(optimizer='SGD',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[metric_accuracy, metric_accuracy_top5])
    #metrics = [metric_accuracy, metric_accuracy_top5], run_eagerly = False)
    m.evaluate(test_ds)

    #
    data_down_dir = '/home/sspark/Datasets/ImageNet_down/'
    write_dir = '/home/sspark/Datasets/ImageNet'

    #    # Construct a tf.data.Dataset
    download_config = tfds.download.DownloadConfig(
        extract_dir=os.path.join(write_dir, 'extracted'),
        manual_dir=data_down_dir
    )

    #
    download_and_prepare_kwargs = {
        'download_dir': os.path.join(write_dir, 'download'),
        'download_config': download_config
    }

    data_dir = os.path.join(write_dir, 'data')
    dataset_name = 'imagenet2012'

    #
    assert False


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

# path_model = './'+exp_set_name
#path_model = os.path.join(root_model, exp_set_name)
#path_model = os.path.join(root_model, model_dataset_name)
path_model_load = os.path.join(root_model_load, model_dataset_name)
path_model_save = os.path.join(root_model_save, model_dataset_name)


####
# glb config set
if conf.path_stat_root=='':
    path_stat_root = path_model_load
else:
    path_stat_root = conf.path_stat_root
#config_glb.path_stat = conf.path_stat
config_glb.path_stat = os.path.join(path_stat_root,conf.path_stat_dir)
config_glb.path_model_load = path_model_load
config_glb.model_name = model_name
config_glb.dataset_name = dataset_name


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


if conf.load_best_model:
    filepath_load = path_model_load
else:
    filepath_load = os.path.join(path_model_load, config_name)

filepath_save = os.path.join(path_model_save, config_name)


########################################
# load dataset
########################################
# dataset load
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
    if conf.dataset == 'ImageNet':
        load_weight = 'imagenet'
        include_top = True
        add_top = False
    else:
        # get latest saved model
        #latest_model = lib_snn.util.get_latest_saved_model(filepath)

        latest_model = lib_snn.util.get_latest_saved_model(filepath_load)
        load_weight = os.path.join(filepath_load, latest_model)
        print('load weight: '+load_weight)
        #pre_model = tf.keras.models.load_model(load_weight)

        if not latest_model.startswith('ep-'):
            assert False, 'the name of latest model should start with ''ep-'''

        if conf.mode=='inference':
            init_epoch = int(re.split('-|\.',latest_model)[1])
        elif conf.mode=='load_and_train':
            init_epoch = 0
        else:
            assert False

        include_top = True
        add_top = False

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
    metric_accuracy, metric_accuracy_top5,
    dataset_name
    )


# read source model - h5 format


#with h5py.File(source_model_file,'r') as f:
#    print('Keys: %s'%f.keys())


# Step1: download and test model
# test
#if True:
if False:
    # ResNet50
    if model_name=='ResNet50':
        from tensorflow.keras.applications.resnet import ResNet50
        #m = ResNet50(weights='/home/sspark/Models/refs/ImageNet/Keras/ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
        m = ResNet50()
        m.compile(optimizer='SGD',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=[metric_accuracy, metric_accuracy_top5])

        m.evaluate(test_ds)
        assert False

    # ResNet101
    if model_name=='ResNet101':
        from tensorflow.keras.applications.resnet import ResNet101
        m = ResNet101()
        m.compile(optimizer='SGD',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=[metric_accuracy, metric_accuracy_top5])

        m.evaluate(test_ds)
        assert False


    # ResNet152
    if model_name=='ResNet152':
        from tensorflow.keras.applications.resnet import ResNet152
        m = ResNet152()
        m.compile(optimizer='SGD',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[metric_accuracy, metric_accuracy_top5])

        m.evaluate(test_ds)
        assert False


    # MobileNet
    if model_name=='MobileNet':
        from tensorflow.keras.applications.mobilenet import MobileNet
        m = MobileNet(weights='/home/sspark/Models/refs/ImageNet/Keras/MobileNet/mobilenet_1_0_224_tf.h5')
        m.compile(optimizer='SGD',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=[metric_accuracy, metric_accuracy_top5])

        m.evaluate(test_ds)
        assert False

    # MobileNetV2
    if model_name=='MobileNetV2':
        #from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        MobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2
        #m = MobileNetV2(weights='/home/sspark/Models/refs/ImageNet/Keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')
        m = MobileNetV2(weights='imagenet')
        m.compile(optimizer='SGD',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=[metric_accuracy, metric_accuracy_top5])

        m.evaluate(test_ds)
        assert False

    # EfficientNetV2S
    if model_name == 'EfficientNetV2S':
        #from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        from models.imagenet_pretrained_models import EfficientNetV2S

        # m = MobileNetV2(weights='/home/sspark/Models/refs/ImageNet/Keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')
        m = EfficientNetV2S(weights='imagenet')
        m.compile(optimizer='SGD',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[metric_accuracy, metric_accuracy_top5])

        m.evaluate(test_ds)
        assert False

    # EfficientNetV2M
    if model_name == 'EfficientNetV2M':
        # from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        from models.imagenet_pretrained_models import EfficientNetV2M

        # m = MobileNetV2(weights='/home/sspark/Models/refs/ImageNet/Keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')
        m = EfficientNetV2M(weights='imagenet')
        m.compile(optimizer='SGD',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[metric_accuracy, metric_accuracy_top5])

        m.evaluate(test_ds)
        assert False

# TODO: make it function
########################################
# read source model file (.h5)
########################################

#
# TODO: make file
#
source_model_path_root = '/home/sspark/Models/refs/ImageNet/Keras'

#
source_model_file_dict = collections.OrderedDict()
source_model_file_dict['VGG16'] = 'VGG16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
source_model_file_dict['ResNet50'] = 'ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
source_model_file_dict['ResNet50V2'] = 'ResNet50V2/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5'
source_model_file_dict['ResNet101'] = 'ResNet101/resnet101_weights_tf_dim_ordering_tf_kernels.h5'
source_model_file_dict['ResNet152'] = 'ResNet152/resnet152_weights_tf_dim_ordering_tf_kernels.h5'
source_model_file_dict['MobileNet'] = 'MobileNet/mobilenet_1_0_224_tf.h5'
source_model_file_dict['MobileNetV2'] = 'MobileNetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'
source_model_file_dict['EfficientNetV2S'] = 'EfficientNetV2S/efficientnetv2-s.h5'



source_model_file = source_model_file_dict[model_name]
source_model_file = os.path.join(source_model_path_root,source_model_file)


print('Keys from weight file: {:}'.format(source_model_file))
f = h5py.File(source_model_file,'r')
print('Keys: %s'%f.keys())

#assert False

# step 2
# load local weight + keras source code (modified)
#model.load_weights(source_model_file)
#assert False

# load local weight + lib_snn code
if True:
    if dataset_name == 'ImageNet':

        imagenet_utils.load_weights(model_name,model,source_model_file)

        if False:
                     model.get_layer('conv1').kernel.assign(f['block1_conv1']['block1_conv1_W_1:0'][:])
                     model.get_layer('conv1').bias.assign(f['block1_conv1']['block1_conv1_b_1:0'][:])
                     model.get_layer('conv1_1').kernel.assign(f['block1_conv2']['block1_conv2_W_1:0'][:])
                     model.get_layer('conv1_1').bias.assign(f['block1_conv2']['block1_conv2_b_1:0'][:])
                     model.get_layer('conv2').kernel.assign(f['block2_conv1']['block2_conv1_W_1:0'][:])
                     model.get_layer('conv2').bias.assign(f['block2_conv1']['block2_conv1_b_1:0'][:])
                     model.get_layer('conv2_1').kernel.assign(f['block2_conv2']['block2_conv2_W_1:0'][:])
                     model.get_layer('conv2_1').bias.assign(f['block2_conv2']['block2_conv2_b_1:0'][:])
                     model.get_layer('conv3').kernel.assign(f['block3_conv1']['block3_conv1_W_1:0'][:])
                     model.get_layer('conv3').bias.assign(f['block3_conv1']['block3_conv1_b_1:0'][:])
                     model.get_layer('conv3_1').kernel.assign(f['block3_conv2']['block3_conv2_W_1:0'][:])
                     model.get_layer('conv3_1').bias.assign(f['block3_conv2']['block3_conv2_b_1:0'][:])
                     model.get_layer('conv3_2').kernel.assign(f['block3_conv3']['block3_conv3_W_1:0'][:])
                     model.get_layer('conv3_2').bias.assign(f['block3_conv3']['block3_conv3_b_1:0'][:])
                     model.get_layer('conv4').kernel.assign(f['block4_conv1']['block4_conv1_W_1:0'][:])
                     model.get_layer('conv4').bias.assign(f['block4_conv1']['block4_conv1_b_1:0'][:])
                     model.get_layer('conv4_1').kernel.assign(f['block4_conv2']['block4_conv2_W_1:0'][:])
                     model.get_layer('conv4_1').bias.assign(f['block4_conv2']['block4_conv2_b_1:0'][:])
                     model.get_layer('conv4_2').kernel.assign(f['block4_conv3']['block4_conv3_W_1:0'][:])
                     model.get_layer('conv4_2').bias.assign(f['block4_conv3']['block4_conv3_b_1:0'][:])
                     model.get_layer('conv5').kernel.assign(f['block5_conv1']['block5_conv1_W_1:0'][:])
                     model.get_layer('conv5').bias.assign(f['block5_conv1']['block5_conv1_b_1:0'][:])
                     model.get_layer('conv5_1').kernel.assign(f['block5_conv2']['block5_conv2_W_1:0'][:])
                     model.get_layer('conv5_1').bias.assign(f['block5_conv2']['block5_conv2_b_1:0'][:])
                     model.get_layer('conv5_2').kernel.assign(f['block5_conv3']['block5_conv3_W_1:0'][:])
                     model.get_layer('conv5_2').bias.assign(f['block5_conv3']['block5_conv3_b_1:0'][:])
                     model.get_layer('fc1').kernel.assign(f['fc1']['fc1_W_1:0'][:])
                     model.get_layer('fc1').bias.assign(f['fc1']['fc1_b_1:0'][:])
                     model.get_layer('fc2').kernel.assign(f['fc2']['fc2_W_1:0'][:])
                     model.get_layer('fc2').bias.assign(f['fc2']['fc2_b_1:0'][:])
                     model.get_layer('predictions').kernel.assign(f['predictions']['predictions_W_1:0'][:])
                     model.get_layer('predictions').bias.assign(f['predictions']['predictions_b_1:0'][:])
    else:
        model.load_weights(load_weight)




#
# remove dir - train model
if not load_model:
    if overwrite_train_model:
        if os.path.isdir(filepath_save):
            shutil.rmtree(filepath_save)

if f_hp_tune:
    path_tensorboard = os.path.join(root_tensorboard, hp_tune_name)

else:
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


####################
# save model
####################

filepath_save = os.path.join(path_model_save, 'ImageNet_Import')
os.makedirs(filepath_save,exist_ok=True)
model.save_weights(filepath_save + '/ep-0000.h5')
print('ImageNet import model saved - {}'.format(filepath_save))

########
# Callbacks
########

#
if train and load_model and (not f_hp_tune):
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
    #filepath=filepath_save + '/ep-{epoch:04d}.hdf5',
    filepath=filepath_save + '/ep-{epoch:04d}.h5',
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
cb_libsnn = lib_snn.callbacks.SNNLIB(conf,path_model_load,test_ds_num,None)
cb_libsnn_ann = lib_snn.callbacks.SNNLIB(conf,path_model_load,test_ds_num)

#
callbacks_train = [cb_tensorboard,cb_libsnn]
#callbacks_train = [cb_tensorboard]
if save_model:
    callbacks_train.append(cb_model_checkpoint)
    callbacks_train.append(cb_manage_saved_model)

callbacks_test = []
# TODO: move to parameters
#dnn_to_snn = True
#if dnn_to_snn:
    #callbacks_test.append(cb_dnntosnn)

callbacks_test = [cb_libsnn]
callbacks_test_ann = [cb_libsnn_ann]

#assert False
#
if True:
    print('Test mode')

    #
    ####
    # run - test dataset
    #
    result = model.evaluate(test_ds, callbacks=callbacks_test)

    print(result)
