


import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB1
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import EfficientNetB5
from tensorflow.keras.applications.efficientnet import EfficientNetB6
from tensorflow.keras.applications.efficientnet import EfficientNetB7




from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.xception import preprocess_input as Xception_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as VGG16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as VGG19_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as ResNet_preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as ResNetV2_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as InceptionV3_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as InceptionResNetV2_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as MobileNet_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as  MobileNetV2_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as DenseNet_preprocess_input
from tensorflow.keras.applications.nasnet import preprocess_input as NASNet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as EfficientNet_preprocess_input


from tensorflow.keras.applications import imagenet_utils

#from tensorflow.keras.preprocessing import img_to_array, load_img

#
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#
import tqdm

import os
import matplotlib as plt

import numpy as np
import argparse
import cv2


#
#tf.config.functions_run_eagerly()

#
gpu_number=0
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)

global input_size

#
#model_name='Xception'
model_name='VGG16'
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


# dataset
dataset_name='CIFAR-10'
#dataset_name='ImageNet'


#
def eager_resize_with_crop(image, label):
    return tf.py_function(resize_with_crop,[image,label],[tf.float32, tf.int64])
    #return resize_with_crop(image,label)

#
#@tf.function
def resize_with_crop(image, label):
    global input_size

    i=image
    i=tf.cast(i,tf.float32)
    #i=tf.image.resize(i,256,preserve_aspect_ratio=True)

    #[w,h,c] = tf.shape(image)
    w=tf.shape(image)[0]
    h=tf.shape(image)[1]

    #s = 270 # 71.43. 90.06
    #s = 260 # 71.37, 90.09
    #s = 256 # 71.26, 90.10
    #s = 250 # 71.13, 90.05
    #print(tf.shape(image))
    s = input_size

    #if w >= h:
    if tf.greater(w,h):
        w = tf.cast(tf.math.multiply(tf.math.divide(w,h),s),tf.int32)
        ##i=tf.image.resize(i,(w,256),method='bicubic',preserve_aspect_ratio=True)
        #i=tf.image.resize(i,(w,256),method='bicubic')
        i=tf.image.resize(i,(w,s),method='lanczos3')
        #i=tf.image.resize(i,(w,s),method='lanczos5')
        #i=tf.image.resize(i,(w,s),method='bicubic')
    else:
        h = tf.cast(tf.math.multiply(tf.math.divide(h,w),s),tf.int32)
        ##i=tf.image.resize(i,(256,h),method='bicubic',preserve_aspect_ratio=True)
        #i=tf.image.resize(i,(256,h),method='bicubic')
        i=tf.image.resize(i,(s,h),method='lanczos3')
        #i=tf.image.resize(i,(s,h),method='lanczos5')
        #i=tf.image.resize(i,(s,h),method='bicubic')

    #i=tf.image.resize_with_crop_or_pad(i,224,224)
    i=tf.image.resize_with_crop_or_pad(i,input_size,input_size)
    i=preprocess_input(i)


    #print(type(i))
    #print(i)
    #print(i.shape)
#
    #print(i,label)
    #print(type(i))
    #print(type(label))
#
    #print(tf.reduce_max(i))

    return (i, label)

#@tf.function
def resize_with_crop_aug(image, label):
    global input_size

    i=image
    i=tf.cast(i,tf.float32)
    #i=tf.image.resize(i,256,preserve_aspect_ratio=True)

    #[w,h,c] = tf.shape(image)
    w=tf.shape(image)[0]
    h=tf.shape(image)[1]

    s = input_size

    #if w >= h:
    if tf.greater(w,h):
        w = tf.cast(tf.math.multiply(tf.math.divide(w,h),s),tf.int32)
        ##i=tf.image.resize(i,(w,256),method='bicubic',preserve_aspect_ratio=True)
        #i=tf.image.resize(i,(w,256),method='bicubic')
        i=tf.image.resize(i,(w,s),method='lanczos3')
        #i=tf.image.resize(i,(w,s),method='lanczos5')
        #i=tf.image.resize(i,(w,s),method='bicubic')
    else:
        h = tf.cast(tf.math.multiply(tf.math.divide(h,w),s),tf.int32)
        ##i=tf.image.resize(i,(256,h),method='bicubic',preserve_aspect_ratio=True)
        #i=tf.image.resize(i,(256,h),method='bicubic')
        i=tf.image.resize(i,(s,h),method='lanczos3')
        #i=tf.image.resize(i,(s,h),method='lanczos5')
        #i=tf.image.resize(i,(s,h),method='bicubic')

    #i=tf.image.resize_with_crop_or_pad(i,input_size,input_size)
    i=tf.image.random_crop(i,[input_size,input_size,3])
    i=tf.image.random_flip_left_right(i)
    i=preprocess_input(i)

    return (i, label)


# models
models = {
    'Xception': Xception,
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'ResNet50V2': ResNet50V2,
    'ResNet101V2': ResNet101V2,
    'ResNet152V2': ResNet152V2,
    'InceptionV3': InceptionV3,
    'InceptionResNetV2': InceptionResNetV2,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'DenseNet201': DenseNet201,
    'NASNetMobile': NASNetMobile,
    'NASNetLarge': NASNetLarge,
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
    'EfficientNetB2': EfficientNetB2,
    'EfficientNetB3': EfficientNetB3,
    'EfficientNetB4': EfficientNetB4,
    'EfficientNetB5': EfficientNetB5,
    'EfficientNetB6': EfficientNetB6,
    'EfficientNetB7': EfficientNetB7,
}

preprocessor_input = {
    'Xception': Xception_preprocess_input,
    'VGG16': VGG16_preprocess_input,
    'VGG19': VGG19_preprocess_input,
    'ResNet50': ResNet_preprocess_input,
    'ResNet101': ResNet_preprocess_input,
    'ResNet152': ResNet_preprocess_input,
    'ResNet50V2': ResNetV2_preprocess_input,
    'ResNet101V2': ResNetV2_preprocess_input,
    'ResNet152V2': ResNetV2_preprocess_input,
    'InceptionV3': InceptionV3_preprocess_input,
    'InceptionResNetV2': InceptionResNetV2_preprocess_input,
    'MobileNet': MobileNet_preprocess_input,
    'MobileNetV2': MobileNetV2_preprocess_input,
    'DenseNet121': DenseNet_preprocess_input,
    'DenseNet169': DenseNet_preprocess_input,
    'DenseNet201': DenseNet_preprocess_input,
    'NASNetMobile': NASNet_preprocess_input,
    'NASNetLarge': NASNet_preprocess_input,
    'EfficientNetB0': EfficientNet_preprocess_input,
    'EfficientNetB1': EfficientNet_preprocess_input,
    'EfficientNetB2': EfficientNet_preprocess_input,
    'EfficientNetB3': EfficientNet_preprocess_input,
    'EfficientNetB4': EfficientNet_preprocess_input,
    'EfficientNetB5': EfficientNet_preprocess_input,
    'EfficientNetB6': EfficientNet_preprocess_input,
    'EfficientNetB7': EfficientNet_preprocess_input,
}

GPU = 'RTX_3090'
# NVIDIA TITAN V (12GB)
if GPU=='NVIDA_TITAN_V':
    input_sizes = {
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

    batch_size_inference_sel ={
        'NASNetLarge': 128,
        'EfficientNetB4': 128,
        'EfficientNetB5': 128,
        'EfficientNetB6': 64,
        'EfficientNetB7': 64,
    }

input_sizes = {
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

batch_size_inference_sel ={
    'NASNetLarge': 128,
    'EfficientNetB4': 128,
    'EfficientNetB5': 128,
    'EfficientNetB6': 64,
    'EfficientNetB7': 64,
}



batch_size_train_sel = {
    'VGG16': 512,
}

model = models[model_name]
preprocess_input = preprocessor_input[model_name]
input_size = input_sizes.get(model_name,224)
batch_size_inference = batch_size_inference_sel.get(model_name,256)
batch_size_train = batch_size_train_sel.get(model_name,256)



## ImageNet Dataset setup
if dataset_name == 'ImageNet':
    # Get ImageNet labels
    #labels_path = tf.keras.utils.get_file('ImageNetwLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    #imagenet_labels = np.array(open(labels_path).read().splitlines())

    #print(imagenet_labels)

    # Set data_dir to a read-only storage of .tar files
    # Set write_dir to a w/r wtorage
    data_dir = '~/Datasets/ImageNet_down/'
    write_dir = '~/Datasets/ImageNet'

    # Construct a tf.data.Dataset
    download_config = tfds.download.DownloadConfig(
        extract_dir=os.path.join(write_dir,'extracted'),
        manual_dir=data_dir
    )

    download_and_prepare_kwargs={
        'download_dir': os.path.join(write_dir, 'download'),
        'download_config': download_config
    }

    ds = tfds.load('imagenet2012',
                   data_dir=os.path.join(write_dir, 'data'),
                   #batch_size=256,
                   #batch_size=64,
                   #batch_size=2,
                   #split='train',
                   split='validation',
                   #split=['train','validation']
                   shuffle_files=False,
                   download=True,
                   as_supervised=True,
                   #with_info=True,
                   download_and_prepare_kwargs=download_and_prepare_kwargs)
elif dataset_name == 'CIFAR-10':
    #(train_ds, val_ds, test_ds) = tfds.load('cifar10',
    #                              split=['train[:90%]','train[90%:100%]','test'],
    #                              as_supervised=True,
    #                              batch_size=-1)

    f_cross_valid = False

    if f_cross_valid:
        train_ds = tfds.load('cifar10',
                             split=[f'train[:{k}%]+train[{k+10}%:]' for k in range(0,100,10)],
                             as_supervised=True)

        valid_ds = tfds.load('cifar10',
                              split=[f'train[{k}%:{k+10}%]' for k in range(0,100,10)],
                              as_supervised=True)

    else:
        train_ds, valid_ds = tfds.load('cifar10',
                             split=['train[:90%]','train[90%:100%]'],
                             as_supervised=True)


    test_ds = tfds.load('cifar10',
                      split=['test'],
                      as_supervised=True)


else:
    assert False



#print(ds.batch_size)

#assert False

#assert isinstance(ds, tf.data.Dataset)


if dataset_name == 'ImageNet':
    include_top = True
else:
    include_top = False

#
image_shape=(input_size,input_size,3)
pretrained_model = model(input_shape=image_shape, include_top=include_top, weights='imagenet')
#pretrained_model = VGG16(include_top=True, weights='imagenet')
#pretrained_model = VGG19(include_top=True, weights='imagenet')
#pretrained_model = ResNet50(include_top=True, weights='imagenet')
#pretrained_model = ResNet101(include_top=True, weights='imagenet')

#pretrained_model.trainable = False







#
# only inference
if dataset_name == 'ImageNet':

    pretrained_model.compile(optimizer='adam',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             #metrics=['accuracy'])
                             #metrics=[tf.keras.metrics.sparse_top_k_categorical_accuracy])
                             metrics=[tf.keras.metrics.sparse_categorical_accuracy, \
                                      tf.keras.metrics.sparse_top_k_categorical_accuracy])

    # Preprocess input
    ds=ds.map(resize_with_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds=ds.map(eager_resize_with_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds=ds.batch(batch_size_inference)
    #ds=ds.batch(250)
    #ds=ds.batch(2)
    ds=ds.prefetch(tf.data.experimental.AUTOTUNE)

    #ds=ds.take(1)

    result = pretrained_model.evaluate(ds)
elif dataset_name == 'CIFAR-10':

    # Preprocess input
    train_ds=train_ds.map(resize_with_crop_aug,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #train_ds=train_ds.batch(batch_size_inference)
    train_ds=train_ds.batch(batch_size_train)
    train_ds=train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    valid_ds=valid_ds.map(resize_with_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #valid_ds=valid_ds.batch(batch_size_inference)
    valid_ds=valid_ds.batch(batch_size_train)
    valid_ds=valid_ds.prefetch(tf.data.experimental.AUTOTUNE)




    #feature = pretrained_model(train_ds)

    #
    pretrained_model.trainable=False
    training_model = tf.keras.Sequential()
    training_model.add(pretrained_model)
    training_model.add(tf.keras.layers.Flatten(name='flatten'))
    training_model.add(tf.keras.layers.Dropout(0.5))
    training_model.add(tf.keras.layers.Dense(4096, activation='relu', name='fc1'))
    training_model.add(tf.keras.layers.BatchNormalization())
    training_model.add(tf.keras.layers.Dropout(0.5))
    #training_model.add(tf.keras.layers.Dense(4096, activation='relu', name='fc2'))
    training_model.add(tf.keras.layers.Dense(1024, activation='relu', name='fc2'))
    training_model.add(tf.keras.layers.BatchNormalization())
    training_model.add(tf.keras.layers.Dropout(0.5))
    training_model.add(tf.keras.layers.Dense(10, activation='softmax', name='predictions'))


    #x = pretrained_model(train_ds)
    #x = tf.keras.layers.Flatten(name='flatten')(x)
    #x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    #x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
    #output = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)


    #training_model = tf.keras.Model(inputs=pretrained_model.input, outputs = output)

    #metric_accuracy = tf.keras.metrics.sparse_categorical_accuracy(name='accuracy')
    #metric_accuracy_top5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(name='accuracy_top5')

    metric_accuracy = tf.keras.metrics.sparse_categorical_accuracy
    metric_accuracy_top5 = tf.keras.metrics.sparse_top_k_categorical_accuracy

    metric_accuracy.name = 'acc'
    metric_accuracy_top5.name = 'acc-5'

    training_model.compile(optimizer='adam',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             #metrics=['accuracy'])
                             #metrics=[tf.keras.metrics.sparse_top_k_categorical_accuracy])
                             metrics=[metric_accuracy, metric_accuracy_top5])



    train_results = training_model.fit(train_ds,epochs=1000,validation_data=valid_ds)


    #result = pretrained_model.evaluate(ds)

else:
    assert False

##result = pretrained_model.predict(ds)
#print(decode_predictions(result,top=1))
##print(dict(zip(pretrained_model.metrics_names, result)))
#print(zip(pretrained_model.metrics_names, result))
#print(pretrained_model.metrics)
#print(result)
#



