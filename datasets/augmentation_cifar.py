
#
# Data augmentation - CIFAR


import tensorflow as tf
#import tensorflow_addons as tfa

#from main import input_size
#from main import input_size_pre_crop_ratio
#from main import model_name
#from main import num_class

#global input_size
#global input_size_pre_crop_ratio
#global model_name
#global num_class

#from models.input_preprocessor import preprocessor_input
import datasets.preprocessing
#from lib_snn.config_glb import model_name
#from lib_snn.config_glb import dataset_name

from config import config

# model_name = config.model_name
# dataset_name = config.dataset_name
dataset_name = "CIFAR10"

#
#from config import conf
#from config_common import conf
from absl import flags
conf = flags.FLAGS

from datasets.preprocessing import preprocessing_input_img

model_name = conf.model_path
#from tensorflow.python.keras.applications.imagenet_utils import preprocess_input as preprocess_input_others
preprocess_input_others = tf.keras.applications.imagenet_utils.preprocess_input

########
# cutmix
########
# mixup data augmentation
# from keras.io

def get_box(lambda_value,input_size):
    cut_rat = tf.math.sqrt(1.0-lambda_value)

    cut_w = input_size * cut_rat
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = input_size * cut_rat
    cut_h = tf.cast(cut_h, tf.int32)

    cut_x = tf.random.uniform((), minval=0, maxval=input_size, dtype=tf.int32)
    cut_y = tf.random.uniform((), minval=0, maxval=input_size, dtype=tf.int32)

    boundaryx1 = tf.clip_by_value(cut_x - cut_w//2, 0, input_size)
    boundaryy1 = tf.clip_by_value(cut_y - cut_h//2, 0, input_size)
    bbx2 = tf.clip_by_value(cut_x + cut_w//2, 0, input_size)
    bby2 = tf.clip_by_value(cut_y + cut_h//2, 0, input_size)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_w, target_h

#
def eager_cutmix(ds_one, ds_two, alpha=1.0):
    return tf.py_function(mixup, [ds_one, ds_two, alpha],[tf.float32,tf.float32])

#
@tf.function
def cutmix(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input):
    (images_one, labels_one), (images_two, labels_two) = train_ds_one, train_ds_two

    # Get a sample from the Beta distribution
    batch_size = 1
    gamma_1_sample = tf.random.gamma(shape=(), alpha=alpha)
    gamma_2_sample = tf.random.gamma(shape=(), alpha=alpha)
    lambda_value = gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_w, target_h = get_box(lambda_value,input_size)

    images_one, labels_one = resize_with_crop_aug(images_one,labels_one,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)
    images_two, labels_two = resize_with_crop_aug(images_two,labels_two,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)

    # Get a patch from the second image
    crop2 = tf.image.crop_to_bounding_box(images_two, boundaryy1, boundaryx1, target_h, target_w)

    # Pad the images_two patch with the same offset
    images_two = tf.image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, input_size, input_size)

    # Get a patch from the first image
    crop1 = tf.image.crop_to_bounding_box(images_one, boundaryy1, boundaryx1, target_h, target_w)

    # Pad the images_one patch with the same offset
    img1 = tf.image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, input_size, input_size)

    # Modifi the first image by subtracting the patch
    images_one = images_one - img1

    # Add the modified images_one and images_two to get the CutMix image
    image = images_one + images_two

    # Adjust Lambda in accordanct to the pixel ration
    lambda_value = 1 - (target_w*target_h)/(input_size*input_size)
    lambda_value = tf.cast(lambda_value,tf.float32)

    # Combine the labels of both images
    label = lambda_value*labels_one + (1-lambda_value)*labels_two

    return (image, label)

########
# mixup
########
# mixup data augmentation
# from keras.io

def eager_mixup(ds_one, ds_two, alpha=1.0):
    return tf.py_function(mixup, [ds_one, ds_two, alpha],[tf.float32,tf.float32])
    #return tf.py_function(mixup, [ds_one, ds_two, alpha],[tf.uint8,tf.uint8,tf.int64),tf.float32])
    #return tf.py_function(mixup, [ds_one, ds_two, alpha],[(tf.uint8,tf.int64),(tf.uint8,tf.int64),tf.float32])

def mixup(ds_one, ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input):

    # unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = 1
    #batch_size = tf.shape(images_one)[0]
    #print(batch_size)
    #assert False

    #
    images_one, labels_one = resize_with_crop_aug(images_one,labels_one,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)
    images_two, labels_two = resize_with_crop_aug(images_two,labels_two,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)

    labels_one = tf.cast(labels_one,tf.float32)
    labels_two = tf.cast(labels_two,tf.float32)

    # sample lambda and reshape it to do the mixup
    gamma_1_sample = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma_2_sample = tf.random.gamma(shape=[batch_size], alpha=alpha)
    l = gamma_1_sample / (gamma_1_sample+gamma_2_sample)
    #xx_l = l
    #x_l = tf.reshape(l, shape=(batch_size,1,1,1))
    #x_l = tf.broadcast_to(x_l, tf.shape(images_one))
    #y_l = tf.reshape(l, shape=(batch_size,1))
    #y_l = tf.broadcast_to(y_l, tf.shape(images_one))
    #y_l = l

    # perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    #print(type(images_one[0]))
    ##print((images_one[0]))
    #assert False
    #images = tf.add(tf.multiply(images_one,x_l),tf.multiply(images_two,(1-x_l)))
    #images = tf.multiply(images_one,x_l)
    #$images = images_one * x_l
    #images = images_one * l
    images = images_one * l + images_two * (1-l)
    #images = x_l*images_one + (1-x_l)*labels_two
    #images=images_one
    labels = labels_one * l + labels_two * (1-l)
    #labels = labels_one * y_l
    #labels = labels_one * 0.2
    #labels = tf.add(tf.multiply())

    return (images,labels)

#
def eager_resize_with_crop(image, label):
    return tf.py_function(resize_with_crop,[image,label],[tf.float32, tf.int64])
    #return resize_with_crop(image,label)

#
#@tf.function
#def resize_with_crop(image, label, dataset_name, input_size,input_size_pre_crop_ratio, num_class, input_prec_mode='torch'):
def resize_with_crop(image, label, dataset_name, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode, preprocess_input):

    i=image
    i=tf.cast(i,tf.float32)

    #[w,h,c] = tf.shape(image)
    #w=tf.shape(image)[0]
    #h=tf.shape(image)[1]

    #print(tf.shape(image))
    #s = input_size_pre_crop

    if dataset_name == 'imagenet2012':
        i = _resize_with_crop_imagenet(image,input_size,input_size_pre_crop_ratio)
    else:
        s = input_size
        if input_prec_mode=='torch':
            i = tf.image.resize_with_crop_or_pad(i, s, s)
        elif input_prec_mode == 'caffe':
            # transfer learning with pre-trained modes in Keras (ImageNet)
            i = tf.image.resize(i, (s, s), method='lanczos3')
        else:
            assert False

    #
    if conf.data_prep=='default':
        try:
            i = preprocess_input(i, mode=input_prec_mode)
        except:
            i=preprocess_input(i)
    else:
        i=preprocessing_input_img(i,mode=conf.data_prep)

    #
    label = tf.one_hot(label,num_class)

    return (i, label)


#@tf.function
#def resize_with_crop_aug(image, label, dataset_name, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode='torch'):
def resize_with_crop_aug(image, label, dataset_name, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode, preprocess_input):

    i=image
    i=tf.cast(i,tf.float32)

    #[w,h,c] = tf.shape(image)
    #w=tf.shape(image)[0]
    #h=tf.shape(image)[1]

    if dataset_name == 'imagenet2012':
        i=_resize_with_crop_imagenet(image,input_size,input_size_pre_crop_ratio)
    else:
        s = input_size * input_size_pre_crop_ratio
        s = tf.cast(s, tf.int32)

        # data augmentation from "A Simple Framework for Contrastive Learning of Visual Representations"

        #i=tf.numpy_function(lambda i: tf.keras.preprocessing.image.random_zoom(i, (0.2,0.2)),[i],tf.float32)
        #i=tf.keras.preprocessing.image.random_zoom(i,[-0.1,0.2])
        #i=tf.keras.preprocessing.image.random_rotation(i,0.3)
        #i=tf.image.random_brightness(i,max_delta=63)
        #i=tf.image.random_contrast(i,lower=0.2,upper=1.8)

        if input_prec_mode=='torch':
            i = tf.image.resize_with_crop_or_pad(i, s, s)
        elif input_prec_mode == 'caffe':
            # transfer learning with pre-trained modes in Keras (ImageNet)
            i = tf.image.resize(i, (s, s), method='lanczos3')
        else:
            assert False

    # color jitter
    #i=tf.image.random_brightness(i,max_delta=0.8)
    #i=tf.image.random_contrast(i,lower=0.2,upper=1.8)
    #i=tf.image.random_saturation(i,lower=0.2,upper=1.8)
    #i=tf.image.random_hue(i,0.2)

    #i=tf.image.random_contrast(i,lower=0.0,upper=0.4)
    #i=tf.image.random_saturation(i,lower=0.0,upper=0.4)
    #i=tf.image.random_hue(i,max_delta=0.1)
    #i=tf.clip_by_value(i,0,1)

    # random transformation
    #i=tf.image.resize_with_crop_or_pad(i,input_size,input_size)
    i=tf.image.random_crop(i,[input_size,input_size,3])
    i=tf.image.random_flip_left_right(i)

    #
    if conf.data_prep=='default':
        try:
            i = preprocess_input(i, mode=input_prec_mode)
        except:
            i=preprocess_input(i)
    else:
        i=datasets.preprocessing.preprocessing_input_img(i,mode=conf.data_prep)


    # one-hot vectorization - label
    label = tf.one_hot(label, num_class)

    return (i, label)



def _resize_with_crop_imagenet(image,input_size,input_size_pre_crop_ratio):
    #from lib_snn.config_glb import model_name
    #print(model_name)

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
    #s = input_size_pre_crop
    s = input_size*input_size_pre_crop_ratio

    #if w >= h:
    if tf.greater(w,h):
        w = tf.cast(tf.math.multiply(tf.math.divide(w,h),s),tf.int32)
        ##i=tf.image.resize(i,(w,256),method='bicubic',preserve_aspect_ratio=True)
        #i=tf.image.resize(i,(w,256),method='bicubic')
        s=tf.cast(s,tf.int32)
        if 'MobileNet' in model_name:
            i=tf.image.resize(i,(w,s),method='bilinear')
            #i=tf.image.resize(i,(w,s),method='bicubic')
            #i=tf.image.resize(i,(w,s),method='lanczos3')   # VGG, ResNet
        elif 'EfficientNet' in model_name:
            i=tf.image.resize(i,(w,s),method='bicubic',preserve_aspect_ratio=True,antialias=True)      # EfficientNet
        else:
            i=tf.image.resize(i,(w,s),method='lanczos3')   # VGG, ResNet
            #i=tf.image.resize(i,(w,s),method='bicubic')
        #i=tf.image.resize(i,(w,s),method='bilinear')
        #i=tf.image.resize(i,(w,s),method='bilinear',preserve_aspect_ratio=True,antialias=True)
        #i=tf.image.resize(i,(w,s),method='lanczos3',preserve_aspect_ratio=True,antialias=True)   #
        #i=tf.image.resize(i,(w,s),method='lanczos5')
        #i=tf.image.resize(i,(w,s),method='bicubic')
        #i=tf.image.resize(i,(w,s),method='bicubic',preserve_aspect_ratio=True,antialias=True)      # EfficientNet
        #i=tf.image.resize(i,(w,s),method='nearest')
        #i=tf.image.resize(i,(w,s),method='mitchellcubic')
        #i=tf.image.resize(i,(w,s),method='mitchellcubic',preserve_aspect_ratio=True,antialias=True)
        #i=tf.image.resize(i,(w,s),method='area')
    else:
        h = tf.cast(tf.math.multiply(tf.math.divide(h,w),s),tf.int32)
        ##i=tf.image.resize(i,(256,h),method='bicubic',preserve_aspect_ratio=True)
        #i=tf.image.resize(i,(256,h),method='bicubic')
        s=tf.cast(s,tf.int32)
        if 'MobileNet' in model_name:
            i=tf.image.resize(i,(s,h),method='bilinear')
            #i=tf.image.resize(i,(s,h),method='bicubic')
            #i=tf.image.resize(i,(s,h),method='lanczos3')   # VGG ,ResNet
        elif 'EfficientNet' in model_name:
            i=tf.image.resize(i,(s,h),method='bicubic',preserve_aspect_ratio=True,antialias=True)      # EfficientNet
        else:
            i=tf.image.resize(i,(s,h),method='lanczos3')   # VGG ,ResNet
        #i=tf.image.resize(i,(s,h),method='bilinear')
        #i=tf.image.resize(i,(s,h),method='bilinear',preserve_aspect_ratio=True,antialias=True)
        #i=tf.image.resize(i,(s,h),method='lanczos3')   # VGG ,ResNet
        #i=tf.image.resize(i,(s,h),method='lanczos3',preserve_aspect_ratio=True,antialias=True)   #
        #i=tf.image.resize(i,(s,h),method='lanczos5')
        #i=tf.image.resize(i,(s,h),method='bicubic')
        #i=tf.image.resize(i,(s,h),method='bicubic',preserve_aspect_ratio=True,antialias=True)      # EfficientNet
        #i=tf.image.resize(i,(s,h),method='nearest')
        #i=tf.image.resize(i,(s,h),method='mitchellcubic')
        #i=tf.image.resize(i,(s,h),method='mitchellcubic',preserve_aspect_ratio=True,antialias=True)
        #i=tf.image.resize(i,(s,h),method='area')

    #i=tf.image.resize_with_crop_or_pad(i,224,224)
    i=tf.image.resize_with_crop_or_pad(i,input_size,input_size)

    return i

