
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

from typing import Any, List, Iterable, Optional, Tuple, Union
import math

from config import config
conf = config.flags

model_name = config.model_name
dataset_name = config.dataset_name

#
#from config import conf
#from config_common import conf
#from absl import flags
#conf = flags.FLAGS

from datasets.preprocessing import preprocessing_input_img

#from tensorflow.python.keras.applications.imagenet_utils import preprocess_input as preprocess_input_others
preprocess_input_others = tf.keras.applications.imagenet_utils.preprocess_input

import lib_snn

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
#def _cutmix(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input):
def _cutmix(train_ds_one, train_ds_two):

    alpha = conf.mix_alpha
    input_size = lib_snn.utils_vis.image_shape_vis(model_name, dataset_name)[0]
    #alpha = 0.5
    #input_size = 32


    (images_one, labels_one), (images_two, labels_two) = train_ds_one, train_ds_two

    # Get a sample from the Beta distribution
    batch_size = 1
    gamma_1_sample = tf.random.gamma(shape=(), alpha=alpha)
    gamma_2_sample = tf.random.gamma(shape=(), alpha=alpha)
    lambda_value = gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_w, target_h = get_box(lambda_value,input_size)

    #images_one, labels_one = resize_with_crop_aug(images_one,labels_one,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)
    #images_two, labels_two = resize_with_crop_aug(images_two,labels_two,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)

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

    #
    return (image, label)



@tf.function
#def _cutmix(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input):
def _cutmix_in_batch(images, labels):

    #alpha = 0.5
    #input_size=32
    alpha = conf.mix_alpha
    input_size = lib_snn.utils_vis.image_shape_vis(model_name, dataset_name)[0]

    (images_one, labels_one) = (images, labels)
    images_two = tf.reverse(images_one, [0])
    labels_two = tf.reverse(labels_one, [0])


    # Get a sample from the Beta distribution
    batch_size = 1
    gamma_1_sample = tf.random.gamma(shape=(), alpha=alpha)
    gamma_2_sample = tf.random.gamma(shape=(), alpha=alpha)
    lambda_value = gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_w, target_h = get_box(lambda_value,input_size)

    #images_one, labels_one = resize_with_crop_aug(images_one,labels_one,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)
    #images_two, labels_two = resize_with_crop_aug(images_two,labels_two,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)

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

    #
    return (image, label)



@tf.function
#def cutmix(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input):
def cutmix(train_ds_one, train_ds_two):

    #def mix_off(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input):
    def mix_off(train_ds_one, train_ds_two):
        (images_one, labels_one) = train_ds_one
        #images_one, labels_one = resize_with_crop_aug(images_one,labels_one,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)
        return (images_one, labels_one)

    #mix_off_iter = 500*200
    mix_off_iter = conf.mix_off_iter

    #return tf.cond(
        #lib_snn.model.train_counter < mix_off_iter,
        #lambda: _cutmix(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input),
        #lambda: mix_off(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input))

    return tf.cond(
        lib_snn.model.train_counter < mix_off_iter,
        lambda: _cutmix(train_ds_one, train_ds_two),
        lambda: mix_off(train_ds_one, train_ds_two))



@tf.function
#def cutmix(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input):
def cutmix_in_batch(images, labels):

    #def mix_off(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input):
    def mix_off(images, labels):
        #(images_one, labels_one) =
        #images_one, labels_one = resize_with_crop_aug(images_one,labels_one,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)
        #return (images_one, labels_one)
        return (images, labels)

    #mix_off_iter = 500*200
    mix_off_iter = conf.mix_off_iter

    #return tf.cond(
        #lib_snn.model.train_counter < mix_off_iter,
        #lambda: _cutmix(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input),
        #lambda: mix_off(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input))

    return tf.cond(
        lib_snn.model.train_counter < mix_off_iter,
        lambda: _cutmix_in_batch(images, labels),
        lambda: mix_off(images, labels))


########
# mixup
########
# mixup data augmentation
# from keras.io

def eager_mixup(ds_one, ds_two, alpha=1.0):
    assert False
    return tf.py_function(mixup, [ds_one, ds_two, alpha],[tf.float32,tf.float32])
    #return tf.py_function(mixup, [ds_one, ds_two, alpha],[tf.uint8,tf.uint8,tf.int64),tf.float32])
    #return tf.py_function(mixup, [ds_one, ds_two, alpha],[(tf.uint8,tf.int64),(tf.uint8,tf.int64),tf.float32])


#
def mixup(ds_one, ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input):

    # unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = 1
    #batch_size = tf.shape(images_one)[0]
    #print(batch_size)
    #assert False

    #
    #images_one, labels_one = resize_with_crop_aug(images_one,labels_one,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)
    #images_two, labels_two = resize_with_crop_aug(images_two,labels_two,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)

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
@tf.function
def _mixup_in_batch(images, labels):

    alpha = conf.mix_alpha
    batch_size = 1

    images_one = images
    images_two = tf.reverse(images, [0])
    labels_one = labels
    labels_two = tf.reverse(labels, [0])

    # sample lambda and reshape it to do the mixup
    gamma_1_sample = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma_2_sample = tf.random.gamma(shape=[batch_size], alpha=alpha)
    l = gamma_1_sample / (gamma_1_sample+gamma_2_sample)

    images = images_one * l + images_two * (1-l)
    labels = labels_one * l + labels_two * (1-l)

    return (images,labels)

@tf.function
def mixup_in_batch(images, labels):

    #def mix_off(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input):
    def mix_off(images, labels):
        #(images_one, labels_one) =
        #images_one, labels_one = resize_with_crop_aug(images_one,labels_one,dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input)
        #return (images_one, labels_one)
        return (images, labels)

    #mix_off_iter = 500*200
    mix_off_iter = conf.mix_off_iter

    #return tf.cond(
        #lib_snn.model.train_counter < mix_off_iter,
        #lambda: _mixup(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input),
        #lambda: mix_off(train_ds_one, train_ds_two, dataset_name, input_size, input_size_pre_crop_ratio, num_class, alpha, input_prec_mode,preprocessor_input))

    return tf.cond(
        tf.logical_or(lib_snn.model.train_counter < mix_off_iter, mix_off_iter < 0),
        lambda: _mixup_in_batch(images, labels),
        lambda: mix_off(images, labels))


#
def eager_resize_with_crop(image, label):
    return tf.py_function(resize_with_crop,[image,label],[tf.float32, tf.int64])
    #return resize_with_crop(image,label)

#
@tf.function
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
    #
    #if conf.data_prep=='default':
    #    try:
    #        i = preprocess_input(i, mode=input_prec_mode)
    #    except:
    #        i=preprocess_input(i)
    #else:
    #    i=preprocessing_input_img(i,mode=conf.data_prep)

    #
    label = tf.one_hot(label,num_class)

    return (i, label)


@tf.function
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


    # TODO: to other function
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

class ImageAugment(object):
  """Image augmentation class for applying image distortions."""

  def distort(
      self,
      image: tf.Tensor
  ) -> tf.Tensor:
    """Given an image tensor, returns a distorted image with the same shape.

    Expect the image tensor values are in the range [0, 255].

    Args:
      image: `Tensor` of shape [height, width, 3] or
        [num_frames, height, width, 3] representing an image or image sequence.

    Returns:
      The augmented version of `image`.
    """
    raise NotImplementedError()

  def distort_with_boxes(
      self,
      image: tf.Tensor,
      bboxes: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Distorts the image and bounding boxes.

    Expect the image tensor values are in the range [0, 255].

    Args:
      image: `Tensor` of shape [height, width, 3] or
        [num_frames, height, width, 3] representing an image or image sequence.
      bboxes: `Tensor` of shape [num_boxes, 4] or [num_frames, num_boxes, 4]
        representing bounding boxes for an image or image sequence.

    Returns:
      The augmented version of `image` and `bboxes`.
    """
    raise NotImplementedError


class RandomErasing(ImageAugment):
  """Applies RandomErasing to a single image.


  Reference: https://arxiv.org/abs/1708.04896


  Implementation is inspired by
  https://github.com/rwightman/pytorch-image-models.
  """


  def __init__(self,
               probability: float = 0.25,
               min_area: float = 0.02,
               max_area: float = 1 / 3,
               min_aspect: float = 0.3,
               max_aspect: Optional[float] = None,
               min_count=1,
               max_count=1,
               trials=10):
    """Applies RandomErasing to a single image.


    Args:
      probability: Probability of augmenting the image. Defaults to `0.25`.
      min_area: Minimum area of the random erasing rectangle. Defaults to
        `0.02`.
      max_area: Maximum area of the random erasing rectangle. Defaults to `1/3`.
      min_aspect: Minimum aspect rate of the random erasing rectangle. Defaults
        to `0.3`.
      max_aspect: Maximum aspect rate of the random erasing rectangle. Defaults
        to `None`.
      min_count: Minimum number of erased rectangles. Defaults to `1`.
      max_count: Maximum number of erased rectangles. Defaults to `1`.
      trials: Maximum number of trials to randomly sample a rectangle that
        fulfills constraint. Defaults to `10`.
    """
    self._probability = probability
    self._min_area = float(min_area)
    self._max_area = float(max_area)
    self._min_log_aspect = math.log(min_aspect)
    self._max_log_aspect = math.log(max_aspect or 1 / min_aspect)
    self._min_count = min_count
    self._max_count = max_count
    self._trials = trials


  def distort(self, image: tf.Tensor) -> tf.Tensor:
    """Applies RandomErasing to single `image`.


    Args:
      image (tf.Tensor): Of shape [height, width, 3] representing an image.


    Returns:
      tf.Tensor: The augmented version of `image`.
    """
    uniform_random = tf.random.uniform(shape=[], minval=0., maxval=1.0)
    mirror_cond = tf.less(uniform_random, self._probability)
    image = tf.cond(mirror_cond, lambda: self._erase(image), lambda: image)
    return image


  @tf.function
  def _erase(self, image: tf.Tensor) -> tf.Tensor:
    """Erase an area."""
    if self._min_count == self._max_count:
      count = self._min_count
    else:
      count = tf.random.uniform(
          shape=[],
          minval=int(self._min_count),
          maxval=int(self._max_count - self._min_count + 1),
          dtype=tf.int32)


    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    area = tf.cast(image_width * image_height, tf.float32)


    for _ in range(count):
      # Work around since break is not supported in tf.function
      is_trial_successfull = False
      for _ in range(self._trials):
        if not is_trial_successfull:
          erase_area = tf.random.uniform(
              shape=[],
              minval=area * self._min_area,
              maxval=area * self._max_area)
          aspect_ratio = tf.math.exp(
              tf.random.uniform(
                  shape=[],
                  minval=self._min_log_aspect,
                  maxval=self._max_log_aspect))


          half_height = tf.cast(
              tf.math.round(tf.math.sqrt(erase_area * aspect_ratio) / 2),
              dtype=tf.int32)
          half_width = tf.cast(
              tf.math.round(tf.math.sqrt(erase_area / aspect_ratio) / 2),
              dtype=tf.int32)


          if 2 * half_height < image_height and 2 * half_width < image_width:
            center_height = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=int(image_height - 2 * half_height),
                dtype=tf.int32)
            center_width = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=int(image_width - 2 * half_width),
                dtype=tf.int32)


            image = _fill_rectangle(
                image,
                center_width,
                center_height,
                half_width,
                half_height,
                #replace=None)
                replace=0)


            is_trial_successfull = True


    return image


def _fill_rectangle(image,
                    center_width,
                    center_height,
                    half_width,
                    half_height,
                    replace=None):
    """Fills blank area."""
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    lower_pad = tf.maximum(0, center_height - half_height)
    upper_pad = tf.maximum(0, image_height - center_height - half_height)
    left_pad = tf.maximum(0, center_width - half_width)
    right_pad = tf.maximum(0, image_width - center_width - half_width)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad)
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype),
        padding_dims,
        constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])

    if replace is None:
        fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
    elif isinstance(replace, tf.Tensor):
        fill = replace
    else:
        fill = tf.ones_like(image, dtype=image.dtype) * replace
    image = tf.where(tf.equal(mask, 0), fill, image)

    return image


