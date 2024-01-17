
import tensorflow as tf

from config import config
conf = config.flags


#RED = [255,0,0]    # on
#GREEN = [0,255,0]  # off

#RED = [1,0,0]
#GREEN = [0,1,0]

#RED = [1,0]      # on
#GREEN = [0,1]    # off

RED = [0.1,0]      # on
GREEN = [0,0.1]    # off

#RED = [1]      # on
#GREEN = [-1]    # off

#RED = [0.1]      # on
#GREEN = [-0.1]    # off

#
# this function is modified based on
# https://github.com/jackd/events-tfds
# - vis.image.as_frame
#
#def as_frame(coords, polarity, shape=None, image=None):
def as_frame(events, labels, shape=None):
    assert False, 'use as_frames or update code'

    num_class=10
    coords = events['coords']
    polarity = events['polarity']

    if shape is None:
        #shape = np.max(coords, axis=0)[-1::-1] + 1
        #shape = tf.reduce_max(coords, axis=0)[-1::-1] + 1
        # only 2D
        shape = tf.reduce_max(coords, axis=0) + 1

        assert False

        #image_shape = (*(shape.numpy()),3)
    else:
        image_shape = shape


    #image = tf.zeros(image_shape,dtype=tf.uint8)
    #images = tf.zeros(image_shape,dtype=tf.int32)
    images = tf.zeros(image_shape,dtype=tf.float32)


    #colors = tf.where(tf.expand_dims(polarity,1),[255,0,0],[0,255,0])
    colors = tf.where(tf.expand_dims(polarity,1),RED,GREEN)
    images = tf.tensor_scatter_nd_update(images,coords,colors)


    #x, y = coords.T
    #if image is None:
        #image = np.zeros((*shape, 3), dtype=np.uint8)
    #image[(shape[0] - y - 1, x)] = np.where(polarity[:, np.newaxis], RED, GREEN)
    #return image

    # resize image
    s=32
    images = tf.image.resize(images, (s, s))


    # one-hot vectorization - label
    labels = tf.one_hot(labels, num_class)


    return (images, labels)




#
# this function is modified based on
# https://github.com/jackd/events-tfds
# - vis.image.as_frames
#
def as_frames(
    events,
    labels,
    #coords,
    #time,
    #polarity=None,
    dt=None,
    num_frames=None,
    shape=None,
    flip_up_down=False,
    augmentation=False,
):

    #
    conf.input_data_time_dim = True

    #
    num_class=10
    coords = events['coords']
    polarity = events['polarity']
    time = events['time']

    coords = tf.cast(coords, dtype=tf.int32)
    time = tf.cast(time, dtype=tf.int32)

    #if time.size == 0:
    if time.shape[0]==0:
        raise ValueError("time must not be empty")
    t_start = time[0]
    t_end = time[-1]
    if dt is None:
        dt = int((t_end - t_start) // (num_frames - 1))
    else:
        num_frames = (t_end - t_start) // dt + 1

    if shape is None:
        #shape = np.max(coords, axis=0)[-1::-1] + 1
        assert False
    else:
        #shape = shape[-1::-1]
        image_shape = (num_frames, *shape)

    #frame_data = np.zeros((num_frames, *shape, 3), dtype=np.uint8)

    #images = tf.zeros(image_shape,dtype=tf.int32)
    images = tf.zeros(image_shape,dtype=tf.float32)

    #if polarity is None:
    #    colors = WHITE
    #else:
    #    colors = np.where(polarity[:, np.newaxis], RED, GREEN)

    colors = tf.where(tf.expand_dims(polarity,1),RED,GREEN)

    idxs_frame = tf.math.floordiv(tf.subtract(time,t_start),dt)
    idxs_frame = tf.expand_dims(idxs_frame,axis=1)

    # (frame, x, y, color)
    idxs = tf.concat([idxs_frame,coords],1)

    images = tf.tensor_scatter_nd_update(images,idxs,colors)

    # resize image
    #s=32
    #crop_size = 36
    s=48
    crop_size = 52
    #crop_size = 50

    if augmentation:
        image_resize_size = crop_size
    else:
        image_resize_size = s
    #images=tf.image.resize(images,(image_resize_size,image_resize_size),method='lanczos3')   # VGG, ResNet
    #images=tf.image.resize(images,(image_resize_size,image_resize_size),method='bilinear')   # VGG, ResNet
    images=tf.image.resize(images,(s,s),method='bilinear')   # VGG, ResNet


    if augmentation:
        images = tf.image.pad_to_bounding_box(images, 1, 1, crop_size, crop_size)  # zero padding
        #images = tf.image.resize(images, (s, s))
        #images = tf.image.random_crop(images, (num_frames,crop_size,crop_size,3))    # random crop
        #images=tf.image.resize(images,(crop_size,crop_size),method='lanczos3')   # VGG, ResNet
        images = tf.image.random_crop(images, (num_frames,s,s,2))    # random crop
        images=tf.image.random_flip_left_right(images)
        images=tf.image.random_flip_up_down(images)



    # one-hot vectorization - label
    labels = tf.one_hot(labels, num_class)

    #
    #idxs_frame = tf.cast(idxs_frame,dtype=tf.int32)
    #i = np.minimum((time - t_start) // dt, num_frames - 1)

    # fi = np.concatenate((i[:, np.newaxis], coords), axis=-1)
    #x, y = coords.T
    #if flip_up_down:
    #    y = shape[0] - y - 1
    # frame_data[(i, shape[0] - y - 1, x)] = colors
    #frame_data[(i, y, x)] = colors
    #images = tf.tensor_scatter_nd_update(images,idxs,colors)

    return (images, labels)
