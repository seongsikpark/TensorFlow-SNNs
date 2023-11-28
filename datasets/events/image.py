
import tensorflow as tf

#
# this function is modified based on
# https://github.com/jackd/events-tfds
# - vis.image.as_frame
#
#def as_frame(coords, polarity, shape=None, image=None):
def as_frame(events, labels, shape=None):

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
    images = tf.zeros(image_shape,dtype=tf.int32)


    colors = tf.where(tf.expand_dims(polarity,1),[255,0,0],[0,128,0])
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