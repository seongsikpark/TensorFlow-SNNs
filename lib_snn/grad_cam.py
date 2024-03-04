################################################
# grad_cam.py
#
# This file is created based on Keras example
################################################

import numpy as np
import tensorflow as tf
import keras

import lib_snn

#
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt

#
from config import config
conf = config.flags


def get_image_array(img_path, size):
    # (w,h,c)
    img = keras.utils.load_img(img_path, target_size=size)
    #
    array = keras.utils.img_to_array(img)
    # add batch dimension
    # (1,w,h,c)
    array = np.expand_dims(array,axis=0)
    return array

def make_gradcam_heatmap_dnn(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the ouput predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:,pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will add also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()




# accumulated spike version
def make_gradcam_heatmap_snn(img_array, model, last_conv_layer_name, neuron_mode, pred_index=None):

    #normalize=True
    normalize=False

    #
    integrated_gradient=False
    #integrated_gradient=True

    #
    #mean_t = True
    mean_t = False

    #
    spike_count_norm = True
    #spike_count_norm = False

    #
    batch_size = 100
    input_shape = (32,32,3)
    classes = 10

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = lib_snn.model.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output], batch_size, input_shape,  classes, conf
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    #with tf.GradientTape() as tape:
    with tf.GradientTape(persistent=True) as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:,pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    #grads = tape.gradient(class_channel, last_conv_layer_output)

    #grads = []
    #grads


    # grad cam / integrated gradient
    #if False:

    if neuron_mode:
        if True:
            source_layer = grad_model.get_layer(last_conv_layer_name)
            g0 = tape.gradient(class_channel, source_layer.act.out.read(0))
            #g0 = tape.gradient(class_channel, source_layer.act.inputs.read(0))
            # ig
            if integrated_gradient:
                g0 = (g0[:-1]+g0[1:])/2
                g0 = tf.reduce_mean(g0, axis=0)
            else:
                g0 = g0[-1]

            g1 = tape.gradient(class_channel, source_layer.act.out.read(1))
            #g1 = tape.gradient(class_channel, source_layer.act.inputs.read(1))
            if integrated_gradient:
                g1 = (g1[:-1]+g1[1:])/2
                g1 = tf.reduce_mean(g1, axis=0)
            else:
                g1 = g1[-1]

            g2 = tape.gradient(class_channel, source_layer.act.out.read(2))
            #g2 = tape.gradient(class_channel, source_layer.act.inputs.read(2))
            if integrated_gradient:
                g2 = (g2[:-1]+g2[1:])/2
                g2 = tf.reduce_mean(g2, axis=0)
            else:
                g2 = g2[-1]

            g3 = tape.gradient(class_channel, source_layer.act.out.read(3))
            #g3 = tape.gradient(class_channel, source_layer.act.inputs.read(3))
            if integrated_gradient:
                g3 = (g3[:-1]+g3[1:])/2
                g3 = tf.reduce_mean(g3, axis=0)
            else:
                g3 = g3[-1]

            a0 = source_layer.act.out.read(0)[-1]
            a1 = source_layer.act.out.read(1)[-1]
            a2 = source_layer.act.out.read(2)[-1]
            a3 = source_layer.act.out.read(3)[-1]

            #a0 = source_layer.act.inputs.read(0)[-1]
            #a1 = source_layer.act.inputs.read(1)[-1]
            #a2 = source_layer.act.inputs.read(2)[-1]
            #a3 = source_layer.act.inputs.read(3)[-1]

            #
            if not mean_t:
                ga0 = g0*a0
                ga1 = g1*a1
                ga2 = g2*a2
                ga3 = g3*a3


                #ga = tf.reduce_mean([ga0,ga1,ga2,ga3],axis=0)
                #ga = tf.reduce_sum([ga0,ga1,ga2,ga3],axis=0)
                if spike_count_norm:
                    #sc = tf.reduce_sum([a0,a1,a2,a3])
                    ga0 = ga0/tf.reduce_sum(a0)
                    ga1 = ga1/tf.reduce_sum(a1)
                    ga2 = ga2/tf.reduce_sum(a2)
                    ga3 = ga3/tf.reduce_sum(a3)

                    ga = tf.reduce_sum([ga0, ga1, ga2, ga3], axis=0)
                else:
                    ga = tf.reduce_sum([ga0,ga1,ga2,ga3],axis=0)
                    ga = ga / 4
            else:
                a = tf.reduce_sum([a0,a1,a2,a3],axis=0)
                g = tf.reduce_sum([g0,g1,g2,g3],axis=0)

                if spike_count_norm:
                    sc = tf.reduce_sum([a0,a1,a2,a3])
                    a = a / sc
                    g = g / sc
                else:
                    a = a / 4
                    g = g / 4

                ga = g*a

            heatmap = ga
        elif False:
        #elif True:
            # spike count
            source_layer = grad_model.get_layer(last_conv_layer_name)
            heatmap = source_layer.act.spike_count[-1]
        else:
            # neuron input
            source_layer = grad_model.get_layer(last_conv_layer_name)

            a0 = source_layer.act.inputs.read(0)[-1]
            a1 = source_layer.act.inputs.read(1)[-1]
            a2 = source_layer.act.inputs.read(2)[-1]
            a3 = source_layer.act.inputs.read(3)[-1]


            heatmap = tf.reduce_mean([a0,a1,a2,a3],axis=0)

    else:
        if False:
            source_layer = grad_model.get_layer(last_conv_layer_name)
            g0 = tape.gradient(class_channel, source_layer._outputs.read(0))
            #g0 = (g0[:-1] + g0[1:]) / 2
            #g0 = tf.reduce_mean(g0, axis=0)
            g0 = g0[-1]
            g1 = tape.gradient(class_channel, source_layer._outputs.read(1))
            #g1 = (g1[:-1] + g1[1:]) / 2
            #g1 = tf.reduce_mean(g1, axis=0)
            g1 = g1[-1]
            g2 = tape.gradient(class_channel, source_layer._outputs.read(2))
            #g2 = (g2[:-1] + g2[1:]) / 2
            #g2 = tf.reduce_mean(g2, axis=0)
            g2 = g2[-1]
            g3 = tape.gradient(class_channel, source_layer._outputs.read(3))
            #g3 = (g3[:-1] + g3[1:]) / 2
            #g3 = tf.reduce_mean(g3, axis=0)
            g3 = g3[-1]

            a0 = source_layer._outputs.read(0)[-1]
            a1 = source_layer._outputs.read(1)[-1]
            a2 = source_layer._outputs.read(2)[-1]
            a3 = source_layer._outputs.read(3)[-1]

            ga0 = g0 * a0
            ga1 = g1 * a1
            ga2 = g2 * a2
            ga3 = g3 * a3

            ga = tf.reduce_mean([ga0, ga1, ga2, ga3], axis=0)

            heatmap = ga
        elif False:
            # grad cam - mean t
            source_layer = grad_model.get_layer(last_conv_layer_name)
            g0 = tape.gradient(class_channel, source_layer._outputs.read(0))
            #g0 = (g0[:-1] + g0[1:]) / 2
            #g0 = tf.reduce_mean(g0, axis=0)
            g0 = g0[-1]
            g1 = tape.gradient(class_channel, source_layer._outputs.read(1))
            #g1 = (g1[:-1] + g1[1:]) / 2
            #g1 = tf.reduce_mean(g1, axis=0)
            g1 = g1[-1]
            g2 = tape.gradient(class_channel, source_layer._outputs.read(2))
            #g2 = (g2[:-1] + g2[1:]) / 2
            #g2 = tf.reduce_mean(g2, axis=0)
            g2 = g2[-1]
            g3 = tape.gradient(class_channel, source_layer._outputs.read(3))
            #g3 = (g3[:-1] + g3[1:]) / 2
            #g3 = tf.reduce_mean(g3, axis=0)
            g3 = g3[-1]
            
            a0 = source_layer._outputs.read(0)[-1]
            a1 = source_layer._outputs.read(1)[-1]
            a2 = source_layer._outputs.read(2)[-1]
            a3 = source_layer._outputs.read(3)[-1]

            a = tf.reduce_mean([a0, a1, a2, a3], axis=0)
            g = tf.reduce_mean([g0, g1, g2, g3], axis=0)

            ga = g*a

            heatmap = ga
        else:
            # syn out - mean t
            source_layer = grad_model.get_layer(last_conv_layer_name)

            a0 = source_layer._outputs.read(0)[-1]
            a1 = source_layer._outputs.read(1)[-1]
            a2 = source_layer._outputs.read(2)[-1]
            a3 = source_layer._outputs.read(3)[-1]

            a = tf.reduce_mean([a0, a1, a2, a3], axis=0)
            heatmap = a

    # original - DNN / SNN (only last layer)
    if False:
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel

        # positive only
        #grads = tf.math.maximum(grads, 0)

        # grad x act
        if grad_x_act:
            grads = tf.math.multiply(grads,last_conv_layer_output)


        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        # test - 240213
        #grads = (grads[:-1]+grads[1:])/2.0

        grads = tf.reduce_mean(grads,axis=0)
        #grads = tf.reduce_mean(grads,axis=(0,1))
        #heatmap = last_conv_layer_output[-1] @ grads[..., tf.newaxis]

        heatmap = tf.multiply(grads,last_conv_layer_output[-1])


    heatmap = tf.maximum(heatmap,0)
    #heatmap = tf.reduce_mean(heatmap,axis=0)

    # dimension reduce - channel
    heatmap = tf.reduce_mean(heatmap,axis=-1)
    heatmap = tf.squeeze(heatmap)

    # positive only
    #pooled_grads = tf.math.maximum(pooled_grads, 0)


    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    #last_conv_layer_output = last_conv_layer_output[0]
    #heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

    #last_conv_layer_n_name = 'n_'+last_conv_layer_name
    #last_conv_layer_n = model.get_layer(last_conv_layer_n_name)

    #
    #last_conv_layer = last_conv_layer_n
    #last_conv_layer_output = last_conv_layer.act.spike_count[0]
    #last_conv_layer_output = last_conv_layer_output[0]
    #last_conv_layer_output = tf.reduce_mean(last_conv_layer_output,axis=0)


    #
    #last_conv_layer_output = model.get_layer(last_conv_layer_name).output_integ[0]
    ##last_conv_layer_output = tf.reduce_mean(last_conv_layer.act.spike_count,axis=0)


    # original
    #heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    #heatmap = tf.squeeze(heatmap)


    # For visualization purpose, we will add also normalize the heatmap between 0 & 1
    #heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)
    #heatmap = tf.maximum(heatmap,0)
    if normalize:
        heatmap = tf.math.divide_no_nan(heatmap,tf.math.reduce_max(heatmap))


    return heatmap.numpy()



