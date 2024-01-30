################################################
# tutorial_grad_cam.py
#
# This file is created based on Keras example
################################################

#
from config_tutorial_grad_cam import config

import numpy as np
import tensorflow as tf
import keras

#
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt

#
from lib_snn import grad_cam


# ImageNet tutorial

# test
#model_builder = keras.applications.xception.Xception
#img_size = (299,299)
#preprocess_input = keras.applications.xception.preprocess_input
#decode_predictions = keras.applications.xception.decode_predictions
#last_conv_layer_name = "block14_sepconv2_act"

# vgg
img_size = (224,224)
model_builder = keras.applications.vgg16.VGG16
preprocess_input = keras.applications.vgg16.preprocess_input
decode_predictions = keras.applications.vgg16.decode_predictions
last_conv_layer_name = "block1_conv1"



# the local path to our target image
img_path = keras.utils.get_file(
    "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
)

#display(Image(img_path))


# prepare image
img_array = preprocess_input(grad_cam.get_image_array(img_path,size=img_size))

# make model
model = model_builder(weights='imagenet')

# remove last layer's softmax
model.layers[-1].activation = None

# print what the top predicted class is
preds = model.predict(img_array)
print("predicted:", decode_predictions(preds, top=1)[0])

# generate class activation heatmap
heatmap = grad_cam.make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# display
plt.matshow(heatmap)
plt.show()



