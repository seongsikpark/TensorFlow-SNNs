import tensorflow as tf
import lib_snn
import tensorflow.keras.layers as nn
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential

from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import  Rearrange

from models.resnet import act_type


import tensorflow as tf

from tensorflow.python.keras.engine import training

import functools

#
#import lib_snn.layers
import lib_snn

from lib_snn.layers import tfn

def spikformer(
        batch_size,
        input_shape,
        conf,
        model_name,
        include_top=True,
        embed_dims=[64,128,256],
        num_heads=[1,2,4],
        classes=1000,
        mlp_ratios=[4,4,4],
        qkv_bias=False,
        qk_scale=None,
        depths=[6,8,6],
        sr_ratios=[8,4,2],
        dataset_name=None,
        **kwargs):

    data_format = conf.data_format

    #
    act_relu = 'relu'
    act_sm = 'softmax'

    #
    if conf.nn_mode=='ANN':
        dropout_conv_r = [0.2, 0.2, 0.0]      # DNN training
    elif conf.nn_mode=='SNN':
        #dropout_conv_r = [0.5, 0.5, 0.0]      # SNN training
        #dropout_conv_r = [0.25, 0.25, 0.25]      # SNN training
        #dropout_conv_r = [0.3, 0.3, 0.3]      # SNN training
        #dropout_conv_r = [0.25, 0.25, 0.25]      # SNN training
        dropout_conv_r = [0.0, 0.0, 0.0]      # SNN training
    else:
        assert False
    #
    initial_channels = kwargs.pop('initial_channels', None)
    #assert initial_channels is not None
    if initial_channels is None:
        initial_channels = 2


    #
    use_bn_feat = conf.use_bn
    use_bn_cls = conf.use_bn

    #
    channels = initial_channels

    #
    if dataset_name=='ImageNet':
        n_dim_cls = 4096
    elif 'CIFAR' in dataset_name:
        n_dim_cls = 512
    else:
        assert False

    #
    k_init = 'glorot_uniform'
    #k_init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0,seed=None)

    # pooling
    if conf.pooling_vgg=='max':
        pool = lib_snn.layers.MaxPool2D
    elif conf.pooling_vgg=='avg':
        pool = lib_snn.layers.AveragePooling2D
    else:
        assert False


    #
    if conf.nn_mode=='ANN':
        act_type = 'relu'
        act_type_out = 'softmax'
    else:
        act_type = conf.n_type
        act_type_out = conf.n_type


    #
    tdbn_first_layer = conf.nn_mode=='SNN' and conf.input_spike_mode=='POISSON' and conf.tdbn
    tdbn = conf.nn_mode=='SNN' and conf.tdbn

    def MLP(x, in_features, hidden_features=None, out_features=None)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        T,B,N,C = tf.shape(x)[0],tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_ = tf.reshape(x, [T*B,N,C])
        x = lib_snn.layers.Dense(hidden_features, name='fc1')
        x = tf.transpose(x, perm=[0,2,1])
        x = self.fc1_bn(x)
        x = tf.transpose(x, perm=[0,2,1])
        x = tf.reshape(x, [T,B,N, self.c_hidden])
        x = self.fc1_lif(x)

        x = tf.reshape(x, [T*B,N, self.c_hidden])
        x = self.fc2_linear(x)
        x = tf.transpose(x, perm=[0,2,1])
        x = self.fc2_bn(x)
        x = tf.transpose(x, perm=[0,2,1])
        x = tf.reshape(x, [T,B,N,C])
        x = self.fc2_lif(x)

        return x

    img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

    input = lib_snn.layers.InputGenLayer(name='in')(img_input)
    if conf.nn_mode =='SNN':
        input = lib_snn.activations.Activation(act_type=act_type,loc='IN',name='n_in')(input)

    sps_c1 = lib_snn.layers.Conv2D()


class MLP(lib_snn.model.Model):
    def __init__(self,in_dim,hidden_dim,out_dim,drop=0.0):
        super(MLP,self).__init__()
        dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1_linear = lib_snn.layers.Dense(hidden_dim,name='fc1')
        self.fc1_bn=lib_snn.layers.BatchNormalization(name='bn_fc1')
        self.fc1_lif=lib_snn.activations.Activation(act_type='LIF',name='n_fc1')

        self.fc2_linear = lib_snn.layers.Dense(dim,name='fc2')
        self.fc2_bn=lib_snn.layers.BatchNormalization(name='bn_fc2')
        self.fc2_lif=lib_snn.activations.Activation(act_type='LIF',name='n_fc2')
        self.c_hidden = hidden_dim
        self.c_output = dim

    def call(self, x):
        T,B,N,C = tf.shape(x)[0],tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

class SSA(lib_snn.model.Model):
    def __init__(self,dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads} "
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear =  lib_snn.layers.Dense(dim,name='lin_q1')
        self.q_bn = lib_snn.layers.BatchNormalization(name='bn_q1')
        self.q_lif = lib_snn.activations.Activation(acy_type='LIF',name='lif_q1')

        self.k_linear =  lib_snn.layers.Dense(dim,name='lin_k1')
        self.k_bn = lib_snn.layers.BatchNormalization(name='bn_k1')
        self.k_lif = lib_snn.activations.Activation(acy_type='LIF',name='lif_k1')

        self.v_linear =  lib_snn.layers.Dense(dim,name='lin_v1')
        self.v_bn = lib_snn.layers.BatchNormalization(name='bn_v1')
        self.v_lif = lib_snn.activations.Activation(acy_type='LIF',name='lif_v1')

        self.attn_Lif = lib_snn.activations.Activation(act_type='LIF',name='lif_attn')
        self.proj_linear = lib_snn.layers.Dense(dim,name='lin_proj')
        self.proj_bn = lib_snn.layers.BatchNormalization(name='bn_proj')
        self.proj_Lif = lib_snn.activations.Activation(act_type='LIF',name='lif_proj')

    def call(self,x):
        T,B,N,C = tf.shape(x)[0],tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_for_qkv= tf.reshape(x, [T*B,N,C])

        q_lin_out = self.q_linear(x_for_qkv)
        q_lin_out = tf.transpose(q_lin_out, perm=[0,2,1])
        q_lin_out = self.q_bn(q_lin_out)
        q_lin_out = tf.transpose(q_lin_out, perm=[0,2,1])
        q_lin_out = tf.reshape(q_lin_out, [T, B, N, C])
        q_lin_out = self.q_lif(q_lin_out)
        q = tf.reshape(q_lin_out, [T,B,N, self.num_heads, C // self.num_heads])
        q = tf.transpose(q, perm=[0, 1, 3, 2, 4])

        k_lin_out = self.k_linear(x_for_qkv)
        k_lin_out = tf.transpose(k_lin_out, perm=[0,2,1])
        k_lin_out = self.k_bn(k_lin_out)
        k_lin_out = tf.transpose(k_lin_out, perm=[0,2,1])
        k_lin_out = tf.reshape(k_lin_out, [T, B, N, C])
        k_lin_out = self.k_lif(k_lin_out)
        k = tf.reshape(k_lin_out, [T,B,N, self.num_heads, C // self.num_heads])
        k = tf.transpose(k, perm=[0, 1, 3, 2, 4])

        v_lin_out = self.v_linear(x_for_qkv)
        v_lin_out = tf.transpose(v_lin_out, perm=[0,2,1])
        v_lin_out = self.v_bn(v_lin_out)
        v_lin_out = tf.transpose(v_lin_out, perm=[0,2,1])
        v_lin_out = tf.reshape(v_lin_out, [T, B, N, C])
        v_lin_out = self.v_lif(v_lin_out)
        v = tf.reshape(v_lin_out, [T,B,N, self.num_heads, C // self.num_heads])
        v = tf.transpose(v, perm=[0, 1, 3, 2, 4])
