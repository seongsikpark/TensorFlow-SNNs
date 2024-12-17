import tensorflow as tf
import lib_snn
import tensorflow.keras.layers as nn
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LayerNormalization

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
        patch_size=4,
        embed_dims=384,
        num_heads=12,
        classes=1000,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        depths=4,
        sr_ratios=8,
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




    def block(x, dim, num_heads,name_num=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
              drop_path=0., norm_layer=LayerNormalization, sr_ratio=1):

        def mlp(x, in_features, hidden_features=None, out_features=None):
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            B,N,C = tf.shape(x)[0],tf.shape(x)[1], tf.shape(x)[2]
            # x_ = tf.reshape(x, [B,N,C])
            x = lib_snn.layers.Dense(hidden_features,kernel_initializer=k_init,  name='MLP_fc1_'+str(name_num))(x)
            # x = tf.transpose(x, perm=[0,2,1])
            x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='MLP_bn_fc1_'+str(name_num))(x)
            # x = tf.transpose(x, perm=[0,2,1])
            # x = tf.reshape(x, [B, N, hidden_features])
            x = lib_snn.activations.Activation(act_type=act_type,name='MLP_n_fc1_'+str(name_num))(x)

            x = tf.reshape(x, [B,N,hidden_features])
            x = lib_snn.layers.Dense(out_features,kernel_initializer=k_init, name='MLP_fc2_'+str(name_num))(x)
            # x = tf.transpose(x, perm=[0,2,1])
            x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='MLP_bn_fc2_'+str(name_num))(x)
            # x = tf.transpose(x, perm=[0,2,1])
            # x = tf.reshape(x, [B, N, out_features])
            x = lib_snn.activations.Activation(act_type=act_type,name='MLP_n_fc2_'+str(name_num))(x)

            return x

        def ssa(x, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):

            assert dim % num_heads ==0, f"dim {dim} should be divisible by num_heads {num_heads}."
            B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
            scale = 0.125
            x_for_qkv = tf.reshape(x, [ B, N, C])

            q_linear_out = lib_snn.layers.Dense(dim,kernel_initializer=k_init, name='q_fc'+str(name_num))(x_for_qkv)
            # q_linear_out = tf.transpose(q_linear_out, perm=[0, 2, 1])
            q_linear_out = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='q_bn'+str(name_num))(q_linear_out)
            # q_linear_out = tf.transpose(q_linear_out, perm=[0, 2, 1])
            # q_linear_out = tf.reshape(q_linear_out, [ B, N, C])
            q_linear_out = lib_snn.activations.Activation(act_type=act_type, name='ssa_q_lif'+str(name_num))(q_linear_out)
            q = tf.reshape(q_linear_out,  [B, N, num_heads, C // num_heads])
            q = tf.transpose(q, perm=[0, 2,1, 3,])  # Rearrange dimensions

            k_linear_out = lib_snn.layers.Dense(dim,kernel_initializer=k_init, name='k_fc'+str(name_num))(x_for_qkv)
            # k_linear_out = tf.transpose(k_linear_out, perm=[0, 2, 1])
            k_linear_out = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='k_bn'+str(name_num))(k_linear_out)
            # k_linear_out = tf.transpose(k_linear_out, perm=[0, 2, 1])
            # k_linear_out = tf.reshape(k_linear_out, [ B, N, C])
            k_linear_out = lib_snn.activations.Activation(act_type=act_type, name='ssa_k_lif'+str(name_num))(k_linear_out)
            k = tf.reshape(k_linear_out, [ B, N, num_heads, C // num_heads])
            k = tf.transpose(k, perm=[0,2, 1, 3])

            v_linear_out = lib_snn.layers.Dense(dim,kernel_initializer=k_init, name='v_fc'+str(name_num))(x_for_qkv)
            # v_linear_out = tf.transpose(v_linear_out, perm=[0, 2, 1])
            v_linear_out = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='v_bn'+str(name_num))(v_linear_out)
            # v_linear_out = tf.transpose(v_linear_out, perm=[0, 2, 1])
            # v_linear_out = tf.reshape(v_linear_out, [ B, N, C])
            v_linear_out = lib_snn.activations.Activation(act_type=act_type, name='ssa_v_lif'+str(name_num))(v_linear_out)
            v = tf.reshape(v_linear_out, [ B, N, num_heads, C // num_heads])
            v = tf.transpose(v, perm=[0, 2, 1, 3])

            attn = tf.matmul(q, k, transpose_b=True) * scale
            x = tf.matmul(attn, v)

            x = tf.transpose(x, perm=[0, 1, 2, 3 ])  # [T, B, N, num_heads, head_dim]
            x = tf.reshape(x, [ B, N, C])  # Combine head_dim and num_heads
            x = lib_snn.activations.Activation(act_type=act_type, name='ssa_att_lif'+str(name_num))(x)

            x = tf.reshape(x, [ B, N, C])  # Flatten T and B axes
            x = lib_snn.layers.Dense(dim,kernel_initializer=k_init, name='proj_fc'+str(name_num))(x)
            x = tf.transpose(x, perm=[0, 2, 1])  # Transpose for BatchNorm
            x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='proj_bn'+str(name_num))(x)
            x = tf.transpose(x, perm=[0, 2, 1])
            x = tf.reshape(x, [ B, N, C])
            x = lib_snn.activations.Activation(act_type=act_type, name='ssa_proj_lif'+str(name_num))(x)

            return x
        norm1 = norm_layer(axis=-1)
        norm2 = norm_layer(axis=-1)

        mlp_hidden_dim = int(dim * mlp_ratio)

        attn_out = ssa(norm1(x), dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        x = x + attn_out

        mlp_out = mlp(norm2(x), in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)
        x = x + mlp_out

        return x

    def sps(x, input_shape, patch_size=4, embed_dims=384):

        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        H, W = input_shape[0] // patch_size[0], input_shape[1] // patch_size[1]
        num_patches = H * W
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [-1, H, W, C])  # T*B, H, W, C
        syn_c1 = lib_snn.layers.Conv2D(embed_dims //8, kernel_size=3, padding='SAME', use_bn=use_bn_feat,kernel_initializer=k_init, name='sps_conv1')(x)
        norm_c1 = tf.reshape(lib_snn.layers.BatchNormalization(en_tdbn=tdbn_first_layer,name='sps_bn1')(syn_c1), [ B, -1, H, W])
        a_c1 = tf.reshape(lib_snn.activations.Activation(act_type=act_type,name='sps_lif1')(norm_c1), [-1, H, W, embed_dims // 8])

        syn_c2 = lib_snn.layers.Conv2D(embed_dims // 4, kernel_size=3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='sps_conv2')(a_c1)
        norm_c2 = tf.reshape(lib_snn.layers.BatchNormalization(tdbn,name='sps_bn2')(syn_c2), [ B, -1, H, W])
        a_c2 = tf.reshape(lib_snn.activations.Activation(act_type=act_type,name='sps_lif2')(norm_c2), [-1, H, W, embed_dims // 4])

        syn_c3 = lib_snn.layers.Conv2D(embed_dims // 2, kernel_size=3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='sps_conv3')(a_c2)
        bn_c3 = tf.reshape(lib_snn.layers.BatchNormalization(tdbn,name='sps_bn3')(syn_c3), [ B, -1, H, W])
        a_c3 = tf.reshape(lib_snn.activations.Activation(act_type=act_type,name='sps_lif3')(bn_c3), [-1, H, W, embed_dims // 2])
        a_p_c3 = lib_snn.layers.MaxPool2D((2,2),(2,2),name='sps_conv3_p')(a_c3) # why strides have to be only (2,2)?

        syn_c4 = lib_snn.layers.Conv2D(embed_dims, kernel_size=3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='sps_conv4')(a_p_c3)
        bn_c4 = tf.reshape(lib_snn.layers.BatchNormalization(tdbn,name='sps_bn4')(syn_c4), [ B, -1, H//2, W//2])
        a_c4 = tf.reshape(lib_snn.activations.Activation(act_type=act_type,name='sps_lif4')(bn_c4), [-1, H//2, W //2, embed_dims])
        a_p_c4 = lib_snn.layers.MaxPool2D((2,2),(2,2),name='sps_conv3_4')(a_c4) # why strides have to be only (2,2)?

        rpe_feat = tf.reshape(a_p_c4, [ B, embed_dims, H // 4, W // 4])
        rpe_c1 = lib_snn.layers.Conv2D(embed_dims, kernel_size=3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='sps_conv_rpe')(a_p_c4)
        rpe_norm_c1= tf.reshape(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='sps_bn_rpe')(rpe_c1), [ B, embed_dims, H // 4, W // 4])
        rpe_a1 = lib_snn.activations.Activation(act_type=act_type, name='sps_lif_rpe')(rpe_norm_c1)
        rpe_out = rpe_a1 + rpe_feat

        x = tf.reshape(rpe_out, [B, -1, embed_dims])
        x = tf.transpose(x, perm=[0, 1, 2])
        return x

    input_tensor =tf.keras.layers.Input(shape=input_shape,batch_size=batch_size)
    input = lib_snn.layers.InputGenLayer(name='in')(input_tensor)
    if conf.nn_mode=='SNN':
        input = lib_snn.activations.Activation(act_type=act_type,loc='IN',name='n_in')(input)

    x = sps(input, input_shape=input_shape, patch_size=patch_size,
            embed_dims=embed_dims)

    # for stage_idx, depth in enumerate(depths):
    for i in range(depths):
        x = block(x, dim=embed_dims, num_heads=num_heads,
                  mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                  norm_layer=LayerNormalization, sr_ratio=sr_ratios,name_num=i)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    output_tensor = lib_snn.layers.Dense(classes, last_layer=True, kernel_initializer=k_init,name='predictions')(x)
    a_p = lib_snn.activations.Activation(act_type='softmax',loc='OUT',name='n_predictions')(output_tensor)
    a_p = lib_snn.activations.Activation(act_type='softmax',name='a_predictions')(output_tensor)
    model = lib_snn.model.Model(input_tensor, a_p, batch_size, input_shape, classes, conf, name=model_name)
    return model