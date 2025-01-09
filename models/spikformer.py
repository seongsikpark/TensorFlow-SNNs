from tensorflow.keras.layers import LayerNormalization
from models.resnet import act_type
import tensorflow as tf
import lib_snn


def mlp(x, in_features,k_init,tdbn, name_num=0, hidden_features=None, out_features=None):
    out_features = out_features
    hidden_features = hidden_features

    B, N, C = x.shape[0],x.shape[1],x.shape[2]
    # x = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(hidden_features,kernel_initializer=k_init,name='MLP_fc1_'+str(name_num)))(x)
    # x = tf.keras.layers.Flatten(name='MLP_f1_'+str(name_num))(x)
    x = lib_snn.layers.Dense(hidden_features, kernel_initializer=k_init, name='MLP_fc1_' + str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B*N,hidden_features)))(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='MLP_bn_fc1_' + str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,hidden_features)))(x)
    x = lib_snn.activations.Activation(act_type=act_type, name='MLP_n_fc1_' + str(name_num))(x)

    # x = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(out_features,kernel_initializer=k_init,name='MLP_fc2_'+str(name_num)))(x)
    # x = tf.keras.layers.Flatten(name='MLP_f2_'+str(name_num))(x)
    x = lib_snn.layers.Dense(out_features, kernel_initializer=k_init, name='MLP_fc2_' + str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B*N,out_features)))(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='MLP_bn_fc2_' + str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,out_features)))(x)
    x = lib_snn.activations.Activation(act_type=act_type, name='MLP_n_fc2_' + str(name_num))(x)
    return x

def mlp_flat(x, in_features,k_init,tdbn, name_num=0, hidden_features=None, out_features=None):
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features

    B, N, C = x.shape[0],x.shape[1],x.shape[2]
    # x = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(hidden_features,kernel_initializer=k_init,name='MLP_fc1_'+str(name_num)))(x)
    # x = tf.keras.layers.Flatten(name='MLP_f1_'+str(name_num))(x)
    x = lib_snn.layers.Dense(hidden_features, kernel_initializer=k_init, name='MLP_fc1_' + str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N*hidden_features)))(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='MLP_bn_fc1_' + str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,hidden_features)))(x)
    x = lib_snn.activations.Activation(act_type=act_type, name='MLP_n_fc1_' + str(name_num))(x)

    # x = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(out_features,kernel_initializer=k_init,name='MLP_fc2_'+str(name_num)))(x)
    # x = tf.keras.layers.Flatten(name='MLP_f2_'+str(name_num))(x)
    x = lib_snn.layers.Dense(out_features, kernel_initializer=k_init, name='MLP_fc2_' + str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N*out_features)))(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='MLP_bn_fc2_' + str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,out_features)))(x)
    x = lib_snn.activations.Activation(act_type=act_type, name='MLP_n_fc2_' + str(name_num))(x)
    return x

def ssa(x, dim,k_init,tdbn, num_heads=12, name_num=0, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
    assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
    B, N, C = x.shape[0],x.shape[1],x.shape[2]
    scale = 0.125
    x_for_qkv = x
    # q_d = lib_snn.layers.Flatten(data_format='channels_last',name='q_f'+str(name_num))(x_for_qkv)
    q_d = lib_snn.layers.Dense(C, kernel_initializer=k_init, name='q_fc' + str(name_num))(x_for_qkv)
    # q_d = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(C, kernel_initializer=k_init, name='q_fc' + str(name_num)))(x_for_qkv)
    # q_d = lib_snn.layers.Flatten(data_format='channels_last',name='q_f'+str(name_num))(q_d)
    #q_d = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B*N,C)))(q_d)
    q_b = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='q_bn' + str(name_num))(q_d)
    #q_b = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,C)))(q_b)
    q_a = lib_snn.activations.Activation(act_type=act_type, name='ssa_q_lif' + str(name_num))(q_b)
    q = tf.keras.layers.Reshape((N,num_heads, C // num_heads))(q_a)
    q= tf.keras.layers.Lambda(lambda x : tf.transpose(x, perm=[0,2,1,3]))(q)

    # k_d = lib_snn.layers.Flatten(data_format='channels_last',name='k_f'+str(name_num))(x_for_qkv)
    k_d = lib_snn.layers.Dense(C, kernel_initializer=k_init, name='k_fc' + str(name_num))(x_for_qkv)
    # k_d = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(C, kernel_initializer=k_init, name='k_fc' + str(name_num)))(x_for_qkv)
    # k_d = tf.keras.layers.Reshape((B*N,C))(k_d)
    #k_d = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B*N,C)))(k_d)
    # k_d = lib_snn.layers.Flatten(data_format='channels_last',name='k_f'+str(name_num))(k_d)
    k_b = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='k_bn' + str(name_num))(k_d)
    #k_b = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,C)))(k_b)
    k_a = lib_snn.activations.Activation(act_type=act_type, name='ssa_k_lif' + str(name_num))(k_b)
    k = tf.keras.layers.Reshape((N, num_heads, C//num_heads))(k_a)
    k= tf.keras.layers.Lambda(lambda x : tf.transpose(x, perm=[0,2,1,3]))(k)

    # v_d = lib_snn.layers.Flatten(data_format='channels_last',name='v_f'+str(name_num))(x_for_qkv)
    v_d = lib_snn.layers.Dense(C, kernel_initializer=k_init, name='v_fc' + str(name_num))(x_for_qkv)
    # v_d = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(C, kernel_initializer=k_init, name='v_c' + str(name_num)))(x_for_qkv)
    # v_d = lib_snn.layers.Flatten(data_format='channels_last',name='v_f'+str(name_num))(v_d)
    #v_d = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B*N,C)))(v_d)
    v_b= lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='v_bn' + str(name_num))(v_d)
    #v_b = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,C)))(v_b)
    v_a = lib_snn.activations.Activation(act_type=act_type, name='ssa_v_lif' + str(name_num))(v_b)
    v = tf.keras.layers.Reshape((N, num_heads, C//num_heads))(v_a)
    v= tf.keras.layers.Lambda(lambda x : tf.transpose(x, perm=[0,2,1,3]))(v)

    attn = tf.keras.layers.Lambda(lambda tensors: tf.matmul(tensors[0],tensors[1]))([q,tf.keras.layers.Lambda(lambda x : tf.transpose(x, perm=[0,1,3,2]))(k)])
    attn = tf.keras.layers.Lambda(lambda x: x*scale)(attn)

    x = tf.keras.layers.Lambda(lambda tensors: tf.matmul(tensors[0],tensors[1]))([attn,v])
    x= tf.keras.layers.Lambda(lambda x : tf.transpose(x, perm=[0,1,3,2]))(x)
    x = tf.keras.layers.Reshape((N,C))(x)
    x = lib_snn.activations.Activation(act_type=act_type, name='ssa_att_lif' + str(name_num))(x)

    # x = lib_snn.layers.Flatten(data_format='channels_last',name='proj_f'+str(name_num))(x)
    x = lib_snn.layers.Dense(C, kernel_initializer=k_init, name='proj_fc' + str(name_num))(x)
    # x = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(C,kernel_initializer=k_init,name='MLP_fc2_'+str(name_num)))(x)
    # x = lib_snn.layers.Flatten(data_format='channels_last',name='proj_f'+str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B*N,C)))(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='proj_bn' + str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,C)))(x)
    x = lib_snn.activations.Activation(act_type=act_type, name='ssa_proj_lif' + str(name_num))(x)

    return x

def ssa_flat(x, dim,k_init,tdbn, num_heads=12, name_num=0, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
    assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
    B, N, C = x.shape[0],x.shape[1],x.shape[2]
    scale = 0.3
    x_for_qkv = x
    # q_d = lib_snn.layers.Flatten(data_format='channels_last',name='q_f'+str(name_num))(x_for_qkv)
    q_d = lib_snn.layers.Dense(C, kernel_initializer=k_init, name='q_fc' + str(name_num))(x_for_qkv)
    # q_d = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(C, kernel_initializer=k_init, name='q_fc' + str(name_num)))(x_for_qkv)
    # q_d = lib_snn.layers.Flatten(data_format='channels_last',name='q_f'+str(name_num))(q_d)
    #q_d = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N*C)))(q_d)
    q_b = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='q_bn' + str(name_num))(q_d)
    #q_b = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,C)))(q_b)
    q_a = lib_snn.activations.Activation(act_type=act_type, name='ssa_q_lif' + str(name_num))(q_b)
    q = tf.keras.layers.Reshape((N,num_heads, C // num_heads))(q_a)
    q= tf.keras.layers.Lambda(lambda x : tf.transpose(x, perm=[0,2,1,3]))(q)

    # k_d = lib_snn.layers.Flatten(data_format='channels_last',name='k_f'+str(name_num))(x_for_qkv)
    k_d = lib_snn.layers.Dense(C, kernel_initializer=k_init, name='k_fc' + str(name_num))(x_for_qkv)
    # k_d = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(C, kernel_initializer=k_init, name='k_fc' + str(name_num)))(x_for_qkv)
    # k_d = tf.keras.layers.Reshape((B*N,C))(k_d)
    #k_d = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N*C)))(k_d)
    # k_d = lib_snn.layers.Flatten(data_format='channels_last',name='k_f'+str(name_num))(k_d)
    k_b = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='k_bn' + str(name_num))(k_d)
    #k_b = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,C)))(k_b)
    k_a = lib_snn.activations.Activation(act_type=act_type, name='ssa_k_lif' + str(name_num))(k_b)
    k = tf.keras.layers.Reshape((N, num_heads, C//num_heads))(k_a)
    k= tf.keras.layers.Lambda(lambda x : tf.transpose(x, perm=[0,2,1,3]))(k)

    # v_d = lib_snn.layers.Flatten(data_format='channels_last',name='v_f'+str(name_num))(x_for_qkv)
    v_d = lib_snn.layers.Dense(C, kernel_initializer=k_init, name='v_fc' + str(name_num))(x_for_qkv)
    # v_d = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(C, kernel_initializer=k_init, name='v_c' + str(name_num)))(x_for_qkv)
    # v_d = lib_snn.layers.Flatten(data_format='channels_last',name='v_f'+str(name_num))(v_d)
    #v_d = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N*C)))(v_d)
    v_b= lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='v_bn' + str(name_num))(v_d)
    #v_b = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,C)))(v_b)
    v_a = lib_snn.activations.Activation(act_type=act_type, name='ssa_v_lif' + str(name_num))(v_b)
    v = tf.keras.layers.Reshape((N, num_heads, C//num_heads))(v_a)
    v= tf.keras.layers.Lambda(lambda x : tf.transpose(x, perm=[0,2,1,3]))(v)

    attn = tf.keras.layers.Lambda(lambda tensors: tf.matmul(tensors[0],tensors[1]))([q,tf.keras.layers.Lambda(lambda x : tf.transpose(x, perm=[0,1,3,2]))(k)])
    attn = tf.keras.layers.Lambda(lambda x: x*scale)(attn)

    x = tf.keras.layers.Lambda(lambda tensors: tf.matmul(tensors[0],tensors[1]))([attn,v])
    x= tf.keras.layers.Lambda(lambda x : tf.transpose(x, perm=[0,1,3,2]))(x)
    x = tf.keras.layers.Reshape((N,C))(x)
    x = lib_snn.activations.Activation(act_type=act_type, name='ssa_att_lif' + str(name_num))(x)

    # x = lib_snn.layers.Flatten(data_format='channels_last',name='proj_f'+str(name_num))(x)
    x = lib_snn.layers.Dense(C, kernel_initializer=k_init, name='proj_fc' + str(name_num))(x)
    # x = tf.keras.layers.TimeDistributed(lib_snn.layers.Dense(C,kernel_initializer=k_init,name='MLP_fc2_'+str(name_num)))(x)
    # x = lib_snn.layers.Flatten(data_format='channels_last',name='proj_f'+str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N*C)))(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='proj_bn' + str(name_num))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (B,N,C)))(x)
    x = lib_snn.activations.Activation(act_type=act_type, name='ssa_proj_lif' + str(name_num))(x)

    return x
def block(x, dim, num_heads,k_init,tdbn, name_num=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
          drop_path=0., norm_layer=LayerNormalization, sr_ratio=1):
    # norm1 = norm_layer(axis=-1)
    # norm2 = norm_layer(axis=-1)
    norm1 = x
    norm2 = x
    mlp_hidden_dim = int(dim * mlp_ratio)
    block_in = x
    attn_out = ssa(norm1, dim=dim,k_init=k_init,tdbn=tdbn, num_heads=num_heads, qkv_bias=qkv_bias,
                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, name_num=name_num)
    x = lib_snn.layers.Add(name='block_attn_'+str(name_num))([block_in,attn_out])

    mlp_out = mlp(norm2, name_num=name_num,k_init=k_init,tdbn=tdbn, in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)
    x = lib_snn.layers.Add(name='block_attn_mlp'+str(name_num))([x,mlp_out])

    return x


def sps(x, input_shape,k_init,tdbn,tdbn_first_layer,use_bn_feat, patch_size=4, embed_dims=384):
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    # B, H, W, C = x.shape[0].x.shape[1],x.shape[2],x.shape[3]
    syn_c1 = lib_snn.layers.Conv2D(embed_dims // 8, kernel_size=3, padding='SAME', use_bn=use_bn_feat,
                                   kernel_initializer=k_init, name='conv1')(x)
    norm_c1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn_first_layer, name='bn_conv1')(syn_c1)
    a_c1 = lib_snn.activations.Activation(act_type=act_type, name='n_conv1')(norm_c1)

    syn_c2 = lib_snn.layers.Conv2D(embed_dims // 4, kernel_size=3, padding='SAME', kernel_initializer=k_init,
                                   name='sps_conv2')(a_c1)
    norm_c2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='sps_bn2')(syn_c2)
    a_c2 = lib_snn.activations.Activation(act_type=act_type, name='sps_lif2')(norm_c2)

    syn_c3 = lib_snn.layers.Conv2D(embed_dims // 2, kernel_size=3, padding='SAME', kernel_initializer=k_init,
                                   name='sps_conv3')(a_c2)
    bn_c3 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='sps_bn3')(syn_c3)
    a_c3 = lib_snn.activations.Activation(act_type=act_type, name='sps_lif3')(bn_c3)
    a_p_c3 = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='sps_conv3_p')(a_c3)  # why strides have to be only (2,2)?

    syn_c4 = lib_snn.layers.Conv2D(embed_dims, kernel_size=3, padding='SAME', kernel_initializer=k_init,
                                   name='sps_conv4')(a_p_c3)
    bn_c4 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='sps_bn4')(syn_c4)
    a_c4 = lib_snn.activations.Activation(act_type=act_type, name='sps_lif4')(bn_c4)
    a_p_c4 = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='sps_conv4_p')(a_c4)  # why strides have to be only (2,2)?

    rpe_feat = lib_snn.layers.Identity(name='rpe_feat')(a_p_c4)
    rpe_c1 = lib_snn.layers.Conv2D(embed_dims, kernel_size=3, padding='SAME', kernel_initializer=k_init,
                                   name='sps_conv_rpe')(a_p_c4)
    rpe_norm_c1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='sps_bn_rpe')(rpe_c1)
    rpe_a1 = lib_snn.activations.Activation(act_type=act_type, name='sps_lif_rpe')(rpe_norm_c1)
    rpe_out = lib_snn.layers.Add(name='sps_out')([rpe_feat,rpe_a1])

    x = tf.keras.layers.Reshape((-1,embed_dims))(rpe_out)
    # x = tf.transpose(x, perm=[0, 1, 2])
    return x

def sps_ImageNet(x, input_shape,k_init,tdbn,tdbn_first_layer,use_bn_feat, patch_size=4, embed_dims=384):
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    # B, H, W, C = x.shape[0].x.shape[1],x.shape[2],x.shape[3]
    syn_c1 = lib_snn.layers.Conv2D(embed_dims // 8, kernel_size=3, padding='SAME', use_bn=use_bn_feat,
                                   kernel_initializer=k_init, name='sps_conv1')(x)
    norm_c1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn_first_layer, name='sps_bn1')(syn_c1)
    a_c1 = lib_snn.activations.Activation(act_type=act_type, name='sps_lif1')(norm_c1)
    a_p_c1 = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='sps_conv1_p')(a_c1)

    syn_c2 = lib_snn.layers.Conv2D(embed_dims // 4, kernel_size=3, padding='SAME', kernel_initializer=k_init,
                                   name='sps_conv2')(a_p_c1)
    norm_c2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='sps_bn2')(syn_c2)
    a_c2 = lib_snn.activations.Activation(act_type=act_type, name='sps_lif2')(norm_c2)
    a_p_c2 = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='sps_conv2_p')(a_c2)

    syn_c3 = lib_snn.layers.Conv2D(embed_dims // 2, kernel_size=3, padding='SAME', kernel_initializer=k_init,
                                   name='sps_conv3')(a_p_c2)
    bn_c3 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='sps_bn3')(syn_c3)
    a_c3 = lib_snn.activations.Activation(act_type=act_type, name='sps_lif3')(bn_c3)
    a_p_c3 = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='sps_conv3_p')(a_c3)  # why strides have to be only (2,2)?

    syn_c4 = lib_snn.layers.Conv2D(embed_dims, kernel_size=3, padding='SAME', kernel_initializer=k_init,
                                   name='sps_conv4')(a_p_c3)
    bn_c4 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='sps_bn4')(syn_c4)
    a_c4 = lib_snn.activations.Activation(act_type=act_type, name='sps_lif4')(bn_c4)
    a_p_c4 = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='sps_conv4_p')(a_c4)  # why strides have to be only (2,2)?

    rpe_feat = lib_snn.layers.Identity(name='rpe_feat')(a_p_c4)
    rpe_c1 = lib_snn.layers.Conv2D(embed_dims, kernel_size=3, padding='SAME', kernel_initializer=k_init,
                                   name='sps_conv_rpe')(a_p_c4)
    rpe_norm_c1 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='sps_bn_rpe')(rpe_c1)
    rpe_a1 = lib_snn.activations.Activation(act_type=act_type, name='sps_lif_rpe')(rpe_norm_c1)
    rpe_out = lib_snn.layers.Add(name='sps_out')([rpe_feat,rpe_a1])

    x = tf.keras.layers.Reshape((-1,embed_dims))(rpe_out)
    # x = tf.transpose(x, perm=[0, 1, 2])
    return x

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
        depths=8,
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



    input_tensor =tf.keras.layers.Input(shape=input_shape,batch_size=batch_size)
    input = lib_snn.layers.InputGenLayer(name='in')(input_tensor)
    if conf.nn_mode=='SNN':
        input = lib_snn.activations.Activation(act_type=act_type,loc='IN',name='n_in')(input)
    if dataset_name == 'ImageNet':
        sps_x = sps_ImageNet(input, input_shape=input_shape, patch_size=patch_size,
                    embed_dims=embed_dims,k_init=k_init,tdbn=tdbn,tdbn_first_layer=tdbn_first_layer,use_bn_feat=use_bn_feat)
    elif dataset_name == 'CIFAR10_DVS':
        sps_x = sps_ImageNet(input, input_shape=input_shape, patch_size=patch_size,
                    embed_dims=embed_dims,k_init=k_init,tdbn=tdbn,tdbn_first_layer=tdbn_first_layer,use_bn_feat=use_bn_feat)
    else:
        sps_x = sps(input, input_shape=input_shape, patch_size=patch_size,
                    embed_dims=embed_dims, k_init=k_init, tdbn=tdbn, tdbn_first_layer=tdbn_first_layer,
                    use_bn_feat=use_bn_feat)
    # for stage_idx, depth in enumerate(depths):
    block_x=sps_x
    for i in range(depths):
        block_x = block(block_x, dim=embed_dims, num_heads=num_heads,
                        mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                        norm_layer=LayerNormalization, sr_ratio=sr_ratios, name_num=i,k_init=k_init,tdbn=tdbn)

    # x_f =tf.keras.layers.Flatten(data_format=data_format,name='flatten')(block_x)
    gap_x = tf.keras.layers.GlobalAveragePooling1D(data_format=data_format)(block_x)
    output_tensor = lib_snn.layers.Dense(classes, last_layer=True, kernel_initializer=k_init,name='predictions')(gap_x)
    a_p = lib_snn.activations.Activation(act_type=act_type_out,loc='OUT',name='n_predictions')(output_tensor)
    a_p = lib_snn.activations.Activation(act_type='softmax',name='a_predictions')(a_p)
    model = lib_snn.model.Model(input_tensor, a_p, batch_size, input_shape, classes, conf, name=model_name)
    # model = lib_snn.model.Model(input_tensor, output_tensor, batch_size, input_shape, classes, conf, name=model_name)
    return model