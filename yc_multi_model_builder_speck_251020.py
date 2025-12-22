# yum_multi_model_builder.py
import tensorflow as tf
from keras import layers
import lib_snn
from tensorflow import keras
from tensorflow.keras import backend as K
value=0.03

class GaussianNoiseInferenceOnly(keras.layers.GaussianNoise):
    def __init__(self, stddev, seed=None, **kwargs):
        super(GaussianNoiseInferenceOnly, self).__init__(stddev, seed=seed, **kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.seed = seed

    def call(self, inputs, training=None):
        def noised():
            return inputs + self._random_generator.random_normal(
                shape=tf.shape(inputs),
                mean=0.,
                stddev=self.stddev,
                dtype=inputs.dtype)

        return K.in_train_phase(inputs, noised(), training=training)

    def get_config(self):
        config = super(GaussianNoiseInferenceOnly, self).get_config()
        return config



def _image_backbone_speck_hw(
    image_input,
    conf,
    act_type="IF",
    k_init="glorot_uniform",
    prefix="vid_",
    emb_dim=256,              # Speck-friendly (match trained a_dim if possible)
):

    pool = lib_snn.layers.AveragePooling2D
    tdbn = conf.nn_mode == 'SNN' and conf.tdbn
    use_bn_feat = conf.use_bn
    data_format = conf.data_format
    # Input & input spike
    x = lib_snn.layers.InputGenLayer(name=f"{prefix}in_audio")(image_input)
    x = lib_snn.activations.Activation(act_type=act_type, loc="IN", name=f"{prefix}n_in")(x)

    # 32x32 -> 16x16 (stride=2) then -> 8x8 by pooling
    x = lib_snn.layers.Conv2D(16, 3, strides=(2, 2), padding="SAME",
                              kernel_initializer=k_init, name=f"{prefix}conv1")(x)
    x = lib_snn.activations.Activation(act_type=act_type, name=f"{prefix}n1")(x)
    x = pool((2, 2), (2, 2), name=f"{prefix}p1")(x)  # 16 -> 8

    # 8 -> 4
    x = lib_snn.layers.Conv2D(32, 3, padding="SAME",
                              kernel_initializer=k_init, name=f"{prefix}conv2")(x)
    x = lib_snn.activations.Activation(act_type=act_type, name=f"{prefix}n2")(x)
    x = pool((2, 2), (2, 2), name=f"{prefix}p2")(x)

    # 4 -> 2
    x = lib_snn.layers.Conv2D(48, 3, padding="SAME",
                              kernel_initializer=k_init, name=f"{prefix}conv3")(x)
    x = lib_snn.activations.Activation(act_type=act_type, name=f"{prefix}n3")(x)
    x = pool((2, 2), (2, 2), name=f"{prefix}p3")(x)

    # 2 -> 1
    x = lib_snn.layers.Conv2D(64, 3, padding="SAME",
                              kernel_initializer=k_init, name=f"{prefix}conv4")(x)
    x = lib_snn.activations.Activation(act_type=act_type, name=f"{prefix}n4")(x)
    x = pool((2, 2), (2, 2), name=f"{prefix}p4")(x)


    x = tf.keras.layers.Flatten(data_format=data_format, name=f'{prefix}flatten')(x)
    x    = lib_snn.layers.Dense(256, kernel_initializer=k_init, name=f'{prefix}fc1')(x)
    x     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_fc1')(x)
    return x



def _image_backbone(image_input, conf, act_type="IF", k_init="glorot_uniform", prefix="img_"):
    tdbn = conf.nn_mode == 'SNN' and conf.tdbn
    use_bn_feat = conf.use_bn
    data_format = conf.data_format
    pool = lib_snn.layers.AveragePooling2D
    channels = 16 * 2  # = 32

    # IN
    in_image   = lib_snn.layers.InputGenLayer(name=f'{prefix}in_image')(image_input)
    # in_image = GaussianNoiseInferenceOnly(value)(in_image)
    n_in       = lib_snn.activations.Activation(act_type=act_type, loc='IN', name=f'{prefix}n_in')(in_image)


    # conv1_1 ~ conv1_3 -> pool
    syn_c1_1   = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat,
                                       kernel_initializer=k_init, name=f'{prefix}conv1_1')(n_in)
    norm_c1_1  = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv1_1')(syn_c1_1)
    # norm_c1_1 = GaussianNoiseInferenceOnly(value)(norm_c1_1)
    a_c1_1     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv1_1')(norm_c1_1)

    syn_c1_2   = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat,
                                       kernel_initializer=k_init, name=f'{prefix}conv1_2')(a_c1_1)
    norm_c1_2  = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv1_2')(syn_c1_2)
    # norm_c1_2 = GaussianNoiseInferenceOnly(value)(norm_c1_2)
    a_c1_2     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv1_2')(norm_c1_2)

    syn_c1_3   = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat,
                                       kernel_initializer=k_init, name=f'{prefix}conv1_3')(a_c1_2)
    norm_c1_3  = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv1_3')(syn_c1_3)
    # norm_c1_3 = GaussianNoiseInferenceOnly(value)(norm_c1_3)
    a_c1_3     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv1_3')(norm_c1_3)
    a_p_c1     = pool((2, 2), (2, 2), name=f'{prefix}conv1_p')(a_c1_3)

    # a_p_c1 = pool((2, 2), (2, 2), name=f'{prefix}conv1_p')(a_c1_1)

    # conv2_1 -> pool
    syn_c2_1   = lib_snn.layers.Conv2D(channels*2, 3, padding='SAME',
                                       kernel_initializer=k_init, name=f'{prefix}conv2_1')(a_p_c1)
    norm_c2_1  = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv2_1')(syn_c2_1)
    # norm_c2_1 = GaussianNoiseInferenceOnly(value)(norm_c2_1)
    a_c2_1     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv2_1')(norm_c2_1)
    a_p_c2_1   = pool((2, 2), (2, 2), name=f'{prefix}conv2_1_p')(a_c2_1)

    # a_p_c2_1 = pool((2, 2), (2, 2), name=f'{prefix}conv2_1_p')(a_p_c1)

    syn_c2_2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name=f'{prefix}conv2_2')(a_p_c2_1)
    norm_c2_2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv2_2')(syn_c2_2)
    # norm_c2_2 = GaussianNoiseInferenceOnly(value)(norm_c2_2)

    a_c2_2 = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv2_2')(norm_c2_2)
    a_p_c2_2 = pool((2, 2), (2, 2), name=f'{prefix}conv2_2_p')(a_c2_2)

    # a_p_c2_2 = pool((2, 2), (2, 2), name=f'{prefix}conv2_2_p')(a_p_c2_1)

    # conv3_1 -> pool
    # syn_c3_1   = lib_snn.layers.Conv2D(channels*2, 3, padding='SAME',
    #                                    kernel_initializer=k_init, name=f'{prefix}conv3_1')(a_p_c2_2)
    # norm_c3_1  = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv3_1')(syn_c3_1)
    # # norm_c3_1 = GaussianNoiseInferenceOnly(value)(norm_c3_1)
    #
    # a_c3_1     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv3_1')(norm_c3_1)
    # a_p_c3_1   = pool((2, 2), (2, 2), name=f'{prefix}conv3_1_p')(a_c3_1)

    a_p_c3_1   = pool((2, 2), (2, 2), name=f'{prefix}conv3_1_p')(a_p_c2_2)

    # conv3_2 -> pool
    # syn_c3_2   = lib_snn.layers.Conv2D(channels*2, 3, padding='SAME',
    #                                    kernel_initializer=k_init, name=f'{prefix}conv3_2')(a_p_c3_1)
    # norm_c3_2  = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv3_2')(syn_c3_2)
    # # norm_c3_2 = GaussianNoiseInferenceOnly(value)(norm_c3_2)
    # a_c3_2     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv3_2')(norm_c3_2)
    # a_p_c3_2   = pool((2, 2), (2, 2), name=f'{prefix}conv3_2_p')(a_c3_2)

    a_p_c3_2   = pool((2, 2), (2, 2), name=f'{prefix}conv3_2_p')(a_p_c3_1)


###########################################################################################################################
    # flatten -> fc1 -> bn -> n_fc1
    a_p_c3_2_f = tf.keras.layers.Flatten(data_format=data_format, name=f'{prefix}flatten')(a_p_c3_2)#<------------------------>
    # a_p_c3_2_f = GaussianNoiseInferenceOnly(value)(a_p_c3_2_f)

    syn_fc1    = lib_snn.layers.Dense(256, kernel_initializer=k_init, name=f'{prefix}fc1')(a_p_c3_2_f)
    norm_fc1   = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_fc1')(syn_fc1)
    a_fc1      = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_fc1')(norm_fc1)

    return a_fc1  # (B, 256)

def _audio_backbone_speck_hw(
    audio_input,
    conf,
    act_type="IF",
    k_init="glorot_uniform",
    prefix="aud_hw_",
    emb_dim=256,
    return_vector=True,
):
    """
    lib_snn version of the Speck-compatible audio backbone.
    Structure matches:
      Conv(inC->32, k3) -> IAF -> AvgPool(2)
      Conv(32->32, k3)  -> IAF -> AvgPool(2)
      Conv(32->64, k3)  -> IAF -> AvgPool(2)
      Conv(64->64, k3)  -> IAF -> AvgPool(2)
      Conv(64->256, k3) -> IAF -> AvgPool(2)   # spatial goes to 1x1
      Conv(256->emb_dim, k1) -> IAF
    Notes:
      - No BatchNorm, No Dense, No Flatten (to mirror the given hardware block).
      - If you need a vector output (B, emb_dim), set return_vector=True to apply time-avg.
    """
    pool = lib_snn.layers.AveragePooling2D

    # --- Input & input-layer spike ---
    x_in  = lib_snn.layers.InputGenLayer(name=f"{prefix}in_audio")(audio_input)
    x     = lib_snn.activations.Activation(act_type=act_type, loc="IN", name=f"{prefix}n_in")(x_in)

    # --- Block 1: Conv  -> IAF -> AvgPool (32x32 -> 16x16) ---
    x = lib_snn.layers.Conv2D(32, 3, padding="SAME",
                              kernel_initializer=k_init,
                              name=f"{prefix}conv1")(x)
    x = lib_snn.activations.Activation(act_type=act_type, name=f"{prefix}n1")(x)
    x = pool((2, 2), (2, 2), name=f"{prefix}p1")(x)

    # --- Block 2: Conv  -> IAF -> AvgPool (16x16 -> 8x8) ---
    x = lib_snn.layers.Conv2D(32, 3, padding="SAME",
                              kernel_initializer=k_init,
                              name=f"{prefix}conv2")(x)
    x = lib_snn.activations.Activation(act_type=act_type, name=f"{prefix}n2")(x)
    x = pool((2, 2), (2, 2), name=f"{prefix}p2")(x)

    # --- Block 3: Conv  -> IAF -> AvgPool (8x8 -> 4x4) ---
    x = lib_snn.layers.Conv2D(64, 3, padding="SAME",
                              kernel_initializer=k_init,
                              name=f"{prefix}conv3")(x)
    x = lib_snn.activations.Activation(act_type=act_type, name=f"{prefix}n3")(x)
    x = pool((2, 2), (2, 2), name=f"{prefix}p3")(x)

    # --- Block 4: Conv  -> IAF -> AvgPool (4x4 -> 2x2) ---
    x = lib_snn.layers.Conv2D(64, 3, padding="SAME",
                              kernel_initializer=k_init,
                              name=f"{prefix}conv4")(x)
    x = lib_snn.activations.Activation(act_type=act_type, name=f"{prefix}n4")(x)
    x = pool((2, 2), (2, 2), name=f"{prefix}p4")(x)

    # --- Block 5: Conv  -> IAF -> AvgPool (2x2 -> 1x1) ---
    x = lib_snn.layers.Conv2D(256, 3, padding="SAME",
                              kernel_initializer=k_init,
                              name=f"{prefix}conv5")(x)
    x = lib_snn.activations.Activation(act_type=act_type, name=f"{prefix}n5")(x)
    x = pool((2, 2), (2, 2), name=f"{prefix}p5")(x)

    # --- Final 1x1 Conv -> IAF (produce emb_dim channels at 1x1 spatial) ---
    x = lib_snn.layers.Conv2D(emb_dim, 1, padding="SAME",
                              kernel_initializer=k_init,
                              name=f"{prefix}conv1x1")(x)
    x = lib_snn.activations.Activation(act_type=act_type, name=f"{prefix}n_last")(x)

    # At this point, shape is (B, T, 1, 1, emb_dim).
    if not return_vector:
        return x

    # Convert (B, T, 1, 1, emb_dim) -> (B, emb_dim) by time-average and squeeze.
    # This keeps the "pure-conv + spike" nature while providing a vector embedding.
    x = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1),  # average over time dimension
        name=f"{prefix}tavg",
    )(x)  # (B, 1, 1, emb_dim)
    x = tf.keras.layers.Reshape((emb_dim,), name=f"{prefix}reshape_vec")(x)  # (B, emb_dim)

    return x


def _audio_backbone(audio_input, conf, act_type="IF", k_init="glorot_uniform", prefix="aud_"):
    tdbn = conf.nn_mode == 'SNN' and conf.tdbn
    use_bn_feat = conf.use_bn
    data_format = conf.data_format
    pool = lib_snn.layers.AveragePooling2D
    channels = 16 * 2  # = 32

    # IN
    in_audio   = lib_snn.layers.InputGenLayer(name=f'{prefix}in_audio')(audio_input)
    n_in       = lib_snn.activations.Activation(act_type=act_type, loc='IN', name=f'{prefix}n_in')(in_audio)


    # conv1_1 ~ conv1_3 -> pool
    syn_c1_1   = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat,
                                       kernel_initializer=k_init, name=f'{prefix}conv1_1')(n_in)
    norm_c1_1  = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv1_1')(syn_c1_1)

    a_c1_1     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv1_1')(norm_c1_1)

    syn_c1_2   = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat,
                                       kernel_initializer=k_init, name=f'{prefix}conv1_2')(a_c1_1)
    norm_c1_2  = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv1_2')(syn_c1_2)

    a_c1_2     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv1_2')(norm_c1_2)

    syn_c1_3   = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat,
                                       kernel_initializer=k_init, name=f'{prefix}conv1_3')(a_c1_2)
    norm_c1_3  = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv1_3')(syn_c1_3)

    a_c1_3     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv1_3')(norm_c1_3)
    a_p_c1     = pool((2, 2), (2, 2), name=f'{prefix}conv1_p')(a_c1_3)



    # conv2_1 -> pool
    syn_c2_1   = lib_snn.layers.Conv2D(channels*2, 3, padding='SAME',
                                       kernel_initializer=k_init, name=f'{prefix}conv2_1')(a_p_c1)
    norm_c2_1  = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv2_1')(syn_c2_1)
    a_c2_1     = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv2_1')(norm_c2_1)
    a_p_c2_1   = pool((2, 2), (2, 2), name=f'{prefix}conv2_1_p')(a_c2_1)
    syn_c2_2 = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name=f'{prefix}conv2_2')(a_p_c2_1)
    norm_c2_2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_conv2_2')(syn_c2_2)

    a_c2_2 = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_conv2_2')(norm_c2_2)
    a_p_c2_2 = pool((2, 2), (2, 2), name=f'{prefix}conv2_2_p')(a_c2_2)
    a_p_c3_1   = pool((2, 2), (2, 2), name=f'{prefix}conv3_1_p')(a_p_c2_2)
    a_p_c3_2   = pool((2, 2), (2, 2), name=f'{prefix}conv3_2_p')(a_p_c3_1)



###########################################################################################################################
    # flatten -> fc1 -> bn -> n_fc1
    a_p_c3_2_f = tf.keras.layers.Flatten(data_format=data_format, name=f'{prefix}flatten')(a_p_c3_2)#<------------------------>
    syn_fc1    = lib_snn.layers.Dense(256, kernel_initializer=k_init, name=f'{prefix}fc1')(a_p_c3_2_f)
    norm_fc1   = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{prefix}bn_fc1')(syn_fc1)
    a_fc1      = lib_snn.activations.Activation(act_type=act_type, name=f'{prefix}n_fc1')(norm_fc1)

    return a_fc1  # (B, 256)

def build_multi_model(
    num_classes,
    batch_size,
    input_shape_audio,
    input_shape_image,
    conf,
    act_type="IF",
    k_init="glorot_uniform",
    return_parts: bool = False,
):
    image_input = tf.keras.layers.Input(shape=input_shape_image, batch_size=batch_size, name='image_input')
    audio_input = tf.keras.layers.Input(shape=input_shape_audio, batch_size=batch_size, name='audio_input')

    img_feat = _image_backbone(image_input, conf, act_type=act_type, k_init=k_init, prefix="img_")
    #aud_feat = _audio_backbone_speck_hw(audio_input, conf, act_type=act_type, k_init=k_init, prefix="aud_")
    #aud_feat = _audio_backbone(audio_input, conf, act_type=act_type, k_init=k_init, prefix="aud_")
    aud_feat= _image_backbone(audio_input, conf, act_type=act_type, k_init=k_init, prefix="aud_")



    #fusion     = layers.Concatenate(name="fusion_head")([aud_feat, img_feat])
    #fusion     = lib_snn.layers.Concatenate(name="fusion_head")([fusion_input_audio_wo_t, fusion_input_img_wo_t])
    fusion     = lib_snn.layers.Concatenate(name="fusion_head")([aud_feat, img_feat])

    # fusion = GaussianNoiseInferenceOnly(value, name=f'gn_fusion')(fusion)

    # fusion = tf.keras.layers.Add(name="fusion_sum")([aud_feat, img_feat])
    # fusion = tf.keras.layers.Average(name="fusion_avg")([aud_feat, img_feat])

    #add dense block - for 2dense
    # tdbn = conf.nn_mode == 'SNN' and conf.tdbn
    # x = lib_snn.layers.Dense(256, kernel_initializer=k_init, name='sh_fc1')(fusion)
    # x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='sh_bn1')(x)
    # x = lib_snn.activations.Activation(act_type=act_type, name='sh_lif1')(x)
    # #

    #
    syn_p      = lib_snn.layers.Dense(num_classes, last_layer=True, kernel_initializer=k_init, name='prediction')(fusion)
    if conf.nn_mode=='SNN':
        n_pred     = lib_snn.activations.Activation(act_type=act_type, loc='OUT', name='n_prediction_integ')(syn_p)
        n_pred     = lib_snn.activations.Activation(act_type='softmax', name='n_prediction')(n_pred)
    else:
        n_pred     = lib_snn.activations.Activation(act_type='softmax', name='n_prediction')(syn_p)


    #fusion_model = lib_snn.model.Model([fusion_input_audio, fusion_input_img], n_pred, batch_size, [[input_shape_audio],[input_shape_image]], name="fusion")
    #fused_output = fusion_model([aud_feat, img_feat])

    model = lib_snn.model.Model(
        [audio_input, image_input],
        n_pred,
        #fused_output,
        batch_size,
        [input_shape_audio, input_shape_image],
        num_classes,
        conf,
        name='model_multi'
    )
    if not return_parts:
        return model

    if False:
        # Encoders still share weights with `model`
        audio_encoder = keras.Model(audio_input, aud_feat, name="audio_encoder")
        image_encoder = keras.Model(image_input, img_feat, name="image_encoder")

        a_dim = int(aud_feat.shape[-1])
        v_dim = int(img_feat.shape[-1])

        # Reuse trained layers from the full model
        pred_layer = model.get_layer("prediction")  # Dense(...) inside full
        act_layer = model.get_layer("n_prediction")  # Activation(...) inside full

        # Build hybrid submodels that REUSE the full head (no separate predictions_head)
        # Vision+Fusion (Speck audio embedding + GPU vision)
        image_in_gpu = keras.Input(shape=input_shape_image, batch_size=batch_size, name="image_input_gpu")
        a_emb_gpu = keras.Input(shape=(a_dim,), batch_size=batch_size, name="a_emb_in_gpu")
        v_emb_gpu = image_encoder(image_in_gpu)  # shared encoder
        fusion_cat_v = layers.Concatenate(name="fusion_concat_v")([a_emb_gpu, v_emb_gpu])
        logits_v = pred_layer(fusion_cat_v)  # reuse full's Dense
        out_v = act_layer(logits_v)  # reuse full's Activation
        gpu_vision_plus_fusion = keras.Model([image_in_gpu, a_emb_gpu], out_v,
                                             name="gpu_vision_plus_fusion")

        # Audio+Fusion (GPU audio + Speck vision embedding)
        audio_in_gpu = keras.Input(shape=input_shape_audio, batch_size=batch_size, name="audio_input_gpu")
        v_emb_gpu_in = keras.Input(shape=(v_dim,), batch_size=batch_size, name="v_emb_in_gpu")
        a_emb_gpu2 = audio_encoder(audio_in_gpu)  # shared encoder
        fusion_cat_a = layers.Concatenate(name="fusion_concat_a")([a_emb_gpu2, v_emb_gpu_in])
        logits_a = pred_layer(fusion_cat_a)  # reuse full's Dense
        out_a = act_layer(logits_a)  # reuse full's Activation
        gpu_audio_plus_fusion = keras.Model([audio_in_gpu, v_emb_gpu_in], out_a,
                                            name="gpu_audio_plus_fusion")

    # sspark, 251016
    feat_ext_audio = lib_snn.model.Model(audio_input, aud_feat, batch_size, input_shape_audio, snn_out_proc='all', name="feat_ext_audio")
    feat_ext_img = lib_snn.model.Model(image_input, img_feat, batch_size, input_shape_image, snn_out_proc='all', name="feat_ext_img")

    #
    # sspark, 251016
    #
    fusion_concat = model.get_layer('fusion_head')
    fusion_dense = model.get_layer('prediction')
    if conf.nn_mode=='SNN':
        fusion_n_pred_integ = model.get_layer('n_prediction_integ')
    fusion_n_pred = model.get_layer('n_prediction')

    # TODO: parameterize time step
    fusion_input_audio = tf.keras.layers.Input(shape=4+aud_feat.shape[1:], batch_size=batch_size, name='fusion_input_aud')
    fusion_input_img = tf.keras.layers.Input(shape=4+img_feat.shape[1:], batch_size=batch_size, name='fusion_input_img')

    fusion_input_audio_wo_t = lib_snn.layers.InputGenLayer(name='fusion_input_aud_wo_t')(fusion_input_audio)
    fusion_input_img_wo_t = lib_snn.layers.InputGenLayer(name='fusion_input_img_wo_t')(fusion_input_img)

    x = fusion_concat([fusion_input_audio_wo_t,fusion_input_img_wo_t])
    x = fusion_dense(x)
    if conf.nn_mode=='SNN':
        x = fusion_n_pred_integ(x)
    x = fusion_n_pred(x)

    fusion_model = lib_snn.model.Model([fusion_input_audio, fusion_input_img], x, batch_size, [[input_shape_audio],[input_shape_image]], name="fusion_model")

    #_fusion_model = lib_snn.model.Model([fusion_input_audio_wo_t, fusion_input_img_wo_t], n_pred, batch_size, [[input_shape_audio],[input_shape_image]], name="_fusion_model")
    #fusion_model = _fusion_model([fusion_input_audio,fusion_input_img])



    #
    parts = {
        #"audio_encoder": audio_encoder,
        #"image_encoder": image_encoder,
        ## no separate 'fusion_head' needed when sharing the full head
        #"gpu_vision_plus_fusion": gpu_vision_plus_fusion,
        #"gpu_audio_plus_fusion": gpu_audio_plus_fusion,
        #"a_dim": a_dim,
        #"v_dim": v_dim,
        "feat_ext_audio": feat_ext_audio,
        "feat_ext_img": feat_ext_img,
        "fusion_model": fusion_model,
    }
    return model, parts