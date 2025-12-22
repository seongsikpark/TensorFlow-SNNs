

from config_snn_multimodal import config

import lib_snn

# from main_multi_input_snn import input_shape_audio
from callbacks import callbacks_snn_train
from lib_snn.optimizers import CosineDecay
from tensorflow import keras
#from yum_image_model_builder import build_image_model
#from yum_audio_model_builder import build_audio_model
from yc_multi_model_builder_speck import build_multi_model
from yc_multi_model_builder_TF import build_multi_model_TF
from yum_load_multi_dataset import load_multimodal_dataset
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import numpy as np

# 파일 상단 import 근처에 추가
from tensorflow import keras
import lib_snn


def _strip_kwargs_patch(Cls, keys=("data_format",)):
    orig_init = Cls.__init__

    def patched_init(self, *args, **kwargs):
        for k in keys:
            # 중복/불필요한 인자 제거 (config에도 있고 내부에서도 넘기는 케이스 방지)
            #   - 값이 None이든 아니든 그냥 제거: 내부에서 다시 세팅됨
            if k in kwargs:
                kwargs.pop(k)
        return orig_init(self, *args, **kwargs)

    Cls.__init__ = patched_init


# Conv2D 외에도 같은 패턴이 있으면 추가
if hasattr(lib_snn.layers, "Conv2D"):
    _strip_kwargs_patch(lib_snn.layers.Conv2D,
                        keys=("data_format", "bias_initializer", "kernel_regularizer", "use_bias"))
# 필요 시:
# if hasattr(lib_snn.layers, "AveragePooling2D"): _strip_kwargs_patch(lib_snn.layers.AveragePooling2D, keys=("data_format",))
# if hasattr(lib_snn.layers, "BatchNormalization"): _strip_kwargs_patch(lib_snn.layers.BatchNormalization, keys=("data_format",))
if hasattr(lib_snn.layers, "Dense"): _strip_kwargs_patch(lib_snn.layers.Dense, keys=('use_bias'))


def _libsnn_custom_objects():
    objs = {}
    # lib_snn.layers 쪽 (프로젝트에 실제로 쓰는 것만 골라 등록)
    for name in [
        "InputGenLayer", "BatchNormalization", "AveragePooling2D",
        "Conv2D", "Dense", "Flatten", "Add", "Average"
    ]:
        if hasattr(lib_snn.layers, name):
            objs[name] = getattr(lib_snn.layers, name)
    # lib_snn.activations
    if hasattr(lib_snn, "activations") and hasattr(lib_snn.activations, "Activation"):
        objs["Activation"] = lib_snn.activations.Activation
    return objs


conf = config.flags
#conf.debug_mode = True
# train_pkl_path='/home/yumin/PycharmProjects/00_Multimodal_SNN/multimodal_train_urban8k_av.pkl'
# test_pkl_path='/home/yumin/PycharmProjects/00_Multimodal_SNN/multimodal_test_urban8k_av.pkl'

# train_pkl_path= '/home/yumin/PycharmProjects/00_Multimodal_SNN/multimodal_train_cremad_96_96.pkl'
# test_pkl_path= '/home/yumin/PycharmProjects/00_Multimodal_SNN/multimodal_test_cremad_96_96.pkl'
#train_pkl_path = '/home/yumin/PycharmProjects/00_Multimodal_SNN/cremad_train.pkl'
#test_pkl_path = '/home/yumin/PycharmProjects/00_Multimodal_SNN/cremad_test.pkl'
#train_pkl_path = '/home/sspark/Projects/02_SNN_training/multimodal/cremad_train.pkl'
#test_pkl_path = '/home/sspark/Projects/02_SNN_training/multimodal/cremad_test.pkl'
train_pkl_path = '/home/sspark/Projects/02_SNN_training/cremad_train_0.7.pkl'
test_pkl_path = '/home/sspark/Projects/02_SNN_training/cremad_test_0.3.pkl'

# train_pkl_path= '/home/yumin/PycharmProjects/00_Multimodal_SNN/multimodal_train_mnist.pkl'
# test_pkl_path= '/home/yumin/PycharmProjects/00_Multimodal_SNN/multimodal_test_mnist_av.pkl'

# model save path
model_save_path = 'saved_models/crema-d/multimodal_crema-d_vggspeck_96_96_rmv3_1_3_2.h5'
rmv = "96x96 + 96x96, cremad multimodal, rmv 3_1_3_2"

h = 32

image_input_shape = (conf.time_step, h, h, 3)
max_disp = h - 1

duration_sec = 3.0  # cifar : 5.0, urban8k : 4.0 #crema-d : 3.0 #nmnist 1.2
num_classes = 6  # cifar : 10, urban : 10, crema : 6
frame_length = 4096
frame_step = 2048
#EPOCHS = 200
EPOCHS = conf.train_epoch
batch_size = 32
batch_size = 64
#batch_size = 128
#model = 'TF'
#model = 'CNN'
#model = 'speck'
model=config.mm_model
# result_plot = True


# configuration
from config_snn_multimodal import config

conf = config.flags

act_type = conf.n_type
act_type_out = conf.n_type
# print("yumin ntype")
# print(conf.n_type) #result : LIF

# ###################################################################
# #load dataset
#
train_ds = load_multimodal_dataset(
    train_pkl_path, batch_size=batch_size, shuffle=True
)
test_ds = load_multimodal_dataset(
    test_pkl_path, batch_size=batch_size, shuffle=False
)
#
# ###################################################################
#
###################################################################
###################################################################
# region check dataset
if False:
    for x_batch, y_batch in train_ds.take(10):
        print("train batch class dist:", np.unique(y_batch.numpy(), return_counts=True))
    for x_batch, y_batch in test_ds.take(10):
        print("Val batch class dist:", np.unique(y_batch.numpy(), return_counts=True))

for (audio_batch, image_batch), label_batch in train_ds.take(1):
    audio_input_shape = audio_batch.shape[1:]  # (T, H, W, C)
    break
print("audio input shape", audio_input_shape)

if model == 'TF':
    model_multi = build_multi_model_TF(
        batch_size,
        audio_input_shape,
        image_input_shape,
        conf,
        model_name='multi_modal_TF',
        dim=32,
        classes=num_classes,
        max_disp=max_disp,
        act_type=conf.n_type)
    model_multi.summary()
elif model == 'CNN':
    model_multi = build_multi_model(
        num_classes,
        batch_size,
        audio_input_shape,
        image_input_shape,
        conf,
        act_type=conf.n_type)
    model_multi.summary()
elif model == 'speck':
    full, parts = build_multi_model(
        num_classes, batch_size, audio_input_shape, image_input_shape, conf,
        act_type=conf.n_type, return_parts=True
    )
    model_multi = full

    #parts["audio_encoder"].save("export/audio_encoder_fp32.keras")
    #parts["image_encoder"].save("export/image_encoder_fp32.keras")

    #parts["gpu_vision_plus_fusion"].save("export/gpu_vision_plus_fusion.keras")

    #parts["gpu_audio_plus_fusion"].save("export/gpu_audio_plus_fusion.keras")
    full.summary()

    # sspark, 251016
    feat_ext_audio = parts["feat_ext_audio"]
    feat_ext_img = parts["feat_ext_img"]
    fusion_model = parts["fusion_model"]


##############################################


#############################################
###############OPTIMIZER#####################
def get_num_samples_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return len(data["label"])


train_sample_count = len(pickle.load(open(train_pkl_path, "rb"))["label"])
train_steps_per_epoch = train_sample_count // batch_size
train_epoch = EPOCHS
#warmup_epochs = 10
warmup_epochs = int(EPOCHS*0.1)

lr_schedule = CosineDecay(
    initial_learning_rate=conf.learning_rate_init,
    decay_steps=train_steps_per_epoch * train_epoch,
    alpha=0.0,
    warmup_target=conf.learning_rate,
    warmup_steps=train_steps_per_epoch * warmup_epochs,
    lr_min=conf.learning_rate_init,
)

#
optimizer = keras.optimizers.experimental.AdamW(
    learning_rate=lr_schedule,
    weight_decay=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6,
)


# step
#step_decay_epoch = conf.step_decay_epoch
#lr_schedule = lib_snn.optimizers.LRSchedule_step(conf.learning_rate, train_steps_per_epoch * step_decay_epoch, 0.1)
#optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, name='SGD')

# ADAM
#optimizer = tf.keras.optimizers.Adam(learning_rate=conf.learning_rate, name='ADAM')

#############################################


metric_accuracy = tf.keras.metrics.sparse_categorical_accuracy
metric_name_acc = config.metric_name_acc
metric_accuracy.name = metric_name_acc

model_multi.compile(
    # optimizer=tf.keras.optimizers.Adam(conf.learning_rate),
    optimizer=optimizer,
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #loss=tf.keras.losses.CategoricalCrossentropy(),
    #metrics=['accuracy'],
    metrics=[metric_accuracy],
    # metrics=[SparseCategoricalAccuracy()],
    run_eagerly=config.eager_mode
)


###########################################
########Call backs#########################
def num_batches(ds):
    c = tf.data.experimental.cardinality(ds)
    if c == tf.data.experimental.UNKNOWN_CARDINALITY:
        return int(ds.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy())
    return int(c.numpy())


#
train_ds_num = num_batches(train_ds) * batch_size
test_ds_num = num_batches(test_ds) * batch_size


callbacks_train, callbacks_test = callbacks_snn_train(
    model_multi,
    train_ds_num=train_ds_num,
    valid_ds=test_ds,
    test_ds_num=test_ds_num,
    #save_path=model_save_path
)
###########################################
###########################################


if conf.mode=='train':
    history = model_multi.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks_train
    )

    model_multi.save(model_save_path)
    print("model save path is ", model_save_path)
# model_multi.load_weights("/home/yumin/PycharmProjects/00_Multimodal_SNN/saved_models/1/multimodal_crema-d_vggspeck.h5",by_name=True)

#scores = model_multi.evaluate(test_ds, return_dict=True, batch_size=batch_size)
# print("score :", scores)
# y_pred = model_multi.predict(test_ds, batch_size=batch_size)

#assert False



#
# ==== HYBRID INFERENCE (Speck + GPU) =========================================
if conf.mode=='inference':

    # load model
    print('load model - '+config.load_weight)
    model_multi.load_weights(config.load_weight)

    #if conf.mode_inf_hybrid:
    if config.mm_model=='speck':

        #
        #if False:
        if True:
            img_bn_conv1 = model_multi.get_layer('img_bn_conv1')
            img_bn_conv2 = model_multi.get_layer('img_bn_conv2')
            img_bn_conv3 = model_multi.get_layer('img_bn_conv3')
            img_bn_conv4 = model_multi.get_layer('img_bn_conv4')
            img_bn_fc1 = model_multi.get_layer('img_bn_fc1')
            model_multi.get_layer('img_conv1').bn_fusion_v2(img_bn_conv1)
            model_multi.get_layer('img_conv2').bn_fusion_v2(img_bn_conv2)
            model_multi.get_layer('img_conv3').bn_fusion_v2(img_bn_conv3)
            model_multi.get_layer('img_conv4').bn_fusion_v2(img_bn_conv4)
            model_multi.get_layer('img_fc1').bn_fusion_v2(img_bn_fc1)


        #
        #if False:
        if True:
            aud_bn_conv1 = model_multi.get_layer('aud_bn_conv1')
            aud_bn_conv2 = model_multi.get_layer('aud_bn_conv2')
            aud_bn_conv3 = model_multi.get_layer('aud_bn_conv3')
            aud_bn_conv4 = model_multi.get_layer('aud_bn_conv4')
            aud_bn_fc1 = model_multi.get_layer('aud_bn_fc1')
            model_multi.get_layer('aud_conv1').bn_fusion_v2(aud_bn_conv1)
            model_multi.get_layer('aud_conv2').bn_fusion_v2(aud_bn_conv2)
            model_multi.get_layer('aud_conv3').bn_fusion_v2(aud_bn_conv3)
            model_multi.get_layer('aud_conv4').bn_fusion_v2(aud_bn_conv4)
            model_multi.get_layer('aud_fc1').bn_fusion_v2(aud_bn_fc1)


        #
        metric = tf.keras.metrics.SparseCategoricalAccuracy()

        #
        for (audio_batch, image_batch), label_batch in test_ds:
            feat_aud = feat_ext_audio(audio_batch)
            feat_img = feat_ext_img(image_batch)
            pred = fusion_model([feat_aud, feat_img])

            #pred_class = tf.argmax(pred,axis=-1,dtype=tf.int32)
            #correct = tf.equal(pred_class,labal_batch)
            metric.update_state(label_batch, pred)



        print("acc: ", metric.result().numpy())
    else:
        result = model_multi.evaluate(test_ds, callbacks=callbacks_test)



assert False



#if conf.mode=='inference' and conf.mode_inf_hybrid:
#    run_hybrid_eval=True
#else:
#    run_hybrid_eval=False

# ori
#run_hybrid_eval = (model == 'speck')


custom_objs = _libsnn_custom_objects()
keras.utils.get_custom_objects().update(custom_objs)
if run_hybrid_eval:


    SPECK_ROLE = "audio"

    if SPECK_ROLE == "audio":
        # gpu_side = tf.keras.models.load_model("export/gpu_vision_plus_fusion.keras", compile=False)
        gpu_side = parts["gpu_vision_plus_fusion"]
    else:
        # gpu_side = tf.keras.models.load_model("export/gpu_audio_plus_fusion.keras", compile=False)
        gpu_side = parts["gpu_audio_plus_fusion"]

    import numpy as np
    import torch
    import samna
    from torch import nn
    import sinabs.layers as sl
    from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
    from sinabs.backend.dynapcnn import DynapcnnNetwork

    _SPECK = {
        "dynapcnn": None,
        "last_core": None,
        "input_hw": None,  # (H, W)
        "emb_dim": None,
    }


    def _build_snn_embed(emb_dim: int):
        emb_dim = min(int(emb_dim), 64)

        snn_embed = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(2, 16, kernel_size=3, padding=1, stride=2, bias=False),
            sl.IAFSqueeze(batch_size=1, min_v_mem=-1.0, spike_threshold=0.5, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(2, 2),  # 16x16 -> 8x8

            # 8x8 -> 4x4
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=False),
            sl.IAFSqueeze(batch_size=1, min_v_mem=-1.0, spike_threshold=0.5, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(2, 2),

            # 4x4 -> 2x2
            nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=1, bias=False),
            sl.IAFSqueeze(batch_size=1, min_v_mem=-1.0, spike_threshold=0.5, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(2, 2),

            # 2x2 -> 1x1
            nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=1, bias=False),
            sl.IAFSqueeze(batch_size=1, min_v_mem=-1.0, spike_threshold=0.5, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(64, emb_dim, kernel_size=1, bias=False),
            sl.IAFSqueeze(batch_size=1, min_v_mem=-1.0, spike_threshold=0.5, surrogate_grad_fn=PeriodicExponential()),
        )
        for m in snn_embed.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
        return snn_embed.eval()


    def _ensure_speck_runtime(H: int, W: int, emb_dim: int):
        need_rebuild = (
                _SPECK["dynapcnn"] is None
                or _SPECK["input_hw"] != (H, W)
                or _SPECK["emb_dim"] != emb_dim
        )
        if not need_rebuild:
            return
        cpu_embed = _build_snn_embed(emb_dim)
        dynap = DynapcnnNetwork(snn=cpu_embed, input_shape=(2, H, W), discretize=True, dvs_input=True)
        dynap.to(device="speck2fdevkit", chip_layers_ordering="auto")
        _SPECK["dynapcnn"] = dynap
        _SPECK["last_core"] = dynap.chip_layers_ordering[-1]
        _SPECK["input_hw"] = (H, W)
        _SPECK["emb_dim"] = emb_dim
        print(f"[Speck] Ready. cores={dynap.chip_layers_ordering}, emb_dim={emb_dim}, input={H}x{W}")


    def _to_on_off_binary(frames_TxHxWxC: np.ndarray, tau: float = 0.05) -> np.ndarray:
        f = frames_TxHxWxC.astype(np.float32)
        if f.shape[-1] > 1:
            f = 0.2989 * f[..., 0] + 0.5870 * f[..., 1] + 0.1140 * f[..., 2] if f.shape[-1] >= 3 else f[..., 0]
        else:
            f = f[..., 0]
        fmin, fmax = f.min(), f.max()
        if fmax > fmin:
            f = (f - fmin) / (fmax - fmin)
        df = np.diff(f, axis=0, prepend=f[0:1])
        on = (df > tau).astype(np.uint8)
        off = (df < -tau).astype(np.uint8)
        return np.stack([off, on], axis=1)  # [T,2,H,W]


    def _frames_to_event_stream(frames_TCHW_bin01: np.ndarray, input_core: int, ts_scale: int = 1):
        T, C, H, W = frames_TCHW_bin01.shape
        stream = []
        for t in range(T):
            for c in range(C):
                ys, xs = np.nonzero(frames_TCHW_bin01[t, c])
                for y, x in zip(ys, xs):
                    spk = samna.speck2f.event.Spike()
                    spk.x = int(x);
                    spk.y = int(y)
                    spk.timestamp = int(t * ts_scale)
                    spk.feature = int(c)
                    spk.layer = input_core
                    stream.append(spk)
        return stream


    def speck_get_embedding(frames_TxHxWxC: np.ndarray, emb_dim_override: int = None) -> np.ndarray:

        arr = np.asarray(frames_TxHxWxC)
        _, H, W, C = arr.shape
        emb_dim = emb_dim_override if emb_dim_override is not None else (_SPECK["emb_dim"] or 256)

        if C == 2 and np.array_equal(arr, arr.astype(bool)):
            frames_TCHW = np.transpose(arr, (0, 3, 1, 2))
        else:
            frames_TCHW = _to_on_off_binary(arr)

        _ensure_speck_runtime(H, W, emb_dim)
        dynap = _SPECK["dynapcnn"]
        last_core = _SPECK["last_core"]

        stream = _frames_to_event_stream(frames_TCHW, input_core=dynap.chip_layers_ordering[0], ts_scale=1)
        out_events = dynap(stream)

        counts = np.zeros(emb_dim, dtype=np.float32)
        feats = [e.feature for e in out_events if e.layer == last_core]
        if feats:
            base = min(feats)
            for f in feats:
                k = f - base
                if 0 <= k < emb_dim:
                    counts[k] += 1.0
        s = counts.sum()
        if s > 0:
            counts = counts / s
        return counts.astype(np.float32)


    correct = 0;
    total = 0
    EMBED_DIM = min(int(parts["a_dim"] if SPECK_ROLE == "audio" else parts["v_dim"]), 128)

    gpu_vf = parts["gpu_vision_plus_fusion"]
    a128_in = keras.Input(shape=(EMBED_DIM,), batch_size=batch_size, name="a_emb_in_128")
    a256 = keras.layers.Dense(256, use_bias=False, name="speck_bridge_a")(a128_in)
    img_in = keras.Input(shape=image_input_shape, batch_size=batch_size, name="img_in_wrap")
    logits = gpu_vf([img_in, a256])
    gpu_side = keras.Model([img_in, a128_in], logits, name="gpu_vf_wrap")

    for (audio_batch, image_batch), label_batch in test_ds:

        B = audio_batch.shape[0]
        label_np = label_batch.numpy()

        if SPECK_ROLE == "audio":
            a_emb_list = []
            for i in range(B):
                a_frames = audio_batch[i].numpy()  # (T,H,W,C)
                a_emb = speck_get_embedding(a_frames, emb_dim_override=EMBED_DIM)  # (D,)
                a_emb_list.append(a_emb)
            a_emb_np = np.stack(a_emb_list, axis=0).astype(np.float32)  # [B, D]
            logits = gpu_side.predict([image_batch, a_emb_np], verbose=0)
        else:
            v_emb_list = []
            for i in range(B):
                v_frames = image_batch[i].numpy()  # (T,H,W,C)
                v_emb = speck_get_embedding(v_frames, emb_dim_override=EMBED_DIM)  # (D,)
                v_emb_list.append(v_emb)
            v_emb_np = np.stack(v_emb_list, axis=0).astype(np.float32)  # [B, D]
            logits = gpu_side.predict([audio_batch, v_emb_np], verbose=0)

        pred = np.argmax(logits, axis=-1)
        correct += (pred == label_np).sum()
        total += B

    print(f"[HYBRID] Speck({SPECK_ROLE}) + GPU fusion accuracy: {correct / total * 100:.2f}%")