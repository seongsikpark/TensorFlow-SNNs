
import tensorflow as tf
import tensorflow_addons as tfa
import keras

import lib_snn

from lib_snn.sim import glb

from absl import flags
conf = flags.FLAGS

from config import config

#
from models.models import model_sel

import utils

import collections


def model_builder(
    num_class,
    train_steps_per_epoch,
    valid_ds
):

    print('Model Builder - {}'.format(conf.nn_mode))
    glb.model_compile_done_reset()



    # temporal first - hold temporal intermediate tensors
    #if conf.nn_mode=='SNN' and 'train' in conf.mode and (not conf.snn_training_spatial_first):
    #    image_shape = (conf.time_step,)+image_shape

    #
    model_name = config.model_name
    dataset_name = config.dataset_name

    #
    eager_mode = config.eager_mode


    data_batch, = valid_ds.take(1)
    images = data_batch[0]
    labels = data_batch[1]

    batch_size = config.batch_size
    #image_shape = lib_snn.utils_vis.image_shape_vis(model_name, dataset_name)

    #batch_size = images.shape[0]
    image_shape = images.shape[1:]

    # train
    train_type = config.train_type
    train_epoch = config.train_epoch

    # model
    model_top = model_sel(model_name,train_type)

    #
    include_top = config.include_top
    load_weight = config.load_weight

    #
    metric_accuracy = tf.keras.metrics.categorical_accuracy
    metric_accuracy_top5 = tf.keras.metrics.top_k_categorical_accuracy

    #metric_name_acc = 'acc'
    #metric_name_acc_top5 = 'acc-5'
    #monitor_cri = 'val_' + metric_name_acc
    metric_name_acc = config.metric_name_acc
    metric_name_acc_top5 = config.metric_name_acc_top5

    metric_accuracy.name = metric_name_acc
    metric_accuracy_top5.name = metric_name_acc_top5
    if conf.model == 'Spikformer' or 'Spikingformer':
        model_top = model_top(batch_size=batch_size, input_shape=image_shape, conf=conf,
                              model_name=model_name, weights=load_weight,
                              dataset_name=dataset_name, classes=num_class,
                              include_top=include_top, patch_size=conf.patch_size, embed_dims=conf.embed_dims,
                              num_heads=conf.num_heads, depths=conf.depths, sr_ratios=conf.sr_ratios)
    else:
        model_top = model_top(batch_size=batch_size, input_shape=image_shape, conf=conf,
                              model_name=model_name, weights=load_weight,
                              dataset_name=dataset_name, classes=num_class,
                              include_top=include_top)


    # set distribute strategy
    dist_strategy = utils.set_gpu()
    model_top.dist_strategy = dist_strategy


    # TODO: parameterize
    # lr schedule
    lr_schedule_first_decay_step = train_steps_per_epoch * 10  # in iteration

    #lr_schedule = hp_lr_schedule
    #train_epoch = hp_train_epoch
    #step_decay_epoch = hp_step_decay_epoch

    #
    opt = conf.optimizer
    learning_rate = conf.learning_rate
    lr_schedule = conf.lr_schedule
    step_decay_epoch = conf.step_decay_epoch
    warmup_target=5E-4
    warmup_epochs = 20
    lr_min=1E-5

    if lr_schedule == 'COS':
        learning_rate = conf.learning_rate_init # initial
        warmup_target = conf.learning_rate
        lr_min=conf.learning_rate_init
        # learning_rate = tf.keras.optimizers.schedules.CosineDecay(learning_rate, train_steps_per_epoch * train_epoch)
        #learning_rate = tf.keras.optimizers.schedules.CosineDecay(learning_rate, train_steps_per_epoch * train_epoch, alpha=1e-4)
        learning_rate = lib_snn.optimizers.CosineDecay(learning_rate, train_steps_per_epoch * train_epoch, alpha=0.0,
                                                       warmup_target=warmup_target, warmup_steps=train_steps_per_epoch*warmup_epochs,
                                                       lr_min=lr_min)
    elif lr_schedule == 'COSR':
        learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(learning_rate, lr_schedule_first_decay_step)
    elif lr_schedule == 'STEP':
        learning_rate = lib_snn.optimizers.LRSchedule_step(learning_rate, train_steps_per_epoch * step_decay_epoch, 0.1)
    elif lr_schedule == 'STEP_WUP':
        learning_rate = lib_snn.optimizers.LRSchedule_step_wup(learning_rate, train_steps_per_epoch * 100, 0.1,
                                                               train_steps_per_epoch * 30)
    else:
        #assert False
        pass

    # optimizer
    if opt == 'SGD':
        if conf.grad_clipnorm == None:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD')
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD',clipnorm=conf.grad_clipnorm)
            #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD',clipnorm=2.0)
            #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD',clipvalue=1.0)
            #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD',global_clipnorm=5.0)
    elif opt == 'ADAM':
        learning_rate = learning_rate
        if conf.grad_clipnorm == None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, name='ADAM')
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, name='ADAM',clipnorm=conf.grad_clipnorm)
    elif opt == 'ADAMW':
        assert config.flags.regularizer is None,'ADAMW -> regularizer should be None'

        learning_rate = learning_rate
        weight_decay = conf.weight_decay_AdamW
        if conf.grad_clipnorm == None:
            #optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay,learning_rate=learning_rate, name='ADAMW')
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay, name='ADAMW')
        else:
            assert False
            #optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay,learning_rate=learning_rate, name='ADAMW')
    else:
        assert False

    #model = model_top.model
    model = model_top

    # set layer nn_mode
    #model.set_layers_nn_mode()



    ##
    use_bn_dict = collections.OrderedDict()
    use_bn_dict['VGG16_ImageNet'] = False

    #
    try:
        config.flags.use_bn = use_bn_dict[config.model_dataset_name]
    except KeyError:
        pass


    # dummy
    #img_input = tf.keras.layers.Input(shape=image_shape, batch_size=batch_size)
    #model(img_input)

    # mixed precision - float16
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)


    #with dist_strategy.scope():
    # compile
    model.compile(optimizer=optimizer,
                  #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=conf.label_smoothing),
                  #loss=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
                  metrics=[metric_accuracy, metric_accuracy_top5], run_eagerly=eager_mode)
                #metrics = [metric_accuracy, metric_accuracy_top5], run_eagerly = False)


    print('-- model compile done')
    glb.model_compile_done()

    return model