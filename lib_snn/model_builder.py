
import tensorflow as tf
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

import functools


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

    #
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

    if lr_schedule == 'COS':
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(learning_rate, train_steps_per_epoch * train_epoch)
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
        learning_rate = learning_rate
        if conf.grad_clipnorm == None:
            optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate, name='ADAMW')
        else:
            optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate, name='ADAMW', clipnorm=conf.grad_clipnorm)
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


    # loss
    if False:
    #if True:
        #for neuron in model.layers_w_neuron:
        for layer in model.layers:
            if isinstance(layer,lib_snn.activations.Activation) and isinstance(layer.act, lib_snn.neurons.Neuron):
                #    #neuron.act.add_loss(lambda: tf.reduce_mean(neuron.act.spike_count_int))
                #    #neuron.act.add_loss(lambda: tf.reduce_sum(neuron.act.spike_count_int))
                neuron = layer
                if neuron.act.loc=='HID':
                    #print(neuron.name)
                    #neuron.act.add_loss(functools.partial(tf.reduce_mean,spike))
                    spike = 0.001*neuron.act.out
                    neuron.act.add_loss(functools.partial(tf.reduce_mean,spike))
                    #model.add_loss(lambda: tf.reduce_mean(0.001*neuron.act.out))

        for layer in model.layers_w_kernel:
            print(layer.name)

    # dummy
    #img_input = tf.keras.layers.Input(shape=image_shape, batch_size=batch_size)
    #model(img_input)

    #with dist_strategy.scope():
    # compile
    model.compile(optimizer=optimizer,
                  #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  #loss=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
                  metrics=[metric_accuracy, metric_accuracy_top5], run_eagerly=eager_mode)
                #metrics = [metric_accuracy, metric_accuracy_top5], run_eagerly = False)


    print('-- model compile done')
    glb.model_compile_done()

    return model