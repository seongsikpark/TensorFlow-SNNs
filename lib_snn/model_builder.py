
import tensorflow as tf
import lib_snn

from lib_snn.sim import glb


def model_builder(
    eager_mode, model_top, batch_size, image_shape, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
    train_epoch, train_steps_per_epoch,
    opt, learning_rate,
    lr_schedule, step_decay_epoch,
    metric_accuracy, metric_accuracy_top5,
    dataset_name,
    dist_strategy=tf.distribute.OneDeviceStrategy
):

    print('Model Builder - {}'.format(conf.nn_mode))
    glb.model_compile_done_reset()

    # model
    model_top = model_top(batch_size=batch_size, input_shape=image_shape, conf=conf,
                          model_name=model_name, weights=load_weight,
                          dataset_name=dataset_name, classes=num_class,
                          include_top=include_top,
                          initial_channels=initial_channels)

    # set distribute strategy
    model_top.dist_strategy = dist_strategy


    # TODO: parameterize
    # lr schedule
    lr_schedule_first_decay_step = train_steps_per_epoch * 10  # in iteration

    #lr_schedule = hp_lr_schedule
    #train_epoch = hp_train_epoch
    #step_decay_epoch = hp_step_decay_epoch

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
        assert False

    # optimizer
    if opt == 'SGD':
        if conf.grad_clipnorm == None:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD')
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD',clipnorm=conf.grad_clipnorm)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD',clipnorm=2.0)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD',clipvalue=1.0)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD',global_clipnorm=1.0)
    elif opt == 'ADAM':
        learning_rate = learning_rate
        if conf.grad_clipnorm == None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, name='ADAM')
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, name='ADAM',clipnorm=conf.grad_clipnorm)
    else:
        assert False

    #model = model_top.model
    model = model_top

    # set layer nn_mode
    #model.set_layers_nn_mode()



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