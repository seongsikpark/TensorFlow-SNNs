


import tensorflow_datasets as tfds
# import events_tfds.events.cifar10_dvs
#from events_tfds.vis.image import as_frames
#from events_tfds.vis.image import as_frame
from datasets.events.image import as_frames
from datasets.events.image import as_frame
# from events_tfds.vis.anim import animate_frames



from datasets.augmentation_cifar import cutmix

import tensorflow as tf

import matplotlib.pyplot as plt

from config import config
conf = config.flags

def load():
    #train_ds = tfds.load("cifar10_dvs", split="train", as_supervised=True)


    batch_size = config.batch_size
    num_parallel = tf.data.AUTOTUNE

    if False:
        for events, labels in train_ds:
            print(events)
            print(labels)


        #train_ds = train_ds.map(lambda events, labels: )

    train_ratio = 0.9
    train_ratio_percent = int(train_ratio*100)
    #train_ds, train_ds_info = tfds.load("cifar10_dvs", split="train", as_supervised=True)
    #train_ds = tfds.load("cifar10_dvs", split="train", as_supervised=True)

    #train_ds = tfds.load("cifar10_dvs", split="train[:"+str(train_ratio_percent)+"%]", as_supervised=True, shuffle_files=True)
    #valid_ds = tfds.load("cifar10_dvs", split="train["+str(train_ratio_percent)+"%:]", as_supervised=True)

    train_ds = tfds.load("cifar10_dvs", split="train[10%:]", as_supervised=True, shuffle_files=True)
    valid_ds = tfds.load("cifar10_dvs", split="train[:10%]", as_supervised=True)


    #train_ds = train_ds.map(lambda events, labels: as_frame())


    num_frames = conf.time_step
    conf.time_dim_size = num_frames

    #image_shape = (128,128,3)
    image_shape = (128,128,2)
    #image_shape = (128,128,1)

    # test
    ##for events, labels in train_ds:
    #ds, = train_ds.take(1)
    #events = ds[0]
    #labels = ds[1]
    #as_frames(events,labels,shape=image_shape,num_frames=num_frames)
    #assert False

    #train_ds = train_ds.map(lambda events,labels: as_frame(events,labels,shape=image_shape))

    #
    #cifa10_dvs_img_size = conf.cifar10_dvs_img_size
    #cifar10_dvs_crop_img_size = conf.cifkhh

    # tf tensor version test
    if False:
    #if True:
        for events, labels in train_ds:
            #frames = as_frames(**{k: v.numpy() for k, v in events.items()}, num_frames=20)
            coords = events['coords']
            polarity = events['polarity']
            frame = as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True)
            #print(labels.numpy())

    #
    train_ds = train_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True))
    # todo - cutmix
#    if config.flags.data_aug_mix == 'mixup' or config.flags.data_aug_mix == 'cutmix':
#        train_ds_1 = train_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True))
#        train_ds_2 = train_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True))
#        train_ds = tf.data.Dataset.zip((train_ds_1, train_ds_2))
#        #train_ds_num = train_ds_1_num
#
#        train_ds = train_ds.map(
#            lambda train_ds_1, train_ds_2: cutmix(train_ds_1, train_ds_2, dataset_name, input_size, input_size_pre_crop_ratio,
#                                                  num_class, 1.0, input_prec_mode,preprocessor_input),
#            num_parallel_calls=num_parallel)
#    else:
#        #train_ds, train_ds_num = default_load_train()
#        train_ds = train_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True))

    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(num_parallel)

    #valid_ds = valid_ds.map(lambda events,labels: as_frame(events,labels,shape=image_shape))
    valid_ds = valid_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames))
    valid_ds = valid_ds.batch(batch_size)
    valid_ds = valid_ds.prefetch(num_parallel)


    if False: # tfds events code
    #if True: # tfds events code
        for events, labels in train_ds:
            #frames = as_frames(**{k: v.numpy() for k, v in events.items()}, num_frames=20)
            coords = events['coords'].numpy()
            polarity = events['polarity'].numpy()
            frame = as_frame(coords,polarity)
            #print(labels.numpy())
            #print(tf.reduce_max(events["coords"], axis=0).numpy())
            #anim = animate_frames(frames, fps=4)

            print(labels.numpy())
            plt.imshow(frame)

    #valid_ds = train_ds
    train_ds_num=10000*train_ratio
    valid_ds_num=10000*(1-train_ratio)

    return train_ds, valid_ds, valid_ds, train_ds_num, valid_ds_num, valid_ds_num
