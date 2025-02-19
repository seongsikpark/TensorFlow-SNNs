
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds


import ssl

import datasets.augmentation_cifar

ssl._create_default_https_context = ssl._create_unverified_context

from datasets.augmentation_cifar import resize_with_crop
from datasets.augmentation_cifar import resize_with_crop_aug
from datasets.augmentation_cifar import mixup
from datasets.augmentation_cifar import cutmix
from datasets.augmentation_cifar import cutmix_in_batch
from datasets.augmentation_cifar import mixup_in_batch


#import keras_cv
import keras_cv_local as keras_cv
#import image_preprocessing



from config import config
conf = config.flags



def load(dataset_name,batch_size,input_size,input_size_pre_crop_ratio,num_class,
         train,input_prec_mode,preprocessor_input):

    dataset_name = dataset_name.lower()

    num_class = num_class
    #batch_size = conf.batch_size
    batch_size = config.batch_size
    #batch_size_inference = conf.batch_size_inf
    batch_size_inference = config.batch_size_inference
    i24nput_size = input_size
    input_size_pre_crop_ratio = input_size_pre_crop_ratio

    if dataset_name=='imagenet':
        data_down_dir = "/data/datasets/ImageNet_down/"
        write_dir = "/data/datasets/ImageNet"

        #    # Construct a tf.data.Dataset
        download_config = tfds.download.DownloadConfig(
            extract_dir=os.path.join(write_dir, 'extracted'),
            manual_dir=data_down_dir
        )

        #
        download_and_prepare_kwargs = {
            'download_dir': os.path.join(write_dir, 'download'),
            'download_config': download_config
        }

        data_dir = os.path.join(write_dir, 'data')
        dataset_name='imagenet2012'

        data_label_train='train'
        data_label_test='validation'
    else:
        download_and_prepare_kwargs = None
        data_dir = None

        data_label_train='train'
        data_label_test='test'

    #
    def default_load_train():
        if config.flags.num_train_data==-1:
            split_train = tfds.core.ReadInstruction(data_label_train, from_=0, to=100, unit='%')
        else:
            idx_train_s = config.flags.idx_train_data
            idx_train_e = idx_train_s + config.flags.num_train_data
            split_train = tfds.core.ReadInstruction(data_label_train, from_=idx_train_s, to=idx_train_e, unit='abs')

        if config.flags.calibration_idx_test or config.flags.vth_search_idx_test or (not config.flags.mode=='train') or (config.flags.f_write_stat):
            train_ds_shuffle = False
        else:
            train_ds_shuffle = True

        train_ds, train_ds_info = tfds.load(dataset_name, split=split_train, shuffle_files=train_ds_shuffle, as_supervised=True, with_info=True,
                                            data_dir = data_dir, download_and_prepare_kwargs = download_and_prepare_kwargs, )

        if config.flags.num_train_data==-1:
            train_ds_num = train_ds_info.splits[data_label_train].num_examples
        else:
            train_ds_num = config.flags.num_train_data

        return train_ds, train_ds_num

    #
    def default_load_test():
        if config.flags.full_test:
            split_test = tfds.core.ReadInstruction(data_label_test, from_=0, to=100, unit='%')
        else:
            idx_test_s = config.flags.idx_test_data
            idx_test_e = idx_test_s + config.flags.num_test_data
            split_test = tfds.core.ReadInstruction(data_label_test, from_=idx_test_s, to=idx_test_e, unit='abs')

        if config.flags.calibration_idx_test or config.flags.vth_search_idx_test or (not config.flags.mode=='train') or (config.flags.f_write_stat):
            train_ds_shuffle = False
        else:
            train_ds_shuffle = True

        valid_ds, valid_ds_info = tfds.load(dataset_name, split=split_test, shuffle_files=False, as_supervised=True, with_info=True,
                                            data_dir = data_dir, download_and_prepare_kwargs = download_and_prepare_kwargs, )

        if config.flags.full_test:
            valid_ds_num = valid_ds_info.splits[data_label_test].num_examples
        else:
            valid_ds_num = config.flags.num_test_data

        return valid_ds, valid_ds_num

    if train:
        if config.flags.data_aug_mix == 'mixup' or config.flags.data_aug_mix == 'cutmix':
            train_ds_1, train_ds_1_num = default_load_train()
            train_ds_2, train_ds_2_num = default_load_train()
            train_ds = tf.data.Dataset.zip((train_ds_1, train_ds_2))
            train_ds_num = train_ds_1_num
        else:
            train_ds, train_ds_num = default_load_train()
    else:
        train_ds, train_ds_num = default_load_train()

    valid_ds, valid_ds_num = default_load_test()


    num_parallel = tf.data.AUTOTUNE
    #
    #input_prec_mode = 'torch'

    #alpha = 0.5

    #
    # data augmentation
    # Preprocess input
    if train:
        if config.flags.data_aug_mix == 'mixup':

            train_ds_1 = train_ds_1.map(lambda image, label: resize_with_crop_aug(image, label, dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input) ,num_parallel_calls=num_parallel) \
                .prefetch(tf.data.AUTOTUNE)

            if config.flags.randaug_en:
                rand_augment = keras_cv.preprocessing.RandAugment(
                    value_range=(0, 255),
                    magnitude=conf.randaug_mag,
                    magnitude_stddev=conf.randaug_mag_std,
                    augmentations_per_image=conf.randaug_n,
                    rate=conf.randaug_rate)

                train_ds_1 = train_ds_1.map(lambda image, label: (rand_augment(tf.cast(image,tf.uint8)),label),num_parallel_calls=num_parallel) \
                    .prefetch(tf.data.AUTOTUNE)

            if config.flags.rand_erase_en:
                random_erase_1 = datasets.augmentation_cifar.RandomErasing()
                train_ds_1 = train_ds_1.map(lambda image, label: (random_erase_1._erase(image),label),num_parallel_calls=num_parallel) \
                    .prefetch(tf.data.AUTOTUNE)
            train_ds_1 = train_ds_1.batch(batch_size, drop_remainder=True)
            mixup= keras_cv.preprocessing.Mixup(alpha=conf.mix_alpha)

            train_ds_1 = train_ds_1.map(lambda image, label: (preprocessor_input(image,mode=input_prec_mode),label)) \
                .map(lambda image, label: mixup({"images":image, "labels":label}),num_parallel_calls=num_parallel) \
                .prefetch(tf.data.AUTOTUNE)

            train_ds = train_ds_1

        elif config.flags.data_aug_mix == 'cutmix':

            train_ds_1 = train_ds_1.map(
                lambda image, label: resize_with_crop_aug(image, label, dataset_name, input_size,
                                                          input_size_pre_crop_ratio, num_class, input_prec_mode,
                                                          preprocessor_input), num_parallel_calls=num_parallel) \
                .prefetch(tf.data.AUTOTUNE)

            if config.flags.randaug_en:
                rand_augment = keras_cv.preprocessing.RandAugment(
                    value_range=(0, 255),
                    magnitude=conf.randaug_mag,
                    magnitude_stddev=conf.randaug_mag_std,
                    augmentations_per_image=conf.randaug_n,
                    rate=conf.randaug_rate)
                train_ds_1 = train_ds_1.map(lambda image, label: (rand_augment(tf.cast(image, tf.uint8)), label),
                                            num_parallel_calls=num_parallel) \
                    .prefetch(tf.data.AUTOTUNE)

            if config.flags.rand_erase_en:
                random_erase_1 = datasets.augmentation_cifar.RandomErasing()
                train_ds_1 = train_ds_1.map(lambda image, label: (random_erase_1._erase(image), label),
                                            num_parallel_calls=num_parallel) \
                    .prefetch(tf.data.AUTOTUNE)

            train_ds_1 = train_ds_1.map(lambda image, label: (preprocessor_input(image, mode=input_prec_mode), label)) \
                .map(cutmix_in_batch, num_parallel_calls=num_parallel) \
                .prefetch(tf.data.AUTOTUNE)

            train_ds = train_ds_1

            #train_ds_1 = train_ds_1.map(lambda image, label: random_erase_1._erase(image),num_parallel_calls=num_parallel).prefetch(tf.data.AUTOTUNE)

            #train_ds_2 = train_ds_2.map(lambda image, label: resize_with_crop_aug(image, label, dataset_name,input_size,input_size_pre_crop_ratio,num_class,input_prec_mode,preprocessor_input) ,num_parallel_calls=num_parallel) \
            #                    .map(lambda image, label: (rand_augment(tf.cast(image,tf.uint8)),label),num_parallel_calls=num_parallel)
            #train_ds_2 = train_ds_2.map(lambda image, label: random_erase_1._erase(image),num_parallel_calls=num_parallel).prefetch(tf.data.AUTOTUNE)

            #train_ds = train_ds_1.map(cutmix_in_batch,num_parallel_calls=num_parallel).prefetch(tf.data.AUTOTUNE)

            #train_ds = tf.data.Dataset.zip((train_ds_1,train_ds_2))

            #train_ds = train_ds.map(
            #    lambda train_ds_1, train_ds_2: cutmix(train_ds_1, train_ds_2),
            #                            num_parallel_calls=num_parallel).prefetch(tf.data.AUTOTUNE)


            #train_ds = train_ds.map(
            #    lambda train_ds_1, train_ds_2: cutmix(train_ds_1, train_ds_2, dataset_name, input_size, input_size_pre_crop_ratio,
            #                                          num_class, alpha, input_prec_mode,preprocessor_input),
            #                            num_parallel_calls=num_parallel).prefetch(tf.data.AUTOTUNE)
            #train_ds = train_ds.map(cutmix,num_parallel_calls=num_parallel).prefetch(tf.data.AUTOTUNE)

            # train_ds = train_ds.map(lambda train_ds_1, train_ds_2: eager_cutmix(train_ds_1, train_ds_2, alpha=0.2),
            #                        num_parallel_calls=num_parallel)


        else:
            train_ds = train_ds.map(
                lambda image, label: resize_with_crop_aug(image, label, dataset_name, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode,preprocessor_input),
                num_parallel_calls=num_parallel) \
                .prefetch(tf.data.AUTOTUNE)

            if config.flags.randaug_en:
                rand_augment = keras_cv.preprocessing.RandAugment(
                    value_range=(0, 255),
                    magnitude=conf.randaug_mag,
                    magnitude_stddev=conf.randaug_mag_std,
                    augmentations_per_image=conf.randaug_n,
                    rate=conf.randaug_rate)

                train_ds = train_ds.map(lambda image, label: (rand_augment(tf.cast(image, tf.uint8)), label),
                                            num_parallel_calls=num_parallel) \
                    .prefetch(tf.data.AUTOTUNE)

            if config.flags.rand_erase_en:
                random_erase_1 = datasets.augmentation_cifar.RandomErasing()
                train_ds = train_ds.map(lambda image, label: (random_erase_1._erase(image), label),
                                            num_parallel_calls=num_parallel) \
                    .prefetch(tf.data.AUTOTUNE)

            train_ds = train_ds.map(lambda image, label: (preprocessor_input(image,mode=input_prec_mode),label)) \
                            .prefetch(tf.data.AUTOTUNE)

        if config.flags.data_aug_mix == 'mixup':
            train_ds = train_ds.prefetch(num_parallel)
        else:
            train_ds = train_ds.batch(batch_size,drop_remainder=True)
            train_ds = train_ds.prefetch(num_parallel)
    else:
        train_ds = train_ds.map(
            lambda image, label: resize_with_crop(image, label, dataset_name, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode,preprocessor_input),
            num_parallel_calls=num_parallel) \
                        .map(lambda image, label: (preprocessor_input(image,mode=input_prec_mode),label)) \
                        .prefetch(tf.data.AUTOTUNE)

        if config.flags.randaug_en:
            rand_augment = keras_cv.preprocessing.RandAugment(
                value_range=(0, 255),
                magnitude=conf.randaug_mag,
                magnitude_stddev=conf.randaug_mag_std,
                augmentations_per_image=conf.randaug_n,
                rate=conf.randaug_rate)

            train_ds = train_ds.map(lambda image, label: (rand_augment(tf.cast(image, tf.uint8)), label),
                                    num_parallel_calls=num_parallel) \
                .prefetch(tf.data.AUTOTUNE)

        if config.flags.rand_erase_en:
            random_erase_1 = datasets.augmentation_cifar.RandomErasing()
            train_ds = train_ds.map(lambda image, label: (random_erase_1._erase(image), label),
                                    num_parallel_calls=num_parallel) \
                .prefetch(tf.data.AUTOTUNE)

        train_ds = train_ds.map(lambda image, label: (preprocessor_input(image, mode=input_prec_mode), label)) \
            .prefetch(tf.data.AUTOTUNE)

        train_ds = train_ds.batch(batch_size,drop_remainder=True)
        train_ds = train_ds.prefetch(num_parallel)

    valid_ds = valid_ds.map(
        lambda image, label: resize_with_crop(image, label, dataset_name, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode,preprocessor_input),
        num_parallel_calls=num_parallel) \
                    .map(lambda image, label: (preprocessor_input(image, mode=input_prec_mode),label)) \
                    .prefetch(tf.data.AUTOTUNE)

    #valid_ds = valid_ds.batch(batch_size,drop_remainder=True)
    valid_ds = valid_ds.batch(batch_size_inference,drop_remainder=True)
    valid_ds = valid_ds.prefetch(num_parallel)

    # data-based weight normalization (DNN-to-SNN conversion)
    if config.flags.f_write_stat and config.flags.f_stat_train_mode:
        valid_ds = train_ds

    return train_ds, valid_ds, valid_ds, train_ds_num, valid_ds_num, valid_ds_num

