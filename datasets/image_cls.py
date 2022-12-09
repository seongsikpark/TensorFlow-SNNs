
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds

from datasets.augmentation_cifar import resize_with_crop
from datasets.augmentation_cifar import resize_with_crop_aug
from datasets.augmentation_cifar import mixup
from datasets.augmentation_cifar import cutmix

f_cross_valid = False

def load(dataset_name,batch_size,input_size,input_size_pre_crop_ratio,num_class,train,num_parallel,conf,input_prec_mode,preprocessor_input):

    dataset_name = dataset_name.lower()

    num_class = num_class
    #batch_size_train = conf.batch_size
    #batch_size_inference = conf.batch_size_inf
    input_size = input_size
    input_size_pre_crop_ratio = input_size_pre_crop_ratio

    if dataset_name=='imagenet':
        data_down_dir = '/home/sspark/Datasets/ImageNet_down/'
        write_dir = '/home/sspark/Datasets/ImageNet'

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
    def default_load():
        if conf.num_train_data==-1:
            split_train = tfds.core.ReadInstruction(data_label_train, from_=0, to=100, unit='%')
        else:
            idx_train_s = conf.idx_train_data
            idx_train_e = idx_train_s + conf.num_train_data
            split_train = tfds.core.ReadInstruction(data_label_train, from_=idx_train_s, to=idx_train_e, unit='abs')

        if conf.full_test:
            split_test = tfds.core.ReadInstruction(data_label_test, from_=0, to=100, unit='%')
        else:
            idx_test_s = conf.idx_test_data
            idx_test_e = idx_test_s + conf.num_test_data
            split_test = tfds.core.ReadInstruction(data_label_test, from_=idx_test_s, to=idx_test_e, unit='abs')

        if conf.calibration_idx_test or conf.vth_search_idx_test or (not conf.mode=='train') or (conf.f_write_stat):
            train_ds_shuffle = False
        else:
            train_ds_shuffle = True

        train_ds, train_ds_info = tfds.load(dataset_name, split=split_train, shuffle_files=train_ds_shuffle, as_supervised=True, with_info=True,
                                            data_dir = data_dir, download_and_prepare_kwargs = download_and_prepare_kwargs, )
        valid_ds, valid_ds_info = tfds.load(dataset_name, split=split_test, shuffle_files=False, as_supervised=True, with_info=True,
                                            data_dir = data_dir, download_and_prepare_kwargs = download_and_prepare_kwargs, )

        if conf.num_train_data==-1:
            train_ds_num = train_ds_info.splits[data_label_train].num_examples
        else:
            train_ds_num = conf.num_train_data

        if conf.full_test:
            valid_ds_num = valid_ds_info.splits[data_label_test].num_examples
        else:
            valid_ds_num = conf.num_test_data

        return train_ds, valid_ds, train_ds_num, valid_ds_num
    #
    if train:
        if f_cross_valid:
            #train_ds = tfds.load('cifar10',
            train_ds, train_ds_info = tfds.load(dataset_name,
                                 split=[f'train[:{k}%]+train[{ k +10}%:]' for k in range(0 ,100 ,10)],
                                 shuffle_files=True,
                                 as_supervised=True,
                                 with_info=True,
                                 data_dir=data_dir,
                                 download_and_prepare_kwargs=download_and_prepare_kwargs,
                                 )

            valid_ds, valid_ds_info = tfds.load(dataset_name,
                                 split=[f'train[{k}%:{ k +10}%]' for k in range(0 ,100 ,10)],
                                 shuffle_files=True,
                                 as_supervised=True,
                                 with_info=True,
                                 data_dir=data_dir,
                                 download_and_prepare_kwargs=download_and_prepare_kwargs,
                                 )

            train_ds_num = train_ds_info.splits[data_label_train].num_examples
            valid_ds_num = valid_ds_info.splits[data_label_test].num_examples

        elif conf.data_aug_mix=='mixup' or conf.data_aug_mix=='cutmix':
            train_ds_1, train_ds_1_info = tfds.load(dataset_name,
                                   split=data_label_train,
                                   shuffle_files=True,
                                   as_supervised=True,
                                   with_info=True,
                                   data_dir=data_dir,
                                   download_and_prepare_kwargs=download_and_prepare_kwargs,
                                   )

            train_ds_2, train_ds_2_info = tfds.load(dataset_name,
                                   split=data_label_train,
                                   shuffle_files=True,
                                   as_supervised=True,
                                   with_info=True,
                                   data_dir=data_dir,
                                   download_and_prepare_kwargs=download_and_prepare_kwargs,
                                   )


            valid_ds, valid_ds_info = tfds.load(dataset_name,
                                 split=data_label_test,
                                 as_supervised=True,
                                 with_info=True,
                                 data_dir=data_dir,
                                 download_and_prepare_kwargs=download_and_prepare_kwargs,
                                 )

            train_ds = tf.data.Dataset.zip((train_ds_1 ,train_ds_2))
            train_ds_num = train_ds_1_info.splits[data_label_train].num_examples
            valid_ds_num = valid_ds_info.splits[data_label_test].num_examples
        else:
            train_ds, valid_ds, train_ds_num, valid_ds_num = default_load()
            #(train_ds, valid_ds), ds_info = tfds.load(dataset_name,
            #                               split=[data_label_train, data_label_test],
            #                               shuffle_files=True,
            #                               as_supervised=True,
            #                               with_info=True,
            #                               data_dir=data_dir,
            #                               download_and_prepare_kwargs=download_and_prepare_kwargs,
            #                               )
            #
            #train_ds_num = ds_info.splits[data_label_train].num_examples
            #valid_ds_num = ds_info.splits[data_label_test].num_examples

    else:
        train_ds, valid_ds, train_ds_num, valid_ds_num = default_load()


    #
    #input_prec_mode = 'torch'

    #
    # data augmentation
    # Preprocess input
    if train:
        if conf.data_aug_mix == 'mixup':
            train_ds = train_ds.map(lambda train_ds_1, train_ds_2: mixup(train_ds_1, train_ds_2, dataset_name, 1.0, input_prec_mode,preprocessor_input),
                                    num_parallel_calls=num_parallel)
            # train_ds=train_ds.map(lambda train_ds_1, train_ds_2: eager_mixup(train_ds_1,train_ds_2,alpha=0.2),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif conf.data_aug_mix == 'cutmix':
            train_ds = train_ds.map(
                lambda train_ds_1, train_ds_2: cutmix(train_ds_1, train_ds_2, dataset_name, input_size, input_size_pre_crop_ratio,
                                                      num_class, 1.0, input_prec_mode,preprocessor_input),
                num_parallel_calls=num_parallel)
            # train_ds = train_ds.map(lambda train_ds_1, train_ds_2: eager_cutmix(train_ds_1, train_ds_2, alpha=0.2),
            #                        num_parallel_calls=num_parallel)
        else:
            train_ds = train_ds.map(
                lambda image, label: resize_with_crop_aug(image, label, dataset_name, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode,preprocessor_input),
                num_parallel_calls=num_parallel)

        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(num_parallel)
    else:
        train_ds = train_ds.map(
            lambda image, label: resize_with_crop(image, label, dataset_name, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode,preprocessor_input),
            num_parallel_calls=num_parallel)

        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(num_parallel)

    # valid_ds=valid_ds.map(resize_with_crop_cifar,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # valid_ds=valid_ds.map(resize_with_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # valid_ds=valid_ds.map(resize_with_crop,num_parallel_calls=num_parallel)
    valid_ds = valid_ds.map(
        lambda image, label: resize_with_crop(image, label, dataset_name, input_size, input_size_pre_crop_ratio, num_class, input_prec_mode,preprocessor_input),
        num_parallel_calls=num_parallel)
    valid_ds = valid_ds.batch(batch_size)
    valid_ds = valid_ds.prefetch(num_parallel)

    # data-based weight normalization (DNN-to-SNN conversion)
    if conf.f_write_stat and conf.f_stat_train_mode:
        valid_ds = train_ds

    return train_ds, valid_ds, valid_ds, train_ds_num, valid_ds_num, valid_ds_num

