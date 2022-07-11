
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds


#def load(conf):
def load(dataset_name,batch_size,input_size,input_size_pre_crop_ratio,num_class,train,num_parallel,conf,input_prec_mode):
#def load(dataset_name, input_size, input_size_pre_crop_ratio, num_class, train, num_parallel, conf, input_prec_mode):

    # Get ImageNet labels
    #labels_path = tf.keras.utils.get_file('ImageNetwLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    #imagenet_labels = np.array(open(labels_path).read().splitlines())
    #print(imagenet_labels)
    #assert False

    #    # Set data_dir to a read-only storage of .tar files
    #    # Set write_dir to a w/r wtorage
    data_dir = '~/Datasets/ImageNet_down/'
    write_dir = '~/Datasets/ImageNet'
    #
    #    # Construct a tf.data.Dataset
    download_config = tfds.download.DownloadConfig(
        extract_dir=os.path.join(write_dir,'extracted'),
        manual_dir=data_dir
    )

    #
    download_and_prepare_kwargs={
        'download_dir': os.path.join(write_dir, 'download'),
        'download_config': download_config
    }


    ds = tfds.load('imagenet2012',
                   data_dir=os.path.join(write_dir, 'data'),
                   #split='train',
                  #split='validation',
                   split=['train','validation'],
                   shuffle_files=True,
                   #download=False,
                   download=True,
                   as_supervised=True,
                   #with_info=True,
                   download_and_prepare_kwargs=download_and_prepare_kwargs,
                   )
#
#   return None, ds, ds

    if conf.mode=='inference':
        shuffle = False
    else:
        shuffle=True

    train_ds, train_ds_info = tfds.load('imagenet2012',
                           data_dir=os.path.join(write_dir, 'data'),
                           split='train',
                           shuffle_files=shuffle,
                           as_supervised=True,
                           with_info=True,
                           download=True,
                           download_and_prepare_kwargs=download_and_prepare_kwargs,
                           )

    valid_ds, valid_ds_info = tfds.load('imagenet2012',
                            data_dir=os.path.join(write_dir, 'data'),
                            split='validation',
                            shuffle_files=False,
                            as_supervised=True,
                            with_info=True,
                            download = True,
                            download_and_prepare_kwargs=download_and_prepare_kwargs,
                            )

    print(train_ds_info)


    train_ds_num = train_ds_info.splits['train'].num_examples
    valid_ds_num = valid_ds_info.splits['validation'].num_examples


    return train_ds, valid_ds, valid_ds, train_ds_num, valid_ds_num, valid_ds_num


