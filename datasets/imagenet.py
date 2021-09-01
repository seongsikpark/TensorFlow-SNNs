
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds


def load(conf):
    # Get ImageNet labels
    labels_path = tf.keras.utils.get_file('ImageNetwLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    print(imagenet_labels)

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

    print('here')
    #
    download_and_prepare_kwargs={
        'download_dir': os.path.join(write_dir, 'download'),
        'download_config': download_config
    }


    ds = tfds.load('imagenet2012',
                   data_dir=os.path.join(write_dir, 'data'),
                   split='validation',
                   #split=['train','validation']
                   shuffle_files=False,
                   download=True,
                   as_supervised=True,
                   #with_info=True,
                   download_and_prepare_kwargs=download_and_prepare_kwargs,
                   )

    global input_size_pre_crop_ratio
    input_size_pre_crop_ratio = 256/224

    return None, ds, ds


