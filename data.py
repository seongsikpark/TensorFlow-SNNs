"""
Written by Matteo Dunnhofer - 2017

Data provider class
"""
import os
import random
from PIL import Image
import scipy.ndimage, scipy.misc
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
#from tensorflow.data import Dataset

import cv2

class ImageNetDataset(object):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.imagenet_mean = [123.68, 116.779, 103.939]
        #self.imagenet_mean = self.imagenet_mean[::-1]
        self.img_shape = [224, 224, 3] # height, width, channel
        self.mode = mode

        self.ids, self.wnids, self.words = self.load_imagenet_meta()
#        data_imgs, data_labels = self.read_image_paths_labels()
        data_imgs, data_labels = self.get_files(self.cfg.data_path_imagenet,'val', None)

        if self.mode == 'train':
            self.dataset = tf.data.Dataset.from_tensor_slices((data_imgs, data_labels))
            self.dataset = self.dataset.map(self.input_parser)
            self.dataset = self.dataset.shuffle(10000).batch(self.cfg.batch_size)
        elif self.mode == 'val':
            self.dataset = tf.data.Dataset.from_tensor_slices((data_imgs, data_labels))
            self.dataset = self.dataset.map(self.input_parser).batch(self.cfg.batch_size)
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices((data_imgs, data_labels))
            self.dataset = self.dataset.map(self.input_parser)



    def load_imagenet_meta(self):
        """
        It reads ImageNet metadata from ILSVRC 2012 dev tool file

        Returns:
            wnids: list of ImageNet wnids labels (as strings)
            words: list of words (as strings) referring to wnids labels and describing the classes

        """
        meta_path = os.path.join(self.cfg.data_path_imagenet, 'data', 'meta.mat')
        metadata = loadmat(meta_path, struct_as_record=False)

        # ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
        synsets = np.squeeze(metadata['synsets'])
        ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets[0:999]]))
        wnids = np.squeeze(np.array([s.WNID for s in synsets[0:999]]))
        words = np.squeeze(np.array([s.words for s in synsets[0:999]]))
        return ids, wnids, words

    def get_files(self, data_dir, data_name, max_rows=None):
        """Reads a \`data_name.txt\` (e.g., \`val.txt\`) from
        http://www.image-net.org/challenges/LSVRC/2012/
        """
        #print(data_dir)
        #print(data_name)
        files, labels = np.split(
            np.genfromtxt("%s/%s.txt" % (data_dir, data_name),
                          dtype=np.str, max_rows=max_rows),
            [1], axis=1)
        #print(files)
        files = files.flatten()
        #print(files)
        labels = np.asarray(labels.flatten(), dtype=np.int)

        paths=[]

        for i in range(0,len(files)):
            paths.append(os.path.join(data_dir,'ILSVRC2012_img_val',files[i]))

        #for path in paths:
        #    print(path)

        #return tf.constant(paths[0:199]), tf.constant(labels[0:199])
        #return tf.constant(paths[:self.cfg.num_test_dataset-1]), tf.constant(labels[:self.cfg.num_test_dataset-1])
        return paths[:self.cfg.num_test_dataset-1], labels[:self.cfg.num_test_dataset-1]


    def read_image_paths_labels(self):
        """
        Reads the paths of the images (from the folders structure)
        and the indexes of the labels (using an annotation file)
        """
        paths = []
        labels = []

        train_dir='ILSVRC2012_img_train'

        if self.mode == 'train':
            for i, wnid in enumerate(self.wnids):
                                #print(self.ids[i], wnid, self.words[i])
                img_names = os.listdir(os.path.join(self.cfg.data_path_imagenet, train_dir, wnid))
                for img_name in img_names:
                    paths.append(os.path.join(self.cfg.data_path_imagenet, train_dir, wnid, img_name))
                    labels.append(i)

            # shuffling the images names and relative labels
            d = zip(paths, labels)
            random.shuffle(d)
            paths, labels = zip(*d)

        else:
            with open(os.path.join(self.cfg.data_path_imagenet, 'data', 'ILSVRC2012_validation_ground_truth.txt')) as f:
                groundtruths = f.readlines()
            groundtruths = [int(x.strip()) for x in groundtruths]

            images_names = sorted(os.listdir(os.path.join(self.cfg.data_path_imagenet, 'ILSVRC2012_img_val')))

            for image_name, gt in zip(images_names, groundtruths):
                paths.append(os.path.join(self.cfg.data_path_imagenet, 'ILSVRC2012_img_val', image_name))
                labels.append(gt)

        self.dataset_size = len(paths)

        #print(type(paths))

        return tf.constant(paths), tf.constant(labels)

    def input_parser(self, img_path, label):
        """
        Parse a single example
        Reads the image tensor (and preprocess it) given its path and produce a one-hot label given an integer index

        Args:
            img_path: a TF string tensor representing the path of the image
            label: a TF int tensor representing an index in the one-hot vector

        Returns:
            a preprocessed tf.float32 tensor of shape (heigth, width, channels)
            a tf.int one-hot tensor
        """
        one_hot = tf.one_hot(label, self.cfg.num_class)

        # image reading
        image = self.read_image(img_path)
        #iamge = self.load_img(img_path, target_size=, crop_size=)


    def load_img(self, paths, grayscale=False, target_size=None, crop_size=None,
                 interp=None):
        image_shape = tf.shape(image)

        # resize of the image (setting largest border to 256px)
        new_h = tf.cond(image_shape[0] < image_shape[1],
                            lambda: tf.div(tf.multiply(256, image_shape[1]), image_shape[0]),
                            lambda: 256)
        new_w = tf.cond(image_shape[0] < image_shape[1],
                            lambda: 256,
                            lambda: tf.div(tf.multiply(256, image_shape[0]), image_shape[1]))

        image = tf.image.resize_images(image, size=[new_h, new_w])

        if self.mode == 'test':
            # take random crops for testing
            patches = []
            for k in range(self.cfg.k_patches):
                patches.append(tf.random_crop(image, size=[self.img_shape[0], self.img_shape[1], self.img_shape[2]]))

            image = patches
        else:
            #image = tf.random_crop(image, size=[self.img_shape[0], self.img_shape[1], self.img_shape[2]])
            image = tf.image.resize_image_with_crop_or_pad(image, self.img_shape[0], self.img_shape[1])

            if self.mode == 'train':
                # some easy data augmentation
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)


        # normalization
        image = tf.to_float(image)

        #print(image_shape)
        #print(image)
        #image = image[:,:,::-1]
        #print(image[0,0,0])
        image = tf.subtract(image, self.imagenet_mean)

        return image, one_hot

    def read_image(self, img_path):
        """
        Given a path of image it reads its content
        into a tf tensor

        Args:
            img_path, a tf string tensor representing the path of the image

        Returns:
            the tf image tensor
        """
        img_file = tf.read_file(img_path)
        #print(img_file)

        #print(type(img_path))
        #print(dir(img_path.eval))
        #print(img_path.eval)

        #img = cv2.imread(img_path.eval)
        #print(img)

        #return tf.image.decode_jpeg(img_file, channels=self.img_shape[2], dct_method="INTEGER_ACCURATE")
        #return tf.image.decode_jpeg(img_file, channels=self.img_shape[2], dct_method="INTEGER_FAST")
        return tf.image.decode_jpeg(img_file, channels=self.img_shape[2])


    def crop(self, img, crop_size, crop_loc=4, crop_grid=(3, 3)):
        if isinstance(crop_loc, list):
            imgs = np.zeros((img.shape[0], len(crop_loc), crop_size, crop_size, 3),
                            np.float32)
            for (i, loc) in enumerate(crop_loc):
                r, c = crop_idx(img.shape[1:3], crop_size, loc, crop_grid)
                imgs[:, i] = img[:, r:r+crop_size, c:c+crop_size, :]
            return imgs
        elif crop_loc == np.prod(crop_grid) + 1:
            imgs = np.zeros((img.shape[0], crop_loc, crop_size, crop_size, 3),
                            np.float32)
            r, c = crop_idx(img.shape[1:3], crop_size, 4, crop_grid)
            imgs[:, 0] = img[:, r:r+crop_size, c:c+crop_size, :]
            imgs[:, 1] = img[:, 0:crop_size, 0:crop_size, :]
            imgs[:, 2] = img[:, 0:crop_size, -crop_size:, :]
            imgs[:, 3] = img[:, -crop_size:, 0:crop_size, :]
            imgs[:, 4] = img[:, -crop_size:, -crop_size:, :]
            imgs[:, 5:] = np.flip(imgs[:, :5], axis=3)
            return imgs
        else:
            r, c = crop_idx(img.shape[1:3], crop_size, crop_loc, crop_grid)
            return img[:, r:r+crop_size, c:c+crop_size, :]


    def load_img(self, paths, grayscale=False, target_size=None, crop_size=None,
                 interp=None):
        assert cv2 is not None, '`load_img` requires `cv2`.'
        if interp is None:
            interp = cv2.INTER_CUBIC
        if not isinstance(paths, list):
            paths = [paths]
        if len(paths) > 1 and (target_size is None or
                               isinstance(target_size, int)):
            raise ValueError('A tuple `target_size` should be provided '
                             'when loading multiple images.')

        def _load_img(path):
            img = cv2.imread(path)
            #print(img)
            if target_size:
                if isinstance(target_size, int):
                    hw_tuple = tuple([x * target_size // min(img.shape[:2])
                                      for x in img.shape[1::-1]])
                else:
                    hw_tuple = (target_size[1], target_size[0])
                if img.shape[1::-1] != hw_tuple:
                    img = cv2.resize(img, hw_tuple, interpolation=interp)
            img = img[:, :, ::-1]
            if len(img.shape) == 2:
                img = np.expand_dims(img, -1)
            return img

        if len(paths) > 1:
            imgs = np.zeros((len(paths),) + target_size + (3,), dtype=np.float32)
            for (i, path) in enumerate(paths):
                imgs[i] = _load_img(path)
        else:
            imgs = np.array([_load_img(paths[0])], dtype=np.float32)

        if crop_size is not None:
            imgs = self.crop(imgs, crop_size)

        return imgs
