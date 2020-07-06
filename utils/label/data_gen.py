# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: data_gen.py 
@time: 2020/03/09
"""

# python packages

# 3rd-party packages
from loguru import logger
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


# self-defined packages


@logger.catch(reraise=True)
class VOCDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, dataset, batch_size,
                 read_data_method, data_process_method=None, image_aug=None,
                 shuffle=True, loop=True):
        """Initialization"""
        self.dataset = dataset  # VOCDataSplitLabels
        self.read_data_method = read_data_method
        self.data_process_method = data_process_method
        self.image_aug = image_aug
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loop = loop
        self.indexes = np.arange(len(self.dataset))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset))
        """Updates indexes after each epoch"""
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        i = self.batch_size * index
        batch_images, batch_labels = [], []
        while len(batch_images) < self.batch_size:
            batch_indexes = []
            for b in range(self.batch_size):
                batch_indexes.append(self.indexes[i])

                i = (i + 1) % self.dataset.nb_samples

            # read data
            images, labels = self.read_data_method(self.dataset.annotations, batch_indexes)

            # append to batch
            batch_images += images
            batch_labels += labels

        # data augmentation
        if self.image_aug is not None:
            batch_images, batch_labels = self.image_aug.aug(batch_images, batch_labels)

        # process data
        if self.data_process_method is not None:
            batch_images, batch_labels = self.data_process_method(batch_images, batch_labels)

        return batch_images, batch_labels

    def get_all(self):
        images, labels = self.read_data_method(self.dataset.annotations, self.indexes)
        return images, labels
