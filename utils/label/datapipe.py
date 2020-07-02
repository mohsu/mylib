# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: datapipe.py 
@time: 2020/05/28
"""

# python packages
from enum import Enum

# 3rd-party packages
import numpy as np
from loguru import logger


# self-defined packages
from utils.label.VOCLabel import VOCAnnotationSet


class Dataset(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def get_indexes(nb_samples, test_split, val_split, is_shuffle):
    indexes = np.arange(nb_samples)

    if is_shuffle:
        np.random.seed(0)
        np.random.shuffle(indexes)

    test_count = int(nb_samples * test_split)
    test_indexes = indexes[:test_count]
    indexes = indexes[test_count:]

    if is_shuffle:
        np.random.seed()
        np.random.shuffle(indexes)

    val_split = int(len(indexes) * val_split)
    val_indexes = indexes[:val_split]
    train_indexes = indexes[val_split:]

    return train_indexes, val_indexes, test_indexes


class VOCDataSubset(VOCAnnotationSet):
    @property
    def training(self):
        return self._get_subset(Dataset.TRAIN)

    @property
    def testing(self):
        return self._get_subset(Dataset.VALIDATION)

    @property
    def validation(self):
        return self._get_subset(Dataset.TEST)

    @property
    def nb_samples(self):
        return len(self)

    def __init__(self, voc_set, val_split=0.1, test_split=0.1, is_shuffle=True, balanced=False):
        super().__init__()
        self.__dict__.update(voc_set.__dict__)
        self._set_indexes = {Dataset.TRAIN: np.array([]), Dataset.VALIDATION: np.array([]), Dataset.TEST: np.array([])}
        self._split_data(val_split, test_split, is_shuffle, balanced)

    def _split_data(self, val_split, test_split, is_shuffle, balanced):
        if val_split == 0 and test_split == 0:
            self._set_indexes[Dataset.TRAIN] = np.arange(self.nb_samples)
        else:
            if balanced:  # only valid for classification
                for klass in self.class_count:
                    class_count = self.class_count[klass]
                    train_indexes, val_indexes, test_indexes = get_indexes(class_count, test_split, val_split,
                                                                           is_shuffle)
                    arr = np.array(self.class_dict[klass])
                    self._set_indexes[Dataset.TEST] = np.append(self._set_indexes[Dataset.TEST], arr[test_indexes])
                    self._set_indexes[Dataset.VALIDATION] = np.append(self._set_indexes[Dataset.VALIDATION],
                                                                      arr[val_indexes])
                    self._set_indexes[Dataset.TRAIN] = np.append(self._set_indexes[Dataset.TRAIN], arr[train_indexes])
                    logger.info(
                        "Label {} has train: {} val: {} test: {}".format(klass, len(train_indexes), len(val_indexes),
                                                                         len(test_indexes)))

            else:
                train_indexes, val_indexes, test_indexes = get_indexes(self.nb_samples, test_split, val_split,
                                                                       is_shuffle)
                self._set_indexes[Dataset.TEST] = test_indexes
                self._set_indexes[Dataset.VALIDATION] = val_indexes
                self._set_indexes[Dataset.TRAIN] = train_indexes

            logger.info(
                f"Total nb_samples {self.nb_samples}, train: {len(self._set_indexes[Dataset.TRAIN])}, "
                f"val: {len(self._set_indexes[Dataset.VALIDATION])}, test: {len(self._set_indexes[Dataset.TEST])}")

    def _get_subset(self, subset):
        voc_set = VOCAnnotationSet(defined_classes=self.defined_classes, debug=self.debug,
                                   for_classification=self.for_classification)
        for i in self._set_indexes[Dataset(subset)]:
            voc_set.add_annotation(self.annotations[i])

        voc_split = VOCDataSubset(voc_set, 0.0, 0.0, is_shuffle=False, balanced=False)
        return voc_split
