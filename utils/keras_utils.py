# -*- coding:utf-8 _*-
"""
@author: Maureen Hsu
@file: gpu_setting.py
@time: 2019/10/21
"""

# python packages

# 3rd-party packages
import tensorflow.keras as keras
from loguru import logger
import matplotlib.pyplot as plt


class LogHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        s = f"[Epoch {epoch+1}/{self.params['epochs']}]"
        for key, value in logs.items():
            s += f" {key}:{value:.4f}"

        logger.debug(s)


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def get_balanced_weight(dic_class):
    total = 0
    dic_weight = {}

    dic_class_idx = {klass: i for i, klass in enumerate(sorted(dic_class.keys()))}

    for klass in dic_class:
        fraction = 1. / dic_class[klass]
        dic_weight[dic_class_idx[klass]] = fraction
        total += fraction

    multiplier = 1.0 / total

    for klass in dic_weight:
        dic_weight[klass] = dic_weight[klass] * multiplier

    logger.debug("Class weights are {}".format(
        ", ".join("{}: {:.4f}".format(klass, weight) for klass, weight in dic_weight.items())))

    return dic_weight


def convert_tf1_model_to_tf2(tf2_model, tf1_weights, model_path=None, weight_path=None):
    tf2_model.load_weight(tf1_weights)
    if model_path is not None:
        tf2_model.save(model_path)
    if weight_path is not None:
        tf2_model.save_weights(weight_path)
