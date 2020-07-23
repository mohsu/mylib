# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: utils.py 
@time: 2020/05/25
"""

# python packages
import tensorflow as tf

# 3rd-party packages
import numpy as np
from loguru import logger

# self-defined packages
from utils.flag_utils import get_flag


@logger.catch(reraise=True)
def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


@logger.catch(reraise=True)
def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


@logger.catch(reraise=True)
def yolo_decode_predictions(model_output, output_images):
    boxes, scores, classes, nums = model_output

    decoded = []
    for i in range(len(output_images)):  # batch_size
        wh = np.flip(output_images[i].shape[0:2])
        objs = []
        for n in range(nums[i]):
            x1y1 = tuple((np.array(boxes[i][n][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][n][2:4]) * wh).astype(np.int32))
            score = float(scores[i][n])
            label = int(classes[i][n])
            objs.append([x1y1 + x2y2, score, label])
        decoded.append(sorted(objs, key=lambda x: (x[0], x[1], x[-1])))
    return decoded


def decode_predictions(model_output, output_images):
    # boxes, scores, classes = model_output
    decoded = []
    for i in range(len(output_images)):  # batch_size
        wh = np.flip(output_images[i].shape[0:2])
        objs = []
        for box, scores, klasses in zip(*model_output[i]):
            x1y1 = tuple((np.array(box[0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(box[2:4]) * wh).astype(np.int32))
            objs.append([x1y1 + x2y2, scores, klasses])
        decoded.append(objs)
    return decoded


def rescale_pred_wh(pred_boxes, output_image_shape):
    wh = np.flip(output_image_shape[0:2])
    rescaled = np.array(pred_boxes * np.repeat(wh, 2)).astype(np.int32)
    return rescaled


def convert2_class(scores, classes):
    if isinstance(scores, list) or scores.ndim == 3:
        return [convert2_class(score, classes) for score in scores]
    scores = tf.split(scores, classes, axis=-1)
    selected_scores, selected_classes = [], []
    for c in range(len(classes)):
        selected_class = tf.argmax(scores[c], axis=-1)
        ind = tf.cast(tf.expand_dims(selected_class, 1), tf.int32)
        ran = tf.expand_dims(tf.range(tf.shape(selected_class)[0], dtype=tf.int32), 1)
        ind = tf.concat([ran, ind], axis=1)
        selected_score = tf.gather_nd(scores[c], ind)
        selected_classes.append(selected_class)
        selected_scores.append(selected_score)

    return np.array(tf.stack(selected_classes, axis=-1)), np.array(tf.stack(selected_scores, axis=-1))


def decode_nms(preds, test_image_shape=None):
    boxes, scores, nums = preds
    if test_image_shape:
        boxes = rescale_pred_wh(boxes, test_image_shape)
    boxes = tf.split(boxes, nums)
    scores = tf.split(scores, nums)

    boxes = [box.numpy() for box in boxes]
    scores = [score.numpy() for score in scores]

    return boxes, scores


def trim_zeros_2d(arr_2d, axis=1):
    mask = ~(arr_2d == 0).all(axis=axis)
    return arr_2d[mask]


def decode_labels(test_labels, output_images):
    decoded = []

    def multiply_scale(x, scale):
        return int(float(x) * scale)

    for i in range(len(output_images)):
        labels = []
        H, W = output_images[i].shape[:2]
        for obj in test_labels[i]:
            xmin, ymin, xmax, ymax = obj[:4]
            klass = np.argmax(obj[4:])
            if sum(obj[:4]) != 0:  # exclude zero padding
                labels.append([(multiply_scale(xmin, W), multiply_scale(ymin, H),
                                multiply_scale(xmax, W), multiply_scale(ymax, H)),
                               klass])
        decoded.append(sorted(labels, key=lambda x: (x[0], x[1], x[-1])))
    return decoded


def find_closet_box(pred, objs, thres=20):
    coords, score, klass = pred

    losses = []
    for i, obj in enumerate(objs):
        if klass != obj[-1]:
            continue
        loss = sum(np.abs(np.array(coords) - np.array(obj[0])))
        if loss < thres:
            losses.append([i, loss])
    if not losses:
        return None, None
    losses = sorted(losses, key=lambda x: x[1])
    return losses[0]
