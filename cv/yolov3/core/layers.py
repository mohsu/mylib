# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: layers.py 
@time: 2020/05/22
"""

# python packages
from absl.flags import FLAGS

# 3rd-party packages
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    ZeroPadding2D,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Add,
    Input,
    UpSampling2D,
    Concatenate,
    Lambda
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,

)
from tensorflow.keras.models import Model

# self-defined packages
from cv.yolov3.core.utils import (
    broadcast_iou,
)
from utils.flag_utils import get_flag


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters=filters, size=3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, size=1)
        x = DarknetConv(x, filters * 2, size=3)
        x = DarknetConv(x, filters, size=1)
        x = DarknetConv(x, filters * 2, size=3)
        x = DarknetConv(x, filters, size=1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    total_class_length = sum(classes) + 5

    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * total_class_length, 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, total_class_length)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


# @tf.function
def calculate_category_loss(class_probs, classes):
    def _calculate_loss(i):
        num_category_class = tf.gather(classes, i)
        start_index = tf.cast(tf.reduce_sum(tf.slice(classes, [0], [i])), tf.int32)
        end_index = tf.cast(start_index + num_category_class, tf.int32)
        loss = tf.sigmoid(class_probs[..., start_index:end_index])
        return loss

    return _calculate_loss


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, sum(classes)), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    # shape = (batch_size, )
    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    """
    This operation performs non_max_suppression on the inputs per batch, across all classes. Prunes away boxes that 
    have high intersection-over-union (IOU) overlap with previously selected boxes. Bounding boxes are supplied as 
    [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners and the 
    coordinates can be provided as normalized (i.e., lying in the interval [0, 1]) or absolute. 
    Note that this algorithm is agnostic to where the origin is in the coordinate system. Also note that this algorithm 
    is invariant to orthogonal transformations and translations of the coordinate system; thus translating or 
    reflections of the coordinate system result in the same boxes being selected by the algorithm. The output of this 
    operation is the final boxes, scores and classes tensor returned after performing non_max_suppression.
    :parameter
    @@ boxes: [batch_size, num_boxes, q, 4] A 4-D float Tensor of shape [batch_size, num_boxes, q, 4]. If q is 1 then 
                same boxes are used for all classes otherwise, if q is equal to number of classes, 
                class-specific boxes are used.
    @@ scores: A 3-D float Tensor of shape [batch_size, num_boxes, num_classes] representing a single score 
                corresponding to each box (each row of boxes). 
    @@ max_output_size_per_class: A scalar integer Tensor representing the maximum number of boxes to be selected by 
                                  non-max suppression per class
    @@ max_total_size: A scalar representing the maximum number of boxes retained over all classes.
    @@ iou_threshold: A float representing the threshold for deciding whether boxes overlap too much with respect to IOU.
    @@ score_threshold: A float representing the threshold for deciding when to remove boxes based on score.
    @@ pad_per_class: If false, the output nmsed boxes, scores and classes are padded/clipped to max_total_size. 
                      If true, the output nmsed boxes, scores and classes are padded to be of length
                       max_size_per_class*num_classes, unless it exceeds max_total_size in which case it is clipped to 
                       max_total_size. Defaults to false.
    @@ clip_boxes: If true, the coordinates of output nmsed boxes will be clipped to [0, 1]. If false, output the box 
                   coordinates as it is. Defaults to true.
    """
    scores = confidence * class_probs
    boxes = tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4))
    scores = tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1]))
    scores_combined = tf.ones(tf.shape(scores)[:-1])
    start_idx = 0
    for num_category_class in classes:
        end_idx = start_idx + num_category_class
        scores_combined *= tf.reduce_max(scores[..., start_idx:end_idx], axis=2)
        start_idx = end_idx
    scores_combined = tf.expand_dims(scores_combined, axis=-1)

    output_boxes, output_scores, output_classes, output_valid_detections = tf.image.combined_non_max_suppression(
        boxes=boxes,
        scores=scores_combined,  # [batch_size, num_boxes, num_classes]
        max_output_size_per_class=get_flag("yolo_max_boxes", 10),
        max_total_size=get_flag("yolo_max_boxes", 10),
        iou_threshold=get_flag("yolo_iou_threshold", 0.45),
        score_threshold=get_flag("yolo_score_threshold", 0.2)
    )

    return output_boxes, output_scores, output_classes, output_valid_detections, scores_combined


@tf.function
def multiclass_nms(pred):
    batch_boxes, batch_scores_combined, batch_scores = tf.split(pred, (4, 1, -1), axis=-1)
    batch_size = tf.shape(batch_boxes)[0]
    boxes = tf.zeros([0, 4])
    scores = tf.zeros([0, tf.shape(batch_scores)[-1]])
    nums = tf.zeros([0], dtype=tf.int32)

    def cond(i, *args):
        return tf.less(i, batch_size)

    @tf.function
    def body(i, boxes, scores, nums):
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            tf.reshape(batch_boxes[i], (-1, 4)),
            tf.reshape(batch_scores_combined[i], (-1,)),
            max_output_size=get_flag("yolo_max_boxes", 10),
            iou_threshold=get_flag("yolo_iou_threshold", 0.45),
            score_threshold=get_flag("yolo_score_threshold", 0.2))
        selected_boxes = tf.gather(batch_boxes[i], selected_indices)
        selected_scores = tf.gather(batch_scores[i], selected_indices)

        boxes = tf.concat([boxes, selected_boxes], axis=0)
        scores = tf.concat([scores, selected_scores], axis=0)
        nums = tf.concat([nums, [tf.shape(selected_indices)[0]]], axis=0)

        return tf.add(i, 1), boxes, scores, nums

    i = tf.constant(0)
    i, boxes, scores, nums = tf.while_loop(cond, body, loop_vars=[i, boxes, scores, nums],
                                           shape_invariants=[i.get_shape(),
                                                             tf.TensorShape([None, 4]),
                                                             tf.TensorShape([None, None]),
                                                             tf.TensorShape([None])],
                                           back_prop=False, parallel_iterations=8)

    return boxes, scores, nums


def multiclass_nms_tensorarray(pred):
    batch_boxes, batch_scores_combined, batch_scores = tf.split(pred, (4, 1, -1), axis=-1)
    boxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False, clear_after_read=True)
    scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False, clear_after_read=True)
    nums = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=False, clear_after_read=True)

    # selected = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False, clear_after_read=True)

    def cond(i, boxes, scores, nums):
        return tf.less(i, tf.shape(batch_boxes)[0])

    def body(i, boxes, scores, nums):
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            tf.reshape(batch_boxes[i], (-1, 4)),
            tf.reshape(batch_scores_combined[i], (-1,)),
            max_output_size=get_flag("yolo_max_boxes", 10),
            iou_threshold=get_flag("yolo_iou_threshold", 0.45),
            score_threshold=get_flag("yolo_score_threshold", 0.2))
        selected_boxes = tf.gather(batch_boxes[i], selected_indices)
        selected_scores = tf.gather(batch_scores[i], selected_indices)

        boxes.write(i, selected_boxes)
        scores.write(i, selected_scores)
        nums.write(i, tf.shape(selected_indices)[0])

        # selected.write(i, tf.concat([boxes, scores], axis=-1))
        # selected.append([boxes, scores])
        return tf.add(i, 1), boxes, scores, nums

    i = tf.constant(0)
    i, boxes, scores, nums = tf.while_loop(cond, body, loop_vars=[i, boxes, scores, nums], back_prop=False)

    boxes = boxes.concat()
    scores = scores.concat()
    nums = nums.stack()

    return boxes, scores, nums


def stack_preds(outputs, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)
    scores = confidence * class_probs
    batch_size = tf.size(bbox[:, 0, 0])
    boxes = tf.reshape(bbox, (batch_size, -1, 4))
    scores = tf.reshape(scores, (batch_size, -1, tf.shape(scores)[-1]))
    scores_combined = tf.ones(tf.shape(scores)[:-1])
    start_idx = 0
    for num_category_class in classes:
        end_idx = start_idx + num_category_class
        scores_combined *= tf.reduce_max(scores[..., start_idx:end_idx], axis=2)
        start_idx = end_idx
    scores_combined = tf.expand_dims(scores_combined, axis=-1)
    preds = tf.concat((boxes, scores_combined, scores), axis=-1)

    return preds

    # for n in tf.range(batch_size):
    #     selected_indices, selected_scores = tf.image.non_max_suppression_padded(
    #         boxes[n], scores_combined[n],
    #         max_output_size=get_flag("yolo_max_boxes", 10),
    #         iou_threshold=get_flag("yolo_iou_threshold", 0.45),
    #         score_threshold=get_flag("yolo_score_threshold", 0.2))
    #     selected_boxes = tf.gather(boxes[n], selected_indices)
    #     selected_scores = tf.gather(scores[n], selected_indices)


def weighted_binary_crossentropy(weights=1):
    def _calculate_weighted_binary_crossentropy(labels, output, from_logits):
        """Calculate weighted binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=tf.float32)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.math.log(output / (1 - output))

        return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output)

    def _weighted_binary_crossentropy(y_true, y_pred, from_logits=False):
        bce = _calculate_weighted_binary_crossentropy(y_true, y_pred, from_logits)
        weighted_bac = bce * tf.convert_to_tensor(weights, tf.float32)
        return K.mean(weighted_bac, axis=-1)

    return _weighted_binary_crossentropy


def calculate_loss(obj_mask, true_class, pred_class):
    def _calculate_loss(i, j):
        return obj_mask * binary_crossentropy(true_class[..., i:j],
                                              pred_class[..., i:j])

    return _calculate_loss


def YoloLoss(anchors, classes=[80], ignore_thresh=0.5, weights=[1, 1, 1, 1]):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class = tf.split(y_true, (4, 1, sum(classes)), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
                  tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask), tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

        class_loss = tf.zeros(tf.shape(obj_loss))
        start_idx = 0
        for num_category_class in classes:
            end_idx = start_idx + num_category_class
            class_loss += calculate_loss(obj_mask, true_class, pred_class)(start_idx, end_idx)
            start_idx = end_idx

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3)) * float(weights[0])
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3)) * float(weights[1])
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3)) * float(weights[2])
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3)) * float(weights[3])

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss
