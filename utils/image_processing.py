import numpy as np
import cv2
from cv2 import *
# # enable X11 Forwarding
# # # 1. add DISPLAY to environment variable
# # # 2.
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import os_path


def show(image, convert_BGR=True, title=None):
    if isinstance(image, list):
        show_images(image, convert_BGR, title)
        return 0
    image = image.astype('uint8')
    if len(image.shape) == 2:
        is_bw = True
    else:
        is_bw = False
    if not is_bw:
        if convert_BGR:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.show()


def show_images(images, convert_BGR, titles):
    if titles is not None:
        for image, title in zip(images, titles):
            show(image, convert_BGR, title)
    else:
        for image in images:
            show(image, convert_BGR)


def to_bytes(image, _format=".png"):
    success, encoded_image = cv2.imencode(_format, image)
    image_bytearr = encoded_image.tobytes()
    return image_bytearr


def read_image_from_io(imgData):
    data = np.fromstring(imgData, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def save_image(image, image_path, make_dir=True, convert_BGR=False):
    if make_dir:
        os_path.make_dir(os_path.os.path.dirname(image_path))

    # if normalized
    if np.max(image) <= 1:
        image = image * 255

    # convert type
    if image.dtype != np.dtype('uint8'):
        image = image.astype('uint8')

    if convert_BGR:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(image_path, image)


def draw_box(image, bbox, color=(255, 0, 0), label=None, font_size=1):
    image = np.copy(image)
    H, W = image.shape[:2]
    h_stride, w_stride = H // 100, W // 100
    if isinstance(bbox, list) or isinstance(bbox, tuple):
        if len(bbox) == 5:
            x1, y1, x2, y2, label = bbox
        else:
            x1, y1, x2, y2 = bbox
    elif hasattr(bbox, "name"):  # is object
        x1, y1, x2, y2, label = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2), bbox.name
    else:
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    if label:
        cv2.putText(image, str(label), (x1 + w_stride, y1 + h_stride), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, color,
                    1,
                    cv2.LINE_AA)
    return image


def draw_boxes(image, bboxes, color=(255, 0, 0), labels=None, font_sizes=1):
    if not isinstance(bboxes, list):
        bboxes = [bboxes]

    if not isinstance(labels, list):
        labels = [labels] * len(bboxes)

    if not isinstance(font_sizes, list):
        font_sizes = [font_sizes] * len(bboxes)

    for bbox, label, font_size in zip(bboxes, labels,
                                      font_sizes):  # can be list of lists, list of objects, or list of bboxes
        image = draw_box(image, bbox, color, label=label, font_size=font_size)

    return image


def draw_annotation(annotation, color=(255, 0, 0), image=None):
    if image is None:
        image = imread(annotation.image_path)
    return draw_boxes(image, annotation.objects, color)
