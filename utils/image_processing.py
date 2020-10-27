import numpy as np
import matplotlib.pyplot as plt
from cv2 import *
from typing import Union, List, Optional

from utils import os_path
from utils.label.VOCLabel import VOCBbox, VOCAnnotation


def show(image: Union[np.ndarray, list], convert_BGR: Optional[bool] = True,
         title: Optional[Union[List[str], str]] = None) -> None:
    if isinstance(image, list):
        return show_images(image, convert_BGR, title)
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


def show_images(images: List[np.ndarray], convert_BGR: bool, titles: List[str]) -> None:
    if titles is not None:
        for image, title in zip(images, titles):
            show(image, convert_BGR, title)
    else:
        for image in images:
            show(image, convert_BGR)


def to_bytes(image: np.ndarray, _format: Optional[str] = ".png"):
    success, encoded_image = cv2.imencode(_format, image)
    image_bytearr = encoded_image.tobytes()
    return image_bytearr


def read_image_from_io(imgData: str) -> np.ndarray:
    data = np.fromstring(imgData, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def save_image(image: np.ndarray, image_path: str, make_dir: Optional[bool] = True,
               convert_BGR: Optional[bool] = False) -> None:
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


def draw_box(image: np.ndarray,
             bbox: Union[list, tuple, np.ndarray, VOCBbox],
             color: Optional[tuple] = (255, 0, 0),
             label: Optional[str, int, float] = None,
             font_size: Optional[int] = 1) -> np.ndarray:
    image = np.copy(image)
    H, W = image.shape[:2]
    h_stride, w_stride = H // 60, W // 200
    bbox_label = None
    if isinstance(bbox, list) or isinstance(bbox, tuple) or isinstance(bbox, np.ndarray):
        if len(bbox) == 5:
            x1, y1, x2, y2, bbox_label = bbox
        else:
            x1, y1, x2, y2 = bbox
    elif hasattr(bbox, "name"):  # is object
        x1, y1, x2, y2, bbox_label = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2), bbox.name
    else:
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    if label:
        cv2.putText(image, str(label), (x1 + w_stride, y1 - h_stride), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, color,
                    1, cv2.LINE_4)
    elif bbox_label:
        cv2.putText(image, str(label), (x1 + w_stride, y1 - h_stride), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, color,
                    1, cv2.LINE_4)
    return image


def draw_boxes(image: np.ndarray,
               bboxes: Union[
                   list, tuple, np.ndarray, VOCBbox, List[list], List[tuple], List[np.ndarray], List[VOCBbox]],
               color: Optional[tuple] = (255, 0, 0),
               labels: Optional[str, int, float, List[str], List[int], List[float]] = None,
               font_sizes: Optional[int, List[int]] = 1) -> np.ndarray:
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


def draw_annotation(annotation: VOCAnnotation,
                    color: Optional[tuple] = (255, 0, 0),
                    image: Optional[np.ndarray] = None) -> np.ndarray:
    if image is None:
        image = imread(annotation.image_path)
    return draw_boxes(image, annotation.objects, color)


def plot_rgb_hist(image: np.ndarray, save_path: Optional[str] = None) -> None:
    # gray scale image
    if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
        plt.hist(image.ravel(), 256, [0, 256])
    else:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
    plt.show()
    if save_path:
        plt.savefig(save_path)


def plot_combined_rgb_hist(image: Union[str, np.ndarray], save_path: Optional[str] = None,
                           title: Optional[str] = None) -> None:
    if isinstance(image, str):
        image = plt.imread(image, format='uint8')
    plt.clf()

    gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])

    plt.subplot(221), plt.imshow(gray, 'gray')
    plt.subplot(222), plt.imshow(image)
    plt.subplot(223), plt.plot(hist_gray)
    plt.subplot(224)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.xlim([0, 256])

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
