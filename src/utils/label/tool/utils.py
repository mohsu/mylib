# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: utils.py 
@time: 2020/05/20
"""

# python packages
import random

# 3rd-party packages
from loguru import logger


# self-defined packages


def is_int(key):
    try:
        int(key)
        return True
    except:
        return False


def get_rect_kargs():
    return {"fill": False,
            "facecolor": (random.random(), random.random(), random.random(), 0.3),
            "linewidth": 1.3, "edgecolor": (0, 0, 0, 0.9)}


def get_coords(rect, type='int'):
    x1, y1 = rect.get_xy()
    x2 = rect.get_width() + x1
    y2 = rect.get_height() + y1
    if isinstance(x1, int):
        return x1, y1, x2, y2
    return x1.astype(type), y1.astype(type), x2.astype(type), y2.astype(type)
