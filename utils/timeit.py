# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: timeit.py 
@time: 2020/06/24
"""

# python packages
import time
# 3rd-party packages
from loguru import logger


# self-defined packages

class Timer:
    def __init__(self, name=None):
        self.name = name
        self.start_time = time.time()
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        logger.debug(
            f"{self.name + ' ' if self.name is not None else ''} Time elapse {self.end_time - self.start_time}s")


def time_function(func):
    def wrap(*args, **kwargs):
        timer = Timer(f"{func.__name__}")
        func(*args, **kwargs)
        timer.stop()
    return wrap
