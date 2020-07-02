# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: timeit.py 
@time: 2020/06/24
"""

# python packages
import timeit

# 3rd-party packages
from loguru import logger


# self-defined packages

class Timer:
    def __init__(self, name=None):
        self.name = name
        self.start_time = timeit.default_timer()
        self.end_time = None

    def start(self):
        self.start_time = timeit.default_timer()

    def stop(self):
        self.end_time = timeit.default_timer()
        logger.debug(
            f"{self.name + ' ' if self.name is not None else ''} Time elapse {self.end_time - self.start_time}s")
