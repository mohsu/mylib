# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: misc.py 
@time: 2019/11/22
"""

# python packages

# 3rd-party packages
from loguru import logger


# self-defined packages

# global variables

# set up log & config
# logger.add(log_path, rotation="5 MB", retention="1 week") # filter=lambda record: record["extra"].get("name") == ""
@logger.catch(reraise=True)
class LabelDict(dict):
    def __init__(self, dic):
        super().__init__(dic)

    def __getitem__(self, item):
        for key, value in self.items():
            if key == item:
                return value
        return -1

    def get_key_from_value(self, value):
        for key in self.keys():
            if self[key] == value:
                return key
        return None
