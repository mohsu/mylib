# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: flag_utils.py 
@time: 2020/06/02
"""

# python packages
from absl.flags import FLAGS

# 3rd-party packages
from loguru import logger


# self-defined packages

@logger.catch(reraise=True)
def get_flag(attr, default=None):
    try:
        if hasattr(FLAGS, attr):
            return getattr(FLAGS, attr)
    except Exception:
        return default
    return default
