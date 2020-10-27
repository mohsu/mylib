# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: flag_utils.py 
@time: 2020/06/02
"""

# python packages
from absl.flags import FLAGS
from typing import Optional, Union

# 3rd-party packages
from loguru import logger


# self-defined packages

@logger.catch(reraise=True)
def get_flag(attr: str, default: Optional[str] = None):
    try:
        if hasattr(FLAGS, attr):
            return getattr(FLAGS, attr)
    except Exception:
        return default
    return default


@logger.catch(reraise=True)
def log_flag():
    logger.debug(f"Flags: {FLAGS.flags_into_string()}")
