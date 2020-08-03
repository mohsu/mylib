# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: gpu_setting.py
@time: 2019/10/21
"""

# python packages

# 3rd-party packages
from loguru import logger
import pynvml


# @logger.catch(reraise=True)
# def set_visible_gpu(num):
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(num)
#
#
# @logger.catch(reraise=True)
# class GPU:
#     def __init__(self):
#         self.gpu_settings = {}
#         self.session = None
#
#     def set(self):  # will change sometime after keras finally can set gpu growth
#         import tensorflow.compat.v1 as tf
#
#         tf.disable_v2_behavior()
#         logger.debug("Set GPU {}".format(self.gpu_settings))
#         gpu_options = tf.GPUOptions(**self.gpu_settings)
#         self.session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
#         tf.keras.backend.set_session(self.session)
#
#     def set_gpu_fraction(self, fraction):
#         self.gpu_settings["per_process_gpu_memory_fraction"] = float(fraction)
#
#     def set_allow_growth(self, allow_growth):
#         self.gpu_settings["allow_growth"] = allow_growth


class GPU:
    def __init__(self):
        import tensorflow as tf
        self.config = tf.config
        self.physical_devices = tf.config.list_physical_devices('GPU')
        self.visible_devices = self.physical_devices
        logger.debug(f"available gpus: {self.physical_devices}")
        self.auto_select_free = True

    def get_memory_info(self, idx):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return meminfo

    def check_available(self, idx, fraction=0.7):
        meminfo = self.get_memory_info(idx)
        used = meminfo.used / meminfo.total
        if used > fraction:
            return False
        else:
            return True

    def set_visible_device(self, visible_device_indexes):
        if visible_device_indexes is None:
            visible_devices = []
        else:
            if isinstance(visible_device_indexes, int):
                visible_device_indexes = [visible_device_indexes]
            visible_devices = []
            for i in visible_device_indexes:
                if self.check_available(i, 0.7):
                    visible_devices.append(self.physical_devices[i])
            if self.auto_select_free and len(visible_devices) < len(visible_device_indexes):
                logger.debug("Not enough free gpu in selection, automatically select free gpu(s).")
                free_gpu_idxes = [j for j in range(len(self.physical_devices)) if
                                  j not in visible_device_indexes and self.check_available(j)]
                visible_devices += [self.physical_devices[free_gpu_idx] for free_gpu_idx in free_gpu_idxes[:(len(visible_device_indexes) - len(visible_devices))]]

        self.config.experimental.set_visible_devices(devices=visible_devices, device_type='GPU')
        self.visible_devices = visible_devices
        logger.debug(f"Set available gpus: {visible_devices}")

    def set_allow_growth(self, allow_growth):
        logger.debug(f"Set gpu memory_growth to be : {allow_growth}")
        for gpu in self.visible_devices:
            try:
                self.config.experimental.set_memory_growth(gpu, allow_growth)
            except Exception as e:
                logger.error(e)
                # Invalid device or cannot modify virtual devices once initialized.

    def set_memory_fraction(self, memory_fraction=None, dic_memory_fraction=None):
        if dic_memory_fraction is None and memory_fraction is not None:
            dic_memory_fraction = {gpu_index: memory_fraction for gpu_index in range(len(self.visible_devices))}

        if dic_memory_fraction is not None:
            dic_memory_limit = {}
            for gpu_index in dic_memory_fraction:
                memory_info = self.get_memory_info(gpu_index)
                memory_limit = int((memory_info.total >> 20) * dic_memory_fraction[gpu_index])
                dic_memory_limit[gpu_index] = memory_limit
            self.set_memory_limit(dic_memory_limit=dic_memory_limit)

    def set_memory_limit(self, memory_limit=None, dic_memory_limit=None):
        if dic_memory_limit is not None:
            for gpu_index in dic_memory_limit:
                logger.debug(f"Set gpu {gpu_index} memory_limit to be : {dic_memory_limit[gpu_index]}")
                self.config.experimental.set_virtual_device_configuration(
                    self.visible_devices[gpu_index],
                    [self.config.experimental.VirtualDeviceConfiguration(memory_limit=int(dic_memory_limit[gpu_index]))]
                )
        elif memory_limit is not None:
            logger.debug(f"Set {self.visible_devices} memory_limit to be : {memory_limit}")
            for gpu in self.visible_devices:
                self.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [self.config.experimental.VirtualDeviceConfiguration(memory_limit=int(memory_limit))]
                )
