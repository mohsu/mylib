# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: gpu_setting.py
@time: 2019/10/21
"""

# python packages
from typing import Any, List, Optional, Dict, Union

# 3rd-party packages
from loguru import logger
import pynvml


class GPU:
    def __init__(self):
        import tensorflow as tf
        self.config = tf.config
        self.physical_devices = tf.config.list_physical_devices('GPU')
        logger.debug(f"physical_devices gpus: {self.physical_devices}")
        self.visible_devices = [self.physical_devices[i] for i in range(len(self.physical_devices)) if
                                self.check_available(i)]
        logger.debug(f"available gpus: {self.visible_devices}")
        self.auto_select_free = True

    def get_memory_info(self, idx: int) -> Any:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return meminfo

    def check_available(self, idx: int, fraction=0.7) -> bool:
        meminfo = self.get_memory_info(idx)
        used = meminfo.used / meminfo.total
        if used > fraction:
            return False
        else:
            return True

    def set_visible_device(self, visible_device_indexes: Optional[Union[int, List[int]]]) -> None:
        if isinstance(visible_device_indexes, int):
            visible_device_indexes = [visible_device_indexes]
        visible_devices = []
        if visible_device_indexes or visible_device_indexes[0] < 0:
            for i in visible_device_indexes:
                if self.physical_devices[i] in self.visible_devices:
                    visible_devices.append(self.physical_devices[i])
            if self.auto_select_free and len(visible_devices) < len(visible_device_indexes):
                logger.debug("Not enough free gpu in selection, automatically select free gpu(s).")
                free_gpu_idxes = [j for j in range(len(self.physical_devices)) if
                                  j not in visible_device_indexes and self.check_available(j)]
                visible_devices += [self.physical_devices[free_gpu_idx] for free_gpu_idx in
                                    free_gpu_idxes[:(len(visible_device_indexes) - len(visible_devices))]]

        self.config.experimental.set_visible_devices(devices=visible_devices, device_type='GPU')
        self.visible_devices = visible_devices
        logger.debug(f"Set available gpus: {visible_devices}")

    def set_allow_growth(self, allow_growth: bool) -> None:
        logger.debug(f"Set gpu memory_growth to be : {allow_growth}")
        for gpu in self.visible_devices:
            try:
                self.config.experimental.set_memory_growth(gpu, allow_growth)
            except Exception as e:
                logger.error(e)
                # Invalid device or cannot modify virtual devices once initialized.

    def set_memory_fraction(self,
                            memory_fraction: Optional[float] = None,
                            dic_memory_fraction: Optional[Dict[int, float]] = None) -> None:
        if dic_memory_fraction is None and memory_fraction is not None:
            dic_memory_fraction = {gpu_index: memory_fraction for gpu_index in range(len(self.visible_devices))}

        if dic_memory_fraction is not None:
            dic_memory_limit = {}
            for gpu_index in dic_memory_fraction:
                memory_info = self.get_memory_info(gpu_index)
                memory_limit = int((memory_info.total >> 20) * dic_memory_fraction[gpu_index])
                dic_memory_limit[gpu_index] = memory_limit
            self.set_memory_limit(dic_memory_limit=dic_memory_limit)

    def set_memory_limit(self,
                         memory_limit: Optional[int] = None,
                         dic_memory_limit: Optional[Dict[int, int]] = None) -> None:
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
