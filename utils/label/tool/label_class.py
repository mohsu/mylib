# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: label_class.py 
@time: 2020/05/20
"""

# python packages
from enum import Enum, Flag

# 3rd-party packages
from loguru import logger

# self-defined packages
from com.enum_card import EnumCardSuit, EnumCardNumber


class KeyboardAction(Enum):
    INTO_LABEL_MODE = 'escape'
    EXIT_LABEL_MODE = 'escape'
    SAVE = 'ctrl+s'
    LAST_STEP = 'ctrl+z'
    DEL_LABEL = 'delete'
    LAST = 'left'
    NEXT = 'right'
    BACKSPACE_LABEL_NAME = 'backspace'
    SURE = 'y'
    NOT_SURE = 'n'
    DEFAULT_LABEL = 'f1'
    DEFAULT_EXTRA_2 = 'f2'
    SWITCH_LOAD_FROM_MODEL = 'f3'
    REFRESH = 'f12'
    DEL_ALL_LABEL_NAME = '-'


class LabelMode(Flag):
    LABEL_MODE = True
    NON_LABEL_MODE = False

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other


class CardName:
    def __init__(self, *args):
        self.number = None
        self.suit = None
        if args:
            for arg in args:
                self.insert(arg)

    def __str__(self):
        name = ''
        if self.number is not None:
            name += str(self.number)
        if self.suit is not None:
            name += str(self.suit)
        return name

    def insert(self, new_key):
        if self.number is None:
            try:
                self.number = EnumCardNumber.get_pair(new_key)
            except:
                pass
            return
        elif self.suit is None:
            try:
                self.suit = EnumCardSuit.get_pair(new_key)
            except:
                pass

    def backspace(self, t=1):
        for _ in range(t):
            if self.suit is not None:
                self.suit = None
            elif self.number is not None:
                self.number = None

    def export(self):
        return [self.number.name, self.suit.name]
