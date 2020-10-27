# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: self_defined_class.py 
@time: 2020/02/20
"""

# python packages
from enum import EnumMeta, Flag
from typing import TypeVar


# 3rd-party


# self-defined packages
class ReversibleEnumMeta(EnumMeta):
    def __getitem__(cls, item):
        # try if it is an integer
        try:
            int(item)
            for name, pair in cls._member_map_.items():
                if int(item) == int(pair):
                    return name
        except:
            pass

        # if name matches --> get value
        if item in cls._member_map_:
            return cls._member_map_[item].value
        # case insensitive try
        for name, pair in cls._member_map_.items():
            if str(name).lower() == str(item).lower():
                return pair.value

        # if string matches --> return get value
        for name, pair in cls._member_map_.items():
            if str(pair).lower() == str(item).lower():
                return pair.value

        # if value matches --> (reverse) return name
        for name, pair in cls._member_map_.items():
            if str(pair.value).lower() == str(item).lower():
                return name

        raise KeyError(item)

    def get_name(cls, item):
        # if value matches --> return name
        for name, pair in cls._member_map_.items():
            if str(pair.value).lower() == str(item).lower():
                return name

        # if string matches --> return name
        for name, pair in cls._member_map_.items():
            if str(pair).lower() == str(item).lower():
                return name

        # if pair matches --> return name
        for name, pair in cls._member_map_.items():
            if isinstance(pair, item) and pair == str(item).lower():
                return name
        raise KeyError(item)

    def get_pair(cls, item):
        if item in cls._member_map_:
            return cls._member_map_[item]
        # if name matches
        for name, pair in cls._member_map_.items():
            if str(name).lower() == str(item).lower():
                return pair
            if str(pair.value).lower() == str(item).lower():
                return pair
            if str(pair).lower() == str(item).lower():
                return pair

        raise KeyError(item)

    def __len__(cls):
        return len(cls._member_map_.keys())

    def items(cls):
        return cls._member_map_.items()

    def keys(cls):
        return [name for name_str, name in cls.items()]

    def names(cls):
        return cls._member_map_.keys()

    def values(cls):
        return [item.value for item in cls._member_map_.values()]


class BoolFlag(Flag, metaclass=ReversibleEnumMeta):
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, type(self)):
            return self != other
        return self.value != other

    def __str__(self):
        return self.name

    def __hash__(self):
        return id(self)

    @classmethod
    def get_pair(cls, item):
        return ReversibleEnumMeta.get_pair(cls, bool(item))


class CombinedEnumElement(list):
    def __str__(self):
        return "".join([str(x) for x in self])


type_myclass = TypeVar('type_myclass', ReversibleEnumMeta, BoolFlag)

# class CombinedReversibleEnum:
#     def __init__(self, *enums, list_overlap=None):
#         self.len_enum = len(enums)
#         self.enums = enums
#
#         # do permutation
#         list_name, list_pair, list_pair_value = self.enum_permutation(enums, list_overlap)
#         self._names = list_name
#         self._keys = list_pair
#         self._values = list_pair_value
#
#         # set attributes
#         for i, _name in enumerate(self._names):
#             self.__dict__["".join(_name)] = CombinedEnumElement(self._keys[i])
#
#     def __len__(self):
#         return len(self._names)
#
#     def __getitem__(self, item):
#         if isinstance(item, tuple):
#             item = list(item)
#
#         if "," in item and item in self._values:
#             item = item.split(",")
#
#         items = item[:]
#         mapped_items = []
#         for i, item in enumerate(items):
#             enum_type = self.enums[i]
#             mapped_items.append(enum_type[item])
#         return mapped_items
#
#     def get_pair(self, item):
#         if isinstance(item, tuple):
#             item = list(item)
#
#         if "," in item:
#             item = item.split(",")
#
#         items = item[:]
#         mapped_items = []
#
#         for i, item in enumerate(items):
#             enum_type = self.enums[i]
#             mapped_items.append(enum_type.get_pair(item))
#
#         return mapped_items
#
#     @staticmethod
#     def enum_permutation(enums, list_overlap=None):
#         list_name, list_pair, list_pair_value, \
#         fixed_list_name, fixed_list_pair, fixed_list_pair_value = [], [], [], [], [], []
#         if list_overlap:
#             list_overlap = [x.capitalize() for x in list_overlap]
#         for i, enum1 in enumerate(enums):
#             if i != 0:
#                 fixed_list_name, fixed_list_pair, fixed_list_pair_value = list_name[:], list_pair[:], list_pair_value[:]
#             list_name, list_pair, list_pair_value = [], [], []
#             if i == 0:
#                 for name, pair in enum1.items():
#                     list_name.append([name])
#                     list_pair.append([pair])
#                     list_pair_value.append(str(pair.value))
#             else:
#                 for name, pair in enum1.items():
#                     for j, (k, p) in enumerate(zip(fixed_list_name, fixed_list_pair)):
#                         if list_overlap and ((name not in list_overlap and any(
#                                 overlap_name in list(flatten(k)) for overlap_name in list_overlap)) or (
#                                                      name in list_overlap and name != list(flatten(k))[0])):
#                             continue
#                         list_name.append(k[:] + [name])
#                         list_pair.append(p[:] + [pair])
#                         list_pair_value.append(f"{fixed_list_pair_value[j]},{pair.value}")
#         return list_name, list_pair, list_pair_value
