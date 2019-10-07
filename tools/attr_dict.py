"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


def nested_attr_dict(dct):
    if type(dct) in (dict, AttrDict):
        dct = AttrDict(dct)
        for k, v in dct.items():
            dct[k] = nested_attr_dict(v)
    return dct


class AttrDict(dict):

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value
