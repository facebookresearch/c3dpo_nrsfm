"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import inspect
import copy
import os
import ast
import yaml
import numpy as np


def convert_to_stringval(cfg_, squeeze=None, stringify_vals=False):
    out = {}
    convert_to_stringval_rec([('ROOT', cfg_)], out,
                             squeeze=squeeze, stringify_vals=stringify_vals)
    return out


def convert_to_stringval_rec(flds, output, squeeze=None, stringify_vals=False):
    for k, v in flds[-1][1].items():
        if isinstance(v, dict):
            flds_cp = copy.deepcopy(flds)
            flds_cp.append((k, v))
            convert_to_stringval_rec(flds_cp, output,
                                     squeeze=squeeze,
                                     stringify_vals=stringify_vals)
        else:
            valname_full = []
            for f in flds[1:]:
                valname_full.append(squeeze_string(f[0], squeeze))
            valname_full.append(squeeze_string(k, squeeze))
            valname_full = ".".join(valname_full)

            if stringify_vals:
                output[valname_full] = str(v)
            else:
                output[valname_full] = v


def squeeze_key_string(f, squeeze_inter, squeeze_tail):

    keys = f.split('.')
    tail = keys[-1]
    inter = keys[0:-1]

    nkeys = len(keys)

    if nkeys > 1:
        take_from_each = int(
            np.floor(float(squeeze_inter-nkeys)/float(nkeys-1)))
        take_from_each = max(take_from_each, 1)
        for keyi in range(nkeys-1):
            s = inter[keyi]
            s = s[0:min(take_from_each, len(s))]
            inter[keyi] = s

    tail = squeeze_string(tail, squeeze_tail)
    inter.append(tail)
    out = ".".join(inter)

    return out


def squeeze_string(f, squeeze):
    if squeeze is None or squeeze > len(f):
        return f
    idx = np.round(np.linspace(0, len(f)-1, squeeze))
    idx = idx.astype(int).tolist()
    f_short = [f[i] for i in idx]
    f_short = str("").join(f_short)
    return f_short


def get_default_args(C):
    # returns dict of keyword args of a callable C
    sig = inspect.signature(C)

    kwargs = {}

    for pname, defval in dict(sig.parameters).items():

        if defval.default == inspect.Parameter.empty:
            # print('skipping %s' % pname)
            continue
        else:
            kwargs[pname] = defval.default

    return kwargs


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


def get_arg_parser(cfg_constructor):

    dargs = get_default_args(cfg_constructor)
    dargs_full_name = convert_to_stringval(dargs, stringify_vals=False)

    parser = argparse.ArgumentParser(
        description='Auto-initialized argument parser'
    )

    for darg, val in dargs_full_name.items():
        tp = type(val) if val is not None else str
        if tp == bool:
            parser.add_argument(
                '--%s' % darg,
                dest=darg,
                help=darg,
                default=val,
                type=str2bool,
            )
        elif tp == list:
            parser.add_argument(
                '--%s' % darg,
                type=arg_as_list,
                default=val,
                help=darg)
        else:
            parser.add_argument(
                '--%s' % darg,
                dest=darg,
                help=darg,
                default=val,
                type=tp,
            )

    return parser


def set_config_from_config(cfg, cfg_set):
    # cfg_set ... dict with nested options
    cfg_dot_separated = convert_to_stringval(cfg_set, stringify_vals=False)
    set_config(cfg, cfg_dot_separated)


def set_config_rec(cfg, tgt_key, val, check_only=False):
    if len(tgt_key) > 1:
        k = tgt_key.pop(0)
        if k not in cfg:
            raise ValueError('no such config key %s' % k)
        set_config_rec(cfg[k], tgt_key, val, check_only=check_only)
    else:
        if check_only:
            assert cfg[tgt_key[0]] == val
        else:
            cfg[tgt_key[0]] = val


def set_config(cfg, cfg_set):
    # cfg_set ... dict with .-separated options

    for cfg_key, cfg_val in cfg_set.items():
        # print('setting %s = %s' % (cfg_key,str(cfg_val)) )
        cfg_key_split = [k for k in cfg_key.split('.') if len(k) > 0]
        set_config_rec(cfg, copy.deepcopy(cfg_key_split), cfg_val)
        set_config_rec(cfg, cfg_key_split, cfg_val, check_only=True)


def set_config_from_file(cfg, cfg_filename):
    # set config from yaml file
    with open(cfg_filename, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    set_config_from_config(cfg, yaml_cfg)


def dump_config(cfg):
    cfg_filename = os.path.join(cfg.exp_dir, 'expconfig.yaml')
    with open(cfg_filename, 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)
