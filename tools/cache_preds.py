"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch._six import container_abcs, string_classes, int_classes
import re
import time
import sys
import torch

from tqdm import tqdm

from tools.utils import has_method, get_net_input


def cache_preds(model, loader, cache_vars=None, stats=None, n_extract=None):

    print("caching model predictions: %s" % str(cache_vars))

    model.eval()

    trainmode = 'test'

    t_start = time.time()

    cached_preds = []

    cache_size = 0.  # in GB ... counts only cached tensor sizes

    n_batches = len(loader)
    if n_extract is not None:
        n_batches = n_extract

    with tqdm(total=n_batches, file=sys.stdout) as pbar:
        for it, batch in enumerate(loader):

            last_iter = it == n_batches-1

            # move to gpu and cast to Var
            net_input = get_net_input(batch)

            with torch.no_grad():
                preds = model(**net_input)

            assert not any(k in preds for k in net_input.keys())
            preds.update(net_input)  # merge everything into one big dict

            if stats is not None:
                stats.update(preds, time_start=t_start, stat_set=trainmode)
                assert stats.it[trainmode] == it, \
                    "inconsistent stat iteration number!"

            # restrict the variables to cache
            if cache_vars is not None:
                preds = {k: preds[k] for k in cache_vars if k in preds}

            # ... gather and log the size of the cache
            preds, preds_size = gather_all(preds)
            cache_size += preds_size

            cached_preds.append(preds)

            pbar.set_postfix(cache_size="%1.2f GB" % cache_size)
            pbar.update(1)

            if last_iter and n_extract is not None:
                break

    cached_preds_cat = concatenate_cache(cached_preds)

    return cached_preds_cat


def gather_all(preds):
    cache_size = 0
    for k in preds:
        if has_method(preds[k], 'cuda'):
            preds[k] = preds[k].data.cpu()
            cache_size += preds[k].numpy().nbytes / 1e9
        elif type(preds[k]) == dict:
            preds[k], size_now = gather_all(preds[k])
            cache_size += size_now
    return preds, cache_size


# cache concatenation - largely taken from pytorch default_collate()
np_str_obj_array_pattern = re.compile(r'[SaUO]')
error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def concatenate_cache(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        return torch.cat(batch, 0, out=out)  # the main difference is here
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))
            return concatenate_cache([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: concatenate_cache([d[key] for d in batch])
                for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):
        return type(batch[0])(*(concatenate_cache(samples)
                                for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):  # also some diffs here
        # just unpack
        return [s_ for s in batch for s_ in s]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))
