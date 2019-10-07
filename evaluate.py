"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from dataset.dataset_zoo import dataset_zoo
from dataset.eval_zoo import eval_zoo
from experiment import init_model_from_dir
from tools.model_io import download_model
from tools.cache_preds import cache_preds
from tabulate import tabulate


def eval_model(dataset_name):

    model_dir = download_model(dataset_name, force_download=False)

    model, _ = init_model_from_dir(model_dir)
    model.eval()

    _, _, dataset_test = dataset_zoo(
        dataset_name=dataset_name, sets_to_load=('val',),
        force_download=False)

    loader_test = torch.utils.data.DataLoader(dataset_test,
                                              num_workers=8, pin_memory=True,
                                              batch_size=1024, shuffle=False)

    eval_script, cache_vars, eval_vars = eval_zoo(dataset_name)

    cached_preds = cache_preds(model, loader_test, cache_vars=cache_vars)
    results, _ = eval_script(cached_preds, eval_vars=eval_vars)

    return results


if __name__ == '__main__':

    results = {}

    for dataset in ('h36m', 'h36m_hourglass', 'pascal3d_hrnet',
                    'pascal3d', 'up3d_79kp'):
        results[dataset] = eval_model(dataset)

    print('\n\nRESULTS:')
    tab_rows = []
    for dataset, result in results.items():
        tab_row = [dataset]
        tab_row.extend([result[m] for m in ('EVAL_MPJPE_best', 'EVAL_stress')])
        tab_rows.append(tab_row)

    print(tabulate(tab_rows, headers=['dataset', 'MPJPE', 'Stress']))

    # RESULTS:
    # dataset               MPJPE      Stress
    # --------------  -----------  ----------
    # h36m             95.6338     41.5864
    # h36m_hourglass  145.021      84.693
    # pascal3d_hrnet   56.8909     40.1775
    # pascal3d         36.6413     31.0768
    # up3d_79kp         0.0672771   0.0406902
