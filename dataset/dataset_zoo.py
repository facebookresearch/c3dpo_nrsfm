"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import copy
from dataset.dataset_configs import IMAGE_ROOTS, DATASET_ROOT
from dataset.keypoints_dataset import KeypointsDataset


def dataset_zoo(dataset_name='h36m',
                sets_to_load=('train', 'val'),
                force_download=False,
                TRAIN={'rand_sample': -1,
                       'limit_to': -1},
                VAL={'rand_sample': -1,
                     'limit_to': -1},
                TEST={'rand_sample': -1,
                      'limit_to': -1},
                **kwargs):

    assert dataset_name in ['h36m', 'h36m_hourglass',
                            'pascal3d', 'pascal3d_hrnet', 'up3d_79kp',
                            'cub_birds', 'cub_birds_hrnet']

    main_root = DATASET_ROOT
    json_train = os.path.join(main_root, dataset_name + '_train.json')
    if dataset_name == 'up3d_79kp':
        # for up3d we eval on test set ...
        json_val = os.path.join(main_root, dataset_name + '_test.json')
    else:
        json_val = os.path.join(main_root, dataset_name + '_val.json')

    image_roots = copy.deepcopy(IMAGE_ROOTS)
    image_roots = image_roots[dataset_name] \
        if dataset_name in image_roots else None
    if image_roots is not None:
        if len(image_roots) == 2:
            image_root_train, image_root_val = image_roots
        elif len(image_roots) == 1:
            image_root_train = image_root_val = image_roots[0]
        else:
            raise ValueError('cant be')
    else:
        image_root_train = image_root_val = None

    # auto-download dataset file if doesnt exist
    for json_file in (json_train, json_val):
        if not os.path.isfile(json_file) or force_download:
            download_dataset_json(json_file)

    dataset_train = None
    dataset_val = None
    dataset_test = None
    if 'train' in sets_to_load:
        dataset_train = KeypointsDataset(
            image_root=image_root_train,
            jsonfile=json_train, train=True, **TRAIN)
    if 'val' in sets_to_load:
        dataset_val = KeypointsDataset(
            image_root=image_root_val,
            jsonfile=json_val, train=False, **VAL)
    dataset_test = dataset_val

    return dataset_train, dataset_val, dataset_test


def download_dataset_json(json_file):
    import urllib.request
    import json
    from dataset.dataset_configs import DATASET_URL, DATASET_MD5
    from tools.utils import md5

    json_dir = '/'.join(json_file.split('/')[0:-1])
    json_name = json_file.split('/')[-1].split('.')[0]
    os.makedirs(json_dir, exist_ok=True)

    url = DATASET_URL[json_name]
    print('downloading dataset json %s from %s' % (json_name, url))

    try:
        urllib.request.urlretrieve(url, json_file)
    except:
        if os.path.isfile(json_file):
            os.remove(json_file)

    print('checking dataset %s' % json_name)
    assert md5(json_file) == DATASET_MD5[json_name], 'bad md5!'
    with open(json_file, 'r') as f:
        dt = json.load(f)
    assert dt['dataset'] == json_name
