"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import json
import copy

import numpy as np
import torch
import pickle

from torch.utils import data
from tools.utils import NumpySeedFix, auto_init_args


class KeypointsDataset(data.Dataset):
    """
    This is a generalized class suitable
    for storing object keypoint annotations

    The input jsonfile needs to be a list of dictionaries
    (one dictionary per pose annotation) of the form:

    {
        # REQUIRED FIELDS #
        "kp_loc" : 2 x N list of keypoints
        "kp_vis" : 1 x N list of 1/0 boolean indicators
        # OPTIONAL FIELDS #
        "file_name": name of file from image_root
        "kp_loc_3d": 3 x N list of 3D GT keypoint locations in camera coords
    }

    """

    def __init__(self,
                 jsonfile=None,
                 train=True,
                 limit_to=0,
                 rand_sample=0,
                 image_root=None,
                 refresh_db=False,
                 ):

        auto_init_args(self)

        self.load_db_file()

        has_classes = 'class_mask' in self.db[0]
        if has_classes:
            self.class_db = self.get_class_db()
        else:
            self.class_db = None

    def load_db_file(self):

        print("loading data from %s" % self.jsonfile)

        ext = self.jsonfile.split('.')[-1]
        if ext == 'json':
            with open(self.jsonfile, 'r') as data_file:
                db = json.load(data_file)
        elif ext == 'pkl':
            with open(self.jsonfile, 'rb') as data_file:
                db = pickle.load(data_file)
        else:
            raise ValueError('bad extension %s' % ext)

        # the gdrive-downloaded jsons have a slightly different format:
        if 'dataset' in db:
            db = db['data']

        print("data train=%d , n frames = %d" % (self.train, len(db)))

        self.db = db

        self.restrict_images()

    def get_class_db(self):
        print('parsing class db ...')
        masks = np.stack([np.array(e['class_mask']) for e in self.db])
        unq_masks = np.unique(masks, axis=0)

        class_db = {tuple(m.tolist()): [] for m in unq_masks}
        for ei, e in enumerate(self.db):
            class_db[tuple(e['class_mask'])].append(ei)
        class_db = list(class_db.values())

        for eis in class_db:  # sanity check
            cls_array = np.stack([self.db[ei]['class_mask'] for ei in eis])
            assert ((cls_array - cls_array[0:1, :])**2).sum() <= 1e-6

        return class_db

    def restrict_images(self):

        if self.limit_to > 0:
            tgtnum = min(self.limit_to, len(self.db))
            with NumpySeedFix():
                prm = np.random.permutation(
                    len(self.db))[0:tgtnum]
            print("limitting dataset to %d samples" % tgtnum)
            self.db = [self.db[i] for i in prm]

    def __len__(self):
        if self.rand_sample > 0:
            return self.rand_sample
        else:
            return len(self.db)

    def __getitem__(self, index):

        if self.rand_sample > 0:
            if self.class_db is not None:
                # in case we have classes, sample first rand class
                # and then image index
                cls_index = np.random.randint(len(self.class_db))
                index = np.random.choice(self.class_db[cls_index])
            else:
                index = np.random.randint(len(self.db))

        entry = copy.deepcopy(self.db[index])

        # convert to torch Tensors where possible
        for fld in ('kp_loc', 'kp_vis', 'kp_loc_3d',
                    'class_mask', 'kp_defined'):
            if fld in entry:
                entry[fld] = torch.FloatTensor(entry[fld])

        if self.image_root is not None and 'image_path' in entry:
            entry['image_path'] = os.path.join(
                self.image_root, entry['image_path'])
        else:
            entry['image_path'] = '<NONE>'

        if 'p3d_info' in entry:  # filter the kp out of bbox
            bbox = torch.FloatTensor(entry['p3d_info']['bbox'])
            bbox_vis, bbox_err = bbox_kp_visibility(
                bbox, entry['kp_loc'], entry['kp_vis'])
            entry['kp_vis'] = entry['kp_vis'] * bbox_vis.float()

        # mask out invisible
        entry['kp_loc'] = entry['kp_loc'] * entry['kp_vis'][None]

        return entry


def bbox_kp_visibility(bbox, keypoints, vis):
    bx, by, bw, bh = bbox
    x = keypoints[0]
    y = keypoints[1]
    ctx_ = 0.1
    in_box = (x >= bx-ctx_*bw) * (x <= bx+bw*(1+ctx_)) * \
        (y >= by-ctx_*bh) * (y <= by+bh*(1+ctx_))

    in_box = in_box * (vis == 1)

    err = torch.stack([(bx-ctx_*bw)-x,
                       x-(bx+bw*(1+ctx_)),
                       (by-ctx_*bh)-y,
                       y-(by+bh*(1+ctx_))])
    err = torch.relu(err) * vis[None].float()
    err = torch.stack((torch.max(err[0], err[1]),
                       torch.max(err[2], err[3]))).max(dim=1)[0]

    return in_box, err
