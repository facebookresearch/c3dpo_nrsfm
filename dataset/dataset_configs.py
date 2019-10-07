"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

EXP_ROOT = './data/exps/c3dpo/'
DATASET_ROOT = './data/datasets/c3dpo/'

DATASET_URL = {
    'pascal3d_val':          'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/pascal3d_val.json',
    'pascal3d_train':        'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/pascal3d_train.json',
    'pascal3d_hrnet_val':    'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/pascal3d_hrnet_val.json',
    'pascal3d_hrnet_train':  'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/pascal3d_hrnet_train.json',
    'h36m_val':              'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/h36m_val.json',
    'h36m_train':            'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/h36m_train.json',
    'h36m_hourglass_val':    'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/h36m_hourglass_val.json',
    'h36m_hourglass_train':  'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/h36m_hourglass_train.json',
    'cub_birds_val':         'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/cub_birds_val.json',
    'cub_birds_train':       'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/cub_birds_train.json',
    'cub_birds_hrnet_val':   'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/cub_birds_hrnet_val.json',
    'cub_birds_hrnet_train': 'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/cub_birds_hrnet_train.json',
    'up3d_79kp_train':       'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/up3d_79kp_train.json',
    'up3d_79kp_val':         'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/up3d_79kp_val.json',
    'up3d_79kp_test':        'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/up3d_79kp_test.json',
}

DATASET_MD5 = {
    "h36m_train": "454e2aee4cad761499265f858fe2e0ff",
    "h36m_val": "d2347fc651e7f704ce3a4da880852fff",
    "h36m_hourglass_train": "d2ffcaf4ce9e49712a65e2b1932814a3",
    "h36m_hourglass_val": "9996a703cb3b24da3b5563baa09da2bd",
    "pascal3d_hrnet_train": "c145b879e7462f8942a258f7c6dcbee4",
    "pascal3d_hrnet_val": "5cb55986b1c19253f0b8213e47688443",
    "pascal3d_train": "a78048a101ef56bc371b01f66c19178b",
    "pascal3d_val": "0128817c43eaa1eff268d5295700c8ad",
    "up3d_79kp_train": "fde2aee038ecd0f145181559eff59c9f",
    "up3d_79kp_test": "7d8bf3405ec085394e9257440e8bcb18",
}

MODEL_URL = {
    'pascal3d':        'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/c3dpo_pretrained/pretrained_pascal3d/model_epoch_00000000.pth',
    'pascal3d_hrnet':  'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/c3dpo_pretrained/pretrained_pascal3d_hrnet/model_epoch_00000000.pth',
    'h36m':            'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/c3dpo_pretrained/pretrained_h36m/model_epoch_00000000.pth',
    'h36m_hourglass':  'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/c3dpo_pretrained/pretrained_h36m_hourglass/model_epoch_00000000.pth',
    # 'cub_birds':       '', TODO(dnovotny)
    'up3d_79kp':       'https://dl.fbaipublicfiles.com/c3dpo_nrsfm/c3dpo_pretrained/pretrained_up3d_79kp/model_epoch_00000000.pth',
}

MODEL_MD5 = {
    "h36m": "280bce4d1074e1140a0cc23806bcf8cf",
    "h36m_hourglass": "4dd849bf6d3b0b6e5d93afbed9cad187",
    "pascal3d_hrnet": "464163c58f2827b45def014135870844",
    "pascal3d": "464163c58f2827b45def014135870844",
    "up3d_79kp": "2de88ac68f0fbb0763dcbce039d74610",
}

# list of root folders containing the dataset images
IMAGE_ROOTS = {}


# ----- connectivity patterns for visualizing the stick-men
STICKS = {
    'pose_track': [[2, 0], [0, 1], [1, 5], [5, 7],
                   [9, 7], [1, 6], [6, 8], [10, 8],
                   [1, 12], [12, 11], [11, 1], [14, 12],
                   [11, 13], [15, 13], [16, 14]],
    'h36m': [[10, 9], [9, 8], [8, 14],
             [14, 15], [15, 16], [8, 11],
             [11, 12], [12, 13], [8, 7],
             [7, 0], [1, 0], [1, 2],
             [2, 3], [0, 4], [4, 5], [5, 6]],
    'cub_birds': [[1, 5], [5, 4], [4, 9],
                  [9, 0], [0, 13], [0, 12],
                  [0, 8], [12, 13], [1, 14],
                  [14, 3], [3, 2], [2, 7],
                  [1, 10], [1, 6], [2, 11],
                  [2, 7], [8, 13]],
    'coco': [[13, 15], [14, 16], [12, 14], [11, 12, ], [11, 13],
             [0, 12], [0, 11], [8, 10], [6, 8],
             [7, 9], [5, 7], [0, 5], [0, 6],
             [0, 3], [0, 4], [0, 2], [0, 1]],
    'freicars': [[0, 8], [0, 4], [4, 10], [8, 10],
                 [10, 9], [9, 11], [8, 11],
                 [11, 6], [9, 2], [2, 6],
                 [4, 1], [5, 1], [0, 5], [5, 7], [1, 3],
                 [7, 3], [3, 2], [7, 6]],
    'pascal3d': {
        'car': [[0, 8], [0, 4], [4, 10], [8, 10],
                [10, 9], [9, 11], [8, 11],
                [11, 6], [9, 2], [2, 6],
                [4, 1], [5, 1], [0, 5], [5, 7], [1, 3],
                [7, 3], [3, 2], [7, 6]],
        'aeroplane': [[2, 5], [1, 4], [5, 3], [3, 7],
                      [7, 0], [0, 5], [5, 7], [5, 6],
                      [6, 0], [6, 3], [2, 4], [2, 1]],
        'motorbike': [[6, 2],
                      [2, 9],
                      [2, 3],
                      [3, 8],
                      [5, 8],
                      [3, 5],
                      [2, 1],
                      [1, 0],
                      [0, 7],
                      [0, 4],
                      [4, 7],
                      [1, 4],
                      [1, 7],
                      [1, 5],
                      [1, 8]],
        'sofa':    [[1, 5],
                    [5, 4],
                    [4, 6],
                    [6, 2],
                    [2, 0],
                    [1, 0],
                    [0, 4],
                    [1, 3],
                    [7, 5],
                    [2, 3],
                    [3, 7],
                    [9, 7],
                    [7, 6],
                    [6, 8],
                    [8, 9]],
        'chair': [[7, 3],
                  [6, 2],
                  [9, 5],
                  [8, 4],
                  [7, 9],
                  [8, 6],
                  [6, 7],
                  [9, 8],
                  [9, 1],
                  [8, 0],
                  [1, 0]],
    },
}
STICKS['cub_birds_hrnet'] = STICKS['cub_birds']

H36M_ACTIONS = ['Directions', 'Discussion', 'Eating', 'Greeting',
                'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting',
                'SittingDown', 'Smoking', 'Waiting', 'WalkDog',
                'Walking', 'WalkTogether']

P3D_NUM_KEYPOINTS = {
    'aeroplane': 8,
    'car': 12,
    'tvmonitor': 8,
    'sofa': 10,
    'motorbike': 10,
    'diningtable': 12,
    'chair': 10,
    'bus': 12,
    'bottle': 7,
    'boat': 7,
    'bicycle': 11,
    'train': 17}

P3D_CLASSES = list(P3D_NUM_KEYPOINTS.keys())

P3D_NUM_IMAGES = {
    'train': {"aeroplane": 1953, "car": 5627,
              "tvmonitor": 1374, "sofa":  669,
              "motorbike":  725, "diningtable":  751,
              "chair": 1186, "bus": 1185,
              "bottle": 1601, "boat": 2046,
              "bicycle":  904, "train": 1113, },
    'val': {"aeroplane":  269, "car":  294,
            "tvmonitor":  206, "sofa":   37,
            "motorbike":  116, "diningtable": 12,
            "chair":  227, "bus":  153,
            "bottle":  249, "boat":  163,
            "bicycle":  115, "train":  109}}
