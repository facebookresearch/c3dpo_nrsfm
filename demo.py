"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import torch
from PIL import Image

from dataset.dataset_configs import STICKS
from experiment import init_model_from_dir
from tools.model_io import download_model
from tools.utils import get_net_input

from tools.vis_utils import show_projections
from visuals.rotating_shape_video import rotating_3d_video


def run_demo():

    net_input = get_net_input(get_test_h36m_sample())

    model_dir = download_model('h36m')

    model, _ = init_model_from_dir(model_dir)
    model.eval()

    preds = model(**net_input)

    # input keypoints
    kp_loc = net_input['kp_loc'][0]

    # predicted 3d keypoints in camera coords
    kp_pred_3d = preds['shape_image_coord'][0]

    sticks = STICKS['h36m']

    # viz = get_visdom_connection()
    im_proj = show_projections(
        kp_loc[None].detach().cpu().numpy(),
        visdom_env='demo_h36m',
        visdom_win='input_keypoints',
        title='input_keypoints',
        cmap__='rainbow',
        markersize=40,
        sticks=sticks,
        stickwidth=2,
    )

    im_proj = Image.fromarray(im_proj)
    im_proj_path = os.path.join(model_dir, 'demo_projections.png')
    print('Saving input keypoints to %s' % im_proj_path)
    im_proj.save(im_proj_path)

    video_path = os.path.join(model_dir, 'demo_shape.mp4')
    rotating_3d_video(kp_pred_3d.detach().cpu(),
                      video_path=video_path,
                      sticks=sticks,
                      title='rotating 3d',
                      cmap='rainbow',
                      visdom_env='demo_h36m',
                      visdom_win='3d_shape',
                      get_frames=7, )


def get_test_h36m_sample():

    kp_loc = \
        [[0.0000,  0.2296,  0.1577,  0.1479, -0.2335, -0.1450,  0.0276,
          0.0090,  0.0065, -0.0022,  0.0566, -0.3193, -0.4960, -0.4642,
          0.3650,  0.8939,  1.3002],
         [0.0000, -0.0311,  0.8875,  1.8011,  0.0319,  0.9565,  1.8620,
         -0.5053, -1.0108, -1.2185, -1.4179, -0.9106, -0.3406,  0.1310,
         -0.9744, -0.7978, -0.8496]]

    kp_vis = [1., 1., 1., 1., 1., 1., 1., 1.,
              1., 1., 1., 1., 1., 1., 1., 1., 1.]

    kp_loc, kp_vis = [torch.FloatTensor(a) for a in (kp_loc, kp_vis)]

    return {'kp_loc': kp_loc[None], 'kp_vis': kp_vis[None]}


if __name__ == '__main__':
    run_demo()
