"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from tools.video_writer import VideoWriter
from tools.vis_utils import matplot_plot_point_cloud, get_visdom_connection
from tools.so3 import so3_exponential_map


def rotating_3d_video(
    shape,
    video_path='/tmp/video.mp4',
    fps=10,
    vlen=4,
    sticks=None,
    title='rotating 3d',
    cmap='rainbow',
    visdom_env=None,
    visdom_win=None,
    get_frames=0,
):

    # center
    mean = shape.sum(1) / shape.shape[1]
    shape = shape - mean[:, None]

    lim = float(torch.topk(shape.view(-1), int(0.95*shape.numel()))[0][0])

    axis = torch.FloatTensor([0, 1, 0])
    angles = torch.linspace(0, np.pi*2, fps*vlen)
    log_rots = axis[None, :] * angles[:, None]
    Rs = so3_exponential_map(log_rots)
    shape_rot = torch.bmm(Rs, shape[None].repeat(len(Rs), 1, 1))

    extract_frames = []
    if get_frames > 0:
        extract_frames = np.round(np.linspace(0, len(Rs)-1, get_frames))

    vw = VideoWriter(out_path=video_path)
    for ii, shape_rot_ in enumerate(shape_rot):
        fig = matplot_plot_point_cloud(shape_rot_.numpy(),
                                       pointsize=300, azim=-90, elev=90,
                                       figsize=(8, 8), title=title,
                                       sticks=sticks, lim=lim,
                                       cmap=cmap, ax=None, subsample=None,
                                       flip_y=True)
        vw.write_frame(fig)

        if ii in extract_frames:
            framefile = os.path.splitext(video_path)[0] + '_%04d.png' % ii
            print('exporting %s' % framefile)
            plt.savefig(framefile)

        plt.close(fig)

    vidpath = vw.get_video(silent=True)

    if visdom_env is not None:
        viz = get_visdom_connection()
        viz.video(videofile=vidpath, opts={'title': title},
                  env=visdom_env, win=visdom_win)

    return vidpath
