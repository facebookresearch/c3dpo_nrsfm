"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from PIL import Image
from visdom import Visdom

from tools.utils import NumpySeedFix

# the visdom connection handle
viz = None


def get_visdom_env(cfg):
    if len(cfg.visdom_env) == 0:
        visdom_env = cfg.exp_dir
    else:
        visdom_env = cfg.visdom_env
    return visdom_env


def get_visdom_connection(server='http://localhost', port=8097):
    global viz
    if viz is None:
        viz = Visdom(server=server, port=port)
    return viz


def denorm_image_trivial(im):
    im = im - im.min()
    im = im / (im.max()+1e-7)
    return im


def ensure_im_width(img, basewidth):
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img


def fig2data(fig, size=None):
    """
    Convert a Matplotlib figure to a numpy array
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    if size:
        im = im.resize(size)
    return np.array(im)


def show_projections(p,
                     visdom_env=None,
                     visdom_win=None,
                     v=None,
                     image_path=None,
                     image=None,
                     title='projs',
                     cmap__='gist_ncar',
                     markersize=None,
                     sticks=None,
                     stickwidth=2,
                     stick_color=None,
                     plot_point_order=False,
                     bbox=None,
                     ):

    if image is None:
        try:
            im = Image.open(image_path).convert('RGB')
            im = np.array(im).transpose(2, 0, 1)
        except:
            im = None
            print('!cant load image %s' % image_path)
    else:
        im = image

    nkp = int(p.shape[2])

    pid = np.linspace(0., 1., nkp)

    if v is not None:
        okp = v > 0
    else:
        okp = np.ones(nkp) == 1

    possible_markers = ['o', 'x', 'd']
    markers = [possible_markers[i % len(possible_markers)]
               for i in range(len(p))]

    if markersize is None:
        msz = 50
        if nkp > 40:
            msz = 5
        markersizes = [msz]*nkp
    else:
        markersizes = [markersize]*nkp

    fig = plt.figure(figsize=[11, 11])

    if im is not None:
        plt.imshow(im.transpose((1, 2, 0)))
        plt.axis('off')

    if sticks is not None:
        if stick_color is not None:
            linecol = stick_color
        else:
            linecol = [0., 0., 0.]

        for p_ in p:
            for stick in sticks:
                if v is not None:
                    if v[stick[0]] > 0 and v[stick[1]] > 0:
                        linestyle = '-'
                    else:
                        continue
                else:
                    linestyle = '-'

                plt.plot(p_[0, stick], p_[1, stick], linestyle,
                         color=linecol, linewidth=stickwidth, zorder=1)

    for p_, marker, msz in zip(p, markers, markersizes):
        plt.scatter(p_[0, okp], p_[1, okp], msz, pid[okp],
                    cmap=cmap__, linewidths=2, marker=marker, zorder=2,
                    vmin=0., vmax=1.)
        if plot_point_order:
            for ii in np.where(okp)[0]:
                plt.text(p_[0, ii], p_[1, ii], '%d' %
                         ii, fontsize=int(msz*0.25))

    if bbox is not None:
        import matplotlib.patches as patches
        # Create a Rectangle patch
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    if im is None:
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        # plt.gca().set_frame_on(False)
        plt.gca().set_axis_off()
    else:  # remove all margins
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().set_frame_on(False)
        plt.gca().set_axis_off()

    # return fig
    improj = np.array(fig2data(fig))
    if visdom_env is not None:
        viz = get_visdom_connection()
        viz.image(np.array(improj).transpose(2, 0, 1),
                  env=visdom_env, opts={'title': title}, win=visdom_win)

    plt.close(fig)

    return improj


def extend_to_3d_skeleton_simple(ptcloud, sticks, line_resol=10, rgb=None):

    H36M_TO_MPII_PERM = [3,  2,  1,  4,  5,  6,
                         0,  8,  9, 10, 16, 15, 14, 11, 12, 13]

    rgb_now = rgb.T if rgb is not None else None
    ptcloud_now = ptcloud.T

    ptcloud = ptcloud.T
    rgb = rgb.T if rgb is not None else rgb

    if ptcloud_now.shape[1] == 16:  # MPII
        sticks_new = []
        for stick in sticks:
            if stick[0] in H36M_TO_MPII_PERM and stick[1] in H36M_TO_MPII_PERM:
                s1 = H36M_TO_MPII_PERM.index(int(stick[0]))
                s2 = H36M_TO_MPII_PERM.index(int(stick[1]))
                sticks_new.append([s1, s2])
        sticks = sticks_new

    for sticki, stick in enumerate(sticks):
        alpha = np.linspace(0, 1, line_resol)[:, None]
        linepoints = ptcloud[stick[0], :][None, :] * alpha + \
            ptcloud[stick[1], :][None, :] * (1. - alpha)
        ptcloud_now = np.concatenate((ptcloud_now, linepoints), axis=0)
        if rgb is not None:
            linergb = rgb[stick[0], :][None, :] * alpha + \
                rgb[stick[1], :][None, :] * (1.-alpha)
            rgb_now = np.concatenate(
                (rgb_now, linergb.astype(np.int32)), axis=0)

    if rgb is not None:
        rgb_now = rgb_now.T

    return ptcloud_now.T, rgb_now


def visdom_plot_pointclouds(viz, pcl, visdom_env, title,
                            plot_legend=True, markersize=2,
                            nmax=5000, sticks=None, win=None):

    if sticks is not None:
        pcl = {k: extend_to_3d_skeleton_simple(v, sticks)[0]
               for k, v in pcl.items()}

    legend = list(pcl.keys())

    cmap = 'tab10'
    npcl = len(pcl)
    rgb = (cm.get_cmap(cmap)(np.linspace(0, 1, 10))
           [:, :3]*255.).astype(np.int32).T
    rgb = np.tile(rgb, (1, int(np.ceil(npcl/10))))[:, 0:npcl]

    rgb_cat = {k: np.tile(rgb[:, i:i+1], (1, p.shape[1])) for
               i, (k, p) in enumerate(pcl.items())}

    rgb_cat = np.concatenate(list(rgb_cat.values()), axis=1)
    pcl_cat = np.concatenate(list(pcl.values()), axis=1)

    if pcl_cat.shape[1] > nmax:
        with NumpySeedFix():
            prm = np.random.permutation(
                pcl_cat.shape[1])[0:nmax]
        pcl_cat = pcl_cat[:, prm]
        rgb_cat = rgb_cat[:, prm]

    win = viz.scatter(pcl_cat.T, env=visdom_env,
                      opts={'title': title, 'markersize': markersize,
                            'markercolor': rgb_cat.T}, win=win)
    # legend
    if plot_legend:
        dummy_vals = np.tile(
            np.arange(npcl)[:, None], (1, 2)).astype(np.float32)
        title = "%s_%s" % (title, legend)
        opts = dict(title=title, legend=legend, width=400, height=400)
        viz.line(dummy_vals.T, env=visdom_env, opts=opts)

    return win


def matplot_plot_point_cloud(ptcloud, pointsize=20, azim=90, elev=90,
                             figsize=(8, 8), title=None, sticks=None, lim=None,
                             cmap='gist_ncar', ax=None, subsample=None,
                             flip_y=False):

    if lim is None:
        lim = np.abs(ptcloud).max()

    nkp = int(ptcloud.shape[1])
    pid = np.linspace(0., 1., nkp)
    rgb = (cm.get_cmap(cmap)(pid)[:, :3]*255.).astype(np.int32)

    if subsample is not None:
        with NumpySeedFix():
            prm = np.random.permutation(nkp)[0:subsample]
        pid = pid[prm]
        rgb = rgb[prm, :]
        ptcloud = ptcloud[:, prm]

    if flip_y:
        ptcloud[1, :] = -ptcloud[1, :]

    if ax is not None:
        fig = None
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=elev, azim=azim)
    if sticks is not None:
        for stick in sticks:
            line = ptcloud[:, [stick[0], stick[1]]]
            xs, ys, zs = line
            ax.plot(xs, ys, zs, color='black')

    xs, ys, zs = ptcloud
    ax.scatter(xs, ys, zs, s=pointsize, c=pid, marker='.', cmap=cmap)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_zticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    if title is not None:
        ax.set_title(title)
    plt.show()

    return fig


# old functions to replace:

def enlarge_box(box, perc, imsz):
    boxw, boxh = box[2]-box[0], box[3]-box[1]
    box[0] -= boxw*perc
    box[1] -= boxh*perc
    box[2] += boxw*perc
    box[3] += boxh*perc

    imh, imw = imsz
    box = np.maximum(np.minimum(box, np.array([imw, imh, imw, imh])), 0.)

    return box
