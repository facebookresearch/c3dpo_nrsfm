"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch


def masked_kp_mean(kp_loc, kp_vis):
    visibility_mass = torch.clamp(kp_vis.sum(1), 1e-4)
    kp_mean = (kp_loc*kp_vis[:, None, :]).sum(2)
    kp_mean = kp_mean / visibility_mass[:, None]
    return kp_mean


def huber(dfsq, scaling=0.03):
    loss = (safe_sqrt(1+dfsq/(scaling*scaling), eps=1e-4)-1) * scaling
    return loss


def avg_l2_huber(x, y, mask=None, scaling=0.03):
    diff = x - y
    dist = (diff*diff).sum(1)
    dist = huber(dist, scaling=float(scaling))
    if mask is not None:
        dist = (dist*mask).sum(1) / \
            torch.clamp(mask.sum(1), 1.)
    else:
        if len(dist.shape) == 2 and dist.shape[1] > 1:
            dist = dist.mean(1)
    dist = dist.mean()
    return dist


def avg_l2_dist(x, y, squared=False, mask=None, eps=1e-4):
    diff = x - y
    dist = (diff*diff).sum(1)
    if not squared:
        dist = safe_sqrt(dist, eps=eps)
    if mask is not None:
        dist = (dist*mask).sum(1) / \
            torch.clamp(mask.sum(1), 1.)
    else:
        if len(dist.shape) == 2 and dist.shape[1] > 1:
            dist = dist.mean(1)
    dist = dist.mean()
    return dist


def argmin_translation(x, y, v=None):
    # find translation "T" st. x + T = y
    x_mu = x.mean(2)
    if v is not None:
        vmass = torch.clamp(v.sum(1, keepdim=True), 1e-4)
        x_mu = (v[:, None, :]*x).sum(2) / vmass
        y_mu = (v[:, None, :]*y).sum(2) / vmass
    T = y_mu - x_mu

    return T


def argmin_scale(x, y, v=None):
    # find scale "s" st.: sx=y
    if v is not None:  # mask invisible
        x = x * v[:, None, :]
        y = y * v[:, None, :]
    xtx = (x*x).sum(1).sum(1)
    xty = (x*y).sum(1).sum(1)
    s = xty / torch.clamp(xtx, 1e-4)

    return s


def safe_sqrt(A, eps=float(1e-4)):
    """
    performs safe differentiable sqrt
    """
    return (torch.clamp(A, float(0))+eps).sqrt()
