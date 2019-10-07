"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import math
import torch.nn.functional as Fu


def rand_rot(N, dtype=None, max_rot_angle=float(math.pi),
             axes=(1, 1, 1), get_ss=False):

    rand_axis = torch.zeros((N, 3)).type(dtype).normal_()

    # apply the axes mask
    axes = torch.Tensor(axes).type(dtype)
    rand_axis = axes[None, :] * rand_axis

    rand_axis = Fu.normalize(rand_axis, dim=1, p=2)
    rand_angle = torch.ones(N).type(dtype).uniform_(0, max_rot_angle)
    R_ss_rand = rand_axis * rand_angle[:, None]
    R_rand = so3_exponential_map(R_ss_rand)

    if get_ss:
        return R_rand, R_ss_rand
    else:
        return R_rand


def so3_exponential_map(log_rot: torch.Tensor, eps: float = 0.0001):
    """
    Convert a batch of logarithmic representations of rotation matrices
    `log_rot` to a batch of 3x3 rotation matrices using Rodrigues formula.
    The conversion has a singularity around 0 which is handled by clamping
    controlled with the `eps` argument.

    Args:
        log_rot: batch of vectors of shape `(minibatch , 3)`
        eps: a float constant handling the conversion singularity around 0

    Returns:
        batch of rotation matrices of shape `(minibatch , 3 , 3)`

    Raises:
        ValueError if `log_rot` is of incorrect shape
    """

    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError('Input tensor shape has to be Nx3.')

    nrms = (log_rot * log_rot).sum(1)
    phis = torch.clamp(nrms, 0.).sqrt()
    phisi = 1. / (phis+eps)
    fac1 = phisi * phis.sin()
    fac2 = phisi * phisi * (1. - phis.cos())
    ss = hat(log_rot)

    R = fac1[:, None, None] * ss + \
        fac2[:, None, None] * torch.bmm(ss, ss) + \
        torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]

    return R


def hat(v: torch.Tensor):
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: batch of vectors of shape `(minibatch , 3)`

    Returns:
        batch of skew-symmetric matrices of shape `(minibatch, 3, 3)`

    Raises:
        ValueError if `v` is of incorrect shape

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError('Input vectors have to be 3-dimensional.')

    h = v.new_zeros(N, 3, 3)

    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h
