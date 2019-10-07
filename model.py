"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from dataset.dataset_configs import STICKS
from tools.so3 import so3_exponential_map, rand_rot
from tools.functions import masked_kp_mean, \
    argmin_translation, argmin_scale, \
    avg_l2_huber
from tools.vis_utils import get_visdom_connection, \
    show_projections, \
    visdom_plot_pointclouds
from tools.utils import auto_init_args
import numpy as np
import torch.nn.functional as Fu
from torch import nn as nn
import torch


class C3DPO(torch.nn.Module):

    def __init__(self, n_keypoints=17,
                 shape_basis_size=10,
                 n_fully_connected=1024,
                 n_layers=6,
                 keypoint_rescale=float(1),
                 keypoint_norm_type='to_mean',
                 projection_type='orthographic',
                 z_augment=True,
                 z_augment_rot_angle=float(np.pi)/8,
                 z_equivariance=True,
                 z_equivariance_rot_angle=float(np.pi)/8,
                 camera_translation=False,
                 camera_xy_translation=False,
                 argmin_translation=False,
                 camera_scale=False,
                 connectivity_setup='NONE',
                 huber_scaling=0.01,
                 reprojection_normalization='kp_total_count',
                 independent_phi_for_aug=False,
                 canonicalization={
                     'use':               True,
                     'n_layers':          6,
                     'n_rand_samples':    4,
                     'rot_angle':         float(np.pi),
                     'n_fully_connected': 1024,
                 },
                 perspective_depth_threshold=0.1,
                 depth_offset=0.,
                 replace_keypoints_with_input=True,
                 root_joint=0,
                 weight_init_std=0.01,
                 loss_weights={'l_reprojection':     1.,
                               'l_canonicalization': 1.},
                 log_vars=[
                     'objective',
                     'dist_reprojection',
                     'l_reprojection',
                     'l_canonicalization'],
                 **kwargs):
        super(C3DPO, self).__init__()

        # autoassign constructor params to self
        auto_init_args(self)

        # factorization net
        self.phi = nn.Sequential(
            *self.make_trunk(dim_in=self.n_keypoints * 3,
                             # 2 dim loc, 1 dim visibility
                             n_fully_connected=self.n_fully_connected,
                             n_layers=self.n_layers))

        # shape coefficient predictor
        self.alpha_layer = conv1x1(self.n_fully_connected,
                                   self.shape_basis_size,
                                   std=weight_init_std)

        # 3D shape predictor
        self.shape_layer = conv1x1(self.shape_basis_size, 3*n_keypoints,
                                   std=weight_init_std)

        # rotation predictor (predicts log-rotation)
        self.rot_layer = conv1x1(self.n_fully_connected, 3,
                                 std=weight_init_std)
        if self.camera_translation:
            # camera translation
            self.translation_layer = conv1x1(self.n_fully_connected, 3,
                                             std=weight_init_std)
        if self.camera_scale:
            # camera scale (with final sofplus to ensure positive outputs)
            self.scale_layer = nn.Sequential(conv1x1(self.n_fully_connected, 3,
                                                     std=weight_init_std),
                                             nn.Softplus())

        if self.canonicalization['use']:
            # canonicalization net:
            self.psi = nn.Sequential(
                *self.make_trunk(dim_in=self.n_keypoints*3,
                                 n_fully_connected=self.canonicalization['n_fully_connected'],
                                 n_layers=self.canonicalization['n_layers']))
            self.alpha_layer_psi = conv1x1(self.n_fully_connected,
                                           self.shape_basis_size,
                                           std=weight_init_std)

    def make_trunk(self,
                   n_fully_connected=None,
                   dim_in=None,
                   n_layers=None,
                   use_bn=True):

        layer1 = ConvBNLayer(dim_in,
                             n_fully_connected,
                             use_bn=use_bn)
        layers = [layer1]

        for l in range(n_layers):
            layers.append(ResLayer(n_fully_connected,
                                   int(n_fully_connected/4)))

        return layers

    def forward(self, kp_loc=None, kp_vis=None,
                class_mask=None, K=None, **kwargs):

        # dictionary with outputs of the fw pass
        preds = {}

        # input sizes ...
        ba, kp_dim, n_kp = kp_loc.shape

        assert kp_dim == 2, 'bad input keypoint dim'
        assert n_kp == self.n_keypoints, 'bad # of keypoints!'

        if self.projection_type == 'perspective':
            assert K is not None
            kp_loc_cal = self.calibrate_keypoints(kp_loc, K)
        else:
            kp_loc_cal = kp_loc

        # normalize keypoints
        kp_loc_norm, kp_mean = \
            self.normalize_keypoints(
                kp_loc_cal, kp_vis, rescale=self.keypoint_rescale)
        # save for later visualisations ...
        preds['kp_loc_norm'] = kp_loc_norm
        preds['kp_mean'] = kp_mean

        # run the shape predictor
        preds['phi'] = self.run_phi(kp_loc_norm, kp_vis, class_mask=class_mask)

        if self.canonicalization['use']:
            preds['l_canonicalization'], preds['psi'] = \
                self.canonicalization_loss(preds['phi'],
                                           class_mask=class_mask)

        # 3D->2D project shape to camera
        kp_reprojected, depth = self.camera_projection(
            preds['phi']['shape_camera_coord'])
        preds['kp_reprojected'] = kp_reprojected

        # compute the repro loss for backpropagation
        if self.reprojection_normalization == 'kp_count_per_image':
            preds['l_reprojection'] = avg_l2_huber(
                kp_reprojected,
                kp_loc_norm,
                mask=kp_vis,
                squared=self.squared_reprojection_loss)

        elif self.reprojection_normalization == 'kp_total_count':
            def flatten_(x): return x.permute(
                1, 2, 0).contiguous().view(1, 2, self.n_keypoints*ba)
            preds['l_reprojection'] = avg_l2_huber(
                flatten_(kp_reprojected),
                flatten_(kp_loc_norm),
                mask=kp_vis.permute(1, 0).contiguous().view(1, -1),
                scaling=self.huber_scaling)

        else:
            raise ValueError('unknown loss normalization %s' %
                             self.loss_normalization)

        # unnormalize the shape projections
        kp_reprojected_image = \
            self.unnormalize_keypoints(kp_reprojected, kp_mean,
                                       rescale=self.keypoint_rescale)

        # projections in the image coordinate frame
        if self.replace_keypoints_with_input and not self.training:
            # use the input points
            kp_reprojected_image = \
                (1-kp_vis[:, None, :]) * kp_reprojected_image + \
                kp_vis[:, None, :] * kp_loc_cal

        preds['kp_reprojected_image'] = kp_reprojected_image

        # projected 3D shape in the image space
        #   = unprojection of kp_reprojected_image
        shape_image_coord = self.camera_unprojection(
            kp_reprojected_image, depth,
            rescale=self.keypoint_rescale)

        if self.projection_type == 'perspective':
            preds['shape_image_coord_cal'] = shape_image_coord
            shape_image_coord = \
                self.uncalibrate_keypoints(shape_image_coord, K)
            preds['kp_reprojected_image_uncal'], _ = \
                self.camera_projection(shape_image_coord)

        preds['shape_image_coord'] = shape_image_coord

        # get the final loss
        preds['objective'] = self.get_objective(preds)
        assert np.isfinite(
            preds['objective'].sum().data.cpu().numpy()), "nans!"

        return preds

    def camera_projection(self, shape):
        depth = shape[:, 2:3, :]
        if self.projection_type == 'perspective':
            if self.perspective_depth_threshold > 0:
                depth = torch.clamp(depth, self.perspective_depth_threshold)
            projections = shape[:, 0:2, :] / depth
        elif self.projection_type == 'orthographic':
            projections = shape[:, 0:2, :]
        else:
            raise ValueError('no such projection type %s' %
                             self.projection_type)

        return projections, depth

    def camera_unprojection(self, kp_loc, depth, rescale=float(1)):
        depth = depth / rescale
        if self.projection_type == 'perspective':
            shape = torch.cat((kp_loc * depth, depth), dim=1)
        elif self.projection_type == 'orthographic':
            shape = torch.cat((kp_loc, depth), dim=1)
        else:
            raise ValueError('no such projection type %s' %
                             self.projection_type)

        return shape

    def calibrate_keypoints(self, kp_loc, K):
        # undo the projection matrix
        assert K is not None
        kp_loc = kp_loc - K[:, 0:2, 2:3]
        focal = torch.stack((K[:, 0, 0], K[:, 1, 1]), dim=1)
        kp_loc = kp_loc / focal[:, :, None]
        return kp_loc

    def uncalibrate_keypoints(self, kp_loc, K):
        assert K is not None
        kp_loc = torch.bmm(K, kp_loc)
        return kp_loc

    def normalize_keypoints(self,
                            kp_loc,
                            kp_vis,
                            rescale=1.,
                            K=None):
        if self.keypoint_norm_type == 'to_root':
            # center around the root joint
            kp_mean = kp_loc[:, :, self.root_joint]
            kp_loc_norm = kp_loc - kp_mean[:, :, None]
        elif self.keypoint_norm_type == 'to_mean':
            # calc the mean of visible points
            kp_mean = masked_kp_mean(kp_loc, kp_vis)
            # remove the mean from the keypoint locations
            kp_loc_norm = kp_loc - kp_mean[:, :, None]
        else:
            raise ValueError('no such kp norm  %s' %
                             self.keypoint_norm_type)

        # rescale
        kp_loc_norm = kp_loc_norm * rescale

        return kp_loc_norm, kp_mean

    def unnormalize_keypoints(self,
                              kp_loc_norm,
                              kp_mean,
                              rescale=1.,
                              K=None):
        kp_loc = kp_loc_norm * (1. / rescale)
        kp_loc = kp_loc + kp_mean[:, :, None]
        return kp_loc

    def run_phi(self,
                kp_loc,
                kp_vis,
                class_mask=None,
                ):

        preds = {}

        # batch size
        ba = kp_loc.shape[0]
        dtype = kp_loc.type()

        kp_loc_orig = kp_loc.clone()

        if self.z_augment and self.training:
            R_rand = rand_rot(ba,
                              dtype=dtype,
                              max_rot_angle=float(self.z_augment_rot_angle),
                              axes=(0, 0, 1))
            kp_loc_in = torch.bmm(R_rand[:, 0:2, 0:2], kp_loc)
        else:
            R_rand = torch.eye(3).type(dtype)[None].repeat((ba, 1, 1))
            kp_loc_in = kp_loc_orig

        if self.z_equivariance and self.training:
            # random xy rot
            R_rand_eq = rand_rot(ba,
                                 dtype=dtype,
                                 max_rot_angle=float(
                                     self.z_equivariance_rot_angle),
                                 axes=(0, 0, 1))
            kp_loc_in = torch.cat(
                (kp_loc_in,
                 torch.bmm(R_rand_eq[:, 0:2, 0:2], kp_loc_in)
                 ), dim=0)
            kp_vis_in = kp_vis.repeat((2, 1))
        else:
            kp_vis_in = kp_vis

        # mask kp_loc by kp_visibility
        kp_loc_masked = kp_loc_in * kp_vis_in[:, None, :]

        # vectorize
        kp_loc_flatten = kp_loc_masked.view(-1, 2*self.n_keypoints)

        # concatenate visibilities and kp locations
        l1_input = torch.cat((kp_loc_flatten, kp_vis_in), dim=1)

        # pass to network
        if self.independent_phi_for_aug and l1_input.shape[0] == 2*ba:
            feats = torch.cat([self.phi(l1_[:, :, None, None]) for
                               l1_ in l1_input.split(ba, dim=0)], dim=0)
        else:
            feats = self.phi(l1_input[:, :, None, None])

        # coefficients into the linear basis
        shape_coeff = self.alpha_layer(feats)[:, :, 0, 0]

        if self.z_equivariance and self.training:
            # use the shape coeff from the second set of preds
            shape_coeff = shape_coeff[ba:]
            # take the feats from the first set
            feats = feats[:ba]

        # shape prediction is just a linear layer implemented as a conv
        shape_canonical = self.shape_layer(
            shape_coeff[:, :, None, None])[:, :, 0, 0]
        shape_canonical = shape_canonical.view(ba, 3, self.n_keypoints)

        if self.keypoint_norm_type == 'to_root':
            # make sure we fix the root at 0
            root_j = shape_canonical[:, :, self.root_joint]
            shape_canonical = shape_canonical - root_j[:, :, None]

        # predict camera params
        # ... log rotation (exponential representation)
        R_log = self.rot_layer(feats)[:, :, 0, 0]

        # convert from the 3D to 3x3 rot matrix
        R = so3_exponential_map(R_log)

        # T vector of the camera
        if self.camera_translation:
            T = self.translation_layer(feats)[:, :, 0, 0]
            if self.camera_xy_translation:  # kill the last z-dim
                T = T * torch.tensor([1., 1., 0.]).type(dtype)[None, :]
        else:
            T = R_log.new_zeros(ba, 3)

        # offset the translation vector of the camera
        if self.depth_offset > 0.:
            T[:, 2] = T[:, 2] + self.depth_offset

        # scale of the camera
        if self.camera_scale:
            scale = self.scale_layer(feats)[:, 0, 0, 0]
        else:
            scale = R_log.new_ones(ba)

        # rotated+scaled shape into the camera ( Y = sRX + T  )
        shape_camera_coord = self.apply_similarity_t(
            shape_canonical, R, T, scale)

        # undo equivariant transformation
        if (self.z_equivariance or self.z_augment) and self.training:
            R_rand_inv = R_rand.transpose(2, 1)
            R = torch.bmm(R_rand_inv, R)
            T = torch.bmm(R_rand_inv, T[:, :, None])[:, :, 0]
            shape_camera_coord = torch.bmm(R_rand_inv, shape_camera_coord)

        # estimate translation
        if self.argmin_translation:
            assert self.projection_type == 'orthographic'
            projection, _ = self.camera_projection(shape_camera_coord)
            T_amin = argmin_translation(projection, kp_loc_orig, v=kp_vis)
            T_amin = Fu.pad(T_amin, (0, 1), 'constant', float(0))
            shape_camera_coord = shape_camera_coord + T_amin[:, :, None]
            T = T + T_amin

        if class_mask is not None:
            shape_camera_coord = shape_camera_coord * class_mask[:, None, :]
            shape_canonical = shape_canonical * class_mask[:, None, :]

        preds['R_log'] = R_log
        preds['R'] = R
        preds['scale'] = scale
        preds['T'] = T
        preds['shape_camera_coord'] = shape_camera_coord
        preds['shape_coeff'] = shape_coeff
        preds['shape_canonical'] = shape_canonical

        return preds

    def apply_similarity_t(self, S, R, T, s):
        return torch.bmm(R, s[:, None, None] * S) + T[:, :, None]

    def canonicalization_loss(self, phi_out, class_mask=None):

        shape_canonical = phi_out['shape_canonical']

        dtype = shape_canonical.type()
        ba = shape_canonical.shape[0]

        n_sample = self.canonicalization['n_rand_samples']

        # rotate the canonical point cloud
        # generate random rotation around all axes
        R_rand = rand_rot(ba * n_sample,
                          dtype=dtype,
                          max_rot_angle=self.canonicalization['rot_angle'],
                          axes=(1, 1, 1))

        unrotated = shape_canonical.repeat(n_sample, 1, 1)
        rotated = torch.bmm(R_rand, unrotated)

        psi_out = self.run_psi(rotated)  # psi3( Rrand X )

        a, b = psi_out['shape_canonical'], unrotated
        l_canonicalization = avg_l2_huber(a, b,
                                          scaling=self.huber_scaling,
                                          mask=class_mask.repeat(n_sample, 1)
                                          if class_mask is not None else None)

        # reshape the outputs in the output list
        psi_out = {k: v.view(
            self.canonicalization['n_rand_samples'],
            ba, *v.shape[1:]) for k, v in psi_out.items()}

        return l_canonicalization, psi_out

    def run_psi(self, shape_canonical):

        preds = {}

        # batch size
        ba = shape_canonical.shape[0]
        assert shape_canonical.shape[1] == 3, '3d inputs only please'

        # reshape and pass to the network ...
        l1_input = shape_canonical.view(ba, 3*self.n_keypoints)

        # pass to network
        feats = self.psi(l1_input[:, :, None, None])

        # coefficients into the linear basis
        shape_coeff = self.alpha_layer_psi(feats)[:, :, 0, 0]
        preds['shape_coeff'] = shape_coeff

        # use the shape_pred_layer from 2d predictor
        shape_pred = self.shape_layer(
            shape_coeff[:, :, None, None])[:, :, 0, 0]
        shape_pred = shape_pred.view(ba, 3, self.n_keypoints)
        preds['shape_canonical'] = shape_pred

        return preds

    def get_objective(self, preds):
        losses_weighted = [preds[k] * float(w) for k, w in
                           self.loss_weights.items()
                           if k in preds]
        if (not hasattr(self, '_loss_weights_printed') or
                not self._loss_weights_printed) and self.training:
            print('-------\nloss_weights:')
            for k, w in self.loss_weights.items():
                print('%20s: %1.2e' % (k, w))
            print('-------')
            self._loss_weights_printed = True
        loss = torch.stack(losses_weighted).sum()
        return loss

    def visualize(self, visdom_env, trainmode,
                  preds, stats, clear_env=False):
        viz = get_visdom_connection(server=stats.visdom_server,
                                    port=stats.visdom_port)
        if not viz.check_connection():
            print("no visdom server! -> skipping batch vis")
            return

        if clear_env:  # clear visualisations
            print("  ... clearing visdom environment")
            viz.close(env=visdom_env, win=None)

        print('vis into env:\n   %s' % visdom_env)

        it = stats.it[trainmode]
        epoch = stats.epoch
        idx_image = 0

        title = "e%d_it%d_im%d" % (epoch, it, idx_image)

        # get the connectivity pattern
        sticks = STICKS[self.connectivity_setup] if \
            self.connectivity_setup in STICKS else None

        var_kp = {'orthographic': 'kp_reprojected_image',
                  'perspective':  'kp_reprojected_image_uncal'
                  }[self.projection_type]

        # show reprojections
        p = np.stack(
            [preds[k][idx_image].detach().cpu().numpy()
             for k in (var_kp, 'kp_loc')])
        v = preds['kp_vis'][idx_image].detach().cpu().numpy()

        show_projections(p, visdom_env=visdom_env, v=v,
                         title='projections_'+title, cmap__='gist_ncar',
                         markersize=50, sticks=sticks,
                         stickwidth=1, plot_point_order=False,
                         image_path=preds['image_path'][idx_image],
                         visdom_win='projections')

        # show 3d reconstruction
        if True:
            var3d = {'orthographic': 'shape_image_coord',
                     'perspective': 'shape_image_coord_cal'
                     }[self.projection_type]
            pcl = {'pred': preds[var3d]
                   [idx_image].detach().cpu().numpy().copy()}
            if 'kp_loc_3d' in preds:
                pcl['gt'] = preds['kp_loc_3d'][idx_image].detach(
                ).cpu().numpy().copy()
                if self.projection_type == 'perspective':
                    # for perspective projections, we dont know the scale
                    # so we estimate it here ...
                    scale = argmin_scale(torch.from_numpy(pcl['pred'][None]),
                                         torch.from_numpy(pcl['gt'][None]))
                    pcl['pred'] = pcl['pred'] * float(scale)
                elif self.projection_type == 'orthographic':
                    # here we depth-center gt and predictions
                    for k in ('pred', 'gt'):
                        pcl_ = pcl[k].copy()
                        meanz = pcl_.mean(1) * np.array([0., 0., 1.])
                        pcl[k] = pcl_ - meanz[:, None]
                else:
                    raise ValueError(self.projection_type)

            visdom_plot_pointclouds(viz, pcl, visdom_env, '3d_'+title,
                                    plot_legend=False, markersize=20,
                                    sticks=sticks, win='3d')


def pytorch_ge12():
    v = torch.__version__
    v = float('.'.join(v.split('.')[0:2]))
    return v >= 1.2


def conv1x1(in_planes, out_planes, std=0.01):
    """1x1 convolution"""
    cnv = nn.Conv2d(in_planes, out_planes, bias=True, kernel_size=1)

    cnv.weight.data.normal_(0., std)
    if cnv.bias is not None:
        cnv.bias.data.fill_(0.)

    return cnv


class ConvBNLayer(nn.Module):

    def __init__(self, inplanes, planes, use_bn=True, stride=1, ):
        super(ConvBNLayer, self).__init__()

        # do a reasonable init
        self.conv1 = conv1x1(inplanes, planes)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
            if pytorch_ge12():
                self.bn1.weight.data.uniform_(0., 1.)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        return out


class ResLayer(nn.Module):

    def __init__(self, inplanes, planes, expansion=4):
        super(ResLayer, self).__init__()
        self.expansion = expansion

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if pytorch_ge12():
            self.bn1.weight.data.uniform_(0., 1.)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if pytorch_ge12():
            self.bn2.weight.data.uniform_(0., 1.)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if pytorch_ge12():
            self.bn3.weight.data.uniform_(0., 1.)
        self.relu = nn.ReLU(inplace=True)
        self.skip = inplanes == (planes*self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.skip:
            out += residual
        out = self.relu(out)

        return out
