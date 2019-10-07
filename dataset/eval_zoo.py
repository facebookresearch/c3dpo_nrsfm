"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import numpy as np

from tabulate import tabulate
from tqdm import tqdm


def eval_zoo(dataset_name, include_debug_vars=False):

    if dataset_name in ('h36m', 'h36m_hourglass'):
        eval_script = eval_h36m
        cache_vars = ['kp_loc_3d', 'h36m_info', 'shape_image_coord']
        eval_vars = ['EVAL_MPJPE_orig', 'EVAL_MPJPE_best', 'EVAL_stress']
    elif dataset_name in ('pascal3d', 'pascal3d_hrnet'):
        eval_script = eval_pascal3d
        cache_vars = ['kp_loc_3d', 'p3d_info', 'class_mask',
                      'shape_image_coord', 'kp_defined', 'kp_vis']
        eval_vars = ['EVAL_MPJPE_orig', 'EVAL_MPJPE_best', 'EVAL_stress']
    elif dataset_name in ('up3d_79kp'):
        eval_script = eval_up3d_79kp
        cache_vars = ['kp_loc_3d', 'shape_image_coord']
        eval_vars = ['EVAL_MPJPE_orig', 'EVAL_MPJPE_best', 'EVAL_stress']
    elif dataset_name in ('cub_birds', 'cub_birds_hrnet'):
        eval_script = eval_dummy
        cache_vars = ['shape_image_coord']
        eval_vars = ['EVAL_dummy']
    else:
        assert False, ("no such dataset eval %s" % dataset_name)

    return eval_script, cache_vars, eval_vars


def eval_dummy(cached_preds, eval_vars=None):
    return {'EVAL_dummy': 0.}, None


def eval_pascal3d(cached_preds, eval_vars=None,
                  N_CLS=12, N_KP=124, N_ENTRIES=1950):
    """
    evaluates 3d error metrics on pascal3d
    """

    from dataset.dataset_configs import P3D_CLASSES, P3D_NUM_IMAGES

    print('PAS 3D evaluation ...')

    gt = np.array(cached_preds['kp_loc_3d'])
    pred = np.array(cached_preds['shape_image_coord'])
    classes = np.array(cached_preds['p3d_info']['p3d_class'])
    class_mask = np.array(cached_preds['class_mask'])
    kp_defined = np.array(cached_preds['kp_defined'])
    eval_mask = class_mask * kp_defined

    assert pred.shape[2] == N_KP

    for arr in (gt, pred, classes, class_mask, kp_defined, eval_mask):
        assert len(arr) == N_ENTRIES, 'wrong n of predictions!'

    results = calc_3d_errs(pred, gt, fix_mean_depth=True,
                           scale=float(1), mask=eval_mask)

    metrics = list(results.keys())

    # check that eval_vars are all evaluated
    if eval_vars is not None:
        for m in eval_vars:
            assert m in metrics, "missing metric %s!" % m
        print("eval vars checks ok!")

    all_avg_results, avg_class_results = \
        eval_results_per_class(classes, results, P3D_CLASSES,
                               N_PER_CLASS=P3D_NUM_IMAGES['val'])

    print_results_per_class(avg_class_results, all_avg_results)

    aux_out = {}
    aux_out['per_sample_err'] = results
    aux_out['per_class_err'] = avg_class_results

    return all_avg_results, aux_out


def eval_up3d_79kp(cached_preds,
                   eval_vars=None,
                   N_ENTRIES=15000):

    print('UP3D evaluation ... (tgt n entries = %d)' % N_ENTRIES)

    gt = np.array(cached_preds['kp_loc_3d'])
    pred = np.array(cached_preds['shape_image_coord'])

    for arr in (gt, pred):
        assert len(arr) == N_ENTRIES, 'wrong n of predictions!'

    results = calc_3d_errs(pred, gt, fix_mean_depth=True)

    metrics = list(results.keys())

    # check that eval_vars are all evaluated
    if eval_vars is not None:
        for m in eval_vars:
            assert m in metrics, "missing metric %s!" % m

    all_avg_results = {}
    for metric in metrics:
        all_avg_results[metric] = float(np.array(results[metric]).mean())
        print("%20s: %20s" % (metric, "%1.4f" % all_avg_results[metric]))

    aux_out = {}
    aux_out['per_sample_err'] = results

    return all_avg_results, aux_out


def eval_h36m(cached_preds,
              eval_vars=None,
              N_ENTRIES=109556,
              norm_to_hip=True):

    from dataset.dataset_configs import H36M_ACTIONS

    print('H36M evaluation ... (tgt n entries = %d)' % N_ENTRIES)

    gt = np.array(cached_preds['kp_loc_3d'])
    pred = np.array(cached_preds['shape_image_coord'])
    scale = np.array(cached_preds['h36m_info']['scale'])
    action_names = cached_preds['h36m_info']['action_name']

    for arr in (gt, pred, scale, action_names):
        assert len(arr) == N_ENTRIES, 'wrong n of predictions!'

    if norm_to_hip:
        pred = pred - pred[:, :, 0:1]

    results = calc_3d_errs(pred, gt,
                           fix_mean_depth=False,
                           scale=scale)

    metrics = list(results.keys())

    # check that eval_vars are all evaluated
    if eval_vars is not None:
        for m in eval_vars:
            assert m in metrics, "missing metric %s!" % m
        # print("eval vars checks ok!")

    all_avg_results, avg_action_results = \
        eval_results_per_class(action_names, results, H36M_ACTIONS)

    print_results_per_class(avg_action_results, all_avg_results)

    aux_out = {}
    aux_out['per_sample_err'] = results
    aux_out['per_class_err'] = avg_action_results

    return all_avg_results, aux_out


def eval_results_per_class(classes, results, CLASS_LIST, N_PER_CLASS=None):

    metrics = list(results.keys())

    avg_cls_results = {}
    for cls_ in CLASS_LIST:
        ok_cls = [ei for ei, _ in enumerate(classes) if classes[ei] == cls_]
        cls_results = {k: v[ok_cls] for k, v in results.items()}
        if N_PER_CLASS is not None:
            assert len(ok_cls) == N_PER_CLASS[cls_]
        if True:  # asserts ...
            for k, v in cls_results.items():
                assert v.size == len(ok_cls)
        avg_cls_results[cls_] = {k: np.array(
            v).mean() for k, v in cls_results.items()}

    all_avg_results = {}
    for metric in metrics:
        avgmetric = [v[metric] for _, v in avg_cls_results.items()]
        all_avg_results[metric] = float(np.array(avgmetric).mean())

    return all_avg_results, avg_cls_results


def print_results_per_class(avg_cls_results, all_avg_results):

    metrics = list(all_avg_results.keys())

    # result printing
    avg_results_print = copy.deepcopy(avg_cls_results)
    avg_results_print['== Mean =='] = all_avg_results
    tab_rows = []
    for cls_, cls_metrics in avg_results_print.items():
        tab_row = [cls_]
        for metric in metrics:
            val = cls_metrics[metric]
            tab_row.append("%1.3f" % val)
        tab_rows.append(tab_row)
    headers = ['classes']
    headers.extend(copy.deepcopy(metrics))
    print(tabulate(tab_rows, headers=headers))


def calc_3d_errs(pred, gt,
                 fix_mean_depth=False,
                 get_best_scale=False,
                 scale=float(1), mask=None):

    pred_flip = np.copy(pred)
    pred_flip[:, 2, :] = -pred_flip[:, 2, :]

    pairs_compare = {'EVAL_MPJPE_orig': pred,
                     'EVAL_MPJPE_flip':  pred_flip}

    results = {}
    for metric, pred_compare in pairs_compare.items():
        results[metric] = calc_dist_err(gt, pred_compare,
                                        fix_mean_depth=fix_mean_depth,
                                        get_best_scale=get_best_scale,
                                        scale=scale,
                                        mask=mask)

    results['EVAL_MPJPE_best'] = np.minimum(results['EVAL_MPJPE_orig'],
                                            results['EVAL_MPJPE_flip'])

    results['EVAL_stress'] = calc_stress_err(gt, pred, mask=mask, scale=scale)

    return results


def calc_stress_err(gt, pred, scale=1., mask=None, get_best_scale=False):

    assert pred.shape[1] == 3
    assert gt.shape[1] == 3
    assert pred.shape[0] == gt.shape[0]
    assert pred.shape[2] == gt.shape[2]

    if get_best_scale:
        argmin_scale = compute_best_scale(pred, gt, v=mask)
        pred = pred.copy() * argmin_scale[:, None, None]

    errs = []
    nkp = gt.shape[2]
    if mask is not None:
        tridx_cache = [np.triu_indices(k, k=1)
                       for k in range(nkp+1)]
        assert mask.shape[1] == pred.shape[2]
        assert mask.shape[0] == pred.shape[0]
    else:
        tridx = np.triu_indices(nkp, k=1)
        assert len(tridx[0]) == (nkp*(nkp-1))/2

    print('stress eval:')
    with tqdm(total=len(gt)) as tq:
        for ii, (g_, p_) in enumerate(zip(gt, pred)):
            if mask is not None:
                mask_ = mask[ii]
            else:
                mask_ = None
            edm_g = calc_edm(g_, squared=False, mask=mask_)
            edm_p = calc_edm(p_, squared=False, mask=mask_)
            stress = np.abs(edm_g - edm_p)
            if mask_ is not None:
                nkp_now = edm_g.shape[0]
                assert mask_.sum() == nkp_now
                tridx = tridx_cache[nkp_now]
            mstress = stress[tridx[0], tridx[1]]
            mstress = mstress.mean()
            # if True:
            # 	triu_mask_ = np.triu(np.ones(nkp),k=1)
            # 	mstress_ = (stress * triu_mask_).sum() / triu_mask_.sum()
            # 	assert np.abs(mstress - mstress_) <= 1e-3

            errs.append(mstress)
            tq.update(1)

    errs = np.array(errs) * scale

    return errs


def calc_dist_err(gt, pred, scale=1.,
                  fix_mean_depth=False,
                  get_best_scale=False,
                  mask=None):

    assert pred.shape[1] == 3
    assert gt.shape[1] == 3
    assert pred.shape[0] == gt.shape[0]
    assert pred.shape[2] == gt.shape[2]

    if fix_mean_depth:
        # print('setting mean depth = 0')
        pred = set_mean_depth_to_0(pred, mask=mask)
        gt = set_mean_depth_to_0(gt,   mask=mask)

    if get_best_scale:
        argmin_scale = compute_best_scale(pred, gt, v=mask)
        pred = pred.copy() * argmin_scale[:, None, None]

    df = pred - gt
    errs = np_safe_sqrt((df*df).sum(1))

    if True:
        errs_ = np.sqrt((df*df).sum(1))
        df__ = np.max(np.abs(errs-errs_))
        assert df__ <= 1e-5
        # print('err diff = %1.2e' % df__)

    if mask is not None:
        assert mask.shape[0] == pred.shape[0]
        assert mask.shape[1] == pred.shape[2]
        assert len(mask.shape) == 2
        errs = (mask*errs).sum(1) / mask.sum(1)
    else:
        errs = errs.mean(1)

    errs = errs * scale
    return errs


def set_mean_depth_to_0(x, mask=None):

    x = x.copy()
    if mask is not None:
        x = x * mask[:, None, :]
        mu_depth = (x.sum(2)/mask.sum(1)[:, None])[:, 2]
    else:
        mu_depth = x.mean(2)[:, 2]

    x[:, 2, :] = x[:, 2, :] - mu_depth[:, None]

    if mask is not None:
        x = x * mask[:, None, :]

    return x


def np_safe_sqrt(x):
    y = np.zeros_like(x)
    assert (x > -1e-5).all()
    x_good = x > 0
    y[x_good] = np.sqrt(x[x_good])
    return y


def calc_edm(x, squared=True, mask=None):
    if mask is not None:
        x = x.copy()[:, mask == 1]
    xx = x.T @ x
    x2 = (x*x).sum(0)
    edm = x2[:, None]+x2[None, :]-2.*xx
    edm = np.maximum(edm, 0.)
    if not squared:
        edm = np_safe_sqrt(edm)
        # edm = np.sqrt(edm)

    # if True:
    # 	import scipy
    # 	import scipy.spatial
    # 	edm_ = scipy.spatial.distance.cdist(x.T,x.T)
    # 	df = np.abs(edm-edm_).max()
    # 	assert df <= edm.mean()/200., '%1.3e' % df
    # 	# print('df = %1.3f' % df)
    # 	# scipy.spatial.distance.pdist(x.T)

    return edm


def compute_best_scale():
    raise NotImplementedError('not yet finished')
