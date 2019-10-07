"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pickle
import torch
import glob
import os
import yaml


def load_stats(flstats):
    try:
        stats, _ = pickle.load(open(flstats, 'rb'))  # dont load the config
    except:
        # print("Cant load stats! %s" % flstats)
        stats = None
    return stats


def get_model_path(fl):
    fl = os.path.splitext(fl)[0]
    flmodel = "%s.pth" % fl
    return flmodel


def get_optimizer_path(fl):
    fl = os.path.splitext(fl)[0]
    flopt = "%s_opt.pth" % fl
    return flopt


def get_stats_path(fl):
    fl = os.path.splitext(fl)[0]
    flstats = "%s_stats.pkl" % fl
    return flstats


def save_model(model, stats, fl, optimizer=None, cfg=None):
    flstats = get_stats_path(fl)
    flmodel = get_model_path(fl)
    print("saving model to %s" % flmodel)
    torch.save(model.state_dict(), flmodel)
    if optimizer is not None:
        flopt = get_optimizer_path(fl)
        print("saving optimizer to %s" % flopt)
        torch.save(optimizer.state_dict(), flopt)
    print("saving model stats and cfg to %s" % flstats)
    pickle.dump((stats, cfg), open(flstats, 'wb'))


def load_model(fl):
    flstats = get_stats_path(fl)
    flmodel = get_model_path(fl)
    flopt = get_optimizer_path(fl)
    model_state_dict = torch.load(flmodel, map_location='cpu')
    stats = load_stats(flstats)
    if os.path.isfile(flopt):
        optimizer = torch.load(flopt, map_location='cpu')
    else:
        optimizer = None

    return model_state_dict, stats, optimizer


def get_checkpoint(exp_dir, epoch):
    fl = os.path.join(exp_dir, 'model_epoch_%08d.pth' % epoch)
    return fl


def find_last_checkpoint(exp_dir, any_path=False):

    if any_path:
        exts = ['.pth', '_stats.pkl', '_opt.pth']
    else:
        exts = ['.pth']

    for ext in exts:
        fls = sorted(glob.glob(os.path.join(
            exp_dir, 'model_epoch_'+'[0-9]'*8+ext)))
        if len(fls) > 0:
            break

    if len(fls) == 0:
        fl = None
    else:
        fl = fls[-1][0:-len(ext)] + '.pth'

    return fl


def purge_epoch(exp_dir, epoch):
    model_path = get_checkpoint(exp_dir, epoch)
    to_kill = [model_path,
               get_optimizer_path(model_path),
               get_stats_path(model_path)]

    for k in to_kill:
        if os.path.isfile(k):
            print('deleting %s' % k)
            os.remove(k)


def download_model(model_name, force_download=False):
    import urllib.request
    from dataset.dataset_configs import MODEL_URL, MODEL_MD5, EXP_ROOT
    from tools.utils import md5

    exp_name = 'pretrained_%s' % model_name

    outdir = os.path.join(EXP_ROOT, exp_name)
    if os.path.isdir(outdir) and not force_download:
        return outdir

    os.makedirs(outdir, exist_ok=True)

    url = MODEL_URL[model_name]
    print('downloading model %s from %s' % (model_name, url))

    model_file = get_checkpoint(outdir, 0)

    try:
        urllib.request.urlretrieve(url, model_file)
    except:
        if os.path.isfile(model_file):
            os.remove(model_file)
        raise BaseException('cant download %s' % model_file)

    assert md5(model_file) == MODEL_MD5[model_name], 'bad md5!'

    # copy the yaml config from our ./cfgs/
    cfg_file_src = './cfgs/%s.yaml' % model_name
    with open(cfg_file_src, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # overwrite some important fields
    cfg['exp_dir'] = outdir
    cfg['exp_name'] = exp_name
    cfg_file_dst = os.path.join(outdir, 'expconfig.yaml')
    print('dumping to %s' % cfg_file_dst)
    with open(cfg_file_dst, 'w') as f:
        yaml.dump(cfg, f)
    assert os.path.isfile(cfg_file_dst)

    return outdir
