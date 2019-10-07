"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import time
import copy
import json

import numpy as np
import torch

from dataset.dataset_zoo import dataset_zoo
from dataset.eval_zoo import eval_zoo
from model import C3DPO

from config import set_config_from_file, set_config, \
    get_arg_parser, dump_config, get_default_args

from tools.attr_dict import nested_attr_dict
from tools.utils import auto_init_args, get_net_input, pprint_dict
from tools.stats import Stats
from tools.vis_utils import get_visdom_env
from tools.model_io import find_last_checkpoint, purge_epoch, \
    load_model, get_checkpoint, save_model
from tools.cache_preds import cache_preds


def init_model_from_dir(exp_dir):
    cfg_file = os.path.join(exp_dir, 'expconfig.yaml')
    if not os.path.isfile(cfg_file):
        print('no config %s!' % cfg_file)
        return None
    exp = ExperimentConfig(cfg_file=cfg_file)
    exp.cfg.exp_dir = exp_dir  # !
    cfg = exp.cfg

    # init the model
    model, _, _ = init_model(cfg, force_load=True, clear_stats=True)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    return model, cfg


def init_model(cfg, force_load=False, clear_stats=False, add_log_vars=None):

    # get the model
    model = C3DPO(**cfg.MODEL)

    # obtain the network outputs that should be logged
    if hasattr(model, 'log_vars'):
        log_vars = copy.deepcopy(model.log_vars)
    else:
        log_vars = ['objective']
    if add_log_vars is not None:
        log_vars.extend(copy.deepcopy(add_log_vars))

    visdom_env_charts = get_visdom_env(cfg) + "_charts"

    # init stats struct
    stats = Stats(log_vars, visdom_env=visdom_env_charts,
                  verbose=False, visdom_server=cfg.visdom_server,
                  visdom_port=cfg.visdom_port)

    # find the last checkpoint
    if cfg.resume_epoch > 0:
        model_path = get_checkpoint(cfg.exp_dir, cfg.resume_epoch)
    else:
        model_path = find_last_checkpoint(cfg.exp_dir)

    optimizer_state = None

    if model_path is not None:
        print("found previous model %s" % model_path)
        if force_load or cfg.resume:
            print("   -> resuming")
            model_state_dict, stats_load, optimizer_state = load_model(
                model_path)
            if not clear_stats:
                stats = stats_load
            else:
                print("   -> clearing stats")
            model.load_state_dict(model_state_dict, strict=True)
            model.log_vars = log_vars
        else:
            print("   -> but not resuming -> starting from scratch")

    # update in case it got lost during load:
    stats.visdom_env = visdom_env_charts
    stats.visdom_server = cfg.visdom_server
    stats.visdom_port = cfg.visdom_port
    stats.plot_file = os.path.join(cfg.exp_dir, 'train_stats.pdf')
    stats.synchronize_logged_vars(log_vars)

    return model, stats, optimizer_state


def init_optimizer(model, optimizer_state,
                   PARAM_GROUPS=(),
                   freeze_bn=False,
                   breed='sgd',
                   weight_decay=0.0005,
                   lr_policy='multistep',
                   lr=0.001,
                   gamma=0.1,
                   momentum=0.9,
                   betas=(0.9, 0.999),
                   milestones=[30, 37, ],
                   max_epochs=43,
                   ):

    # init the optimizer
    if hasattr(model, '_get_param_groups') and model.custom_param_groups:
        # use the model function
        p_groups = model._get_param_groups(lr, wd=weight_decay)
    else:
        allprm = [prm for prm in model.parameters() if prm.requires_grad]
        p_groups = [{'params': allprm, 'lr': lr}]

    if breed == 'sgd':
        optimizer = torch.optim.SGD(p_groups, lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    elif breed == 'adagrad':
        optimizer = torch.optim.Adagrad(p_groups, lr=lr,
                                        weight_decay=weight_decay)

    elif breed == 'adam':
        optimizer = torch.optim.Adam(p_groups, lr=lr,
                                     betas=betas,
                                     weight_decay=weight_decay)

    else:
        raise ValueError("no such solver type %s" % breed)
    print("  -> solver type = %s" % breed)

    if lr_policy == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma)
    else:
        raise ValueError("no such lr policy %s" % lr_policy)

    # add the max epochs here!
    scheduler.max_epochs = max_epochs

    if optimizer_state is not None:
        print("  -> setting loaded optimizer state")
        optimizer.load_state_dict(optimizer_state)

    optimizer.zero_grad()

    return optimizer, scheduler


def run_training(cfg):
    """
    run the training loops
    """

    # torch gpu setup
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_idx)
    if cfg.model_zoo is not None:
        os.environ["TORCH_MODEL_ZOO"] = cfg.model_zoo

    # make the exp dir
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # set the seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # set cudnn to reproducibility mode
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # dump the exp config to the exp dir
    dump_config(cfg)

    # setup datasets
    dset_train, dset_val, dset_test = dataset_zoo(**cfg.DATASET)

    # init loaders
    trainloader = torch.utils.data.DataLoader(dset_train,
                                              num_workers=cfg.num_workers,
                                              pin_memory=True,
                                              batch_size=cfg.batch_size,
                                              shuffle=True)

    if dset_val is not None:
        valloader = torch.utils.data.DataLoader(dset_val,
                                                num_workers=cfg.num_workers,
                                                pin_memory=True,
                                                batch_size=cfg.batch_size,
                                                shuffle=False)
    else:
        valloader = None

    # test loaders
    if dset_test is not None:
        testloader = torch.utils.data.DataLoader(dset_test,
                                                 num_workers=cfg.num_workers,
                                                 pin_memory=True,
                                                 batch_size=cfg.batch_size,
                                                 shuffle=False)
        _, _, eval_vars = eval_zoo(cfg.DATASET.dataset_name)
    else:
        testloader = None
        eval_vars = None

    # init the model
    model, stats, optimizer_state = init_model(cfg, add_log_vars=eval_vars)
    start_epoch = stats.epoch + 1

    # move model to gpu
    if torch.cuda.is_available():
        model.cuda()

    # init the optimizer
    optimizer, scheduler = init_optimizer(
        model, optimizer_state=optimizer_state, **cfg.SOLVER)

    # loop through epochs
    scheduler.last_epoch = start_epoch
    for epoch in range(start_epoch, cfg.SOLVER.max_epochs):
        with stats:  # automatic new_epoch and plotting at every epoch start

            print("scheduler lr = %1.2e" % float(scheduler.get_lr()[-1]))

            # train loop
            trainvalidate(model, stats, epoch, trainloader, optimizer, False,
                          visdom_env_root=get_visdom_env(cfg), **cfg)

            # val loop
            if valloader is not None:
                trainvalidate(model, stats, epoch, valloader, optimizer, True,
                              visdom_env_root=get_visdom_env(cfg), **cfg)

            # eval loop (optional)
            if testloader is not None:
                eval_result = run_eval(cfg, model, testloader, stats=stats)
                dump_eval_result(cfg, eval_result)

            assert stats.epoch == epoch, "inconsistent stats!"

            # delete previous models if required
            if cfg.store_checkpoints_purge > 0 and cfg.store_checkpoints:
                for prev_epoch in range(epoch-cfg.store_checkpoints_purge):
                    purge_epoch(cfg.exp_dir, prev_epoch)

            # save model
            if cfg.store_checkpoints:
                outfile = get_checkpoint(cfg.exp_dir, epoch)
                save_model(model, stats, outfile, optimizer=optimizer)

            scheduler.step()

    # the final eval
    if testloader is not None:
        eval_result = run_eval(cfg, model, testloader, stats=None)
        dump_eval_result(cfg, eval_result)
        return eval_result
    else:
        return None


def trainvalidate(model,
                  stats,
                  epoch,
                  loader,
                  optimizer,
                  validation,
                  bp_var='objective',
                  metric_print_interval=5,
                  visualize_interval=100,
                  visdom_env_root='trainvalidate',
                  **kwargs):

    if validation:
        model.eval()
        trainmode = 'val'
    else:
        model.train()
        trainmode = 'train'

    t_start = time.time()

    # clear the visualisations on the first run in the epoch
    clear_visualisations = True

    # get the visdom env name
    visdom_env_imgs = visdom_env_root + "_images_" + trainmode

    n_batches = len(loader)
    for it, batch in enumerate(loader):

        last_iter = it == n_batches-1

        # move to gpu where possible
        net_input = get_net_input(batch)

        # the forward pass
        if (not validation):
            optimizer.zero_grad()
            preds = model(**net_input)
        else:
            with torch.no_grad():
                preds = model(**net_input)

        # make sure we dont overwrite something
        assert not any(k in preds for k in net_input.keys())
        preds.update(net_input)  # merge everything into one big dict

        # update the stats logger
        stats.update(preds, time_start=t_start, stat_set=trainmode)
        assert stats.it[trainmode] == it, "inconsistent stat iteration number!"

        # print textual status update
        if (it % metric_print_interval) == 0 or last_iter:
            stats.print(stat_set=trainmode, max_it=n_batches)

        # visualize results
        if (visualize_interval > 0) and (it % visualize_interval) == 0:
            model.visualize(visdom_env_imgs, trainmode,
                            preds, stats, clear_env=clear_visualisations)
            clear_visualisations = False

        # optimizer step
        if (not validation):
            loss = preds[bp_var]
            loss.backward()
            optimizer.step()


def dump_eval_result(cfg, results):
    # dump results of eval to cfg.exp_dir
    resfile = os.path.join(cfg.exp_dir, 'eval_results.json')
    with open(resfile, 'w') as f:
        json.dump(results, f)


def run_eval(cfg, model, loader, stats=None):
    eval_script, cache_vars, eval_vars = eval_zoo(cfg.DATASET.dataset_name)
    cached_preds = cache_preds(
        model, loader, stats=stats, cache_vars=cache_vars)
    results, _ = eval_script(cached_preds, eval_vars=eval_vars)
    if stats is not None:
        stats.update(results, stat_set='test')
        stats.print(stat_set='test')

    return results


class ExperimentConfig(object):
    def __init__(self,
                 cfg_file=None,
                 model_zoo='./data/torch_zoo/',
                 exp_name='test',
                 exp_idx=0,
                 exp_dir='./data/exps/default/',
                 gpu_idx=0,
                 resume=True,
                 seed=0,
                 resume_epoch=-1,
                 store_checkpoints=True,
                 store_checkpoints_purge=3,
                 batch_size=256,
                 num_workers=8,
                 visdom_env='',
                 visdom_server='http://localhost',
                 visdom_port=8097,
                 metric_print_interval=5,
                 visualize_interval=0,
                 SOLVER=get_default_args(init_optimizer),
                 DATASET=get_default_args(dataset_zoo),
                 MODEL=get_default_args(C3DPO),
                 ):

        self.cfg = get_default_args(ExperimentConfig)
        if cfg_file is not None:
            set_config_from_file(self.cfg, cfg_file)
        else:
            auto_init_args(self, tgt='cfg', can_overwrite=True)
        self.cfg = nested_attr_dict(self.cfg)


def run_experiment_from_cfg_file(cfg_file):

    if not os.path.isfile(cfg_file):
        print('no config %s!' % cfg_file)
        return None

    exp = ExperimentConfig(cfg_file=cfg_file)

    results = run_training(exp.cfg)

    return results


if __name__ == '__main__':

    exp = ExperimentConfig()
    parser = get_arg_parser(type(exp))
    parsed = vars(parser.parse_args())
    if parsed['cfg_file'] is not None:
        print('setting config from cfg file %s' % parsed['cfg_file'])
        set_config_from_file(exp.cfg, parsed['cfg_file'])
        defaults = vars(parser.parse_args(''))
        rest = {k: v for k, v in parsed.items() if defaults[k] != parsed[k]}
        print('assigning remaining args: %s' % str(list(rest.keys())))
        set_config(exp.cfg, rest)
    else:
        print('setting config from argparser')
        set_config(exp.cfg, parsed)
    pprint_dict(exp.cfg)
    run_training(exp.cfg)

else:
    pass
