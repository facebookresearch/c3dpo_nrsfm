"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from itertools import cycle
from collections.abc import Iterable
from tools.vis_utils import get_visdom_connection


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.history = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, epoch=0):

        # make sure the history is of the same len as epoch
        while len(self.history) <= epoch:
            self.history.append([])

        self.history[epoch].append(val / n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_epoch_averages(self, epoch=-1):
        if len(self.history) == 0:  # no stats here
            return None
        elif epoch == -1:
            return [float(np.array(x).mean()) for x in self.history]
        else:
            return float(np.array(self.history[epoch]).mean())

    def get_all_values(self):
        all_vals = [np.array(x) for x in self.history]
        all_vals = np.concatenate(all_vals)
        return all_vals

    def get_epoch(self):
        return len(self.history)


class Stats(object):
    """
    stats logging object useful for gathering statistics of training
    a deep net in pytorch

    Example:

    # init stats structure that logs statistics 'objective' and 'top1e'
    stats = Stats( ('objective','top1e') )

    network = init_net() # init a pytorch module (=nueral network)
    dataloader = init_dataloader() # init a dataloader

    for epoch in range(10):

        # start of epoch -> call new_epoch
        stats.new_epoch()

        # iterate over batches
        for batch in dataloader:

            # run network and save into a dict of output variables "output"
            output = network(batch)

            # stats.update() automatically parses the 'objective' and 'top1e'
            # from the "output" dict and stores this into the db
            stats.update(output)
            stats.print() # prints the averages over given epoch

        # stores the training plots into '/tmp/epoch_stats.pdf'
        # and plots into a visdom server running at localhost (if running)
        stats.plot_stats(plot_file='/tmp/epoch_stats.pdf')
    """

    def __init__(self, log_vars, verbose=False,
                 epoch=-1, visdom_env='main',
                 do_plot=True, plot_file=None,
                 visdom_server='http://localhost',
                 visdom_port=8097):

        self.verbose = verbose
        self.log_vars = log_vars
        self.visdom_env = visdom_env
        self.visdom_server = visdom_server
        self.visdom_port = visdom_port
        self.plot_file = plot_file
        self.do_plot = do_plot
        self.hard_reset(epoch=epoch)

    # some sugar to be used with "with stats:" at the beginning of the epoch
    def __enter__(self):
        if self.do_plot and self.epoch >= 0:
            self.plot_stats(self.visdom_env)
        self.new_epoch()

    def __exit__(self, type, value, traceback):
        iserr = not(type is None) and issubclass(type, Exception)
        iserr = iserr or (type is KeyboardInterrupt)
        if iserr:
            print("error inside 'with' block")
            return
        if self.do_plot:
            self.plot_stats(self.visdom_env)

    def reset(self):  # to be called after each epoch
        stat_sets = list(self.stats.keys())
        if self.verbose:
            print("stats: epoch %d - reset" % self.epoch)
        self.it = {k: -1 for k in stat_sets}
        for stat_set in stat_sets:
            for stat in self.stats[stat_set]:
                self.stats[stat_set][stat].reset()

    def hard_reset(self, epoch=-1):  # to be called during object __init__
        self.epoch = epoch
        if self.verbose:
            print("stats: epoch %d - hard reset" % self.epoch)
        self.stats = {}

        # reset
        self.reset()

    def new_epoch(self):
        if self.verbose:
            print("stats: new epoch %d" % (self.epoch+1))
        self.epoch += 1
        self.reset()  # zero the stats + increase epoch counter

    def gather_value(self, val):
        if type(val) == float:
            pass
        else:
            val = val.data.cpu().numpy()
            val = float(val.sum())
        return val

    def update(self, preds, time_start=None,
               freeze_iter=False, stat_set='train'):

        if self.epoch == -1:  # uninitialized
            print(
                "warning: epoch==-1 means uninitialized stats structure\
                 -> new_epoch() called")
            self.new_epoch()

        if stat_set not in self.stats:
            self.stats[stat_set] = {}
            self.it[stat_set] = -1

        if not freeze_iter:
            self.it[stat_set] += 1

        epoch = self.epoch
        it = self.it[stat_set]

        for stat in self.log_vars:

            if stat not in self.stats[stat_set]:
                self.stats[stat_set][stat] = AverageMeter()

            if stat == 'sec/it':  # compute speed
                if time_start is None:
                    elapsed = 0.
                else:
                    elapsed = time.time() - time_start
                time_per_it = float(elapsed) / float(it+1)
                val = time_per_it
            else:
                if stat in preds:
                    try:
                        val = self.gather_value(preds[stat])
                    except:
                        raise ValueError("could not extract prediction %s\
                                          from the prediction dictionary" %
                                         stat)
                else:
                    val = None

            if val is not None:
                self.stats[stat_set][stat].update(val, epoch=epoch, n=1)

    def get_epoch_averages(self, epoch=None):

        stat_sets = list(self.stats.keys())

        if epoch is None:
            epoch = self.epoch
        if epoch == -1:
            epoch = list(range(self.epoch))

        outvals = {}
        for stat_set in stat_sets:
            outvals[stat_set] = {'epoch': epoch,
                                 'it': self.it[stat_set],
                                 'epoch_max': self.epoch}

            for stat in self.stats[stat_set].keys():
                if self.stats[stat_set][stat].count == 0:
                    continue
                if isinstance(epoch, Iterable):
                    avgs = self.stats[stat_set][stat].get_epoch_averages()
                    avgs = [avgs[e] for e in epoch]
                else:
                    avgs = self.stats[stat_set][stat].get_epoch_averages(
                        epoch=epoch)
                outvals[stat_set][stat] = avgs

        return outvals

    def print(self, max_it=None, stat_set='train',
              vars_print=None, get_str=False):

        epoch = self.epoch
        stats = self.stats

        str_out = ""

        it = self.it[stat_set]
        stat_str = ""
        stats_print = sorted(stats[stat_set].keys())
        for stat in stats_print:
            if stats[stat_set][stat].count == 0:
                continue
            stat_str += " {0:.12}: {1:1.3f} |".format(
                stat, stats[stat_set][stat].avg)

        head_str = "[%s] | epoch %3d | it %5d" % (stat_set, epoch, it)
        if max_it:
            head_str += "/ %d" % max_it

        str_out = "%s | %s" % (head_str, stat_str)

        if get_str:
            return str_out
        else:
            print(str_out)

    def plot_stats(self, visdom_env=None, plot_file=None,
                   visdom_server=None, visdom_port=None):

        # use the cached visdom env if none supplied
        if visdom_env is None:
            visdom_env = self.visdom_env
        if visdom_server is None:
            visdom_server = self.visdom_server
        if visdom_port is None:
            visdom_port = self.visdom_port
        if plot_file is None:
            plot_file = self.plot_file

        stat_sets = list(self.stats.keys())

        print("printing charts to visdom env '%s' (%s:%d)" %
              (visdom_env, visdom_server, visdom_port))

        novisdom = False

        viz = get_visdom_connection(server=visdom_server, port=visdom_port)
        if not viz.check_connection():
            print("no visdom server! -> skipping visdom plots")
            novisdom = True

        lines = []

        # plot metrics
        if not novisdom:
            viz.close(env=visdom_env, win=None)

        for stat in self.log_vars:
            vals = []
            stat_sets_now = []
            for stat_set in stat_sets:
                val = self.stats[stat_set][stat].get_epoch_averages()
                if val is None:
                    continue
                else:
                    val = np.array(val)[:, None]
                    stat_sets_now.append(stat_set)
                vals.append(val)

            if len(vals) == 0:
                continue

            vals = np.concatenate(vals, axis=1)
            x = np.arange(vals.shape[0])

            lines.append((stat_sets_now, stat, x, vals,))

        if not novisdom:
            for idx, (tmodes, stat, x, vals) in enumerate(lines):
                title = "%s" % stat
                opts = dict(title=title, legend=list(tmodes))
                if vals.shape[1] == 1:
                    vals = vals[:, 0]
                viz.line(Y=vals, X=x, env=visdom_env, opts=opts)

        if plot_file:
            print("exporting stats to %s" % plot_file)
            ncol = 3
            nrow = int(np.ceil(float(len(lines))/ncol))
            matplotlib.rcParams.update({'font.size': 5})
            color = cycle(plt.cm.tab10(np.linspace(0, 1, 10)))
            fig = plt.figure(1)
            plt.clf()
            for idx, (tmodes, stat, x, vals) in enumerate(lines):
                c = next(color)
                plt.subplot(nrow, ncol, idx+1)
                for vali, vals_ in enumerate(vals.T):
                    c_ = c * (1. - float(vali) * 0.3)
                    plt.plot(x, vals_, c=c_, linewidth=1)
                plt.ylabel(stat)
                plt.xlabel("epoch")
                plt.gca().yaxis.label.set_color(c[0:3]*0.75)
                plt.legend(tmodes)
                gcolor = np.array(mcolors.to_rgba('lightgray'))
                plt.grid(b=True, which='major', color=gcolor,
                         linestyle='-', linewidth=0.4)
                plt.grid(b=True, which='minor', color=gcolor,
                         linestyle='--', linewidth=0.2)
                plt.minorticks_on()

            plt.tight_layout()
            plt.show()
            fig.savefig(plot_file)

    def synchronize_logged_vars(self, log_vars, default_val=float('NaN')):

        stat_sets = list(self.stats.keys())

        # remove the additional log_vars
        for stat_set in stat_sets:
            for stat in self.stats[stat_set].keys():
                if stat not in log_vars:
                    print("additional stat %s:%s -> removing" %
                          (stat_set, stat))

            self.stats[stat_set] = {
                stat: v for stat, v in self.stats[stat_set].items()
                if stat in log_vars
            }

        self.log_vars = log_vars  # !!!

        for stat_set in stat_sets:
            reference_stat = list(self.stats[stat_set].keys())[0]
            for stat in log_vars:
                if stat not in self.stats[stat_set]:
                    print("missing stat %s:%s -> filling with default values (%1.2f)" %
                          (stat_set, stat, default_val))
                elif len(self.stats[stat_set][stat].history) != self.epoch+1:
                    h = self.stats[stat_set][stat].history
                    if len(h) == 0:  # just never updated stat ... skip
                        continue
                    else:
                        print("incomplete stat %s:%s -> reseting with default values (%1.2f)" %
                              (stat_set, stat, default_val))
                else:
                    continue

                self.stats[stat_set][stat] = AverageMeter()
                self.stats[stat_set][stat].reset()

                lastep = self.epoch+1
                for ep in range(lastep):
                    self.stats[stat_set][stat].update(
                        default_val, n=1, epoch=ep)
                epoch_self = self.stats[stat_set][reference_stat].get_epoch()
                epoch_generated = self.stats[stat_set][stat].get_epoch()
                assert epoch_self == epoch_generated, \
                    "bad epoch of synchronized log_var! %d vs %d" % \
                    (epoch_self, epoch_generated)
