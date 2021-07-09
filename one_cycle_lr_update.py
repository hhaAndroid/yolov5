from mmcv.runner import LrUpdaterHook
from mmcv.runner.hooks import HOOKS

import math
import numpy as np


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


class OneCycleLrUpdaterHook(LrUpdaterHook):
    def __init__(self, **kwargs):
        super(OneCycleLrUpdaterHook, self).__init__(**kwargs)
        self.warmup_iters = 1000
        self.xi = [0, self.warmup_iters]
        self.warmup_bias_lr = 0.1
        self.warmup_epochs = 3
        self.warmup_momentum = 0.8
        self.momentum = 0.937
        self.lf = one_cycle(1, 0.2, 300)
        self.warmup_end = False

    def before_train_iter(self, runner):

        warmup_total_iters = max(round(self.warmup_epochs * runner.nb), 1000)

        cur_iters = runner.iter
        cur_epoch = runner.epoch
        optimizer = runner.optimizer
        xi = [0, warmup_total_iters]
        if cur_iters <= warmup_total_iters:
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(cur_iters, xi,
                                    [self.warmup_bias_lr if j == 2 else 0.0, self.base_lr[j] * self.lf(cur_epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(cur_iters, xi, [self.warmup_momentum, self.momentum])
            print('mmdet=', cur_iters, [x['lr'] for x in optimizer.param_groups])
        else:
            self.warmup_end = True

    def after_train_epoch(self, runner):
        if self.warmup_end:
            cur_epoch = runner.epoch
            optimizer = runner.optimizer
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = self.base_lr[j]*self.lf(cur_epoch)
            print('mmdet=', 1, [x['lr'] for x in optimizer.param_groups])
