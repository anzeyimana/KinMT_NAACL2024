# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch DataLoader for TFRecords"""

import torch

from torch.optim.lr_scheduler import _LRScheduler
import math

class AnnealingLR(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    DECAY_STYLES = ['linear', 'cosine', 'exponential', 'constant', 'None']

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters, decay_style=None, last_iter=-1, min_lr=1e-6):
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.num_iters = last_iter + 1
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None
        self.step(self.num_iters)
        self.gamma = 0.995
        self.iters_to_steps = 80.0

    def get_lr(self):
        # https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            lr = float(self.start_lr) * self.num_iters / self.warmup_iter
        else:
            if self.decay_style == self.DECAY_STYLES[0]:
                lr = self.start_lr*((self.end_iter-(self.num_iters-self.warmup_iter))/self.end_iter)
            elif self.decay_style == self.DECAY_STYLES[1]:
                if (self.num_iters - self.warmup_iter) < self.end_iter:
                    lr = self.start_lr / 2.0 * (math.cos(math.pi * (self.num_iters - self.warmup_iter) / self.end_iter) + 1)
                else:
                    lr = self.min_lr
            elif self.decay_style == self.DECAY_STYLES[2]:
                lr = self.start_lr * pow(self.gamma, (self.num_iters - self.warmup_iter) / self.iters_to_steps)
            else:
                lr = self.start_lr
        if lr < self.min_lr:
            return self.min_lr
        return lr

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def state_dict(self):
        sd = {
                'start_lr': self.start_lr,
                'min_lr': self.min_lr,
                'warmup_iter': self.warmup_iter,
                'num_iters': self.num_iters,
                'decay_style': self.decay_style,
                'end_iter': self.end_iter
        }
        return sd

    def load_state_dict(self, sd):
        self.start_lr = sd['start_lr']
        if 'min_lr' in sd:
            self.min_lr = sd['min_lr']
        self.warmup_iter = sd['warmup_iter']
        self.num_iters = sd['num_iters']
        self.end_iter = sd['end_iter']
        self.decay_style = sd['decay_style']
        self.step(self.num_iters)
class InverseSQRT_LRScheduler(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    def __init__(self, optimizer, start_lr=0.0005, warmup_iter=4000, num_iters=200000, warmup_init_lr=1e-8, last_iter=0):
        self.optimizer = optimizer
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = start_lr * warmup_iter**0.5
        self.num_iters = last_iter + 1
        # initial learning rate
        self.lr = warmup_init_lr
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.end_iter = num_iters
        self.step(self.num_iters)
        #print('learning rate decaying', decay_style)

    def get_lr(self):
        # https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return float(self.start_lr) * self.num_iters / self.warmup_iter
        else:
            return self.decay_factor * self.num_iters**-0.5

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def state_dict(self):
        sd = {
                'start_lr': self.start_lr,
                'warmup_iter': self.warmup_iter,
                'num_iters': self.num_iters,
                'end_iter': self.end_iter
        }
        return sd

    def load_state_dict(self, sd):
        self.start_lr = sd['start_lr']
        self.warmup_iter = sd['warmup_iter']
        self.num_iters = sd['num_iters']
        self.end_iter = sd['end_iter']
        self.step(self.num_iters)
