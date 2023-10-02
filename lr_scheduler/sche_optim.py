# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch

import math

# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                # warmup_factor = self.warmup_factor * self.last_epoch
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def make_optimizer(optim_name, model, base_lr, weight_decay, bias_lr_factor, momentum):
    """
    :param optim_name:
    :param model:
    :param base_lr:
    :param weight_decay:
    :param bias_lr_factor:
    :param momentum:
    :return:
    """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        if "bias" in key:
            lr = base_lr * bias_lr_factor
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if optim_name == 'SGD':
        optimizer = getattr(torch.optim, optim_name)(params, momentum=momentum)
    elif optim_name == 'AdamW':
        optimizer = getattr(torch.optim, optim_name)(params)
    else:
        optimizer = getattr(torch.optim, optim_name)(params)
    return optimizer

'''
Bag of Tricks for Image Classification with Convolutional Neural Networks
'''
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        warmup_epochs=10,
        eta_min=1e-8,
        last_epoch=-1,
    ):
        self.max_epochs = max_epochs - 1
        self.eta_min=eta_min
        self.warmup_epochs = warmup_epochs
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = [base_lr * (self.last_epoch+1) / (self.warmup_epochs + 1e-32) for base_lr in self.base_lrs]
        else:
            lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]
        return lr


class CosineStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        step_epochs=2,
        gamma=0.3,
        eta_min=0,
        last_epoch=-1,
    ):
        self.max_epochs = max_epochs
        self.eta_min=eta_min
        self.step_epochs = step_epochs
        self.gamma = gamma
        self.last_cosine_lr = 0
        super(CosineStepLR, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.last_epoch < self.max_epochs - self.step_epochs:
            lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch) / (self.max_epochs - self.step_epochs))) / 2
                    for base_lr in self.base_lrs]
            self.last_cosine_lr = lr
        else:
            lr = [self.gamma ** (self.step_epochs - self.max_epochs + self.last_epoch + 1) * base_lr for base_lr in self.last_cosine_lr]

        return lr

class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer,  max_epochs, warmup_iters=10, eta_min=0, last_epoch=-1, delay=30, verbose=False):
        self.T_max = max_epochs
        self.eta_min = eta_min
        self.delay = delay
        self.warmup_epochs = warmup_iters
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
           return [base_lr * (self.last_epoch+1) / (self.warmup_epochs + 1e-32) for base_lr in self.base_lrs]
        else:
            if self.last_epoch < self.delay-1:
                return [base_lr  for base_lr in self.base_lrs]
            else:
                if self.last_epoch == 0:
                    return [group['lr'] for group in self.optimizer.param_groups]
                elif self._step_count == 1 and self.last_epoch > 0:
                    return [self.eta_min + (base_lr - self.eta_min) *
                            (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                            for base_lr, group in
                            zip(self.base_lrs, self.optimizer.param_groups)]
                elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
                    return [group['lr'] + (base_lr - self.eta_min) *
                            (1 - math.cos(math.pi / self.T_max)) / 2
                            for base_lr, group in
                            zip(self.base_lrs, self.optimizer.param_groups)]
                return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                        (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                        (group['lr'] - self.eta_min) + self.eta_min
                        for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.01,
                 min_lr : float = 7.5e-5,
                 warmup_steps : int = 3000,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def make_warmup_scheduler(scheduler_name, optimizer, max_epochs, milestones=[40,70,100], gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=3000,
                          warmup_method="linear",
                          last_epoch=-1, min_lr = 1e-7, delay=30):

    if scheduler_name =="Warm_MultiStep":
        scheduler = WarmupMultiStepLR(optimizer, milestones, gamma, warmup_factor, warmup_iters, warmup_method,
                                    last_epoch=last_epoch)
    elif scheduler_name == "Warm_Cosine":
        scheduler = WarmupCosineLR(optimizer, max_epochs, warmup_iters, last_epoch=last_epoch, eta_min=min_lr)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, max_epochs, warmup_iters, eta_min=min_lr, delay=delay)

    return scheduler
