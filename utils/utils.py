import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings

def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_optimizer(config, params):
    """Return a flax optimizer object based on `config`."""
    if config.type == 'Adam':
        optimizer = optim.Adam(params, lr=config.lr, betas=(config.beta1, config.beta1), eps=config.eps,
                               weight_decay=config.weight_decay)
    elif config.type == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.lr, amsgrad=True, weight_decay=1e-12)
    else:
        raise NotImplementedError(
            f'Optimizer {config.type} not supported yet!'
        )
    return optimizer


# customize exp lr scheduler with min lr
class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


def get_scheduler(config, optimizer):
    if config.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.factor,
            patience=config.patience,
        )
    elif config.type == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=config.factor,
            min_lr=config.min_lr,
        )
    elif config.type == 'expmin_milestone':
        gamma = np.exp(np.log(config.factor) / config.milestone)
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=gamma,
            min_lr=config.min_lr,
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % config.type)