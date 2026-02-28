"""
Optimizers and schedulers for Exphormer_Max.
Adapted from graphgps/optimizer/extra_optimizers.py — register decorators removed.
"""
import math
import torch.optim as optim
from torch.optim import AdamW, Adam, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_optimizer(model_params, cfg):
    """Create optimizer from cfg."""
    opt_name = cfg.optim.optimizer.lower()
    lr = cfg.optim.base_lr
    wd = cfg.optim.weight_decay

    if opt_name == 'adamw':
        return AdamW(model_params, lr=lr, weight_decay=wd)
    elif opt_name == 'adam':
        return Adam(model_params, lr=lr, weight_decay=wd)
    elif opt_name == 'adagrad':
        return Adagrad(model_params, lr=lr, weight_decay=wd)
    elif opt_name == 'sgd':
        return optim.SGD(model_params, lr=lr, weight_decay=wd,
                         momentum=cfg.optim.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, cfg):
    """Create LR scheduler from cfg."""
    sched = cfg.optim.scheduler

    if sched == 'cosine_with_warmup':
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.optim.num_warmup_epochs,
            num_training_steps=cfg.optim.max_epoch)

    elif sched == 'linear_with_warmup':
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.optim.num_warmup_epochs,
            num_training_steps=cfg.optim.max_epoch)

    elif sched in ('reduce_on_plateau', 'plateau'):
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=cfg.optim.reduce_factor,
            patience=cfg.optim.schedule_patience,
            min_lr=cfg.optim.min_lr,
        )
        # ReduceLROnPlateau lacks get_last_lr; add it for compatibility
        if not hasattr(scheduler, 'get_last_lr'):
            def get_last_lr(self):
                return self._last_lr
            scheduler.get_last_lr = get_last_lr.__get__(scheduler)
            scheduler._last_lr = [g['lr'] for g in optimizer.param_groups]
        return scheduler

    elif sched == 'none':
        return get_constant_schedule(optimizer)

    else:
        raise ValueError(f"Unknown scheduler: {sched}")


# ---------------------------------------------------------------------------
# LambdaLR implementations (HuggingFace-style)
# ---------------------------------------------------------------------------

def get_constant_schedule(optimizer, last_epoch=-1):
    return optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1, last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                    num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps,
                                    num_training_steps, num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
