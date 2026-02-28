"""
Custom training loop for Exphormer_Max.
Replaces graphgps/train/custom_train.py — no graphgym dependencies.
"""
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from loss.losses import compute_loss


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def _is_eval_epoch(cur_epoch, eval_period, max_epoch):
    return (cur_epoch % eval_period == 0) or (cur_epoch == max_epoch - 1)


def _is_ckpt_epoch(cur_epoch, ckpt_period):
    return cur_epoch % ckpt_period == 0


# ---------------------------------------------------------------------------
# ogbn-arxiv special cross-entropy
# ---------------------------------------------------------------------------

def _arxiv_cross_entropy(pred, true, split_idx):
    true = true.squeeze(-1)
    pred_score = F.log_softmax(pred[split_idx], dim=-1)
    loss = F.nll_loss(pred_score, true[split_idx])
    return loss, pred_score


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_ckpt(model, optimizer, scheduler, epoch, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, 'ckpt.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, ckpt_path)


def _load_ckpt(model, optimizer, scheduler, run_dir, epoch_resume=-1):
    ckpt_path = os.path.join(run_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        return 0
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    logging.info(f'Resumed from checkpoint at epoch {ckpt["epoch"]}')
    return start_epoch


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------

def _flatten_dict(perf):
    """Flatten list-of-lists perf into a flat dict for wandb logging."""
    prefixes = ['train', 'val', 'test']
    result = {}
    for i, prefix in enumerate(prefixes[:len(perf)]):
        if perf[i]:
            stats = perf[i][-1]
            result.update({f"{prefix}/{k}": v for k, v in stats.items()})
    return result


def _cfg_to_dict(cfg_node):
    from yacs.config import CfgNode
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    cfg_dict = dict(cfg_node)
    for k, v in cfg_dict.items():
        cfg_dict[k] = _cfg_to_dict(v)
    return cfg_dict


def _make_wandb_name(cfg):
    dataset_name = cfg.dataset.format
    if dataset_name.startswith('OGB'):
        dataset_name = dataset_name[3:]
    if dataset_name.startswith('PyG-'):
        dataset_name = dataset_name[4:]
    if dataset_name in ['GNNBenchmarkDataset', 'TUDataset']:
        dataset_name = ''
    if cfg.dataset.name != 'none':
        dataset_name += '-' if dataset_name else ''
        dataset_name += cfg.dataset.name
    model_name = cfg.model.type
    model_name += f'.{cfg.name_tag}' if cfg.name_tag else ''
    run_id = getattr(cfg, 'run_id', 0)
    return f'{dataset_name}.{model_name}.r{run_id}'


# ---------------------------------------------------------------------------
# Single-epoch passes
# ---------------------------------------------------------------------------

def train_epoch(logger, loader, model, optimizer, scheduler, cfg):
    model.train()
    optimizer.zero_grad()
    device = torch.device(cfg.device)
    batch_accumulation = cfg.optim.batch_accumulation
    time_start = time.time()

    for iter_idx, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(device)

        pred, true = model(batch)

        if cfg.dataset.name == 'ogbn-arxiv':
            split_idx = loader.dataset.split_idx['train'].to(device)
            loss, pred_score = _arxiv_cross_entropy(pred, true, split_idx)
            _true = true[split_idx].detach().cpu()
            _pred = pred_score.detach().cpu()
        else:
            loss, pred_score = compute_loss(pred, true, cfg.model.loss_fun)
            _true = true.detach().cpu()
            _pred = pred_score.detach().cpu()

        loss.backward()

        if ((iter_idx + 1) % batch_accumulation == 0) or \
                (iter_idx + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=getattr(cfg, 'params', 0),
            dataset_name=cfg.dataset.name,
        )
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split, cfg):
    model.eval()
    device = torch.device(cfg.device)
    time_start = time.time()

    for batch in loader:
        batch.split = split
        batch.to(device)

        pred, true = model(batch)

        if cfg.dataset.name == 'ogbn-arxiv':
            index_split = loader.dataset.split_idx[split].to(device)
            loss, pred_score = _arxiv_cross_entropy(pred, true, index_split)
            _true = true[index_split].detach().cpu()
            _pred = pred_score.detach().cpu()
        else:
            loss, pred_score = compute_loss(pred, true, cfg.model.loss_fun)
            _true = true.detach().cpu()
            _pred = pred_score.detach().cpu()

        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=0,
            time_used=time.time() - time_start,
            params=getattr(cfg, 'params', 0),
            dataset_name=cfg.dataset.name,
        )
        time_start = time.time()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def custom_train(loggers, loaders, model, optimizer, scheduler, cfg):
    """
    Full training loop.

    Args:
        loggers: [train_logger, val_logger, test_logger]
        loaders: [train_loader, val_loader, test_loader]
        model:   nn.Module
        optimizer, scheduler: PyTorch optimizer/scheduler
        cfg:     YACS CfgNode
    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = _load_ckpt(model, optimizer, scheduler,
                                  cfg.run_dir, cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found — task already done.')
        return

    logging.info('Starting from epoch %d', start_epoch)

    # WandB setup
    wandb_run = None
    if cfg.wandb.use:
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb is not installed. '
                              'Install it or set wandb.use False.')
        wandb_name = cfg.wandb.name or _make_wandb_name(cfg)
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=wandb_name,
        )
        wandb_run.config.update(_cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    eval_period = cfg.train.eval_period
    max_epoch = cfg.optim.max_epoch

    for cur_epoch in range(start_epoch, max_epoch):
        t0 = time.perf_counter()

        # --- Train ---
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, cfg)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        # --- Val / Test (on eval epochs) ---
        if _is_eval_epoch(cur_epoch, eval_period, max_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1], cfg=cfg)
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1] if perf[i] else {})

        # --- Scheduler step ---
        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1].get('loss', 0))
        else:
            scheduler.step()

        full_epoch_times.append(time.perf_counter() - t0)

        # --- Regular checkpoint ---
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best and \
                _is_ckpt_epoch(cur_epoch, cfg.train.ckpt_period):
            _save_ckpt(model, optimizer, scheduler, cur_epoch, cfg.run_dir)

        # --- WandB log ---
        if wandb_run is not None:
            wandb_run.log(_flatten_dict(perf), step=cur_epoch)

        # --- Best-epoch summary (on eval epochs) ---
        if _is_eval_epoch(cur_epoch, eval_period, max_epoch) and val_perf:
            best_epoch = int(np.array([vp.get('loss', 0)
                                       for vp in val_perf]).argmin())
            best_train = best_val = best_test = ''
            if cfg.metric_best != 'auto':
                m = cfg.metric_best
                arr = np.array([vp.get(m, 0) for vp in val_perf])
                best_epoch = int(getattr(arr, cfg.metric_agg)())
                if m in perf[0][best_epoch]:
                    best_train = f'train_{m}: {perf[0][best_epoch][m]:.4f}'
                if len(perf) > 1 and m in perf[1][best_epoch]:
                    best_val = f'val_{m}: {perf[1][best_epoch][m]:.4f}'
                if len(perf) > 2 and perf[2] and m in perf[2][best_epoch]:
                    best_test = f'test_{m}: {perf[2][best_epoch][m]:.4f}'

                if wandb_run is not None:
                    bstats = {'best/epoch': best_epoch}
                    for i, s in enumerate(['train', 'val', 'test'][:num_splits]):
                        if perf[i]:
                            bstats[f'best/{s}_loss'] = \
                                perf[i][best_epoch].get('loss', 0)
                            if m in perf[i][best_epoch]:
                                bstats[f'best/{s}_{m}'] = \
                                    perf[i][best_epoch][m]
                                wandb_run.summary[f'best_{s}_perf'] = \
                                    perf[i][best_epoch][m]
                    wandb_run.log(bstats, step=cur_epoch)
                    wandb_run.summary['full_epoch_time_avg'] = \
                        np.mean(full_epoch_times)
                    wandb_run.summary['full_epoch_time_sum'] = \
                        np.sum(full_epoch_times)

            # Best-epoch checkpoint
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                _save_ckpt(model, optimizer, scheduler, cur_epoch, cfg.run_dir)

            best_train_loss = perf[0][best_epoch].get('loss', float('nan'))
            best_val_loss = (perf[1][best_epoch].get('loss', float('nan'))
                             if perf[1] else float('nan'))
            best_test_loss = (perf[2][best_epoch].get('loss', float('nan'))
                              if len(perf) > 2 and perf[2] else float('nan'))
            logging.info(
                f'> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s '
                f'(avg {np.mean(full_epoch_times):.1f}s) | '
                f'Best so far: epoch {best_epoch}\t'
                f'train_loss: {best_train_loss:.4f} {best_train}\t'
                f'val_loss: {best_val_loss:.4f} {best_val}\t'
                f'test_loss: {best_test_loss:.4f} {best_test}'
            )

    logging.info(f'Avg time per epoch: {np.mean(full_epoch_times):.2f}s')
    logging.info(
        f'Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h')
    for logger in loggers:
        logger.close()

    if wandb_run is not None:
        wandb_run.finish()

    logging.info('Task done, results saved in %s', cfg.run_dir)
