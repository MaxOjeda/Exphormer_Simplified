"""
Exphormer_Max entry point.
Standalone reimplementation of Exphormer — no graphgym / GraphGPS dependency.

Usage:
    python main.py --cfg configs/Exphormer/cifar10.yaml wandb.use False
    python main.py --cfg configs/Exphormer/voc.yaml --repeat 3 wandb.use False
    python main.py --cfg configs/Exphormer/cifar10.yaml \
        optim.max_epoch 1 train.batch_size 4 wandb.use False
"""
import argparse
import datetime
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# Allow imports relative to Exphormer_Max root regardless of CWD
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import cfg, load_cfg
from loader.master_loader import load_dataset, create_loaders
from loss.losses import compute_loss  # noqa: imported to confirm no errors
from network.model import create_model
from optimizer.schedulers import build_optimizer, build_scheduler
from train.logger import create_loggers
from train.trainer import custom_train


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'logging.log')
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Argument parsing (compatible with graphgym's style)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Exphormer_Max')
    parser.add_argument('--cfg', dest='cfg_file', required=True,
                        help='Path to the YAML config file')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of times to repeat the experiment')
    parser.add_argument('--mark_done', action='store_true',
                        help='Rename cfg file to *_done after completion')
    # Remaining positional overrides in the format  key value key value ...
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Optional config key-value overrides')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def auto_select_device():
    if cfg.device == 'auto':
        cfg.defrost()
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.freeze()
    logging.info(f'Device: {cfg.device}')


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def set_out_dir(cfg_fname, name_tag):
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    if name_tag:
        run_name += f'-{name_tag}'
    cfg.defrost()
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)
    cfg.freeze()


def set_run_dir(run_id):
    run_dir = os.path.join(cfg.out_dir, str(run_id))
    cfg.defrost()
    cfg.run_dir = run_dir
    cfg.freeze()
    if cfg.train.auto_resume:
        os.makedirs(run_dir, exist_ok=True)
    else:
        if os.path.exists(run_dir):
            import shutil
            shutil.rmtree(run_dir)
        os.makedirs(run_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Multi-seed / multi-split loop settings
# ---------------------------------------------------------------------------

def run_loop_settings(args):
    if len(cfg.run_multiple_splits) == 0:
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        if args.repeat != 1:
            raise NotImplementedError(
                'Multiple repeats of multiple splits not supported.')
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


# ---------------------------------------------------------------------------
# Seed / reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dim_in / Dim_out inference from dataset
# ---------------------------------------------------------------------------

def infer_dims(dataset, cfg):
    """Return (dim_in, dim_out) from the dataset."""
    data0 = dataset[0]

    task_type = cfg.dataset.task_type

    # KGC: KGCNodeEncoder creates features from scratch (ignores dim_in),
    # and KGCHead does not use dim_out. Return (1, 1) to avoid crashes from
    # dataset._data access that KGCSplitWrapper does not support.
    if task_type == 'kgc_ranking':
        return 1, 1

    if data0.x is not None:
        dim_in = data0.x.shape[-1]
    else:
        dim_in = 1  # fallback (Constant or OneHotDegree)

    if task_type == 'regression':
        # Regression: output dim is the number of target values per sample
        if data0.y is not None:
            y = data0.y.squeeze()
            dim_out = 1 if y.ndim <= 1 else y.shape[-1]
        else:
            dim_out = 1
    elif task_type == 'classification_binary':
        dim_out = 1
    else:
        # Classification: prefer dataset.num_classes, else infer from label range
        if hasattr(dataset, 'num_classes') and dataset.num_classes is not None:
            dim_out = int(dataset.num_classes)
        elif data0.y is not None:
            dim_out = int(dataset._data.y.max().item()) + 1
        else:
            dim_out = 1

    return dim_in, dim_out


# ---------------------------------------------------------------------------
# Simple aggregation across seeds
# ---------------------------------------------------------------------------

def agg_runs(out_dir, metric_best, metric_agg, round_n):
    """Aggregate JSON-lines stats files across seed sub-directories."""

    def _is_seed(s):
        try:
            int(s)
            return True
        except Exception:
            return False

    # Collect results per split
    results = {'train': [], 'val': [], 'test': []}

    for entry in os.listdir(out_dir):
        if not _is_seed(entry):
            continue
        seed_dir = os.path.join(out_dir, entry)
        for split in ['train', 'val', 'test']:
            stats_path = os.path.join(seed_dir, f'{split}_stats.json')
            if not os.path.exists(stats_path):
                continue
            with open(stats_path) as f:
                stats_list = [json.loads(line) for line in f if line.strip()]
            results[split].append(stats_list)

    if not any(results.values()):
        logging.info('agg_runs: no seed results found.')
        return

    # Find best epoch based on val performance
    val_results = results.get('val', [])
    if not val_results:
        return

    # Transpose: list-of-seeds × epochs → list-of-epochs (each a list over seeds)
    n_epochs = min(len(r) for r in val_results)
    best_epoch = 0

    if metric_best == 'auto':
        # Pick 'auc' if available, else 'accuracy'
        sample = val_results[0][0]
        metric = 'auc' if 'auc' in sample else 'accuracy'
    else:
        metric = metric_best

    metric_values = []
    for ep in range(n_epochs):
        vals = [r[ep].get(metric, float('nan')) for r in val_results
                if ep < len(r)]
        metric_values.append(np.nanmean(vals))

    if metric_agg == 'argmax':
        best_epoch = int(np.argmax(metric_values))
    else:
        best_epoch = int(np.argmin(metric_values))

    # Aggregate across seeds at best epoch
    agg_dir = os.path.join(out_dir, 'agg')
    os.makedirs(agg_dir, exist_ok=True)

    agg_result = {}
    for split, split_results in results.items():
        if not split_results:
            continue
        best_stats = [r[best_epoch] for r in split_results
                      if best_epoch < len(r)]
        if not best_stats:
            continue
        agg = {'epoch': best_stats[0]['epoch']}
        for key in best_stats[0]:
            if key == 'epoch':
                continue
            vals = [s[key] for s in best_stats if key in s]
            agg[key] = round(float(np.mean(vals)), round_n)
            agg[f'{key}_std'] = round(float(np.std(vals)), round_n)
        agg_result[split] = agg

    agg_path = os.path.join(agg_dir, 'best.json')
    with open(agg_path, 'w') as f:
        json.dump(agg_result, f, indent=2)

    logging.info(f'Aggregated results saved to {agg_path}')
    for split, stats in agg_result.items():
        logging.info(f'  {split}: {stats}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    # ---------------------------------------------------------------------------
    # Multi-GPU DDP setup
    # ---------------------------------------------------------------------------
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
    is_distributed = (LOCAL_RANK >= 0)
    if is_distributed:
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(hours=2),  # default 10min kills long eval
        )
        torch.cuda.set_device(LOCAL_RANK)
    rank = dist.get_rank() if is_distributed else 0
    is_main = (rank == 0)

    # Load config
    load_cfg(args.cfg_file, args.opts)
    set_out_dir(args.cfg_file, cfg.name_tag)
    if is_main:
        os.makedirs(cfg.out_dir, exist_ok=True)
        # Dump config once (to out_dir)
        cfg_dump_path = os.path.join(cfg.out_dir, 'config.yaml')
        with open(cfg_dump_path, 'w') as f:
            f.write(cfg.dump())
    if is_distributed:
        dist.barrier()   # ensure rank 0 has created out_dir before others proceed

    torch.set_num_threads(cfg.num_threads)

    # Multi-seed / multi-split loop
    for run_id, seed, split_index in zip(*run_loop_settings(args)):
        if is_main:
            set_run_dir(run_id)
        else:
            # Non-main ranks just update cfg.run_dir without touching the filesystem
            run_dir = os.path.join(cfg.out_dir, str(run_id))
            cfg.defrost()
            cfg.run_dir = run_dir
            cfg.freeze()
        if is_distributed:
            dist.barrier()  # ensure rank 0 has created run_dir before others

        if is_main:
            setup_logging(cfg.run_dir)
        else:
            logging.basicConfig(level=logging.WARNING, force=True)

        cfg.defrost()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        cfg.freeze()

        # Each rank uses a different seed so training draws different random batches
        seed_everything(seed + rank)
        auto_select_device()
        # DDP: override device to the local GPU assigned to this rank
        if is_distributed:
            cfg.defrost()
            cfg.device = f'cuda:{LOCAL_RANK}'
            cfg.freeze()

        logging.info(f'[*] Run ID {run_id}: seed={seed}, '
                     f'split_index={split_index}')
        logging.info(f'    Starting now: {datetime.datetime.now()}')

        # ---- Dataset + loaders ----
        dataset = load_dataset(cfg)

        dim_in, dim_out = infer_dims(dataset, cfg)
        cfg.defrost()
        cfg.share.dim_in = dim_in
        cfg.freeze()

        logging.info(f'dim_in={dim_in}, dim_out={dim_out}')

        loaders = create_loaders(cfg, dataset)

        # ---- Loggers ----
        loggers = create_loggers(cfg, out_dir=cfg.run_dir)

        # ogbn-arxiv: triplicate loggers (node classification workaround)
        if cfg.dataset.name in ('ogbn-arxiv', 'ogbn-proteins'):
            loggers_2 = create_loggers(cfg, out_dir=cfg.run_dir)
            loggers_3 = create_loggers(cfg, out_dir=cfg.run_dir)
            loggers_2[0].name = 'val'
            loggers_3[0].name = 'test'
            loggers = loggers + loggers_2 + loggers_3

        # ---- Model ----
        model = create_model(cfg, dim_in, dim_out)
        model.to(torch.device(cfg.device))
        if is_distributed:
            model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
            logging.info(f'DDP enabled on {dist.get_world_size()} GPUs.')

        # ---- Optimizer / scheduler ----
        optimizer = build_optimizer(model.parameters(), cfg)
        scheduler = build_scheduler(optimizer, cfg)

        # ---- Param count ----
        n_params = sum(p.numel() for p in model.parameters())
        cfg.defrost()
        cfg.params = n_params
        cfg.freeze()

        logging.info(model)
        logging.info(f'Num parameters: {n_params:,}')

        # ---- Train ----
        custom_train(loggers, loaders, model, optimizer, scheduler, cfg,
                     dataset=dataset)

    if is_main:
        # ---- Aggregate results across seeds ----
        try:
            agg_runs(cfg.out_dir,
                     metric_best=cfg.metric_best,
                     metric_agg=cfg.metric_agg,
                     round_n=cfg.round)
        except Exception as e:
            logging.info(f'Failed to aggregate runs: {e}')

        # ---- Mark done ----
        if args.mark_done:
            os.rename(args.cfg_file, f'{args.cfg_file}_done')

    if is_distributed:
        dist.destroy_process_group()

    logging.info(f'[*] All done: {datetime.datetime.now()}')
