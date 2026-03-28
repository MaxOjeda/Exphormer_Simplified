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
from torch_geometric.data import Data

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
    max_iter = cfg.train.max_iter   # 0 = unlimited; >0 stops after N batches
    time_start = time.time()

    for iter_idx, batch in enumerate(loader):
        if max_iter > 0 and iter_idx >= max_iter:
            break
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
# Full-graph expander helper
# ---------------------------------------------------------------------------

def _tile_expander(exp_ei_single, B, N, device):
    """
    Replicate a single-graph expander edge_index (2, E_exp) for B graph copies.

    Each copy i gets node offsets [i*N], producing (2, B*E_exp).
    Returns None if exp_ei_single is None.
    """
    if exp_ei_single is None:
        return None
    exp_ei = exp_ei_single.to(device)          # (2, E_exp)
    E_exp  = exp_ei.shape[1]
    offsets = (torch.arange(B, device=device) * N).repeat_interleave(E_exp)
    return exp_ei.repeat(1, B) + offsets.unsqueeze(0)  # (2, B*E_exp)


# ---------------------------------------------------------------------------
# KGC full-graph evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_epoch_kgc(logger, loader, model, split, cfg, dataset):
    """
    Full-graph KGC evaluation with filtered MRR / Hits@K.

    Batches eval_batch_size queries into a single forward pass by constructing
    a flat graph covering B × num_entities nodes (B copies of the full KG with
    different anchor/relation per copy).  KGCNodeEncoder and KGCHead handle
    variable-B batches via batch.ptr, so no model changes are needed.

    Args:
        logger:  CustomLogger — accumulates ranks via update_stats_kgc.
        loader:  unused (we iterate KGCDataset triples directly).
        model:   MultiModel (nn.Module).
        split:   'val' or 'test'.
        cfg:     YACS CfgNode — reads kgc.eval_batch_size.
        dataset: KGCSplitWrapper — provides filter_dict and full graph tensors.
    """
    model.eval()
    device = torch.device(cfg.device)
    # Release any cached GPU memory from the training step so the large eval
    # batch (B full graphs × 347k+ edges) has maximum headroom.
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    kgc_ds      = dataset.train_ds
    filter_dict  = kgc_ds.all_triples_filter   # {(h, r) -> set(known tails)}
    head_filter  = kgc_ds.head_filter          # {(t, r_orig) -> set(known heads)}
    base_num_rel = kgc_ds.num_base_relations   # for detecting reciprocal queries
    N            = kgc_ds.num_entities

    full_edge_index = kgc_ds.full_edge_index.to(device)  # (2, E)
    full_edge_attr  = kgc_ds.full_edge_attr.to(device)   # (E,)
    E = full_edge_index.shape[1]

    queries   = kgc_ds.val_triples if split == 'val' else kgc_ds.test_triples

    # Standard KGC evaluation uses BOTH tail and head prediction for each triple.
    # For head prediction of (h, r, t), we query (t, r_inv, ?) and rank h.
    # With reciprocal training, r_inv = r + num_base_relations.
    if getattr(cfg.kgc, 'reciprocal', False) and hasattr(kgc_ds, 'num_base_relations'):
        base_num_rel = kgc_ds.num_base_relations
        rec_queries = torch.stack([
            queries[:, 2],                       # t becomes anchor
            queries[:, 1] + base_num_rel,        # r_inv = r + num_base_rel
            queries[:, 0],                       # h becomes true answer
        ], dim=1)
        queries = torch.cat([queries, rec_queries], dim=0)

    n_queries = len(queries)
    eval_bs   = cfg.kgc.eval_batch_size

    logging.info(
        f'eval_epoch_kgc [{split}]: {n_queries} queries '
        f'on full graph ({N} nodes, eval_batch_size={eval_bs})'
    )

    # Precompute per-edge node offsets for the maximum batch size.
    # offset_table[j] = (j // E) * N  for j in [0, eval_bs * E).
    # For a chunk of B graphs, slice [:B*E] to get the right offsets.
    # Shape: (eval_bs * E,)  e.g. [0,...,0, N,...,N, 2N,...,2N, ...]
    offset_table = (
        torch.arange(eval_bs, device=device) * N
    ).repeat_interleave(E)                         # (eval_bs * E,)

    # Tile the static expander (generated once at dataset build time) for
    # eval_bs graph copies.  Sliced per chunk for the last batch (B < eval_bs).
    # cfg.prep.exp=False → full_expander_edge_index is None → no expander edges.
    eval_exp_ei = _tile_expander(kgc_ds.full_expander_edge_index, eval_bs, N, device)
    E_exp = kgc_ds.full_expander_edge_index.shape[1] \
        if kgc_ds.full_expander_edge_index is not None else 0

    processed  = 0
    time_start = time.time()

    for chunk_start in range(0, n_queries, eval_bs):
        chunk  = queries[chunk_start: chunk_start + eval_bs]
        B      = len(chunk)

        chunk_h = chunk[:, 0].tolist()
        chunk_r = chunk[:, 1].tolist()
        chunk_t = chunk[:, 2].tolist()

        # ---------------------------------------------------------------
        # Build a single flat Data representing B copies of the full KG.
        # Node index space: graph i owns nodes [i*N, (i+1)*N).
        # ---------------------------------------------------------------
        # edge_index: repeat full graph B times, adding per-graph offset
        rep_ei = full_edge_index.repeat(1, B)                 # (2, B*E)
        rep_ei = rep_ei + offset_table[:B * E].unsqueeze(0)   # (2, B*E)

        # edge_attr: same relation IDs, just repeated
        rep_ea = full_edge_attr.repeat(B)                      # (B*E,)

        # Batch bookkeeping
        batch_assign = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)
        ptr          = torch.arange(B + 1, device=device) * N               # (B+1,)

        data = Data(
            x              = torch.zeros(B * N, 1, device=device),
            edge_index     = rep_ei,
            edge_attr      = rep_ea,
            anchor_idx     = torch.tensor(chunk_h, dtype=torch.long, device=device),
            query_relation = torch.tensor(chunk_r, dtype=torch.long, device=device),
            y              = torch.tensor(chunk_t, dtype=torch.long, device=device),
            num_nodes      = B * N,
        )
        data.batch      = batch_assign
        data.ptr        = ptr
        data.num_graphs = B

        # Expander edges: slice the pre-generated tensor to the actual batch size.
        # ExpanderEdgeFixer detects expander_edge_index and adds it to the
        # attention graph alongside add_edge_index KG edges (if enabled).
        if eval_exp_ei is not None:
            data.expander_edge_index = eval_exp_ei[:, :B * E_exp]

        pred, _ = model(data)   # (B, N)

        # ---------------------------------------------------------------
        # Filtered ranking for each query in the chunk
        # ---------------------------------------------------------------
        ranks = []
        for i in range(B):
            h, r, t = chunk_h[i], chunk_r[i], chunk_t[i]
            scores = pred[i].clone()          # (N,)
            if r >= base_num_rel:
                # Reciprocal head-prediction query (t_orig, r_inv, h_orig).
                # Use head_filter[(t_orig, r_orig)] to mask ALL known heads
                # across train+val+test (not just training heads).
                r_orig = r - base_num_rel
                for kt in head_filter.get((h, r_orig), set()):
                    if kt != t:
                        scores[kt] = float('-inf')
            else:
                for kt in filter_dict.get((h, r), set()):
                    if kt != t:
                        scores[kt] = float('-inf')
            ranks.append(int((scores >= scores[t]).sum().item()))

        logger.update_stats_kgc(
            ranks=ranks,
            time_used=time.time() - time_start,
            lr=0.0,
            params=getattr(cfg, 'params', 0),
        )
        time_start = time.time()

        processed += B
        if processed % 500 < B or processed >= n_queries:
            logging.info(f'  eval_kgc [{split}]: {processed}/{n_queries}')


# ---------------------------------------------------------------------------
# Full-graph KGC training (NBFNet-style)
# ---------------------------------------------------------------------------

def train_epoch_kgc_full(logger, model, optimizer, scheduler, cfg, dataset):
    """
    NBFNet-style full-graph training epoch for KGC.

    Each call randomly samples `kgc.train_steps_per_epoch * kgc.train_batch_size`
    training triples and runs each mini-batch as B full-graph copies through the
    model, optimising with filtered binary cross-entropy.  This eliminates the
    train/eval structural mismatch caused by subgraph extraction.

    Args:
        logger:    CustomLogger — accumulates per-step stats via update_stats.
        model:     MultiModel (nn.Module).
        optimizer: PyTorch optimizer.
        scheduler: PyTorch LR scheduler (stepped once per epoch in custom_train).
        cfg:       YACS CfgNode — reads kgc.train_steps_per_epoch,
                   kgc.train_batch_size, optim.batch_accumulation,
                   optim.clip_grad_norm, train.max_iter.
        dataset:   KGCSplitWrapper — exposes full graph tensors and filter_dict.
    """
    from loss.losses import kgc_full_graph_ce

    model.train()
    optimizer.zero_grad()
    device = torch.device(cfg.device)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    kgc_ds      = dataset.train_ds
    filter_dict = kgc_ds.all_triples_filter        # {(h, r) -> set(tails)}
    N           = kgc_ds.num_entities
    bnr         = kgc_ds.num_base_relations        # 11 for WN18RR
    nr          = kgc_ds.num_relations             # 22 = 2 * bnr

    full_edge_index = kgc_ds.full_edge_index.to(device)   # (2, E)
    full_edge_attr  = kgc_ds.full_edge_attr.to(device)    # (E,)

    train_triples = kgc_ds.train_triples           # (N_train, 3) on CPU
    n_train       = len(train_triples)

    train_bs    = cfg.kgc.train_batch_size         # queries per forward pass
    steps       = cfg.kgc.train_steps_per_epoch    # total steps this epoch
    batch_accum = cfg.optim.batch_accumulation
    max_iter    = cfg.train.max_iter               # 0 = no limit

    # Random draw of (steps * train_bs) triples, cycling over the training set.
    needed  = steps * train_bs
    reps    = (needed + n_train - 1) // n_train
    indices = torch.randperm(n_train).repeat(reps)[:needed]  # (needed,)

    # Static expander (generated once at dataset build time); tiled per step for B copies.
    # cfg.prep.exp=False → full_expander_edge_index is None → no expander edges.
    _base_exp_ei = kgc_ds.full_expander_edge_index   # (2, E_exp) or None

    time_start = time.time()

    for step_idx in range(steps):
        if max_iter > 0 and step_idx >= max_iter:
            break

        batch_idx = indices[step_idx * train_bs: (step_idx + 1) * train_bs]
        chunk     = train_triples[batch_idx]       # (B, 3)
        B         = len(chunk)

        chunk_h = chunk[:, 0].tolist()
        chunk_r = chunk[:, 1].tolist()
        chunk_t = chunk[:, 2]                      # (B,) long, on CPU

        # NBFNet-style query-edge removal (Zhu et al. 2021, Appendix B):
        # Fully vectorized: tile B copies of the full graph then mask all
        # query answer edges in one GPU pass — no Python for-loop.
        #
        # For graph copy i, remove edges (u→v, r) where {u,v}=={h_i,t_i}
        # and r%bnr == chunk_r[i]%bnr (all 4 forms of the same base fact
        # share the same %bnr value, so one check covers all).
        E = full_edge_index.shape[1]
        graph_idx = torch.arange(B, device=device).repeat_interleave(E)  # (B*E,)

        # Tile edge index with per-graph offsets in one shot
        src_tiled = full_edge_index[0].repeat(B) + graph_idx * N  # (B*E,)
        dst_tiled = full_edge_index[1].repeat(B) + graph_idx * N  # (B*E,)
        rel_tiled = full_edge_attr.repeat(B)                       # (B*E,)

        # Per-edge query h/t/r_orig (broadcast from per-graph tensors)
        h_g = torch.tensor(chunk_h, device=device, dtype=torch.long)[graph_idx]   # (B*E,)
        t_g = chunk_t.to(device)[graph_idx]                                        # (B*E,)
        r_g = torch.tensor(
            [r % bnr for r in chunk_r], device=device, dtype=torch.long
        )[graph_idx]                                                               # (B*E,)

        # Local src/dst within each graph copy (strip offset for comparison)
        src_loc = full_edge_index[0].repeat(B)  # same as src_tiled - graph_idx*N
        dst_loc = full_edge_index[1].repeat(B)

        is_ht = (
            ((src_loc == h_g) & (dst_loc == t_g))
            | ((src_loc == t_g) & (dst_loc == h_g))
        )
        keep = ~(is_ht & (rel_tiled % bnr == r_g))

        rep_ei = torch.stack([src_tiled[keep], dst_tiled[keep]], dim=0)  # (2, E')
        rep_ea = rel_tiled[keep]                                           # (E',)

        batch_assign = torch.arange(B, device=device).repeat_interleave(N)
        ptr          = torch.arange(B + 1, device=device) * N

        data = Data(
            x              = torch.zeros(B * N, 1, device=device),
            edge_index     = rep_ei,
            edge_attr      = rep_ea,
            anchor_idx     = chunk[:, 0].to(device),
            query_relation = chunk[:, 1].to(device),
            y              = chunk_t.to(device),
            num_nodes      = B * N,
        )
        data.batch      = batch_assign
        data.ptr        = ptr
        data.num_graphs = B

        # Expander edges: tile the static expander for B graph copies.
        # ExpanderEdgeFixer merges them with the KG edges (add_edge_index).
        # cfg.prep.exp=False → _base_exp_ei is None → no expander edges.
        if _base_exp_ei is not None:
            data.expander_edge_index = _tile_expander(_base_exp_ei, B, N, device)

        pred, _ = model(data)   # (B, N)

        loss, pred_score = kgc_full_graph_ce(
            pred, chunk_t.to(device), filter_dict, chunk_h, chunk_r,
            label_smoothing=getattr(cfg.kgc, 'label_smoothing', 0.0),
            head_filter=kgc_ds.head_filter,
            base_num_rel=kgc_ds.num_base_relations,
        )

        loss.backward()

        if ((step_idx + 1) % batch_accum == 0) or (step_idx + 1 == steps):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        logger.update_stats(
            true=chunk_t.detach().cpu(),
            pred=pred_score.detach().cpu(),
            loss=loss.detach().cpu().item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=getattr(cfg, 'params', 0),
            dataset_name=cfg.dataset.name,
        )
        time_start = time.time()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def custom_train(loggers, loaders, model, optimizer, scheduler, cfg, dataset=None):
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

    is_kgc = (cfg.dataset.task_type == 'kgc_ranking')
    use_full_graph_eval  = is_kgc and getattr(cfg.kgc, 'eval_full_graph',  False)
    use_full_graph_train = is_kgc and getattr(cfg.kgc, 'train_full_graph', False)

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
        if use_full_graph_train:
            train_epoch_kgc_full(loggers[0], model, optimizer, scheduler,
                                 cfg, dataset)
        else:
            train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, cfg)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        # --- Val / Test (on eval epochs) ---
        if _is_eval_epoch(cur_epoch, eval_period, max_epoch):
            for i in range(1, num_splits):
                split_name = split_names[i - 1]
                if use_full_graph_eval:
                    eval_epoch_kgc(loggers[i], loaders[i], model,
                                   split_name, cfg, dataset)
                else:
                    eval_epoch(loggers[i], loaders[i], model,
                               split=split_name, cfg=cfg)
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

            # Best-epoch checkpoint.
            # best_epoch is an index into perf[], not the actual epoch number,
            # so comparing with cur_epoch breaks on auto_resume.  Instead check
            # whether the most recent eval produced the best result so far.
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == len(val_perf) - 1:
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
