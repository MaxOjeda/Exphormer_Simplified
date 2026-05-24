"""
Diagnostic script for the inductive ceiling problem (sesión 33+).

Combines:
- A1-A4: per-query rank capture + structural features (degree, shortest path,
  relation frequency) for any list of checkpoints.
- C1-C3: weight-surgery (swap matching params between two checkpoints, eval
  the resulting model) + weight-norm trajectory across all checkpoints.

Usage:
    python scripts/diagnose_inductive.py --cfg <config.yaml> \\
        --ckpt-dir results/<name>/0 \\
        --output-dir analysis/<name> \\
        --modes per_query,norms,surgery \\
        --query-epochs 5 6 10 20 29 \\
        --surgery-base 20 --surgery-source 6 \\
        --surgery-patterns "K.weight,V.weight,ff_linear,post_mp.rel_emb,encoder"

Run from project root so relative imports resolve.

Outputs:
    <output-dir>/per_query.csv     # one row per (epoch, split, query)
    <output-dir>/weight_norms.csv  # one row per (epoch, param_name) → ||W||_F
    <output-dir>/surgery.csv       # one row per (pattern, base_mrr, swapped_mrr)
    <output-dir>/structural.csv    # static per-query structural features
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Data

# Project imports — must run from repo root
sys.path.insert(0, os.path.abspath('.'))
from config import cfg, load_cfg                                    # noqa: E402
from loader.master_loader import load_dataset                       # noqa: E402
from network.model import create_model                              # noqa: E402
from train.trainer import _tile_expander                            # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('diag')


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def build_model_from_cfg(cfg, device):
    """Construct the model + dataset for this config; load nothing yet."""
    dataset = load_dataset(cfg)
    # KGC: encoders/head bypass dim_in/dim_out → hardcode (1, 1) like main.py.
    dim_in, dim_out = 1, 1
    cfg.defrost(); cfg.share.dim_in = dim_in; cfg.freeze()
    model = create_model(cfg, dim_in, dim_out).to(device)
    return model, dataset


def load_ckpt(model, ckpt_path):
    blob = torch.load(ckpt_path, map_location='cpu')
    sd = blob['model_state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        log.warning(f'  missing={len(missing)} unexpected={len(unexpected)}')
    return blob.get('epoch', -1)


# ---------------------------------------------------------------------------
# Eval — per-query rank capture (no aggregation, no logger)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_per_query(model, cfg, dataset, split):
    """
    Drop-in copy of train.trainer.eval_epoch_kgc but yields per-query rows
    (split, query_idx, h, r, t, rank) instead of aggregating to logger.
    """
    model.eval()
    device = torch.device(cfg.device)
    kgc_ds = dataset.train_ds
    base_num_rel = kgc_ds.num_base_relations

    _inductive_test = (split == 'test' and hasattr(kgc_ds, 'test_num_entities'))
    if _inductive_test:
        filter_dict = kgc_ds.test_all_triples_filter
        head_filter = kgc_ds.test_head_filter
        N = kgc_ds.test_num_entities
        full_edge_index = kgc_ds.test_full_edge_index.to(device)
        full_edge_attr = kgc_ds.test_full_edge_attr.to(device)
        _eval_expander = kgc_ds.test_full_expander_edge_index
    else:
        filter_dict = kgc_ds.all_triples_filter
        head_filter = kgc_ds.head_filter
        N = kgc_ds.num_entities
        full_edge_index = kgc_ds.full_edge_index.to(device)
        full_edge_attr = kgc_ds.full_edge_attr.to(device)
        _eval_expander = kgc_ds.full_expander_edge_index

    E = full_edge_index.shape[1]
    queries = kgc_ds.val_triples if split == 'val' else kgc_ds.test_triples

    if getattr(cfg.kgc, 'reciprocal', False):
        rec = torch.stack([queries[:, 2],
                           queries[:, 1] + base_num_rel,
                           queries[:, 0]], dim=1)
        queries = torch.cat([queries, rec], dim=0)

    n_queries = len(queries)
    eval_bs = cfg.kgc.eval_batch_size

    offset_table = (torch.arange(eval_bs, device=device) * N).repeat_interleave(E)
    eval_exp_ei = _tile_expander(_eval_expander, eval_bs, N, device)
    E_exp = _eval_expander.shape[1] if _eval_expander is not None else 0

    rows = []
    for chunk_start in range(0, n_queries, eval_bs):
        chunk = queries[chunk_start: chunk_start + eval_bs]
        B = len(chunk)
        chunk_h, chunk_r, chunk_t = chunk[:, 0].tolist(), chunk[:, 1].tolist(), chunk[:, 2].tolist()

        rep_ei = full_edge_index.repeat(1, B) + offset_table[:B * E].unsqueeze(0)
        rep_ea = full_edge_attr.repeat(B)
        batch_assign = torch.arange(B, device=device).repeat_interleave(N)
        ptr = torch.arange(B + 1, device=device) * N

        data = Data(
            x=torch.zeros(B * N, 1, device=device),
            edge_index=rep_ei,
            edge_attr=rep_ea,
            anchor_idx=torch.tensor(chunk_h, dtype=torch.long, device=device),
            query_relation=torch.tensor(chunk_r, dtype=torch.long, device=device),
            y=torch.tensor(chunk_t, dtype=torch.long, device=device),
            num_nodes=B * N,
        )
        data.batch = batch_assign
        data.ptr = ptr
        data.num_graphs = B
        data.edge_rel_idx = rep_ea
        if eval_exp_ei is not None:
            data.expander_edge_index = eval_exp_ei[:, :B * E_exp]

        pred, _ = model(data)   # (B, N)

        for i in range(B):
            h, r, t = chunk_h[i], chunk_r[i], chunk_t[i]
            scores = pred[i].clone()
            if r >= base_num_rel:
                r_orig = r - base_num_rel
                for kt in head_filter.get((h, r_orig), set()):
                    if kt != t:
                        scores[kt] = float('-inf')
            else:
                for kt in filter_dict.get((h, r), set()):
                    if kt != t:
                        scores[kt] = float('-inf')
            rank = int((scores >= scores[t]).sum().item())
            rows.append({
                'split': split, 'query_idx': chunk_start + i,
                'h': h, 'r': r, 't': t, 'rank': rank,
                'is_reciprocal': int(r >= base_num_rel),
            })
    return rows


# ---------------------------------------------------------------------------
# Structural features (one-shot, shared across all epochs)
# ---------------------------------------------------------------------------

def compute_structural_features(dataset):
    """
    Returns:
        feats: dict
            'train': per-node degree (1d ndarray, len N_train)
            'test':  per-node degree (1d ndarray, len N_test)
            'rel_freq': dict r → count in training triples
            'sp_train': sparse shortest-path matrix on train graph (CSR)
            'sp_test':  sparse shortest-path matrix on test graph (CSR)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    kgc_ds = dataset.train_ds

    # Degrees (count incident edges, both directions)
    def _degree(edge_index, N):
        deg = np.zeros(N, dtype=np.int64)
        ei = edge_index.cpu().numpy()
        np.add.at(deg, ei[0], 1)
        np.add.at(deg, ei[1], 1)
        return deg

    train_ei = kgc_ds.full_edge_index
    train_deg = _degree(train_ei, kgc_ds.num_entities)

    if hasattr(kgc_ds, 'test_full_edge_index'):
        test_ei = kgc_ds.test_full_edge_index
        test_deg = _degree(test_ei, kgc_ds.test_num_entities)
    else:
        test_ei, test_deg = None, None

    # Relation frequency in train triples
    rel_freq = {}
    for r in kgc_ds.train_triples[:, 1].cpu().tolist():
        rel_freq[r] = rel_freq.get(r, 0) + 1

    # Shortest path matrices (treat as unweighted, undirected)
    def _sp(edge_index, N):
        ei = edge_index.cpu().numpy()
        data = np.ones(ei.shape[1], dtype=np.int8)
        adj = csr_matrix((data, (ei[0], ei[1])), shape=(N, N))
        adj = adj + adj.T
        adj.data[:] = 1
        sp = shortest_path(adj, method='D', unweighted=True)
        return sp

    log.info('Computing shortest-path matrices (may take ~30s)...')
    sp_train = _sp(train_ei, kgc_ds.num_entities)
    sp_test = _sp(test_ei, kgc_ds.test_num_entities) if test_ei is not None else None

    return {
        'train_deg': train_deg, 'test_deg': test_deg,
        'rel_freq': rel_freq,
        'sp_train': sp_train, 'sp_test': sp_test,
    }


def write_structural_csv(feats, dataset, output_dir):
    """Per-query static features that don't depend on the model."""
    kgc_ds = dataset.train_ds
    out = Path(output_dir) / 'structural.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'query_idx', 'h', 'r', 't',
                    'is_reciprocal', 'tail_degree', 'head_degree',
                    'shortest_path', 'rel_freq_in_train'])

        base_nr = kgc_ds.num_base_relations
        for split, queries in (('val', kgc_ds.val_triples), ('test', kgc_ds.test_triples)):
            rec = torch.stack([queries[:, 2], queries[:, 1] + base_nr, queries[:, 0]], dim=1)
            queries = torch.cat([queries, rec], dim=0)
            if split == 'test':
                deg = feats['test_deg']; sp = feats['sp_test']
            else:
                deg = feats['train_deg']; sp = feats['sp_train']
            for i, (h, r, t) in enumerate(queries.cpu().tolist()):
                is_rec = int(r >= base_nr)
                r_orig = r - base_nr if is_rec else r
                w.writerow([split, i, h, r, t, is_rec,
                            int(deg[t]) if deg is not None else -1,
                            int(deg[h]) if deg is not None else -1,
                            int(sp[h, t]) if (sp is not None and np.isfinite(sp[h, t])) else -1,
                            feats['rel_freq'].get(r_orig, 0)])
    log.info(f'Wrote {out}')


# ---------------------------------------------------------------------------
# Weight norms across epochs
# ---------------------------------------------------------------------------

def weight_norms(ckpt_dir, output_dir):
    """For each ckpt_epoch_*.pt, dump ||p||_F for every parameter."""
    paths = sorted(Path(ckpt_dir).glob('ckpt_epoch_*.pt'))
    log.info(f'  found {len(paths)} per-epoch checkpoints')
    out = Path(output_dir) / 'weight_norms.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'param', 'frob_norm', 'numel'])
        for p in paths:
            ep = int(re.search(r'epoch_(\d+)', p.name).group(1))
            sd = torch.load(p, map_location='cpu')['model_state_dict']
            for name, tensor in sd.items():
                if tensor.is_floating_point():
                    w.writerow([ep, name, float(tensor.detach().float().norm()), tensor.numel()])
    log.info(f'Wrote {out}')


# ---------------------------------------------------------------------------
# Weight surgery — swap params matching a pattern, eval the resulting model
# ---------------------------------------------------------------------------

def filtered_mrr(rows):
    """Compute MRR over a list of {rank: ...} rows."""
    if not rows:
        return 0.0
    return float(np.mean([1.0 / r['rank'] for r in rows]))


def hits_at(rows, k):
    if not rows:
        return 0.0
    return float(np.mean([1.0 if r['rank'] <= k else 0.0 for r in rows]))


def surgery(cfg, dataset, ckpt_dir, output_dir, base_ep, source_ep, patterns):
    """
    For each pattern in patterns:
        load base_ep ckpt → eval test → base_mrr
        load source_ep ckpt for those params only → eval test → swapped_mrr
    Writes one row per pattern.
    """
    device = torch.device(cfg.device)
    base_path = Path(ckpt_dir) / f'ckpt_epoch_{base_ep:03d}.pt'
    source_path = Path(ckpt_dir) / f'ckpt_epoch_{source_ep:03d}.pt'
    if not base_path.exists() or not source_path.exists():
        log.error(f'missing checkpoints: {base_path} or {source_path}')
        return

    base_sd = torch.load(base_path, map_location='cpu')['model_state_dict']
    src_sd = torch.load(source_path, map_location='cpu')['model_state_dict']

    model, _ = build_model_from_cfg(cfg, device)
    out = Path(output_dir) / 'surgery.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pattern', 'n_swapped_params', 'base_mrr', 'swapped_mrr',
                    'base_h10', 'swapped_h10', 'base_ep', 'source_ep'])

        # Reference: pure base eval
        model.load_state_dict(base_sd)
        rows_base = eval_per_query(model, cfg, dataset, 'test')
        base_mrr, base_h10 = filtered_mrr(rows_base), hits_at(rows_base, 10)
        log.info(f'  base_ep={base_ep}: test_mrr={base_mrr:.4f}  h@10={base_h10:.4f}')

        # Reference: pure source eval (for context)
        model.load_state_dict(src_sd)
        rows_src = eval_per_query(model, cfg, dataset, 'test')
        src_mrr, src_h10 = filtered_mrr(rows_src), hits_at(rows_src, 10)
        log.info(f'  source_ep={source_ep}: test_mrr={src_mrr:.4f}  h@10={src_h10:.4f}')
        w.writerow(['(pure source — reference)', -1, base_mrr, src_mrr, base_h10, src_h10,
                    base_ep, source_ep])

        for pat in patterns:
            rx = re.compile(pat)
            merged = dict(base_sd)
            n_swapped = 0
            for name in base_sd.keys():
                if rx.search(name):
                    merged[name] = src_sd[name].clone()
                    n_swapped += 1
            model.load_state_dict(merged)
            rows = eval_per_query(model, cfg, dataset, 'test')
            mrr, h10 = filtered_mrr(rows), hits_at(rows, 10)
            log.info(f'  swap "{pat}" (n={n_swapped} params): '
                     f'test_mrr {base_mrr:.4f} → {mrr:.4f}  Δ={mrr - base_mrr:+.4f}')
            w.writerow([pat, n_swapped, base_mrr, mrr, base_h10, h10, base_ep, source_ep])

    log.info(f'Wrote {out}')


# ---------------------------------------------------------------------------
# Per-query mode: load each query-epoch, eval, save rows
# ---------------------------------------------------------------------------

def per_query_mode(cfg, dataset, model, ckpt_dir, epochs, output_dir):
    out = Path(output_dir) / 'per_query.csv'
    first = True
    for ep in epochs:
        path = Path(ckpt_dir) / f'ckpt_epoch_{ep:03d}.pt'
        if not path.exists():
            log.warning(f'  missing {path}, skipping')
            continue
        load_ckpt(model, path)
        t0 = time.time()
        rows = eval_per_query(model, cfg, dataset, 'val') + \
               eval_per_query(model, cfg, dataset, 'test')
        mode = 'w' if first else 'a'
        with open(out, mode, newline='') as f:
            w = csv.writer(f)
            if first:
                w.writerow(['epoch', 'split', 'query_idx', 'h', 'r', 't', 'rank', 'is_reciprocal'])
            for row in rows:
                w.writerow([ep, row['split'], row['query_idx'], row['h'], row['r'],
                            row['t'], row['rank'], row['is_reciprocal']])
        # Quick summary
        for split in ('val', 'test'):
            split_rows = [r for r in rows if r['split'] == split]
            if split_rows:
                mrr = filtered_mrr(split_rows); h10 = hits_at(split_rows, 10)
                log.info(f'  ep{ep} {split}: mrr={mrr:.4f}  h@10={h10:.4f}  '
                         f'({len(split_rows)} queries, {time.time() - t0:.1f}s)')
        first = False
    log.info(f'Wrote {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--modes', default='per_query,norms,surgery,structural',
                    help='comma-separated subset of: per_query,norms,surgery,structural')
    ap.add_argument('--query-epochs', nargs='+', type=int, default=[5, 6, 10, 20, 29])
    ap.add_argument('--surgery-base', type=int, default=20)
    ap.add_argument('--surgery-source', type=int, default=6)
    ap.add_argument('--surgery-patterns', default='K\\.weight,V\\.weight,ff_linear,post_mp\\.rel_emb,encoder')
    ap.add_argument('--opts', nargs='*', default=[],
                    help='extra YACS overrides, e.g. kgc.eval_batch_size 16')
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    modes = set(args.modes.split(','))

    # Load config + dataset + model
    load_cfg(args.cfg, args.opts)
    cfg.defrost()
    cfg.out_dir = '_diag_tmp'; cfg.run_dir = '_diag_tmp'
    if cfg.device == 'auto':
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.freeze()
    device = torch.device(cfg.device)
    log.info(f'device={device}')
    model, dataset = build_model_from_cfg(cfg, device)

    # Structural — always cheap; do first
    if 'structural' in modes or 'per_query' in modes:
        feats = compute_structural_features(dataset)
        if 'structural' in modes:
            write_structural_csv(feats, dataset, args.output_dir)

    if 'norms' in modes:
        log.info('=== weight norms ===')
        weight_norms(args.ckpt_dir, args.output_dir)

    if 'per_query' in modes:
        log.info('=== per-query (epochs %s) ===', args.query_epochs)
        per_query_mode(cfg, dataset, model, args.ckpt_dir, args.query_epochs, args.output_dir)

    if 'surgery' in modes:
        log.info('=== surgery (base=%d source=%d) ===', args.surgery_base, args.surgery_source)
        patterns = [p for p in args.surgery_patterns.split(',') if p]
        surgery(cfg, dataset, args.ckpt_dir, args.output_dir,
                args.surgery_base, args.surgery_source, patterns)

    log.info('done')


if __name__ == '__main__':
    main()
