# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Exphormer_Max is a standalone reimplementation of Exphormer (sparse transformer for graphs, ICML 2023) **without any GraphGPS/graphgym dependency**. It supports all 18 experiment configs from `configs/Exphormer/`.

## Environment

Use the `exphormer` conda environment (Python 3.12):
```bash
# IMPORTANT: conda run does not work — use direct path
/nfs_ssd/mojeda_imfd/miniconda3/envs/exphormer/bin/python main.py ...

# Or activate the env first
conda activate exphormer
python main.py ...
```

## Running Experiments

```bash
# Basic run
python main.py --cfg configs/Exphormer/cifar10.yaml wandb.use False

# Quick smoke test (1 epoch, small batch — faster with no-PE dataset)
python main.py --cfg configs/Exphormer/malnettiny.yaml wandb.use False optim.max_epoch 1 train.batch_size 4

# Override any config key (flat dot notation)
python main.py --cfg configs/Exphormer/cifar10.yaml wandb.use False optim.max_epoch 1

# Multiple seeds
python main.py --cfg configs/Exphormer/cifar10.yaml --repeat 3 wandb.use False

# Auto-resume from checkpoint
python main.py --cfg configs/Exphormer/cifar10.yaml train.auto_resume True
```

Results are written to `results/<config-name>/<run_id>/`. Each run produces `stats_train.json`, `stats_val.json`, `stats_test.json` (JSON-lines), and checkpoints.

## Architecture

### Config System
YACS `CfgNode` defined in `config.py`. All defaults cover the 18 configs. Loaded with `load_cfg(yaml_file, opts)` where `opts` is a flat list of `[key, value, ...]` overrides. Key config sections:
- `dataset.*`: format, name, task, encoders, splits
- `gt.*`: layer_type, layers, n_heads, dim_hidden, dropout
- `gnn.*`: head, pre/post MP layers, dim_inner (must equal `gt.dim_hidden`)
- `prep.*`: expander graph settings (exp, exp_algorithm, exp_deg, num_virt_node)
- `posenc_LapPE.*`, `posenc_EquivStableLapPE.*`: PE configuration
- `optim.*`: optimizer, lr, scheduler, max_epoch
- `train.*`: batch_size, eval_period, ckpt_period, auto_resume

### Data Flow
```
main.py
  → load_cfg()                          # config.py
  → load_dataset(cfg)                   # loader/master_loader.py
      → load base dataset (PyG/OGB/custom)   # loader/dataset/*.py
      → precompute PE (LapPE/EquivStableLapPE)   # transform/posenc_stats.py
      → add expander edges               # transform/expander_edges.py
      → prepare_splits()                 # loader/split_generator.py
      → cache result to disk (MD5-keyed)
  → create_loaders(cfg, dataset)        # [train, val, test] DataLoaders
  → create_loggers(cfg)                 # train/logger.py
  → create_model(cfg, dim_in, dim_out)  # network/model.py → MultiModel
  → build_optimizer(), build_scheduler()  # optimizer/schedulers.py
  → custom_train(...)                   # train/trainer.py
  → agg_runs(...)                       # main.py — aggregates across seeds
```

### Model Structure
`MultiModel`: `FeatureEncoder → [optional pre_mp] → N×MultiLayer → Head`

- **FeatureEncoder**: node encoder + BN + edge encoder + BN + `ExpanderEdgeFixer`
  - `ExpanderEdgeFixer` concatenates real edges, expander edges, and virtual node edges into `batch.expander_edge_index` / `batch.expander_edge_attr`
- **MultiLayer**: local GNN (GatedGCN/GCN/GINE/GAT) + Exphormer attention, summed → shared FFN
  - Each sub-model uses residual + norm + dropout
- **Heads**: `GraphHead` (global pooling → MLP) or `InductiveNodeHead` (MLP on nodes)

### Layer Type String
`gt.layer_type` uses `+`-separated names: `CustomGatedGCN+Exphormer`, `GCN+Exphormer`, etc. Each token becomes a separate `LocalModel` or `GlobalModel` in `MultiLayer`.

### No Registration System
Direct if/elif dispatch in factory functions:
- `build_node_encoder(cfg, dim_in)` — dispatches on `cfg.dataset.node_encoder_name`
- `build_edge_encoder(cfg)` — dispatches on `cfg.dataset.edge_encoder_name`
- `build_head(cfg, dim_in, dim_out)` — dispatches on `cfg.gnn.head`
- `create_model(cfg, dim_in, dim_out)` — dispatches on `cfg.model.type`

### Preprocessing Cache
Expensive operations (PE computation, expander graph generation) are cached to disk using a deterministic MD5 hash of the relevant config params. Cache files live alongside the dataset. Delete cache files to force recomputation.

### Metrics & Logging
`CustomLogger` (`train/logger.py`) tracks predictions/labels/loss per epoch and computes:
- Classification: accuracy, F1, AUC
- Regression: MAE, MSE, R2, Spearman
- SBM (PATTERN/CLUSTER): `accuracy_SBM`

The metric used for best-model tracking is set via `cfg.metric_best` (e.g., `accuracy`, `mae`).

### Expander / Virtual Nodes
- Expander edges are d-regular random graphs per data point, stored in `data.expander_edges` pre-batching
- `ExpanderEdgeFixer` merges them with real edges and optional virtual-node edges at forward time
- `ExphormerAttention` operates exclusively on `batch.expander_edge_index` (not `batch.edge_index`)

## Key Bugs / Gotchas

- **BatchNorm in FeatureEncoder**: `nn.BatchNorm1d` must be applied to `batch.x` (not the batch object). Fixed in `network/model.py` `FeatureEncoder.forward()`.
- **`scipy.ravel`**: Removed in newer scipy — use `np.ravel`. Fixed in `transform/dist_transforms.py:171`.
- **`add_reverse_edges` / `add_self_loops`**: Located in `transform/dist_transforms.py`, not `transforms.py`.
- **`gnn.dim_inner` must equal `gt.dim_hidden`**: Enforced by assertion in `MultiModel.__init__`.
- **`dim_edge` default**: If `None` in config, set equal to `gt.dim_hidden` in `FeatureEncoder`.
- **`EquivStableLapPE`**: Stored as `batch.pe_EquivStableLapPE`, NOT appended to `batch.x` (unlike LapPE). GatedGCN reads it directly from the batch.
- **ogbn-arxiv**: Single-graph transductive dataset — `create_loaders` returns the same DataLoader × 3.
- **Configs `*_h.yaml`**: Higher-capacity hyperparameter variants (more heads, larger hidden dim, more layers).

## Extending the Codebase

To add a **new dataset**: implement a `Dataset` class in `loader/dataset/`, then add a dispatch branch in `loader/master_loader.py:load_dataset()`.

To add a **new node/edge encoder**: implement the module and add an elif branch to `build_node_encoder()` / `build_edge_encoder()` in `encoder/node_encoders.py` / `encoder/edge_encoders.py`.

To add a **new layer type**: implement the layer module in `layer/`, then add a dispatch branch in `MultiLayer.__init__()` in `network/model.py` (look for the layer_type string parsing).
