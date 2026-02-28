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

# Override any config key
python main.py --cfg configs/Exphormer/cifar10.yaml wandb.use False optim.max_epoch 1

# Multiple seeds
python main.py --cfg configs/Exphormer/cifar10.yaml --repeat 3 wandb.use False

# Auto-resume
python main.py --cfg configs/Exphormer/cifar10.yaml train.auto_resume True
```

Results are written to `results/<config-name>/<run_id>/`.

## Architecture

### Config System
YACS `CfgNode` defined in `config.py`. All defaults cover the 18 configs. Loaded with `load_cfg(yaml_file, opts)` where `opts` is a flat list of `[key, value, ...]` overrides.

### Data Flow
```
main.py
  → load_cfg()                          # config.py
  → load_dataset(cfg)                   # loader/master_loader.py
      → precompute PE (LapPE/EquivStableLapPE)   # transform/posenc_stats.py
      → add expander edges               # transform/expander_edges.py
      → prepare_splits()                 # loader/split_generator.py
  → create_loaders(cfg, dataset)        # [train, val, test] DataLoaders
  → create_loggers(cfg)                 # train/logger.py
  → create_model(cfg, dim_in, dim_out)  # network/model.py → MultiModel
  → build_optimizer(), build_scheduler()  # optimizer/schedulers.py
  → custom_train(...)                   # train/trainer.py
```

### Model Structure
`MultiModel`: `FeatureEncoder → [optional pre_mp] → N×MultiLayer → Head`

- `FeatureEncoder`: node encoder + BN + edge encoder + BN + `ExpanderEdgeFixer`
- `MultiLayer`: local GNN (GatedGCN/GCN/GINE/GAT) + Exphormer attention, summed → shared FFN
- Heads: `GraphHead` (graph classification/regression) or `InductiveNodeHead` (node tasks)

### Layer Type String
`gt.layer_type` uses `+`-separated names: `CustomGatedGCN+Exphormer`, `GCN+Exphormer`, etc.

### No Registration System
Direct if/elif dispatch in factory functions:
- `build_node_encoder(cfg, dim_in)` — dispatches on `cfg.dataset.node_encoder_name`
- `build_edge_encoder(cfg)` — dispatches on `cfg.dataset.edge_encoder_name`
- `build_head(cfg, dim_in, dim_out)` — dispatches on `cfg.gnn.head`
- `create_model(cfg, dim_in, dim_out)` — dispatches on `cfg.model.type`

## Key Bugs / Gotchas

- **BatchNorm in FeatureEncoder**: `nn.BatchNorm1d` must be applied to `batch.x` (not the batch object). Fixed in `network/model.py` `FeatureEncoder.forward()`.
- **`scipy.ravel`**: Removed in newer scipy — use `np.ravel`. Fixed in `transform/dist_transforms.py:171`.
- **`add_reverse_edges` / `add_self_loops`**: Located in `transform/dist_transforms.py`, not `transforms.py`.
- **ogbn-arxiv**: Single-graph transductive dataset — loaders return same DataLoader × 3.
