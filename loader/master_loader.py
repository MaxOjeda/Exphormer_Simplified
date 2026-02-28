"""
Master dataset loader for Exphormer_Max.
Replaces graphgym's loader infrastructure.
Supports all datasets from configs/Exphormer/*.yaml.
"""
import hashlib
import json
import logging
import os
import os.path as osp
from functools import partial

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (Amazon, Coauthor, GNNBenchmarkDataset)
from torch_geometric.loader import DataLoader

from loader.dataset.coco_superpixels import COCOSuperpixels
from loader.dataset.malnet_tiny import MalNetTiny
from loader.dataset.voc_superpixels import VOCSuperpixels
from loader.split_generator import prepare_splits, set_dataset_splits
from transform.posenc_stats import compute_posenc_stats
from transform.transforms import (pre_transform_in_memory, generate_splits,
                                  typecast_x, concat_x_and_pos)
from transform.dist_transforms import add_reverse_edges, add_self_loops
from transform.expander_edges import generate_random_expander


# ---------------------------------------------------------------------------
# Disk cache for expensive PE + expander edge preprocessing
# ---------------------------------------------------------------------------

def _preproc_cache_path(cfg, name):
    """Return a deterministic cache path based on the preprocessing config."""
    params = {
        'name': name,
        # PE params
        'equivstable_pe': cfg.posenc_EquivStableLapPE.enable,
        'equivstable_maxfreqs': cfg.posenc_EquivStableLapPE.eigen.max_freqs,
        'equivstable_norm': cfg.posenc_EquivStableLapPE.eigen.laplacian_norm,
        'lap_pe': cfg.posenc_LapPE.enable,
        'lap_maxfreqs': cfg.posenc_LapPE.eigen.max_freqs,
        'lap_norm': cfg.posenc_LapPE.eigen.laplacian_norm,
        # Expander params
        'exp': cfg.prep.exp,
        'exp_deg': cfg.prep.exp_deg,
        'exp_algo': cfg.prep.exp_algorithm,
        'exp_count': cfg.prep.exp_count,
        'add_edge_index': cfg.prep.add_edge_index,
        'num_virt_node': cfg.prep.num_virt_node,
    }
    key = hashlib.md5(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:12]
    cache_dir = osp.join(cfg.dataset.dir, '_preproc_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return osp.join(cache_dir, f'{name}_{key}.pt')


def _cache_save(dataset, cache_path):
    torch.save({
        'data': dataset._data,
        'slices': dataset.slices,
    }, cache_path)
    logging.info(f'Saved preprocessed data to cache: {cache_path}')


def _cache_load(dataset, cache_path):
    """Overwrite dataset internal storage with cached tensors. Returns True on success."""
    if not osp.exists(cache_path):
        return False
    try:
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        dataset._data = cached['data']
        dataset.slices = cached['slices']
        dataset._data_list = None   # invalidate item-level cache
        logging.info(f'Loaded preprocessed data from cache: {cache_path}')
        return True
    except Exception as e:
        logging.warning(f'Cache load failed ({e}), recomputing.')
        return False


# ---------------------------------------------------------------------------
# Dataset format dispatch
# ---------------------------------------------------------------------------

def load_dataset(cfg):
    """
    Load and preprocess the dataset specified by cfg.
    Returns a PyG dataset with split information attached.
    """
    fmt = cfg.dataset.format
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir

    if fmt.startswith('PyG-'):
        pyg_id = fmt.split('-', 1)[1]
        ds_dir = osp.join(dataset_dir, pyg_id)

        if pyg_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(ds_dir, name)

        elif pyg_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(ds_dir, feature_set=name)

        elif pyg_id == 'Amazon':
            dataset = Amazon(ds_dir, name)
            pre_transform_in_memory(dataset, partial(generate_splits,
                                                     g_split=cfg.dataset.split[0]))
            pre_transform_in_memory(dataset, add_reverse_edges)
            if cfg.prep.add_self_loops:
                pre_transform_in_memory(dataset, add_self_loops)

        elif pyg_id == 'Coauthor':
            dataset = Coauthor(ds_dir, name)
            pre_transform_in_memory(dataset, partial(generate_splits,
                                                     g_split=cfg.dataset.split[0]))
            pre_transform_in_memory(dataset, add_reverse_edges)
            if cfg.prep.add_self_loops:
                pre_transform_in_memory(dataset, add_self_loops)

        elif pyg_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(ds_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(ds_dir, name,
                                                cfg.dataset.slic_compactness)

        else:
            raise ValueError(f"Unknown PyG dataset: {pyg_id}")

    elif fmt == 'OGB':
        if name.startswith('ogbn'):
            dataset = preformat_ogbn(dataset_dir, name, cfg)
        else:
            raise ValueError(f"Unsupported OGB dataset: {name}")

    else:
        raise ValueError(f"Unknown dataset format: {fmt}")

    _log_dataset_info(dataset, fmt, name)

    # ------------------------------------------------------------------
    # Determine which PE types are needed
    # ------------------------------------------------------------------
    pe_types = []
    for attr in ['posenc_LapPE', 'posenc_EquivStableLapPE']:
        pecfg = getattr(cfg, attr, None)
        if pecfg is not None and pecfg.enable:
            pe_name = attr.split('_', 1)[1]
            pe_types.append(pe_name)

    needs_preproc = pe_types or cfg.prep.exp

    # ------------------------------------------------------------------
    # Try to load from disk cache (skips PE + expander recomputation)
    # ------------------------------------------------------------------
    cache_loaded = False
    if needs_preproc and fmt != 'OGB':
        cache_path = _preproc_cache_path(cfg, name)
        cache_loaded = _cache_load(dataset, cache_path)

    # ------------------------------------------------------------------
    # Precompute positional encodings (if not cached)
    # ------------------------------------------------------------------
    if not cache_loaded and pe_types:
        is_undirected = all(
            d.is_undirected() for d in dataset[:min(10, len(dataset))])
        logging.info(
            f"Precomputing PE ({pe_types}), undirected={is_undirected}...")
        pre_transform_in_memory(
            dataset,
            partial(compute_posenc_stats, pe_types=pe_types,
                    is_undirected=is_undirected, cfg=cfg),
            show_progress=True)

    # ------------------------------------------------------------------
    # Add expander edges (if not cached)
    # ------------------------------------------------------------------
    if not cache_loaded and cfg.prep.exp:
        for j in range(cfg.prep.exp_count):
            logging.info(f"Adding expander edges (round {j})...")
            pre_transform_in_memory(
                dataset,
                partial(generate_random_expander,
                        degree=cfg.prep.exp_deg,
                        algorithm=cfg.prep.exp_algorithm,
                        rng=None,
                        max_num_iters=cfg.prep.exp_max_num_iters,
                        exp_index=j),
                show_progress=True)

    # ------------------------------------------------------------------
    # Save to cache after first-time computation
    # ------------------------------------------------------------------
    if needs_preproc and not cache_loaded and fmt != 'OGB':
        _cache_save(dataset, cache_path)

    # ------------------------------------------------------------------
    # Split setup (skip for ogbn which has its own split_idx)
    # ------------------------------------------------------------------
    if name not in ('ogbn-arxiv', 'ogbn-proteins'):
        prepare_splits(dataset, cfg)

    return dataset


def create_loaders(cfg, dataset):
    """
    Create train/val/test DataLoaders from the dataset.
    Returns a list of 3 loaders.
    """
    name = cfg.dataset.name
    bs = cfg.train.batch_size

    if name in ('ogbn-arxiv', 'ogbn-proteins'):
        # Single-graph transductive: same dataset replicated 3×
        loader = DataLoader([dataset[0]], batch_size=1, shuffle=False)
        loader.dataset.split_idx = dataset.split_idx
        return [loader, loader, loader]

    # Check how splits are stored
    data0 = dataset._data

    if hasattr(data0, 'train_graph_index'):
        # Graph-level splits stored as index arrays
        train_idx = data0.train_graph_index
        val_idx   = data0.val_graph_index
        test_idx  = data0.test_graph_index
        train_loader = DataLoader(dataset[train_idx], batch_size=bs,
                                  shuffle=True)
        val_loader   = DataLoader(dataset[val_idx],   batch_size=bs,
                                  shuffle=False)
        test_loader  = DataLoader(dataset[test_idx],  batch_size=bs,
                                  shuffle=False)
        return [train_loader, val_loader, test_loader]

    elif hasattr(data0, 'train_mask'):
        # Node-level tasks with pre-existing masks
        loader = DataLoader(dataset, batch_size=bs, shuffle=False)
        return [loader, loader, loader]

    else:
        raise RuntimeError(
            "Dataset has no recognized split information. "
            "Expected train_graph_index or train_mask in dataset._data.")


# ---------------------------------------------------------------------------
# Dataset-specific preformatting
# ---------------------------------------------------------------------------

def preformat_GNNBenchmarkDataset(dataset_dir, name):
    if name in ('MNIST', 'CIFAR10'):
        tf_list = [concat_x_and_pos, partial(typecast_x, type_str='float')]
    elif name in ('PATTERN', 'CLUSTER'):
        tf_list = []
    else:
        raise ValueError(f"Unsupported GNNBenchmarkDataset name: {name}")

    dataset = _join_splits(
        [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
         for split in ['train', 'val', 'test']]
    )
    if tf_list:
        pre_transform_in_memory(dataset, T.Compose(tf_list))
    return dataset


def preformat_MalNetTiny(dataset_dir, feature_set):
    if feature_set in ('none', 'Constant'):
        tf = T.Constant()
    elif feature_set == 'OneHotDegree':
        tf = T.OneHotDegree()
    elif feature_set == 'LocalDegreeProfile':
        tf = T.LocalDegreeProfile()
    else:
        raise ValueError(f"Unexpected MalNetTiny feature_set: {feature_set}")

    dataset = MalNetTiny(dataset_dir)
    dataset.name = 'MalNetTiny'
    pre_transform_in_memory(dataset, tf)

    split_dict = dataset.get_idx_split()
    dataset.split_idxs = [split_dict['train'],
                          split_dict['valid'],
                          split_dict['test']]
    return dataset


def preformat_VOCSuperpixels(dataset_dir, name, slic_compactness):
    dataset = _join_splits(
        [VOCSuperpixels(root=dataset_dir, name=name,
                        slic_compactness=slic_compactness, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_COCOSuperpixels(dataset_dir, name, slic_compactness):
    dataset = _join_splits(
        [COCOSuperpixels(root=dataset_dir, name=name,
                         slic_compactness=slic_compactness, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_ogbn(dataset_dir, name, cfg):
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
    if name == 'ogbn-arxiv':
        pre_transform_in_memory(dataset, add_reverse_edges)
        if cfg.prep.add_self_loops:
            pre_transform_in_memory(dataset, add_self_loops)
    split_dict = dataset.get_idx_split()
    split_dict['val'] = split_dict.pop('valid')
    dataset.split_idx = split_dict
    return dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _join_splits(datasets):
    """Join train/val/test PyG datasets into one with split_idxs attribute."""
    assert len(datasets) == 3
    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = ([datasets[0].get(i) for i in range(n1)] +
                 [datasets[1].get(i) for i in range(n2)] +
                 [datasets[2].get(i) for i in range(n3)])

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    datasets[0].split_idxs = [list(range(n1)),
                               list(range(n1, n1 + n2)),
                               list(range(n1 + n2, n1 + n2 + n3))]
    return datasets[0]


def _log_dataset_info(dataset, fmt, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{fmt}'")
    logging.info(f"  num graphs: {len(dataset)}")
    try:
        if dataset._data.x is not None:
            logging.info(f"  num node features: {dataset.num_node_features}")
        if dataset._data.edge_attr is not None:
            logging.info(f"  num edge features: {dataset.num_edge_features}")
    except Exception:
        pass
