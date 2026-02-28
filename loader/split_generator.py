"""
Train/val/test split generation.
Adapted from graphgps/loader/split_generator.py — cfg passed as argument.
"""
import logging
import torch
from sklearn.model_selection import KFold, StratifiedKFold


def prepare_splits(dataset, cfg):
    """
    Verify or generate train/val/test splits for the dataset.
    Modifies dataset in-place by setting train_mask/val_mask/test_mask or
    train_graph_index/val_graph_index/test_graph_index.
    """
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')
    else:
        split_mode = cfg.dataset.split_mode
        if split_mode == 'standard':
            setup_standard_split(dataset)
        elif split_mode == 'random':
            setup_random_split(dataset, cfg)
        elif split_mode.startswith('cv-'):
            k = int(split_mode.split('-')[1])
            setup_cv_split(dataset, cfg, k)
        else:
            raise ValueError(f"Unknown split mode: {split_mode}")


def set_dataset_splits(dataset, split_idxs):
    """
    Apply given split indices to dataset as train/val/test index attributes.
    """
    assert len(split_idxs) == 3, "split_idxs must be [train, val, test]"
    train_idxs, val_idxs, test_idxs = split_idxs

    # Determine if this is a graph-level or node-level task by inspecting data.
    data0 = dataset[0]
    if hasattr(data0, 'y') and data0.y is not None and data0.y.shape[0] > 1:
        # Node-level task: set masks
        # Actually for graph-level we set graph_index attributes
        pass

    # Use graph-level index attributes
    dataset.data['train_graph_index'] = torch.tensor(train_idxs, dtype=torch.long)
    dataset.data['val_graph_index'] = torch.tensor(val_idxs, dtype=torch.long)
    dataset.data['test_graph_index'] = torch.tensor(test_idxs, dtype=torch.long)


def setup_standard_split(dataset):
    """Use pre-defined splits already in the dataset (train_mask etc.)."""
    # GNNBenchmarkDataset has splits stored as masks in each data object
    # We construct index arrays from those masks
    data0 = dataset[0]
    if hasattr(data0, 'train_mask'):
        # Node-level with masks already set
        return
    logging.warning("Standard split requested but no train_mask found in data.")


def setup_random_split(dataset, cfg, train_ratio=None, val_ratio=None):
    """Generate random train/val/test split of graphs."""
    split = cfg.dataset.split
    if train_ratio is None:
        train_ratio = split[0]
    if val_ratio is None:
        val_ratio = split[1]

    n = len(dataset)
    idx = torch.randperm(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idxs = idx[:n_train].tolist()
    val_idxs = idx[n_train:n_train + n_val].tolist()
    test_idxs = idx[n_train + n_val:].tolist()
    set_dataset_splits(dataset, [train_idxs, val_idxs, test_idxs])


def setup_cv_split(dataset, cfg, k):
    """Set up k-fold cross-validation split."""
    split_index = cfg.dataset.split_index
    n = len(dataset)
    kf = KFold(n_splits=k, shuffle=True, random_state=cfg.seed)
    all_splits = list(kf.split(range(n)))
    train_val_idx, test_idx = all_splits[split_index]
    val_size = int(len(train_val_idx) * 0.1)
    val_idx = train_val_idx[:val_size]
    train_idx = train_val_idx[val_size:]
    set_dataset_splits(dataset, [train_idx.tolist(), val_idx.tolist(), test_idx.tolist()])
