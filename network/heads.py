"""
Prediction heads for Exphormer_Max.
Replaces graphgym's head_dict registry.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


def _build_mlp(dim_in, dim_out, num_layers, dropout=0.0, act='relu'):
    """Build a simple MLP."""
    assert num_layers >= 1
    act_fn = {'relu': nn.ReLU, 'gelu': nn.GELU}.get(act, nn.ReLU)
    layers = []
    if num_layers == 1:
        layers.append(nn.Linear(dim_in, dim_out))
    else:
        layers += [nn.Linear(dim_in, dim_in), act_fn(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(dim_in, dim_in), act_fn(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dim_in, dim_out))
    return nn.Sequential(*layers)


class GraphHead(nn.Module):
    """
    Graph-level prediction head (replaces graphgym's 'default' head).
    Global pooling → MLP → (pred, label).
    """

    _POOL = {
        'mean': global_mean_pool,
        'max':  global_max_pool,
        'add':  global_add_pool,
    }

    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        pooling = cfg.model.graph_pooling
        if pooling not in self._POOL:
            raise ValueError(f"Unknown graph_pooling: {pooling}")
        self.pool = self._POOL[pooling]
        self.mlp = _build_mlp(dim_in, dim_out,
                               num_layers=cfg.gnn.layers_post_mp,
                               dropout=cfg.gnn.dropout,
                               act=cfg.gnn.act)

    def forward(self, batch):
        x = self.pool(batch.x, batch.batch)
        pred = self.mlp(x)
        label = batch.y
        return pred, label


class InductiveNodeHead(nn.Module):
    """
    Node-level prediction head (replaces graphgym's 'inductive_node' head).
    MLP → (pred, label).
    """

    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.mlp = _build_mlp(dim_in, dim_out,
                               num_layers=cfg.gnn.layers_post_mp,
                               dropout=cfg.gnn.dropout,
                               act=cfg.gnn.act)

    def forward(self, batch):
        pred = self.mlp(batch.x)
        label = batch.y
        return pred, label


def build_head(cfg, dim_in, dim_out):
    """Factory: return the appropriate head."""
    head_name = cfg.gnn.head
    if head_name == 'default':
        return GraphHead(dim_in, dim_out, cfg)
    elif head_name == 'inductive_node':
        return InductiveNodeHead(dim_in, dim_out, cfg)
    else:
        raise ValueError(f"Unknown head: '{head_name}'. Supported: default, inductive_node")
