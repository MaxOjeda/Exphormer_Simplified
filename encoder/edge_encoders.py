"""
Edge encoders for Exphormer_Max.
Merges: linear_edge_encoder, dummy_edge_encoder, voc_superpixels_encoder (edge part).
"""
import torch
import torch.nn as nn


class LinearEdgeEncoder(nn.Module):
    """Linear projection of edge features."""

    # Dataset-specific raw edge feature dimensions
    _EDGE_DIM = {
        'MNIST': 1, 'CIFAR10': 1,
        'ogbn-proteins': 8,
    }

    def __init__(self, dim_emb, dataset_name=None):
        super().__init__()
        dim_in = self._EDGE_DIM.get(dataset_name, 1)
        self.encoder = nn.Linear(dim_in, dim_emb)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.float())
        return batch


class DummyEdgeEncoder(nn.Module):
    """Single learnable embedding for edges that have no features."""

    def __init__(self, dim_emb):
        super().__init__()
        self.encoder = nn.Embedding(1, dim_emb)

    def forward(self, batch):
        dummy_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
        batch.edge_attr = self.encoder(dummy_attr)
        return batch


class VOCEdgeEncoder(nn.Module):
    """Edge encoder for VOC/COCO superpixel datasets."""

    def __init__(self, dim_emb, edge_input_dim=2):
        super().__init__()
        self.encoder = nn.Linear(edge_input_dim, dim_emb)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.float())
        return batch


class RelationEmbeddingEncoder(nn.Module):
    """
    Encodes integer relation IDs (batch.edge_attr) to dense vectors of dim_edge.

    Embedding table size = num_edge_types, which is set at load time:
      - reciprocal=True:  num_edge_types = num_relations (e.g. 22 for WN18RR)
      - reciprocal=False: num_edge_types = 2*num_relations (structural reverses added)

    Expander edges are NOT handled here — ExpanderEdgeFixer gives them a
    separate learnable embedding (exp_edge_attr) and concatenates it later.

    Args:
        num_edge_types (int): cfg.dataset.num_edge_types — exact embedding table size.
        dim_edge (int): Output embedding dimension (= gt.dim_edge).
    """

    def __init__(self, num_edge_types: int, dim_edge: int):
        super().__init__()
        self.emb = nn.Embedding(num_edge_types, dim_edge)

    def forward(self, batch):
        batch.edge_attr = self.emb(batch.edge_attr.long())  # (E,) → (E, dim_edge)
        return batch


def build_edge_encoder(cfg):
    """
    Return the correct edge encoder given cfg.dataset.edge_encoder_name.
    """
    name = cfg.dataset.edge_encoder_name
    dim_h = cfg.gt.dim_edge  # set equal to dim_hidden in config.py

    if name == 'RelationEmbedding':
        return RelationEmbeddingEncoder(
            num_edge_types=cfg.dataset.num_edge_types,
            dim_edge=cfg.gt.dim_edge,
        )

    elif name == 'LinearEdge':
        return LinearEdgeEncoder(dim_h, dataset_name=cfg.dataset.name)

    elif name == 'DummyEdge':
        return DummyEdgeEncoder(dim_h)

    elif name == 'VOCEdge':
        # VOC: 2 if edge_wt_region_boundary else 1
        edge_input_dim = 2 if cfg.dataset.name == 'edge_wt_region_boundary' else 1
        return VOCEdgeEncoder(dim_h, edge_input_dim=edge_input_dim)

    else:
        raise ValueError(f"Unknown edge encoder: '{name}'. "
                         f"Supported: RelationEmbedding, LinearEdge, DummyEdge, VOCEdge")
