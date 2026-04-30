"""
Exphormer_Max model architecture.
Merges graphgps/network/multi_model.py + graphgps/layer/multi_model_layer.py.
All graphgym dependencies removed; replaced by direct imports and if/elif dispatch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_scatter import scatter

from layer.exphormer import ExphormerAttention
from layer.gatedgcn import GatedGCNLayer
from encoder.node_encoders import build_node_encoder
from encoder.edge_encoders import build_edge_encoder
from encoder.exp_edge_fixer import ExpanderEdgeFixer
from network.heads import build_head


# ---------------------------------------------------------------------------
# Feature encoder (node + edge + expander edge fixer)
# ---------------------------------------------------------------------------

class FeatureEncoder(nn.Module):
    """Encodes node features, edge features, and prepares expander edges."""

    def __init__(self, cfg, dim_in):
        super().__init__()
        self.dim_in = dim_in

        # Resolve dim_edge
        if cfg.gt.dim_edge is None:
            cfg.defrost()
            cfg.gt.dim_edge = cfg.gt.dim_hidden
            cfg.freeze()

        if cfg.dataset.node_encoder:
            self.node_encoder = build_node_encoder(cfg, dim_in)
            self.dim_in = cfg.gnn.dim_inner
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = nn.BatchNorm1d(cfg.gnn.dim_inner)

        if cfg.dataset.edge_encoder:
            self.edge_encoder = build_edge_encoder(cfg)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = nn.BatchNorm1d(cfg.gt.dim_edge)

        if 'Exphormer' in cfg.gt.layer_type:
            # num_relations in KGC mode: needed for sentinel index in batch.edge_rel_idx
            # and for the dim_edge projection if dim_hidden != dim_edge.
            _is_kgc = (cfg.dataset.format == 'KGC' and cfg.dataset.num_relations > 0)
            _exp_num_rel = cfg.dataset.num_relations if _is_kgc else None
            self.exp_edge_fixer = ExpanderEdgeFixer(
                add_edge_index=cfg.prep.add_edge_index,
                num_virt_node=cfg.prep.num_virt_node,
                dim_edge=cfg.gt.dim_edge,
                dim_hidden=cfg.gt.dim_hidden,
                num_relations=_exp_num_rel)

    def forward(self, batch):
        if hasattr(self, 'node_encoder'):
            batch = self.node_encoder(batch)
            if hasattr(self, 'node_encoder_bn'):
                batch.x = self.node_encoder_bn(batch.x)
        if hasattr(self, 'edge_encoder'):
            batch = self.edge_encoder(batch)
            if hasattr(self, 'edge_encoder_bn'):
                batch.edge_attr = self.edge_encoder_bn(batch.edge_attr)
        if hasattr(self, 'exp_edge_fixer'):
            batch = self.exp_edge_fixer(batch)
        return batch


# ---------------------------------------------------------------------------
# Local GNN model (MPNN layer)
# ---------------------------------------------------------------------------

class LocalModel(nn.Module):
    """Wraps a local message-passing GNN layer."""

    def __init__(self, dim_h, local_gnn_type, edge_type, edge_attr_type,
                 num_heads, equivstable_pe=False, dropout=0.0,
                 layer_norm=False, batch_norm=True):
        super().__init__()
        self.dim_h = dim_h
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.edge_type = edge_type
        self.edge_attr_type = edge_attr_type
        self.local_gnn_type = local_gnn_type

        if local_gnn_type == 'CustomGatedGCN':
            gnn_norm_type = 'layer' if layer_norm else ('batch' if batch_norm else 'none')
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=equivstable_pe,
                                             norm_type=gnn_norm_type)
        elif local_gnn_type == 'GCN':
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(
                pygnn.Linear(dim_h, dim_h), nn.ReLU(),
                pygnn.Linear(dim_h, dim_h))
            self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(dim_h, dim_h // num_heads,
                                              heads=num_heads, edge_dim=dim_h)
        else:
            raise ValueError(f"Unsupported local GNN: {local_gnn_type}")

        if layer_norm and batch_norm:
            raise ValueError("Cannot use both layer_norm and batch_norm.")
        if layer_norm:
            self.norm1_local = nn.LayerNorm(dim_h)
        if batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h

        edge_index = getattr(batch, self.edge_type)
        edge_attr  = getattr(batch, self.edge_attr_type)

        if self.local_gnn_type == 'CustomGatedGCN':
            es_data = None
            if self.equivstable_pe:
                es_data = batch.pe_EquivStableLapPE
            local_out = self.local_model(Batch(batch=batch,
                                               x=h,
                                               edge_index=edge_index,
                                               edge_attr=edge_attr,
                                               pe_EquivStableLapPE=es_data))
            h_local = local_out.x
            setattr(batch, self.edge_attr_type, local_out.edge_attr)
        elif self.local_gnn_type == 'GCN':
            h_local = self.local_model(h, edge_index)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local
        else:
            h_local = self.local_model(h, edge_index, edge_attr)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local

        if self.layer_norm:
            h_local = self.norm1_local(h_local)
        if self.batch_norm:
            h_local = self.norm1_local(h_local)
        return h_local


# ---------------------------------------------------------------------------
# Global attention model (Exphormer)
# ---------------------------------------------------------------------------

class GlobalModel(nn.Module):
    """Wraps the Exphormer global sparse attention layer."""

    def __init__(self, dim_h, num_heads, dropout=0.0, attn_dropout=0.0,
                 layer_norm=False, batch_norm=True, exp_edges_cfg=None,
                 use_query_conditioning=False, num_relations=None):
        super().__init__()
        self.dim_h = dim_h
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        use_virt = (exp_edges_cfg is not None and exp_edges_cfg.num_virt_node > 0)
        self.self_attn = ExphormerAttention(
            dim_h, dim_h, num_heads,
            use_bias=False,
            use_virt_nodes=use_virt,
            use_query_conditioning=use_query_conditioning,
            num_relations=num_relations)

        if layer_norm and batch_norm:
            raise ValueError("Cannot use both layer_norm and batch_norm.")
        if layer_norm:
            self.norm1_attn = nn.LayerNorm(dim_h)
        if batch_norm:
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h
        h_attn = self.self_attn(batch)
        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn)
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        return h_attn


# ---------------------------------------------------------------------------
# Multi-layer (local + global combined with FFN)
# ---------------------------------------------------------------------------

class MultiLayer(nn.Module):
    """
    Combines one or more local/global sub-models per layer,
    followed by a 2-layer Feed-Forward block.

    gt.layer_type is a '+'-separated string, e.g. 'CustomGatedGCN+Exphormer'.
    """

    def __init__(self, dim_h, model_types, num_heads,
                 equivstable_pe=False, dropout=0.0, attn_dropout=0.0,
                 layer_norm=False, batch_norm=True, exp_edges_cfg=None,
                 use_query_conditioning=False, num_relations=None):
        super().__init__()
        self.dim_h = dim_h
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.model_types = model_types

        models = []
        for layer_spec in model_types:
            parts = layer_spec.split('__')
            if len(parts) == 3:
                layer_type, edge_type, edge_attr_type = parts
            elif len(parts) == 2:
                layer_type, edge_type = parts
                edge_attr_type = None
            else:
                layer_type    = parts[0]
                edge_type     = 'edge_index'
                edge_attr_type = 'edge_attr'

            if layer_type == 'Exphormer':
                models.append(GlobalModel(
                    dim_h=dim_h, num_heads=num_heads,
                    dropout=dropout, attn_dropout=attn_dropout,
                    layer_norm=layer_norm, batch_norm=batch_norm,
                    exp_edges_cfg=exp_edges_cfg,
                    use_query_conditioning=use_query_conditioning,
                    num_relations=num_relations))
            elif layer_type in ('CustomGatedGCN', 'GCN', 'GINE', 'GAT'):
                models.append(LocalModel(
                    dim_h=dim_h, local_gnn_type=layer_type,
                    edge_type=edge_type, edge_attr_type=edge_attr_type,
                    num_heads=num_heads, equivstable_pe=equivstable_pe,
                    dropout=dropout, layer_norm=layer_norm, batch_norm=batch_norm))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        self.models = nn.ModuleList(models)

        # 2-layer Feed-Forward block (always active)
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if layer_norm:
            self.norm2 = nn.LayerNorm(dim_h)
        if batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h_out_list = []
        for model in self.models:
            h_out_list.append(model(batch))

        h = sum(h_out_list)

        # FFN
        h = h + self.ff_dropout2(
            self.ff_linear2(
                self.ff_dropout1(F.relu(self.ff_linear1(h)))))

        if self.layer_norm:
            h = self.norm2(h)
        if self.batch_norm:
            h = self.norm2(h)

        # Bellman-Ford residual: re-inject initial representation at each layer.
        # x0_anchor = rel_emb_enc[r_q], x0_others = 0 → anchor signal maintained.
        if hasattr(batch, 'x0'):
            h = h + batch.x0

        batch.x = h
        return batch

    def extra_repr(self):
        return f'dim_h={self.dim_h}, model_types={self.model_types}'


# ---------------------------------------------------------------------------
# Top-level MultiModel
# ---------------------------------------------------------------------------

class MultiModel(nn.Module):
    """
    Full Exphormer model: FeatureEncoder → N×MultiLayer → Head.
    """

    def __init__(self, cfg, dim_in, dim_out):
        super().__init__()

        use_query_cond = getattr(cfg.gt, 'use_query_conditioning', False)

        # Single canonical query-relation embedding (KGC mode only).
        # Looked up once per forward() → batch.query_emb (B, d).
        # All downstream modules (KGCNodeEncoder, ExphormerAttention × L,
        # ExpanderEdgeFixer, KGCHead) read batch.query_emb and apply
        # their own per-role linear projections. KnowFormer-style design.
        if use_query_cond and cfg.dataset.num_relations > 0:
            self.query_rel_emb = nn.Embedding(cfg.dataset.num_relations, cfg.gt.dim_hidden)
            nn.init.normal_(self.query_rel_emb.weight, std=0.01)

        self.encoder = FeatureEncoder(cfg, dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            pre_mp_layers = []
            for _ in range(cfg.gnn.layers_pre_mp):
                pre_mp_layers += [nn.Linear(dim_in, cfg.gnn.dim_inner), nn.ReLU()]
                dim_in = cfg.gnn.dim_inner
            self.pre_mp = nn.Sequential(*pre_mp_layers)

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner, \
            f"gt.dim_hidden ({cfg.gt.dim_hidden}) must equal gnn.dim_inner ({cfg.gnn.dim_inner})"
        assert cfg.gt.dim_hidden == dim_in, \
            (f"Model dim_in after encoder ({dim_in}) must equal gt.dim_hidden "
             f"({cfg.gt.dim_hidden}). Check node encoder output dim or layers_pre_mp.")

        model_types = cfg.gt.layer_type.split('+')

        # KGC mode: thread num_relations to ExphormerAttention for the bilinear gate (C2).
        _layer_num_rel = cfg.dataset.num_relations if (use_query_cond and cfg.dataset.num_relations > 0) else None

        self.layers = nn.Sequential(*[
            MultiLayer(
                dim_h=cfg.gt.dim_hidden,
                model_types=model_types,
                num_heads=cfg.gt.n_heads,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                exp_edges_cfg=cfg.prep,
                use_query_conditioning=use_query_cond,
                num_relations=_layer_num_rel)
            for _ in range(cfg.gt.layers)
        ])

        self.post_mp = build_head(cfg, cfg.gnn.dim_inner, dim_out)

        self.grad_checkpoint = getattr(cfg.train, 'grad_checkpoint', False)

    def forward(self, batch):
        # Single lookup — all downstream modules read batch.query_emb (B, d).
        query_emb = None
        if hasattr(self, 'query_rel_emb'):
            query_emb = self.query_rel_emb(batch.query_relation)
            batch.query_emb = query_emb

        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch.x = self.pre_mp(batch.x)

        # Save initial representation h(0) for Bellman-Ford residual in each layer.
        # x0_anchor = query_rel_emb[r_q], x0_others = 0 (set by KGCNodeEncoder).
        batch.x0 = batch.x

        if self.training and self.grad_checkpoint:
            from torch.utils.checkpoint import checkpoint
            # KnowFormer-style: remove query_emb from batch before the checkpoint loop
            # and pass it as an explicit positional arg to checkpoint(). PyTorch treats
            # explicit positional args as saved inputs (not retained activations), so
            # a requires_grad tensor from the shared embedding doesn't get serialized
            # across all L backward replays via the captured batch closure.
            if query_emb is not None:
                del batch.query_emb

            for layer in self.layers:
                x_in  = batch.x
                ea_in = batch.edge_attr

                if query_emb is not None:
                    def _run(x, ea, qemb, _layer=layer, _batch=batch):
                        _batch.x = x
                        _batch.edge_attr = ea
                        _batch.query_emb = qemb
                        result = _layer(_batch)
                        del _batch.query_emb
                        return result.x, result.edge_attr
                    x_out, ea_out = checkpoint(_run, x_in, ea_in, query_emb, use_reentrant=False)
                else:
                    def _run(x, ea, _layer=layer, _batch=batch):
                        _batch.x = x
                        _batch.edge_attr = ea
                        result = _layer(_batch)
                        return result.x, result.edge_attr
                    x_out, ea_out = checkpoint(_run, x_in, ea_in, use_reentrant=False)

                batch.x = x_out
                batch.edge_attr = ea_out

            # Restore for post_mp (KGCHead reads batch.query_emb).
            if query_emb is not None:
                batch.query_emb = query_emb
        else:
            batch = self.layers(batch)

        return self.post_mp(batch)


def create_model(cfg, dim_in, dim_out):
    """Factory function to create the model."""
    model_type = cfg.model.type
    if model_type == 'MultiModel':
        return MultiModel(cfg, dim_in, dim_out)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Only 'MultiModel' is supported.")
