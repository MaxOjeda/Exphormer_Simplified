"""
Exphormer sparse attention layer (clean QC-Exphormer architecture for KGC).

Query-conditioned sparse attention over the interaction graph H = KG ∪ expander.
This is the architecture of the best transductive result (WN18RR MRR 0.566),
stripped of all experimental flags (no nbf_v / distmult_v / rel_matrix_v / pna /
gate_rel_mult / alpha_mix_qk / inductive_routing / V-NBF / FiLM).

KGC mode (use_query_conditioning=True and batch.x0 present):
    Q_v = W_Q(h0_v) + proj_q(c_q)        # anchored to boundary condition h0
    K_w = W_K(h_w)  + proj_k(c_q)        # entity routing + query bias (standard)
    E   = W_E(φ(r)) + proj_e(c_q)        # additive query conditioning (NOT FiLM)
    g   = V_gate(φ(r)) + proj_vg(c_q)    # additive gate, NO sigmoid
    V_w = W_V(h_w)
    s   = exp(clip((Q ⊙ K ⊙ E)·1 / √d, -5, 5))
    m_v = Σ_w s_{w→v} · (V_w ⊙ g_{wv})   # sum aggregation (NBFNet-style, no /Z)

c_q = shared_rel_emb_table[r_q] is a per-layer query-relation embedding.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class ExphormerAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, use_bias, dim_edge=None,
                 use_virt_nodes=False, use_edge_gating=False,
                 use_query_conditioning=False, num_relations=None):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not divisible by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias
        self.use_edge_gating = use_edge_gating
        self.use_query_conditioning = use_query_conditioning

        if dim_edge is None:
            dim_edge = in_dim

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(dim_edge, self.out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

        if use_edge_gating:
            self.V_gate = nn.Linear(dim_edge, self.out_dim * num_heads, bias=False)

        if use_query_conditioning:
            assert num_relations is not None, \
                "num_relations must be provided when use_query_conditioning=True"
            d_out = self.out_dim * num_heads
            # Per-layer query-relation embedding, projected separately into the
            # Q, K, E and gate spaces. Near-zero init keeps training stable at start.
            self.shared_rel_emb_table = nn.Embedding(num_relations, in_dim)
            nn.init.normal_(self.shared_rel_emb_table.weight, mean=0.0, std=0.01)
            self.proj_q = nn.Linear(in_dim, d_out, bias=False)
            self.proj_k = nn.Linear(in_dim, d_out, bias=False)
            self.proj_e = nn.Linear(in_dim, d_out, bias=False)
            nn.init.normal_(self.proj_q.weight, std=0.01)
            nn.init.normal_(self.proj_k.weight, std=0.01)
            nn.init.normal_(self.proj_e.weight, std=0.01)
            if use_edge_gating:
                self.proj_vg = nn.Linear(in_dim, d_out, bias=False)
                nn.init.normal_(self.proj_vg.weight, std=0.01)

    def propagate_attention(self, batch, edge_index):
        src = batch.K_h[edge_index[0].to(torch.long)]   # (E, heads, out_dim)
        dest = batch.Q_h[edge_index[1].to(torch.long)]  # (E, heads, out_dim)
        score = torch.mul(src, dest)

        score = score / np.sqrt(self.out_dim)
        score = torch.mul(score, batch.E)
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))

        # Value per edge, optionally gated by the (relation, query) gate.
        v_src = batch.V_h[edge_index[0].to(torch.long)]  # (E, heads, out_dim)
        if self.use_edge_gating:
            v_src = v_src * batch.E_gate                 # gate, no sigmoid

        msg = v_src * score
        batch.wV = torch.zeros_like(batch.V_h)
        scatter(msg, edge_index[1], dim=0, out=batch.wV, reduce='add')

    def forward(self, batch):
        edge_attr = batch.expander_edge_attr
        edge_index = batch.expander_edge_index
        h = batch.x
        num_node = batch.batch.shape[0]
        if self.use_virt_nodes:
            h = torch.cat([h, batch.virt_h], dim=0)
            edge_index = torch.cat([edge_index, batch.virt_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr, batch.virt_edge_attr], dim=0)

        # Q/K routing (KGC mode, when batch.x0 exists):
        #   Q uses x0 (boundary condition): anchor gets rel_emb[q], others get 0.
        #   K uses accumulated h: carries neighborhood info, differentiates nodes.
        #   V always uses accumulated h for full expressivity.
        # Non-KGC tasks (no batch.x0) fall back to h for both Q and K.
        if hasattr(batch, 'x0'):
            x0 = batch.x0                                        # (num_node, in_dim)
            if self.use_virt_nodes and h.shape[0] > num_node:
                pad = x0.new_zeros(h.shape[0] - num_node, x0.shape[1])
                x0_q = torch.cat([x0, pad], dim=0)
            else:
                x0_q = x0
            Q_h = self.Q(x0_q)
            K_h = self.K(h)
        else:
            Q_h = self.Q(h)
            K_h = self.K(h)
        E = self.E(edge_attr)           # (num_edges, heads * out_dim)
        V_h = self.V(h)                 # standard W_V projection

        # Query conditioning: per-layer relation embedding projected into Q, K, E.
        shared_edge = None
        if self.use_query_conditioning and hasattr(batch, 'query_relation'):
            query_per_node = batch.query_relation[batch.batch]   # (num_node,) rel idx

            # Clamp source node index to real nodes (safe when virtual nodes present).
            src = edge_index[0].clamp(max=num_node - 1).long()
            edge_graph = batch.batch[src]                        # (num_edges,) graph idx
            query_per_edge = batch.query_relation[edge_graph]    # (num_edges,) rel idx

            shared_node = self.shared_rel_emb_table(query_per_node)  # (num_node, in_dim)
            shared_edge = self.shared_rel_emb_table(query_per_edge)  # (num_edges, in_dim)
            Q_cond_bias  = self.proj_q(shared_node)              # (num_node, heads*out_dim)
            K_cond_bias  = self.proj_k(shared_node)
            E_query_bias = self.proj_e(shared_edge)              # (num_edges, heads*out_dim)
            if self.use_virt_nodes and h.shape[0] > num_node:
                pad = Q_h.new_zeros(h.shape[0] - num_node, Q_cond_bias.shape[1])
                Q_cond_bias = torch.cat([Q_cond_bias, pad], dim=0)
                K_cond_bias = torch.cat([K_cond_bias, pad], dim=0)

            Q_h = Q_h + Q_cond_bias
            K_h = K_h + K_cond_bias
            E = E + E_query_bias         # additive query conditioning (not FiLM)

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        if self.use_edge_gating:
            E_gate = self.V_gate(edge_attr)                      # (num_edges, heads*out_dim)
            if shared_edge is not None:
                # Additive query conditioning of the gate, no sigmoid.
                E_gate = E_gate + self.proj_vg(shared_edge)
            batch.E_gate = E_gate.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch, edge_index)

        h_out = batch.wV  # sum aggregation (NBFNet-style)
        h_out = h_out.view(-1, self.out_dim * self.num_heads)

        batch.virt_h = h_out[num_node:]
        h_out = h_out[:num_node]

        return h_out


def get_activation(activation):
    if activation == 'relu':
        return 2, nn.ReLU()
    elif activation == 'gelu':
        return 2, nn.GELU()
    elif activation == 'silu':
        return 2, nn.SiLU()
    elif activation == 'glu':
        return 1, nn.GLU()
    else:
        raise ValueError(f'activation function {activation} is not valid!')


class ExphormerFullLayer(nn.Module):
    """Exphormer attention + FFN (used as standalone layer)."""

    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0, dim_edge=None,
                 layer_norm=False, batch_norm=True,
                 activation='relu', residual=True,
                 use_bias=False, use_virt_nodes=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = ExphormerAttention(in_dim, out_dim, num_heads,
                                            use_bias=use_bias,
                                            dim_edge=dim_edge,
                                            use_virt_nodes=use_virt_nodes)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        factor, self.activation_fn = get_activation(activation=activation)
        self.FFN_h_layer2 = nn.Linear(out_dim * factor, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, batch):
        h = batch.x
        h_in1 = h

        h_attn_out = self.attention(batch)
        h = h_attn_out.view(-1, self.out_channels)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.residual:
            h = h_in1 + h

        if self.layer_norm:
            h = self.layer_norm1_h(h)
        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h

        h = self.FFN_h_layer1(h)
        h = self.activation_fn(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)
        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels, self.out_channels,
            self.num_heads, self.residual)
