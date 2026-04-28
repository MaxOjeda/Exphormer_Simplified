"""
Exphormer sparse attention layer.
Copied from graphgps/layer/Exphormer.py — graphgym dependencies removed.

V mechanism: W_V(h) * gate(r_uv, r_q)  [architecture that achieved 0.566 transductive MRR]
  KGC mode (use_query_conditioning=True):
      V_h  = W_V(h)
      gate = V_gate(edge_attr) + proj_vg(batch.query_emb)   (no sigmoid — critical)
      msg  = V_h[src] * gate * score
  batch.query_emb (B, d) is set once by MultiModel.forward() from the single
  shared query_rel_emb table; this layer has no own embedding table.
  Non-KGC mode: standard V = W_V(h), no gate.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class ExphormerAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, use_bias, dim_edge=None,
                 use_virt_nodes=False, use_query_conditioning=False,
                 use_film_e=True):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not divisible by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias
        self.use_query_conditioning = use_query_conditioning
        self.use_film_e = use_film_e

        if dim_edge is None:
            dim_edge = in_dim

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(dim_edge, self.out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

        if use_query_conditioning:
            d_out = self.out_dim * num_heads

            # Per-layer projections applied to batch.query_emb (B, in_dim).
            # No per-layer embedding table — the single shared table lives in MultiModel.
            self.proj_q = nn.Linear(in_dim, d_out, bias=False)
            self.proj_k = nn.Linear(in_dim, d_out, bias=False)
            if use_film_e:
                self.proj_e = nn.Linear(in_dim, d_out, bias=False)
            nn.init.normal_(self.proj_q.weight, std=0.01)
            nn.init.normal_(self.proj_k.weight, std=0.01)
            if use_film_e:
                nn.init.normal_(self.proj_e.weight, std=0.01)

            # V gate: gate(r_uv, r_q) = V_gate(edge_attr) + proj_vg(query_emb)
            # Applied WITHOUT sigmoid — removing sigmoid was +0.08 MRR (CLAUDE.md).
            self.V_gate = nn.Linear(dim_edge, d_out, bias=False)
            self.proj_vg = nn.Linear(in_dim, d_out, bias=False)
            nn.init.normal_(self.proj_vg.weight, std=0.01)

    def propagate_attention(self, batch, edge_index):
        src_idx = edge_index[0].long()
        dst_idx = edge_index[1].long()

        src  = batch.K_h[src_idx]   # (E, heads, out_dim)
        dest = batch.Q_h[dst_idx]   # (E, heads, out_dim)
        score = torch.mul(src, dest)
        score = score / np.sqrt(self.out_dim)
        score = torch.mul(score, batch.E)
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))

        v_src = batch.V_h[src_idx]                      # (E, heads, out_dim)
        if hasattr(batch, 'E_gate'):
            v_src = v_src * batch.E_gate                # gate is already (E, heads, out_dim)

        msg = v_src * score
        batch.wV = torch.zeros_like(batch.V_h)
        scatter(msg, dst_idx, dim=0, out=batch.wV, reduce='add')

        batch.Z = score.new_zeros(batch.V_h.size(0), self.num_heads, 1)
        scatter(score, dst_idx, dim=0, out=batch.Z, reduce='add')

    def forward(self, batch):
        edge_attr  = batch.expander_edge_attr
        edge_index = batch.expander_edge_index
        h = batch.x
        num_node = batch.batch.shape[0]

        if self.use_virt_nodes:
            h = torch.cat([h, batch.virt_h], dim=0)
            edge_index = torch.cat([edge_index, batch.virt_edge_index], dim=1)
            edge_attr  = torch.cat([edge_attr,  batch.virt_edge_attr],  dim=0)

        # Q: anchored to x0 (boundary condition) — anchor gets W_Q(query_emb[i]),
        # non-anchor nodes get proj_q(query_emb[i]) only (x0=0 for them).
        if hasattr(batch, 'x0'):
            x0 = batch.x0
            if self.use_virt_nodes and h.shape[0] > num_node:
                pad = x0.new_zeros(h.shape[0] - num_node, x0.shape[1])
                h_q = torch.cat([x0, pad], dim=0)
            else:
                h_q = x0
        else:
            h_q = h
        Q_h = self.Q(h_q)
        K_h = self.K(h)
        E   = self.E(edge_attr)
        V_h = self.V(h)

        if self.use_query_conditioning and hasattr(batch, 'query_emb'):
            # Broadcast batch.query_emb (B, in_dim) → per-node and per-edge.
            src_e      = edge_index[0].clamp(max=num_node - 1).long()
            edge_graph = batch.batch[src_e]

            shared_node = batch.query_emb[batch.batch]    # (N, in_dim)
            shared_edge = batch.query_emb[edge_graph]     # (E, in_dim)

            Q_cond = self.proj_q(shared_node)
            K_cond = self.proj_k(shared_node)

            if self.use_virt_nodes and h.shape[0] > num_node:
                pad    = Q_h.new_zeros(h.shape[0] - num_node, Q_cond.shape[1])
                Q_cond = torch.cat([Q_cond, pad], dim=0)
                K_cond = torch.cat([K_cond, pad], dim=0)

            Q_h = Q_h + Q_cond
            K_h = K_h + K_cond
            if self.use_film_e:
                E = E * (1.0 + self.proj_e(shared_edge))

            # V gate: per-edge, function of (r_uv, r_q). No sigmoid.
            E_gate = self.V_gate(edge_attr) + self.proj_vg(shared_edge)
            batch.E_gate = E_gate.view(-1, self.num_heads, self.out_dim)
        elif hasattr(batch, 'E_gate'):
            del batch.E_gate

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E   = E.view(-1,   self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch, edge_index)

        h_out = batch.wV.view(-1, self.out_dim * self.num_heads)  # sum aggregation (NBFNet-style)

        batch.virt_h = h_out[num_node:]
        h_out        = h_out[:num_node]

        return h_out
