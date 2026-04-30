"""
Exphormer sparse attention layer.
Copied from graphgps/layer/Exphormer.py — graphgym dependencies removed.

V mechanism (KGC mode, use_query_conditioning=True):
    V_h  = h  (identity — no W_V projection)
    gate = gate_base[r_uv] + fc_zq(query_emb).view(R+1, d)[r_uv]   (bilinear cross-term)
    msg  = h[src] * gate * score
  This makes msg = h_w ⊙ gate(r_uv, r_q) × score — same structure as NBFNet DistMult
  with a *full-rank* (r_uv × r_q) cross-term (Cambio C2).

  gate_base : nn.Parameter (R+1, d_out) — per-relation baseline (std=1 init), no
              query dependence. Plays the role the old V_gate(edge_attr) had.
  fc_zq     : Linear(d_in, (R+1)*d_out) — produces M_q (B, R+1, d_out) per query;
              std=0.01 init so bilinear modulation starts subtle and grows in training.
  Indexed per-edge by batch.edge_rel_idx (sentinel = num_relations for expander edges,
  built in ExpEdgeFixer).

  Replaces the previous additive gate `V_gate(edge_attr) + proj_vg(query_emb)` which
  had only rank-1 in the (r_uv, r_q) cross-term. KnowFormer's fc_z(z) is the same
  construction (`fc_z(z).reshape(R, d)[r]`).

  h starts from 0 for non-anchors, so V is purely relational at layer 0 and stays
  relational in subsequent layers. W_V is absent: no entity-specific projection.
  Q/K conditioned by +proj_q/k(query_emb) per-node bias.
  batch.query_emb (B, d) is set once by MultiModel.forward() from the single
  shared query_rel_emb table; this layer has no own embedding table.

  Requires: batch.edge_rel_idx (set by ExpEdgeFixer when num_relations is not None)
            num_virt_node = 0 (virt nodes incompatible — no edge_rel_idx for virt edges)

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
                 num_relations=None):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not divisible by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias
        self.use_query_conditioning = use_query_conditioning

        if dim_edge is None:
            dim_edge = in_dim

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(dim_edge, self.out_dim * num_heads, bias=use_bias)

        if use_query_conditioning:
            # KGC mode: no W_V — V = h directly (NBFNet-style identity pass-through).
            # gate provides the relational modulation: msg = h ⊙ gate(r_uv,r_q) × score.
            if use_virt_nodes:
                raise ValueError(
                    "use_query_conditioning=True is incompatible with use_virt_nodes=True: "
                    "the bilinear gate indexes by batch.edge_rel_idx, which is not built "
                    "for virt edges. Set num_virt_node=0 in KGC configs.")
            if num_relations is None:
                raise ValueError(
                    "num_relations must be passed when use_query_conditioning=True. "
                    "Thread cfg.dataset.num_relations through MultiModel/MultiLayer/GlobalModel.")
            d_out = self.out_dim * num_heads

            # Per-layer projections applied to batch.query_emb (B, in_dim).
            # No per-layer embedding table — the single shared table lives in MultiModel.
            self.proj_q = nn.Linear(in_dim, d_out, bias=False)
            self.proj_k = nn.Linear(in_dim, d_out, bias=False)
            nn.init.normal_(self.proj_q.weight, std=0.01)
            nn.init.normal_(self.proj_k.weight, std=0.01)

            # C2 — Bilinear V gate.
            # gate(r, q) = gate_base[r] + M_q[r], with M_q = fc_zq(query_emb).view(R+1, d_out).
            # +1 slot is the sentinel index for expander/non-KG edges (set by ExpEdgeFixer).
            # Replaces additive `V_gate(edge_attr) + proj_vg(query_emb)` which had only
            # rank-1 in the (r, q) cross. Bilinear has full rank.
            # No sigmoid (preserves the +0.08 MRR finding from sesión 1).
            self.num_relation_slots = num_relations + 1
            self.gate_base = nn.Parameter(torch.empty(self.num_relation_slots, d_out))
            nn.init.normal_(self.gate_base, std=1.0)

            self.fc_zq = nn.Linear(in_dim, self.num_relation_slots * d_out, bias=False)
            nn.init.normal_(self.fc_zq.weight, std=0.01)
        else:
            # Non-KGC mode: standard learned V projection.
            self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

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
        V_h = h if self.use_query_conditioning else self.V(h)

        if self.use_query_conditioning and hasattr(batch, 'query_emb'):
            # Broadcast batch.query_emb (B, in_dim) → per-node and per-edge.
            src_e      = edge_index[0].clamp(max=num_node - 1).long()
            edge_graph = batch.batch[src_e]

            shared_node = batch.query_emb[batch.batch]    # (N, in_dim)

            Q_cond = self.proj_q(shared_node)
            K_cond = self.proj_k(shared_node)

            Q_h = Q_h + Q_cond
            K_h = K_h + K_cond

            # C2 — Bilinear V gate (1D-flat indexing for backward speed).
            # Equivalent to: gate[e] = gate_base[r_uv[e]] + M_q[graph_idx[e], r_uv[e]]
            # with M_q = fc_zq(query_emb).view(B, R+1, d_out), but we flatten the (B, R+1)
            # axis so that backward becomes a single 1D scatter_add (CUDA-optimized) instead
            # of 2D advanced-indexing gather/scatter. Same math, ~3-5× faster on H100.
            # No sigmoid — applied directly as multiplicative factor on V_h.
            d_out      = self.out_dim * self.num_heads
            M_q_flat   = self.fc_zq(batch.query_emb).view(-1, d_out)      # (B*(R+1), d_out)
            edge_rel   = batch.edge_rel_idx.long()                        # (E,)
            flat_idx   = edge_graph * self.num_relation_slots + edge_rel  # (E,)
            gate       = (self.gate_base.index_select(0, edge_rel)
                          + M_q_flat.index_select(0, flat_idx))           # (E, d_out)
            batch.E_gate = gate.view(-1, self.num_heads, self.out_dim)
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
