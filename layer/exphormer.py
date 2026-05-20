"""
Exphormer sparse attention layer.
Copied from graphgps/layer/Exphormer.py — graphgym dependencies removed.

KGC mode (use_query_conditioning=True) — Q/K stream + V-NBF stream (KnowFormer-aligned).

  Step 1 (Q/K stream): Q and K are no longer Linear projections of x0/h with a
    query-bias. They are produced by an NBF stream analogous to KnowformerQKLayer:

      qk_x = fc_qk_x(cat([h, noise], -1))             # (N, d), noise ~ N(0, qk_noise_std)
      for _ in range(num_qk_layers):                   # default 2
          qk_x = QKLayerNBF(qk_x, KG_edges, q_emb)     # DistMult scatter + MLP + LN + shortcut
      Q_h, K_h = fc_to_qk(qk_x).chunk(2, dim=-1)       # node-specific, relationally derived

    Why: in the prior design, Q_v ≈ proj_q(q) and K_v ≈ proj_k(q) for non-anchor
    nodes (since x0=0), making attention scores constant across non-anchor pairs.
    The NBF stream gives Q/K node-level discrimination via path propagation, the
    same property that makes KnowFormer work for inductive (0.752 MRR on WN18RR v1).

  V stream (V-NBF): V is recomputed from ZEROS each outer layer. No chaining.

      v_x = zeros(num_node, d); v_x[anchor] = 1.0       # fresh structural one-hot
      v_x = fc_v_x(cat([h, v_x], -1))                   # mix with accumulated h
      v_x = v_nbf(v_x);  v_x = v_nbf2(v_x)              # two DistMult scatters
      gate = gate_base[r_uv] + fc_zq(q).view(R+1,d)[r_uv]  (bilinear C2)
      msg_attn[v] = Σ_w score(Q,K,E) · v_x[w] · gate[r_uv]   (sum aggregation)

  Modules added by Step 1:
    fc_qk_x   : Linear(d_in + 1, d_in)        — projects [h, scalar_noise] to qk_x.
    qk_layers : ModuleList[QKLayerNBF × num_qk_layers]
    fc_to_qk  : Linear(d_in, 2 * d_out)       — splits final qk_x into (Q_h, K_h).

  All scatters are over batch.expander_edge_index (KG ∪ Expander). Step 3 of the
  plan splits this into separate KG-only edges for the NBF streams; not yet done.

  Requires: batch.edge_rel_idx (set by ExpEdgeFixer when num_relations is not None)
            batch.anchor_idx, batch.ptr (set by KGC dataloader)
            num_virt_node = 0 (incompatible with bilinear gate indexing)

Non-KGC mode: standard Q,K,V = W_{Q,K,V}(h), no gate, no streams.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class VLayerNBF(nn.Module):
    """
    One DistMult-style NBF iteration for the V-stream.

    output[v] = Σ_{(u,r,v) in edges} fc_z(query_emb).view(B, R+1, d)[b, r] ⊙ v_x[u]

    This is the same message operator as NBFNet (DistMult) but over KG∪Expander edges.
    The expander edges use the sentinel slot (index = num_relations) of fc_z.
    """

    def __init__(self, d, num_relation_slots):
        super().__init__()
        self.d = d
        self.num_relation_slots = num_relation_slots
        # Default Kaiming init: messages must be non-trivial at init
        # so the anchor signal propagates through V from the first step.
        self.fc_z = nn.Linear(d, num_relation_slots * d, bias=False)

    def forward(self, v_x, edge_index, edge_rel_idx, query_emb, edge_graph_idx):
        """
        v_x:            (N, d) — current V state
        edge_index:     (2, E) — src/dst into v_x
        edge_rel_idx:   (E,)  — relation index; expander sentinel = num_relations
        query_emb:      (B, d)
        edge_graph_idx: (E,)  — graph index of each edge's src node
        Returns:        (N, d) — v_x + aggregated messages (NBFNet BF residual)
        """
        src = edge_index[0].long()
        dst = edge_index[1].long()
        B, d = query_emb.shape

        z_all = self.fc_z(query_emb).view(B, self.num_relation_slots, d)   # (B, R+1, d)
        flat_idx = edge_graph_idx * self.num_relation_slots + edge_rel_idx.long()
        z_edge = z_all.reshape(-1, d).index_select(0, flat_idx)             # (E, d)

        msg = v_x[src] * z_edge   # DistMult: h[src] ⊙ z[r_uv]
        # NBFNet BF residual within the V stream: out[v] = v_x[v] + Σ msg.
        # Without this, anchor signal is erased after scatter (no self-loops).
        out = v_x.clone()
        scatter(msg, dst, dim=0, out=out, reduce='add')
        return out


class QKLayerNBF(nn.Module):
    """
    One DistMult NBF iteration for the Q/K-stream — analogous to
    KnowformerQKLayer (Knowformer/src/model.py:30).

    output[v] = LN(fc_out(scatter(z[r] ⊙ qk_x[u]) + alpha * qk_x[v])) + qk_x[v]

    Differences vs VLayerNBF:
      - fc_out: 2-layer MLP applied to (aggregated msgs + alpha * qk_x_self)
      - alpha:  learnable per-dim scalar mixing the self-message in
      - norm:   LayerNorm (per-node, inductive-safe)
      - shortcut: explicit residual added at the end

    These are the components missing from the bare VLayerNBF that give
    KnowFormer's Q/K stream its expressivity and stability.
    """

    def __init__(self, d, num_relation_slots):
        super().__init__()
        self.d = d
        self.num_relation_slots = num_relation_slots
        self.fc_z = nn.Linear(d, num_relation_slots * d, bias=False)
        self.fc_out = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.alpha = nn.Parameter(torch.empty(1, d))
        nn.init.normal_(self.alpha)
        self.norm = nn.LayerNorm(d)

    def forward(self, qk_x, edge_index, edge_rel_idx, query_emb, edge_graph_idx):
        src = edge_index[0].long()
        dst = edge_index[1].long()
        B, d = query_emb.shape

        z_all = self.fc_z(query_emb).view(B, self.num_relation_slots, d)
        flat_idx = edge_graph_idx * self.num_relation_slots + edge_rel_idx.long()
        z_edge = z_all.reshape(-1, d).index_select(0, flat_idx)

        msg = qk_x[src] * z_edge
        agg = torch.zeros_like(qk_x)
        scatter(msg, dst, dim=0, out=agg, reduce='add')

        x_shortcut = qk_x
        out = self.fc_out(agg + self.alpha * qk_x)
        out = self.norm(out)
        return out + x_shortcut


class ExphormerAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, use_bias, dim_edge=None,
                 use_virt_nodes=False, use_query_conditioning=False,
                 num_relations=None, qk_noise_std=4.0, num_qk_layers=2):
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

        d_out = self.out_dim * num_heads
        self.E = nn.Linear(dim_edge, d_out, bias=use_bias)

        if use_query_conditioning:
            if use_virt_nodes:
                raise ValueError(
                    "use_query_conditioning=True is incompatible with use_virt_nodes=True: "
                    "the bilinear gate indexes by batch.edge_rel_idx, which is not built "
                    "for virt edges. Set num_virt_node=0 in KGC configs.")
            if num_relations is None:
                raise ValueError(
                    "num_relations must be passed when use_query_conditioning=True. "
                    "Thread cfg.dataset.num_relations through MultiModel/MultiLayer/GlobalModel.")

            self.qk_noise_std = qk_noise_std
            self.num_qk_layers = num_qk_layers
            self.num_relation_slots = num_relations + 1

            # Step 1 — Q/K stream (KnowFormer-aligned).
            # qk_x = fc_qk_x([h, scalar_noise])  →  num_qk_layers × QKLayerNBF
            #   →  Q_h, K_h = fc_to_qk(qk_x).chunk(2, dim=-1)
            # Replaces the prior  Q = W_Q(x0) + proj_q(q),  K = W_K(x0) + proj_k(q)
            # which collapsed to a constant per query for all non-anchor nodes.
            self.fc_qk_x  = nn.Linear(in_dim + 1, in_dim, bias=False)
            self.qk_layers = nn.ModuleList([
                QKLayerNBF(in_dim, self.num_relation_slots)
                for _ in range(num_qk_layers)])
            self.fc_to_qk = nn.Linear(in_dim, 2 * d_out, bias=False)

            # C2 — Bilinear V gate.
            # gate(r, q) = gate_base[r] + M_q[r], M_q = fc_zq(query_emb).view(R+1, d_out).
            # No sigmoid (preserves the +0.08 MRR finding from sesion 1).
            self.gate_base = nn.Parameter(torch.empty(self.num_relation_slots, d_out))
            nn.init.normal_(self.gate_base, std=1.0)

            self.fc_zq = nn.Linear(in_dim, self.num_relation_slots * d_out, bias=False)
            nn.init.normal_(self.fc_zq.weight, std=0.01)

            # V-NBF stream: KnowFormer-aligned.
            # fc_v_x: mixes accumulated h with the one-hot anchor marker.
            # v_nbf, v_nbf2: two DistMult scatter iterations (2-hop per outer layer).
            self.fc_v_x = nn.Linear(in_dim * 2, in_dim, bias=False)
            self.v_nbf  = VLayerNBF(in_dim, self.num_relation_slots)
            self.v_nbf2 = VLayerNBF(in_dim, self.num_relation_slots)
        else:
            # Non-KGC mode: standard learned Q,K,V projections of h.
            self.Q = nn.Linear(in_dim, d_out, bias=use_bias)
            self.K = nn.Linear(in_dim, d_out, bias=use_bias)
            self.V = nn.Linear(in_dim, d_out, bias=use_bias)

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

        E = self.E(edge_attr)

        if self.use_query_conditioning and hasattr(batch, 'query_emb'):
            query_emb  = batch.query_emb                                  # (B, in_dim)
            src_e      = edge_index[0].clamp(max=num_node - 1).long()
            edge_graph = batch.batch[src_e]                               # (E,) graph idx
            d_out      = self.out_dim * self.num_heads

            # Step 1 — Q/K stream.
            # qk_x starts from [h, scalar_noise] mixed via fc_qk_x. The scalar
            # noise (training only) breaks symmetry between non-anchor nodes that
            # would otherwise be identical (h ≈ 0 in early layers). num_qk_layers
            # NBF iterations propagate relation-weighted messages, giving each
            # node a query-conditioned representation. fc_to_qk then splits into
            # (Q_h, K_h) via chunking, exactly like KnowformerLayer.
            h_node = h[:num_node]
            if self.training and self.qk_noise_std > 0.0:
                qk_noise = torch.randn(num_node, 1, device=h.device,
                                        dtype=h.dtype) * self.qk_noise_std
            else:
                qk_noise = h.new_zeros(num_node, 1)
            qk_x = self.fc_qk_x(torch.cat([h_node, qk_noise], dim=-1))
            for layer in self.qk_layers:
                qk_x = layer(qk_x, edge_index, batch.edge_rel_idx,
                              query_emb, edge_graph)
            qk_split = self.fc_to_qk(qk_x)                                 # (N, 2*d_out)
            Q_h = qk_split[:, :d_out]
            K_h = qk_split[:, d_out:]

            if self.use_virt_nodes and h.shape[0] > num_node:
                pad = Q_h.new_zeros(h.shape[0] - num_node, d_out)
                Q_h = torch.cat([Q_h, pad], dim=0)
                K_h = torch.cat([K_h, pad], dim=0)

            # C2 — Bilinear V gate (1D-flat indexing for backward speed).
            M_q_flat   = self.fc_zq(query_emb).view(-1, d_out)            # (B*(R+1), d_out)
            edge_rel   = batch.edge_rel_idx.long()                         # (E,)
            flat_idx   = edge_graph * self.num_relation_slots + edge_rel   # (E,)
            gate       = (self.gate_base.index_select(0, edge_rel)
                          + M_q_flat.index_select(0, flat_idx))            # (E, d_out)
            batch.E_gate = gate.view(-1, self.num_heads, self.out_dim)

            # V-NBF stream: KnowFormer-aligned.
            # Fresh zeros every outer layer — V never accumulates entity-specific state.
            # Anchor marked with 1.0 (structural one-hot). fc_v_x mixes with
            # accumulated h. Two internal NBF iterations give 2-hop coverage.
            anchor_global = (batch.ptr[:-1] + batch.anchor_idx).long()    # (B,)
            v_x = h.new_zeros(num_node, h.shape[-1])                       # fresh zeros
            v_x[anchor_global] = 1.0                                       # structural one-hot
            v_x = self.fc_v_x(torch.cat([h_node, v_x], dim=-1))            # mix with h
            v_x = self.v_nbf(v_x, edge_index, batch.edge_rel_idx,
                              query_emb, edge_graph)                        # 1st iteration
            v_x = self.v_nbf2(v_x, edge_index, batch.edge_rel_idx,
                               query_emb, edge_graph)                       # 2nd iteration
            V_h = v_x
        else:
            if hasattr(batch, 'E_gate'):
                del batch.E_gate
            Q_h = self.Q(h)
            K_h = self.K(h)
            V_h = self.V(h)

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E   = E.view(-1,   self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch, edge_index)

        h_out = batch.wV.view(-1, self.out_dim * self.num_heads)  # sum aggregation

        batch.virt_h = h_out[num_node:]
        h_out        = h_out[:num_node]

        return h_out
