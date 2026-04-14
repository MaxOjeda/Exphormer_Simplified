"""
Exphormer sparse attention layer.
Copied from graphgps/layer/Exphormer.py — graphgym dependencies removed.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class ExphormerAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, use_bias, dim_edge=None,
                 use_virt_nodes=False, use_edge_gating=False,
                 use_query_conditioning=False, num_relations=None,
                 gate_rel_mult=False, use_alpha_mix_qk=False,
                 inductive_routing=False, use_nbf_v=False, use_pna=False,
                 use_distmult_v=False):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not divisible by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias
        self.use_edge_gating = use_edge_gating
        self.use_query_conditioning = use_query_conditioning
        # Multiplicative gate conditioning (vs additive).
        self.gate_rel_mult = gate_rel_mult
        # α-blend for Q/K: α*x0 + (1-α)*h per layer (only used when batch.x0 exists).
        self.use_alpha_mix_qk = use_alpha_mix_qk
        if use_alpha_mix_qk:
            # Init at 1.0 → sigmoid(1.0) ≈ 0.73: starts mostly x0.
            self.alpha_qk = nn.Parameter(torch.ones(1))

        # Inductive routing: K_h = proj_k(shared_rel[r_q]) only — no W_K(h).
        # Score becomes a function of (r_ij, r_q, Q_j) but NOT of K_i specific to the
        # train graph. Eliminates the main source of non-inductive generalisation.
        # Requires use_query_conditioning=True (needs proj_k and shared_rel_emb_table).
        self.inductive_routing = inductive_routing

        # NBFNet-style DistMult V: msg(i→j,r) = h_i ⊙ W_r
        # Replaces W_V + V_gate entirely. W_r is a free embedding per relation type,
        # learned directly from gradients — no shared linear projection across relations.
        # Requires num_relations (passed via use_query_conditioning path or standalone).
        self.use_nbf_v = use_nbf_v
        if use_nbf_v:
            assert num_relations is not None, "use_nbf_v requires num_relations"
            assert not use_edge_gating, "use_nbf_v and use_edge_gating are mutually exclusive"
            # +1 for expander edge sentinel (index = num_relations, near-zero init row)
            self.nbf_rel_emb = nn.Embedding(num_relations + 1, self.out_dim * num_heads)
            nn.init.xavier_uniform_(self.nbf_rel_emb.weight.view(
                num_relations + 1, num_heads, self.out_dim
            ).reshape(num_relations + 1, -1))

        # DistMult-inside-attention: msg(i→j,r) = W_V(h_i) ⊙ W_r
        # Keeps the W_V projection (unlike use_nbf_v which uses raw h) for stability,
        # then applies relation-specific element-wise scaling after projection.
        # This lets the model select dimensions per relation — the gate V_gate(r)
        # only scales post-projection and cannot unmix projected dimensions.
        # Mutually exclusive with use_nbf_v and use_edge_gating.
        self.use_distmult_v = use_distmult_v
        if use_distmult_v:
            assert num_relations is not None, "use_distmult_v requires num_relations"
            assert not use_edge_gating, "use_distmult_v and use_edge_gating are mutually exclusive"
            assert not use_nbf_v, "use_distmult_v and use_nbf_v are mutually exclusive"
            # +1 for expander edge sentinel (index = num_relations)
            self.msg_rel_emb = nn.Embedding(num_relations + 1, self.out_dim * num_heads)
            nn.init.xavier_uniform_(self.msg_rel_emb.weight.view(
                num_relations + 1, num_heads, self.out_dim
            ).reshape(num_relations + 1, -1))

        if dim_edge is None:
            dim_edge = in_dim

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(dim_edge, self.out_dim * num_heads, bias=use_bias)
        if not use_nbf_v:
            self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

        if use_edge_gating:
            self.V_gate = nn.Linear(dim_edge, self.out_dim * num_heads, bias=False)

        if use_query_conditioning:
            assert num_relations is not None, \
                "num_relations must be provided when use_query_conditioning=True"
            d_out = self.out_dim * num_heads
            # Shared relation embedding: one table in in_dim space, projected separately
            # for Q conditioning, K conditioning, E conditioning, and V gate conditioning.
            # Near-zero init keeps training stable at init (projections ≈ 0).
            self.shared_rel_emb_table = nn.Embedding(num_relations, in_dim)
            nn.init.normal_(self.shared_rel_emb_table.weight, mean=0.0, std=0.01)
            self.proj_q  = nn.Linear(in_dim, d_out, bias=False)
            self.proj_k  = nn.Linear(in_dim, d_out, bias=False)
            self.proj_e  = nn.Linear(in_dim, d_out, bias=False)
            nn.init.normal_(self.proj_q.weight, std=0.01)
            nn.init.normal_(self.proj_k.weight, std=0.01)
            nn.init.normal_(self.proj_e.weight, std=0.01)
            if use_edge_gating:
                self.proj_vg = nn.Linear(in_dim, d_out, bias=False)
                nn.init.normal_(self.proj_vg.weight, std=0.01)

        # PNA aggregation: concat(sum, mean, max) → Linear → out_dim.
        # sum  = Σ score_i * v_i      (NBFNet-style)
        # mean = sum / Z              (softmax-normalized)
        # max  = max_i(score_i * v_i) (peak detection)
        self.use_pna = use_pna
        if use_pna:
            d_flat = self.out_dim * num_heads
            self.pna_proj = nn.Linear(3 * d_flat, d_flat, bias=False)

    def propagate_attention(self, batch, edge_index):
        src = batch.K_h[edge_index[0].to(torch.long)]   # (num_edges) x num_heads x out_dim
        dest = batch.Q_h[edge_index[1].to(torch.long)]  # (num_edges) x num_heads x out_dim
        score = torch.mul(src, dest)

        score = score / np.sqrt(self.out_dim)
        score = torch.mul(score, batch.E)
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))

        # Value for each edge.
        if self.use_distmult_v:
            # DistMult-inside-attention: msg(i→j,r) = W_V(h_i) ⊙ W_r
            # batch.V_h = W_V(h) reshaped to (N, heads, out_dim) — see forward().
            # msg_rel_emb[r] provides per-relation dimension selection after projection.
            rel_w = self.msg_rel_emb(batch.edge_rel_idx.to(torch.long))  # (E, heads*out_dim)
            rel_w = rel_w.view(-1, self.num_heads, self.out_dim)
            v_src = batch.V_h[edge_index[0].to(torch.long)] * rel_w      # (E, heads, out_dim)
        elif self.use_nbf_v:
            # NBFNet-style DistMult: msg(i→j,r) = h_i ⊙ W_r
            # batch.V_h contains raw h (no W_V projection) reshaped to (N, heads, out_dim).
            # batch.edge_rel_idx contains relation indices per combined edge (KG + expander).
            rel_w = self.nbf_rel_emb(batch.edge_rel_idx.to(torch.long))  # (E, heads*out_dim)
            rel_w = rel_w.view(-1, self.num_heads, self.out_dim)
            v_src = batch.V_h[edge_index[0].to(torch.long)] * rel_w      # (E, heads, out_dim)
        else:
            v_src = batch.V_h[edge_index[0].to(torch.long)]  # (E, heads, out_dim)
            if self.use_edge_gating:
                gate = batch.E_gate                           # (E, heads, out_dim)
                v_src = v_src * gate

        msg = v_src * score
        batch.wV = torch.zeros_like(batch.V_h)
        scatter(msg, edge_index[1], dim=0, out=batch.wV, reduce='add')

        batch.Z = score.new_zeros(batch.V_h.size(0), self.num_heads, 1)
        scatter(score, edge_index[1], dim=0, out=batch.Z, reduce='add')

        if self.use_pna:
            # max aggregation: nodes with no incoming msgs stay 0
            batch.wV_max = torch.zeros_like(batch.V_h)
            scatter(msg, edge_index[1], dim=0, out=batch.wV_max, reduce='max')

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
        #     → Q encodes "what we're searching for" — query-anchored signal.
        #   K uses accumulated h: from layer 1+ h carries neighborhood info for all nodes.
        #     → K encodes "what each node has" — structurally differentiated.
        #   V always uses accumulated h for full expressivity.
        # For non-KGC tasks (no batch.x0), falls back to h for both Q and K as before.
        if hasattr(batch, 'x0'):
            x0 = batch.x0                                        # (num_node, in_dim)
            if self.use_virt_nodes and h.shape[0] > num_node:
                pad = x0.new_zeros(h.shape[0] - num_node, x0.shape[1])
                x0_q = torch.cat([x0, pad], dim=0)
            else:
                x0_q = x0
            if self.use_alpha_mix_qk:
                alpha = torch.sigmoid(self.alpha_qk)
                h_q = alpha * x0_q + (1.0 - alpha) * h
            else:
                h_q = x0_q
            Q_h = self.Q(h_q)
            K_h = self.K(h)
        else:
            Q_h = self.Q(h)
            K_h = self.K(h)
        E = self.E(edge_attr)           # (num_edges, heads * out_dim)
        # Priority for values:
        #   1. batch.V_stream — from V-RMPNN (KnowFormer-style, if use_vrmpnn=True)
        #   2. use_nbf_v      — raw h (DistMult scaling in propagate_attention)
        #   3. default        — W_V(h) linear projection
        if hasattr(batch, 'V_stream'):
            V_h = batch.V_stream          # (N, dim) — anchor-conditioned NBFNet values
        elif self.use_nbf_v:
            V_h = h                        # raw h, DistMult applied in propagate_attention
        else:
            V_h = self.V(h)               # standard W_V projection

        # Query conditioning: shared relation embedding projected into Q, E, and V-gate spaces.
        query_per_edge = None
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
            E_query_bias = self.proj_e(shared_edge)              # (num_edges, heads*out_dim)

            K_cond_bias  = self.proj_k(shared_node)              # (num_node, heads*out_dim)
            if self.use_virt_nodes and h.shape[0] > num_node:
                pad = Q_h.new_zeros(h.shape[0] - num_node, Q_cond_bias.shape[1])
                Q_cond_bias = torch.cat([Q_cond_bias, pad], dim=0)
                K_cond_bias = torch.cat([K_cond_bias, pad], dim=0)
            Q_h = Q_h + Q_cond_bias

            if self.inductive_routing:
                # K is only a function of the query relation — not of h_i.
                # Eliminates train-graph-specific routing: score(i→j) no longer
                # depends on which specific node i is, only on (r_ij, r_q, Q_j).
                K_h = K_cond_bias
            else:
                K_h = K_h + K_cond_bias

            # Multiplicative conditioning on edge features.
            E = E * (1.0 + E_query_bias)

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        if self.use_edge_gating:
            E_gate = self.V_gate(edge_attr)                      # (num_edges, heads*out_dim)
            if shared_edge is not None:
                # Condition V gate on query relation via shared embedding.
                if self.gate_rel_mult:
                    E_gate = E_gate * (1.0 + self.proj_vg(shared_edge))
                else:
                    E_gate = E_gate + self.proj_vg(shared_edge)
            batch.E_gate = E_gate.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch, edge_index)

        if self.use_pna:
            d = self.out_dim * self.num_heads
            wV_sum  = batch.wV.view(-1, d)
            wV_mean = (batch.wV / (batch.Z + 1e-6)).view(-1, d)
            wV_max  = batch.wV_max.view(-1, d)
            h_out = self.pna_proj(torch.cat([wV_sum, wV_mean, wV_max], dim=-1))
        else:
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
