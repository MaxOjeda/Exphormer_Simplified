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

    def __init__(self, in_dim, out_dim, num_heads, use_bias, dim_edge=None, use_virt_nodes=False):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not divisible by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias

        if dim_edge is None:
            dim_edge = in_dim

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(dim_edge, self.out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, batch, edge_index):
        src = batch.K_h[edge_index[0].to(torch.long)]   # (num_edges) x num_heads x out_dim
        dest = batch.Q_h[edge_index[1].to(torch.long)]  # (num_edges) x num_heads x out_dim
        score = torch.mul(src, dest)

        score = score / np.sqrt(self.out_dim)
        score = torch.mul(score, batch.E)
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))

        msg = batch.V_h[edge_index[0].to(torch.long)] * score
        batch.wV = torch.zeros_like(batch.V_h)
        scatter(msg, edge_index[1], dim=0, out=batch.wV, reduce='add')

        batch.Z = score.new_zeros(batch.V_h.size(0), self.num_heads, 1)
        scatter(score, edge_index[1], dim=0, out=batch.Z, reduce='add')

    def forward(self, batch):
        edge_attr = batch.expander_edge_attr
        edge_index = batch.expander_edge_index
        h = batch.x
        num_node = batch.batch.shape[0]
        if self.use_virt_nodes:
            h = torch.cat([h, batch.virt_h], dim=0)
            edge_index = torch.cat([edge_index, batch.virt_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr, batch.virt_edge_attr], dim=0)

        Q_h = self.Q(h)
        K_h = self.K(h)
        E = self.E(edge_attr)
        V_h = self.V(h)

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch, edge_index)

        h_out = batch.wV / (batch.Z + 1e-6)
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
