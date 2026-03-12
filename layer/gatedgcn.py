"""
GatedGCN layer.
Copied from graphgps/layer/gatedgcn_layer.py — graphgym dependencies removed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
    GatedGCN layer.
    Residual Gated Graph ConvNets: https://arxiv.org/pdf/1711.07553.pdf
    """

    def __init__(self, in_dim, out_dim, dropout, residual, equivstable_pe=False,
                 norm_type='batch', **kwargs):
        super().__init__(**kwargs)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        if norm_type == 'layer':
            self.bn_node_x = nn.LayerNorm(out_dim)
            self.bn_edge_e = nn.LayerNorm(out_dim)
        elif norm_type == 'batch':
            self.bn_node_x = nn.BatchNorm1d(out_dim)
            self.bn_edge_e = nn.BatchNorm1d(out_dim)
        else:  # 'none'
            self.bn_node_x = nn.Identity()
            self.bn_edge_e = nn.Identity()
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax, PE=pe_LapPE)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = F.relu(x)
        e = F.relu(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e
        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        dim_size = Bx.shape[0]
        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size, reduce='sum')
        denominator_eta_xj = scatter(sigma_ij, index, 0, None, dim_size, reduce='sum')
        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out
