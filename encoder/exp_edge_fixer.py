"""
ExpanderEdgeFixer: combines real edges, expander edges, and virtual node edges.
Adapted from graphgps/encoder/exp_edge_fixer.py — cfg passed explicitly.
"""
import torch
import torch.nn as nn


class ExpanderEdgeFixer(nn.Module):
    """
    Merges graph edge types into batch.expander_edge_index / batch.expander_edge_attr
    used by ExphormerAttention.

    Three optional edge sources:
      1. Real graph edges (batch.edge_index + batch.edge_attr)
      2. Expander graph edges (batch.expander_edges, stored per-graph)
      3. Virtual node edges (learnable embeddings per graph)
    """

    def __init__(self, add_edge_index=False, num_virt_node=0,
                 dim_edge=64, dim_hidden=64):
        super().__init__()

        self.add_edge_index = add_edge_index
        self.num_virt_node = num_virt_node

        # Learnable embedding for expander edges (single embedding shared)
        self.exp_edge_attr = nn.Embedding(1, dim_edge)
        # NOTE: use_exp_edges / prep.exp are handled upstream before this module
        # is called; the presence of batch.expander_edges signals to use them.

        if self.num_virt_node > 0:
            self.virt_node_emb = nn.Embedding(self.num_virt_node, dim_hidden)
            self.virt_edge_out_emb = nn.Embedding(self.num_virt_node, dim_edge)
            self.virt_edge_in_emb = nn.Embedding(self.num_virt_node, dim_edge)

    def forward(self, batch):
        device = self.exp_edge_attr.weight.device
        edge_index_sets = []
        edge_attr_sets = []

        # 1. Real edges
        if self.add_edge_index:
            edge_index_sets.append(batch.edge_index)
            edge_attr_sets.append(batch.edge_attr)

        num_node = batch.batch.shape[0]
        num_graphs = batch.num_graphs

        # 2. Expander edges (stored as per-graph edge lists)
        if hasattr(batch, 'expander_edges'):
            data_list = batch.to_data_list()
            exp_edges = []
            cumulative = 0
            for data in data_list:
                exp_edges.append(data.expander_edges + cumulative)
                cumulative += data.num_nodes
            exp_edges = torch.cat(exp_edges, dim=0).t()  # (2, E_exp)
            edge_index_sets.append(exp_edges)
            edge_attr_sets.append(
                self.exp_edge_attr(
                    torch.zeros(exp_edges.shape[1], dtype=torch.long, device=device)
                )
            )

        # 3. Virtual nodes
        if self.num_virt_node > 0:
            global_h = []
            virt_edges = []
            virt_edge_attrs = []
            for idx in range(self.num_virt_node):
                virt_idx = torch.zeros(num_graphs, dtype=torch.long, device=device) + idx
                global_h.append(self.virt_node_emb(virt_idx))

                # node → virtual
                virt_edge_index = torch.cat([
                    torch.arange(num_node, device=device).view(1, -1),
                    (batch.batch + (num_node + idx * num_graphs)).view(1, -1)
                ], dim=0)
                virt_edges.append(virt_edge_index)
                virt_edge_attrs.append(
                    self.virt_edge_in_emb(
                        torch.zeros(virt_edge_index.shape[1], dtype=torch.long, device=device) + idx
                    )
                )

                # virtual → node
                virt_edge_index_rev = torch.cat([
                    (batch.batch + (num_node + idx * num_graphs)).view(1, -1),
                    torch.arange(num_node, device=device).view(1, -1)
                ], dim=0)
                virt_edges.append(virt_edge_index_rev)
                virt_edge_attrs.append(
                    self.virt_edge_out_emb(
                        torch.zeros(virt_edge_index_rev.shape[1], dtype=torch.long, device=device) + idx
                    )
                )

            batch.virt_h = torch.cat(global_h, dim=0)
            batch.virt_edge_index = torch.cat(virt_edges, dim=1)
            batch.virt_edge_attr = torch.cat(virt_edge_attrs, dim=0)

        # Combine
        if len(edge_index_sets) > 1:
            edge_index = torch.cat(edge_index_sets, dim=1)
            edge_attr = torch.cat(edge_attr_sets, dim=0)
        elif len(edge_index_sets) == 1:
            edge_index = edge_index_sets[0]
            edge_attr = edge_attr_sets[0]
        else:
            raise ValueError("No edge sets available for Exphormer. "
                             "Set add_edge_index=True or enable expander edges.")

        if hasattr(batch, 'expander_edges'):
            del batch.expander_edges

        batch.expander_edge_index = edge_index
        batch.expander_edge_attr = edge_attr
        return batch
