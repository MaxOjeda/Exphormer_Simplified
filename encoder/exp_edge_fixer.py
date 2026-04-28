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
      1. Real graph edges  (batch.edge_index + batch.edge_attr, if add_edge_index=True)
      2. Expander edges — two mutually exclusive modes:
           a) batch.expander_edge_index  (2, B*E_exp) — pre-computed with global offsets,
              set by the trainer for full-graph mode.  cfg.prep.exp controls generation.
           b) batch.expander_edges       (E_exp, 2) per-graph — produced by __getitem__
              in subgraph DataLoader mode and assembled here via to_data_list().
      3. Virtual node edges (learnable embeddings per graph, if num_virt_node > 0)
    """

    def __init__(self, add_edge_index=False, num_virt_node=0,
                 dim_edge=64, dim_hidden=64, num_relations=None):
        super().__init__()

        self.add_edge_index = add_edge_index
        self.num_virt_node = num_virt_node
        # When set, builds batch.edge_rel_idx: KG edges keep their rel idx,
        # expander edges get sentinel index num_relations (near-zero init row).
        self.num_relations = num_relations

        # Learnable embedding for expander edges — uniform fallback (non-KGC or no query).
        self.exp_edge_attr = nn.Embedding(1, dim_edge)
        # KGC mode: project batch.query_emb (dim_hidden → dim_edge) for expander edge features.
        if num_relations is not None:
            self.proj_exp_edge = nn.Linear(dim_hidden, dim_edge, bias=False)
            nn.init.normal_(self.proj_exp_edge.weight, std=0.01)
        # NOTE: use_exp_edges / prep.exp are handled upstream before this module
        # is called; the presence of batch.expander_edges signals to use them.

        if self.num_virt_node > 0:
            self.virt_node_emb = nn.Embedding(self.num_virt_node, dim_hidden)
            self.virt_edge_out_emb = nn.Embedding(self.num_virt_node, dim_edge)
            self.virt_edge_in_emb = nn.Embedding(self.num_virt_node, dim_edge)

    def _exp_attr(self, exp_ei, batch, device):
        """Return edge features for expander edges (E_exp, dim_edge)."""
        if self.num_relations is not None and hasattr(self, 'proj_exp_edge') \
                and hasattr(batch, 'query_emb'):
            src = exp_ei[0].clamp(max=batch.batch.shape[0] - 1).long()
            graph_idx = batch.batch[src]
            return self.proj_exp_edge(batch.query_emb[graph_idx])  # (E_exp, dim_edge)
        return self.exp_edge_attr(
            torch.zeros(exp_ei.shape[1], dtype=torch.long, device=device)
        )

    def forward(self, batch):
        device = self.exp_edge_attr.weight.device
        edge_index_sets = []
        edge_attr_sets = []
        rel_idx_sets = []   # parallel list for edge_rel_idx (only when num_relations set)

        # 1. Real edges
        if self.add_edge_index:
            edge_index_sets.append(batch.edge_index)
            edge_attr_sets.append(batch.edge_attr)
            if self.num_relations is not None and hasattr(batch, 'edge_rel_idx'):
                rel_idx_sets.append(batch.edge_rel_idx)

        num_node = batch.batch.shape[0]
        num_graphs = batch.num_graphs

        # 2. Expander edges — two sources, mutually exclusive:
        #    a) pre-computed global edge_index (full-graph mode, set by trainer)
        #    b) per-graph edge lists from DataLoader batching (subgraph mode)
        if hasattr(batch, 'expander_edge_index'):
            # Full-graph mode: expander_edge_index is already globally offset
            # (2, B*E_exp). Consumed here; overwritten with combined result below.
            exp_ei = batch.expander_edge_index
            edge_index_sets.append(exp_ei)
            edge_attr_sets.append(self._exp_attr(exp_ei, batch, device))
            if self.num_relations is not None:
                # Expander edges get the sentinel index (num_relations row ≈ near-zero).
                rel_idx_sets.append(
                    torch.full((exp_ei.shape[1],), self.num_relations,
                               dtype=torch.long, device=device)
                )
        elif hasattr(batch, 'expander_edges'):
            # Subgraph mode: per-graph edge lists assembled by DataLoader.
            data_list = batch.to_data_list()
            exp_edges = []
            cumulative = 0
            for data in data_list:
                exp_edges.append(data.expander_edges + cumulative)
                cumulative += data.num_nodes
            exp_edges = torch.cat(exp_edges, dim=0).t()  # (2, E_exp)
            edge_index_sets.append(exp_edges)
            edge_attr_sets.append(self._exp_attr(exp_edges, batch, device))
            if self.num_relations is not None:
                rel_idx_sets.append(
                    torch.full((exp_edges.shape[1],), self.num_relations,
                               dtype=torch.long, device=device)
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
        # expander_edge_index (pre-computed) is overwritten by the combined result below.

        batch.expander_edge_index = edge_index
        batch.expander_edge_attr = edge_attr

        # Build combined edge_rel_idx for use_relational_v (DistMult-style V).
        # Mirrors the order of edge_index_sets: KG edges first, expander edges second.
        if self.num_relations is not None and rel_idx_sets:
            batch.edge_rel_idx = torch.cat(rel_idx_sets) if len(rel_idx_sets) > 1 \
                else rel_idx_sets[0]

        return batch
