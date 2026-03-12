"""
KGCDataset: Knowledge Graph Completion dataset for Exphormer.

Implements subgraph-based query representation for link-prediction tasks
on WN18RR (and similar benchmarks with the same file format).

Data source: PyG's WordNet18RR (torch_geometric.datasets.WordNet18RR),
which downloads raw train/valid/test.txt files from:
  https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/

Each item in the dataset corresponds to one KGC query triple (h, r, t).
The __getitem__ method extracts a k-hop subgraph around the anchor entity h
from the full training graph and packages it as a torch_geometric.data.Data.

Assumptions:
- Triples are (head, relation, tail) directed.
- During subgraph extraction we use ALL training edges (plus their reverses)
  so message passing can flow both ways.
- True answer forced into training subgraphs if missing.
- For val/test subgraphs, the true tail is NOT forced in (to avoid leakage),
  but answer_in_subgraph is set False for diagnostics.
- Reciprocal triples: if cfg.kgc.reciprocal=True, add (t, r+|R|, h) for each
  training triple, doubling |R|. Reciprocal queries are also added for training.
"""

import logging
import os
import os.path as osp
from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import WordNet18RR
from torch_geometric.utils import k_hop_subgraph

# FIX-03: import expander edge generator for per-subgraph expander edges
from transform.expander_edges import generate_random_expander


# ---------------------------------------------------------------------------
# Helper: build filter dict for filtered evaluation
# ---------------------------------------------------------------------------

def _build_filter_dict(all_triples: torch.Tensor):
    """
    Build a dict mapping (head, relation) -> set of known tails across all splits.
    Also build (tail, relation) -> set of known heads (for head-prediction queries).

    Args:
        all_triples: LongTensor of shape (N, 3) with columns [head, relation, tail].

    Returns:
        tail_filter: dict[(h, r)] -> set(known tails)
        head_filter: dict[(t, r)] -> set(known heads)
    """
    tail_filter = defaultdict(set)
    head_filter = defaultdict(set)
    for triple in all_triples.tolist():
        h, r, t = triple
        tail_filter[(h, r)].add(t)
        head_filter[(t, r)].add(h)
    return dict(tail_filter), dict(head_filter)


# ---------------------------------------------------------------------------
# KGCDataset
# ---------------------------------------------------------------------------

class KGCDataset(Dataset):
    """
    Knowledge Graph Completion dataset with subgraph-based query representation.

    Args:
        root (str): Root directory to store raw/processed data.
        split (str): One of 'train', 'val', 'test'.
        cfg: YACS CfgNode with kgc.* keys set.
        all_triples_filter (dict, optional): Pre-built (h,r)->set(tails) dict
            for filtered evaluation. If None, built from this split's triples.

    Attributes:
        num_entities (int): Total number of unique entities.
        num_relations (int): Number of relation types (doubled if reciprocal).
        train_triples (Tensor): (N_train, 3) — [head, relation, tail]
        val_triples   (Tensor): (N_val, 3)
        test_triples  (Tensor): (N_test, 3)
        all_triples_filter (dict): (h, r) -> set(tails) over all splits.
        head_filter    (dict): (t, r) -> set(heads) over all splits.
        full_edge_index (Tensor): (2, E_full) — all training edges + reverses,
            used for subgraph extraction.
        full_edge_attr  (Tensor): (E_full,) — relation IDs for full_edge_index.
    """

    # Maps the 11 WN18RR relation strings to integer IDs (same as PyG).
    RELATION2ID = {
        '_also_see': 0,
        '_derivationally_related_form': 1,
        '_has_part': 2,
        '_hypernym': 3,
        '_instance_hypernym': 4,
        '_member_meronym': 5,
        '_member_of_domain_region': 6,
        '_member_of_domain_usage': 7,
        '_similar_to': 8,
        '_synset_domain_topic_of': 9,
        '_verb_group': 10,
    }
    BASE_NUM_RELATIONS = 11

    def __init__(
        self,
        root: str,
        split: str,
        cfg,
        # Shared state passed from the factory (to avoid re-loading for each split)
        _shared: Optional[dict] = None,
    ):
        assert split in ('train', 'val', 'test'), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        self.split = split
        self.cfg = cfg
        self._reciprocal = cfg.kgc.reciprocal
        self._num_hops = cfg.kgc.subgraph_hops
        self._max_nodes = cfg.kgc.max_nodes

        # If shared state provided (same WordNet18RR loaded once), reuse it.
        if _shared is not None:
            self._init_from_shared(_shared)
        else:
            self._init_load(root)

        # Select the split-specific query triples
        if split == 'train':
            self._queries = self.train_triples
        elif split == 'val':
            self._queries = self.val_triples
        else:
            self._queries = self.test_triples

        # Subgraph cache: base Data objects WITHOUT expander edges.
        # On first run, subgraphs are computed and saved to a .pt file on disk so
        # subsequent runs (and restarts) skip the expensive k_hop_subgraph step.
        # ensure_disk_cache() is called in create_kg_datasets() (main process,
        # before DataLoader workers are forked) so workers inherit the full cache.
        rec_tag = '_rec' if self._reciprocal else ''
        _cache_key = (
            f'{split}_h{self._num_hops}_n{self._max_nodes}{rec_tag}'
        )
        _name_tag = cfg.dataset.name.lower().replace('-', '').replace('_', '')
        _cache_dir = osp.join(cfg.dataset.dir, _name_tag, 'subgraph_cache')
        os.makedirs(_cache_dir, exist_ok=True)
        self._disk_cache_path = osp.join(_cache_dir, f'{_cache_key}.pt')

        self._subgraph_cache: dict = {}
        if osp.exists(self._disk_cache_path):
            logging.info(f'Loading subgraph cache: {self._disk_cache_path}')
            self._subgraph_cache = torch.load(
                self._disk_cache_path, map_location='cpu', weights_only=False,
            )
            logging.info(
                f'  {len(self._subgraph_cache)}/{len(self._queries)} '
                f'subgraphs loaded for {split} split'
            )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_load(self, root):
        """Dispatch to the dataset-specific loader based on cfg.dataset.name."""
        name = self.cfg.dataset.name.upper().replace('-', '').replace('_', '')
        if name == 'WN18RR':
            self._init_load_wn18rr(root)
        elif name == 'FB15K237':
            self._init_load_fb15k237(root)
        else:
            raise ValueError(
                f"Unknown KGC dataset '{self.cfg.dataset.name}'. "
                f"Supported: WN18RR, FB15k-237"
            )

    def _init_load_wn18rr(self, root):
        """
        Load WordNet18RR using PyG.

        PyG provides a single Data object with all edges and per-edge train/val/test
        masks, which we split into triple tensors.
        """
        pyg_root = osp.join(root, 'WN18RR')
        logging.info(f'Loading WordNet18RR from {pyg_root}')
        wn = WordNet18RR(root=pyg_root)
        data = wn[0]  # single Data with edge_index, edge_type, *_mask

        edge_index = data.edge_index  # (2, E_all)
        edge_type  = data.edge_type   # (E_all,)
        num_nodes  = data.num_nodes   # 40943

        def _make_triples(mask):
            src = edge_index[0, mask]
            dst = edge_index[1, mask]
            rel = edge_type[mask]
            return torch.stack([src, rel, dst], dim=1).long()

        train_triples = _make_triples(data.train_mask)
        val_triples   = _make_triples(data.val_mask)
        test_triples  = _make_triples(data.test_mask)

        self._build_from_triples(train_triples, val_triples, test_triples,
                                 num_nodes, self.BASE_NUM_RELATIONS)

    def _init_load_fb15k237(self, root):
        """
        Load FB15k-237 using PyG's FB15k_237.

        PyG stores each split as a separate Data object (edge_index + edge_type).
        The entity/relation vocabularies are built jointly from all split files,
        so IDs are consistent across splits.

        Stats: 14,541 entities, 237 base relations, 272,115 / 17,535 / 20,466
        train/val/test triples.
        """
        from torch_geometric.datasets import FB15k_237
        pyg_root = osp.join(root, 'FB15k237')
        logging.info(f'Loading FB15k-237 from {pyg_root}')

        train_data = FB15k_237(root=pyg_root, split='train')[0]
        val_data   = FB15k_237(root=pyg_root, split='val')[0]
        test_data  = FB15k_237(root=pyg_root, split='test')[0]

        # num_nodes is shared across all splits (set from the full node vocab).
        num_nodes = int(train_data.num_nodes)  # 14541

        def _make_triples(d):
            return torch.stack([
                d.edge_index[0],   # head (src)
                d.edge_type,       # relation
                d.edge_index[1],   # tail (dst)
            ], dim=1).long()

        train_triples = _make_triples(train_data)
        val_triples   = _make_triples(val_data)
        test_triples  = _make_triples(test_data)

        # Infer base relation count from data (should be 237 for standard FB15k-237).
        num_base_rel = int(
            max(train_triples[:, 1].max(),
                val_triples[:, 1].max(),
                test_triples[:, 1].max()).item()
        ) + 1
        logging.info(f'FB15k-237: {num_nodes} entities, {num_base_rel} base relations')

        self._build_from_triples(train_triples, val_triples, test_triples,
                                 num_nodes, num_base_rel)

    def _build_from_triples(self, train_triples, val_triples, test_triples,
                             num_nodes, num_base_rel):
        """
        Common post-processing for any KG dataset:
          1. Optionally add reciprocal triples to training set.
          2. Build filtered evaluation dicts over all splits.
          3. Build full_edge_index (training KG + bidirectional structural reverses).
          4. Store all state as instance attributes.

        Args:
            train_triples: (N_train, 3) LongTensor [head, relation, tail]
            val_triples:   (N_val,   3) LongTensor
            test_triples:  (N_test,  3) LongTensor
            num_nodes:     total number of entities
            num_base_rel:  number of base (non-reciprocal) relation types
        """
        # Reciprocal triples: (t, r + |R|, h) for each training (h, r, t).
        # Added only to the training set (standard KGC practice).
        if self._reciprocal:
            inv_triples = torch.stack([
                train_triples[:, 2],
                train_triples[:, 1] + num_base_rel,
                train_triples[:, 0],
            ], dim=1)
            train_triples = torch.cat([train_triples, inv_triples], dim=0)
            num_relations = num_base_rel * 2
        else:
            num_relations = num_base_rel

        # Filter dict: (h, r) → set(all known tails) across all splits.
        all_triples = torch.cat([train_triples, val_triples, test_triples], dim=0)
        tail_filter, head_filter = _build_filter_dict(all_triples)

        # Full graph for subgraph extraction: training edges + bidirectional
        # structural reverses (r + num_relations).  Val/test edges are excluded
        # to prevent leakage into subgraph neighbourhoods.
        train_src = train_triples[:, 0]
        train_dst = train_triples[:, 2]
        train_rel = train_triples[:, 1]

        rev_rel = train_rel + num_relations  # structural reverse relation IDs
        full_src = torch.cat([train_src, train_dst], dim=0)
        full_dst = torch.cat([train_dst, train_src], dim=0)
        full_rel = torch.cat([train_rel, rev_rel],   dim=0)
        full_edge_index = torch.stack([full_src, full_dst], dim=0).long()
        full_edge_attr  = full_rel.long()

        self.num_entities       = int(num_nodes)
        self.num_relations      = int(num_relations)
        self.num_base_relations = int(num_base_rel)
        self.train_triples      = train_triples
        self.val_triples        = val_triples
        self.test_triples       = test_triples
        self.all_triples_filter = tail_filter
        self.head_filter        = head_filter
        self.full_edge_index    = full_edge_index
        self.full_edge_attr     = full_edge_attr

        logging.info(
            f'KGCDataset loaded: {self.num_entities} entities, '
            f'{self.num_relations} relations (reciprocal={self._reciprocal}), '
            f'{len(train_triples)} train / '
            f'{len(val_triples)} val / '
            f'{len(test_triples)} test triples'
        )

    def _init_from_shared(self, shared: dict):
        """Copy pre-loaded state from shared dict (avoids re-loading PyG dataset)."""
        self.num_entities       = shared['num_entities']
        self.num_relations      = shared['num_relations']
        self.num_base_relations = shared['num_base_relations']
        self.train_triples      = shared['train_triples']
        self.val_triples        = shared['val_triples']
        self.test_triples       = shared['test_triples']
        self.all_triples_filter = shared['all_triples_filter']
        self.head_filter        = shared['head_filter']
        self.full_edge_index    = shared['full_edge_index']
        self.full_edge_attr     = shared['full_edge_attr']

    def get_shared_state(self) -> dict:
        """Return a dict of shared state for constructing sibling split datasets."""
        return {
            'num_entities':       self.num_entities,
            'num_relations':      self.num_relations,
            'num_base_relations': self.num_base_relations,
            'train_triples':      self.train_triples,
            'val_triples':        self.val_triples,
            'test_triples':       self.test_triples,
            'all_triples_filter': self.all_triples_filter,
            'head_filter':        self.head_filter,
            'full_edge_index':    self.full_edge_index,
            'full_edge_attr':     self.full_edge_attr,
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._queries)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _make_base_data(self, idx: int) -> Data:
        """
        Return a *cloned* base Data (no expander edges) for query idx.
        Populates _subgraph_cache[idx] on first call (cache miss).
        """
        if idx in self._subgraph_cache:
            return self._subgraph_cache[idx].clone()

        triple = self._queries[idx]
        head_global = int(triple[0])
        relation    = int(triple[1])
        tail_global = int(triple[2])
        is_train    = (self.split == 'train')

        node_ids, local_edge_index, local_edge_attr, anchor_local, answer_local = \
            self._extract_subgraph(head_global, tail_global, relation, is_train)

        answer_in_subgraph = (answer_local is not None)
        answer_local_idx   = torch.tensor(
            answer_local if answer_in_subgraph else 0, dtype=torch.long
        )

        data = Data(
            x              = node_ids.long(),
            edge_index     = local_edge_index.long(),
            edge_attr      = local_edge_attr.long(),
            y              = answer_local_idx,
            anchor_idx     = torch.tensor(anchor_local, dtype=torch.long),
            query_relation = torch.tensor(relation,     dtype=torch.long),
            true_tail      = torch.tensor(tail_global,  dtype=torch.long),
            query_head     = torch.tensor(head_global,  dtype=torch.long),
            answer_in_subgraph = answer_in_subgraph,
            num_nodes      = node_ids.shape[0],
        )
        self._subgraph_cache[idx] = data
        return data.clone()

    def ensure_disk_cache(self):
        """
        Precompute base subgraphs for all queries in this split and persist to disk.

        Idempotent: returns immediately if the disk file already exists and is
        fully loaded.  Called in the main process (via create_kg_datasets) before
        DataLoader workers are forked, so workers inherit the full in-memory cache
        via copy-on-write and never need to recompute subgraphs.
        """
        n = len(self._queries)

        # Already fully loaded from disk — nothing to do.
        if len(self._subgraph_cache) == n:
            return

        # Disk file exists but wasn't loaded (shouldn't happen, but guard anyway).
        if osp.exists(self._disk_cache_path) and len(self._subgraph_cache) == n:
            return

        logging.info(
            f'Precomputing subgraph cache [{self.split}]: '
            f'{n} queries → {self._disk_cache_path}'
        )
        for i in range(n):
            if i not in self._subgraph_cache:
                self._make_base_data(i)
            if (i + 1) % 5000 == 0 or (i + 1) == n:
                logging.info(f'  [{self.split}] {i + 1}/{n}')

        # Atomic save: write to .tmp then rename to avoid corrupt files on crash.
        tmp_path = self._disk_cache_path + '.tmp'
        logging.info(f'Saving subgraph cache → {self._disk_cache_path} ...')
        torch.save(self._subgraph_cache, tmp_path)
        os.replace(tmp_path, self._disk_cache_path)
        logging.info(f'  Saved {len(self._subgraph_cache)} subgraphs.')

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Data:
        """
        Returns a Data object for query index idx.

        Data fields:
            x              (N_sub,)     LongTensor — global entity IDs
            edge_index     (2, E_sub)   LongTensor — local node indices
            edge_attr      (E_sub,)     LongTensor — relation IDs
            y              ()           LongTensor — local index of true answer
            anchor_idx     ()           LongTensor — local index of anchor (h)
            query_relation ()           LongTensor — the query relation r
            true_tail      ()           LongTensor — global entity ID of true tail
            query_head     ()           LongTensor — global entity ID of anchor
            answer_in_subgraph bool     — whether true answer is in subgraph
            expander_edges (E_exp, 2)   LongTensor — expander edges (if cfg.prep.exp)
        """
        data = self._make_base_data(idx)

        # Generate fresh expander edges (random each epoch, not cached).
        if self.cfg.prep.exp:
            if data.num_nodes > 1:
                generate_random_expander(
                    data,
                    degree=self.cfg.prep.exp_deg,
                    algorithm=self.cfg.prep.exp_algorithm,
                    max_num_iters=self.cfg.prep.exp_max_num_iters,
                    rng=None,
                    check_spectral=self.cfg.prep.exp_check_spectral,
                )
            else:
                data.expander_edges = torch.zeros((0, 2), dtype=torch.long)

        return data

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def _extract_subgraph(
        self,
        anchor: int,
        true_tail: int,
        relation: int,
        is_train: bool,
    ):
        """
        Extract a k-hop subgraph around `anchor` from the full training graph.

        Algorithm:
        1. Run k_hop_subgraph(anchor, num_hops, full_edge_index) to get local nodes.
        2. If is_train and true_tail not in subgraph, forcibly include it
           (add node to node_ids, add a direct edge anchor -> true_tail with
           the query relation).
        3. Cap at max_nodes by randomly dropping non-anchor, non-answer nodes.
        4. Relabel to local indices, filter edge_index/edge_attr accordingly.

        Returns:
            node_ids         (N_sub,)   LongTensor — global entity IDs
            local_edge_index (2, E_sub) LongTensor — local indices
            local_edge_attr  (E_sub,)   LongTensor — relation IDs
            anchor_local     int        — local index of anchor
            answer_local     int or None — local index of true tail (None if not present)
        """
        num_nodes = self.num_entities

        # Step 1: k-hop subgraph extraction (bidirectional, using full_edge_index)
        # k_hop_subgraph returns:
        #   subset       (N_sub,)  — global node IDs in subgraph
        #   sub_ei       (2, E')   — local edge_index (relabelled)
        #   mapping      (scalar)  — local index of anchor node
        #   edge_mask    (E_full,) — bool mask over full_edge_index
        subset, sub_ei, mapping, edge_mask = k_hop_subgraph(
            node_idx=anchor,
            num_hops=self._num_hops,
            edge_index=self.full_edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            flow='source_to_target',
        )
        # mapping is a tensor; convert to scalar
        anchor_local_in_sub = int(mapping) if mapping.ndim == 0 else int(mapping[0])

        # Get edge attributes for the extracted subgraph
        sub_ea = self.full_edge_attr[edge_mask]  # (E',)

        # Step 2 (training only): force true_tail into subgraph if missing
        tail_in_sub = (subset == true_tail).any().item()
        forced_tail = False

        if is_train and not tail_in_sub:
            # Append true_tail as a new node
            subset = torch.cat([subset, torch.tensor([true_tail], dtype=torch.long)])
            new_node_local = subset.shape[0] - 1  # last index

            # Add direct edge: anchor -> true_tail with query relation
            new_src = torch.tensor([anchor_local_in_sub], dtype=torch.long)
            new_dst = torch.tensor([new_node_local], dtype=torch.long)
            new_edge = torch.stack([new_src, new_dst], dim=0)  # (2, 1)
            new_attr = torch.tensor([relation], dtype=torch.long)

            sub_ei = torch.cat([sub_ei, new_edge], dim=1)
            sub_ea = torch.cat([sub_ea, new_attr], dim=0)
            forced_tail = True
            tail_in_sub = True

        # Step 3: cap subgraph size
        if subset.shape[0] > self._max_nodes:
            subset, sub_ei, sub_ea, anchor_local_in_sub = self._cap_subgraph(
                subset, sub_ei, sub_ea, anchor_local_in_sub, true_tail,
                forced_tail, is_train
            )
            # Recompute tail presence after capping
            tail_in_sub = (subset == true_tail).any().item()

        # Step 4: find local answer index
        if tail_in_sub:
            answer_local = int((subset == true_tail).nonzero(as_tuple=True)[0][0])
        else:
            answer_local = None

        # Final anchor local index (after potential capping may have changed order)
        anchor_matches = (subset == anchor).nonzero(as_tuple=True)[0]
        anchor_local = int(anchor_matches[0]) if len(anchor_matches) > 0 else 0

        return subset, sub_ei, sub_ea, anchor_local, answer_local

    def _cap_subgraph(
        self,
        node_ids:       torch.Tensor,  # (N,) global entity IDs
        edge_index:     torch.Tensor,  # (2, E) local indices
        edge_attr:      torch.Tensor,  # (E,)
        anchor_local:   int,
        true_tail:      int,
        forced_tail:    bool,
        is_train:       bool,
    ):
        """
        Randomly drop nodes to reduce subgraph to max_nodes.
        Always keeps the anchor and (if is_train) the true_tail.

        Returns the new (node_ids, edge_index, edge_attr, anchor_local).
        """
        max_n = self._max_nodes
        n = node_ids.shape[0]

        # Identify protected nodes (must keep)
        anchor_global = int(node_ids[anchor_local])
        protected_globals = {anchor_global}
        if is_train:
            protected_globals.add(true_tail)

        # Protected local indices
        protected_locals = set()
        for i, gid in enumerate(node_ids.tolist()):
            if gid in protected_globals:
                protected_locals.add(i)

        # Randomly select non-protected nodes to keep
        non_protected = [i for i in range(n) if i not in protected_locals]
        num_to_keep = max_n - len(protected_locals)
        if num_to_keep > 0 and non_protected:
            perm = torch.randperm(len(non_protected))
            keep_non = set(non_protected[j] for j in perm[:num_to_keep].tolist())
        else:
            keep_non = set()

        keep_mask_list = [(i in protected_locals or i in keep_non) for i in range(n)]
        keep_mask = torch.tensor(keep_mask_list, dtype=torch.bool)
        new_node_ids = node_ids[keep_mask]

        # Build global->new_local mapping
        old_to_new = {}
        new_idx = 0
        for old_i, keep in enumerate(keep_mask_list):
            if keep:
                old_to_new[old_i] = new_idx
                new_idx += 1

        # Filter edges: keep only edges where both endpoints are retained
        if edge_index.shape[1] > 0:
            src_ok = torch.tensor(
                [keep_mask_list[int(e)] for e in edge_index[0].tolist()],
                dtype=torch.bool)
            dst_ok = torch.tensor(
                [keep_mask_list[int(e)] for e in edge_index[1].tolist()],
                dtype=torch.bool)
            edge_keep = src_ok & dst_ok
            new_ei_raw = edge_index[:, edge_keep]
            new_ea = edge_attr[edge_keep]
            # Relabel
            src_new = torch.tensor(
                [old_to_new[int(x)] for x in new_ei_raw[0].tolist()],
                dtype=torch.long)
            dst_new = torch.tensor(
                [old_to_new[int(x)] for x in new_ei_raw[1].tolist()],
                dtype=torch.long)
            new_edge_index = torch.stack([src_new, dst_new], dim=0)
        else:
            new_edge_index = edge_index
            new_ea = edge_attr

        # New anchor local index
        anchor_new_local = old_to_new.get(anchor_local, 0)

        return new_node_ids, new_edge_index, new_ea, anchor_new_local


# ---------------------------------------------------------------------------
# Factory: create all three split datasets sharing the same loaded state
# ---------------------------------------------------------------------------

def create_kg_datasets(root: str, cfg) -> tuple:
    """
    Create (train_dataset, val_dataset, test_dataset) for a KGC benchmark.

    Loads the dataset once and shares the extracted triples + filter dicts
    across all three splits to avoid redundant I/O.

    Args:
        root (str): Root directory for storing data (cfg.dataset.dir).
        cfg:        YACS CfgNode with kgc.* fields.

    Returns:
        (KGCDataset, KGCDataset, KGCDataset) for train/val/test splits.
    """
    # Load once using the 'train' split object (it loads all triples internally)
    train_ds = KGCDataset(root=root, split='train', cfg=cfg)
    shared   = train_ds.get_shared_state()

    val_ds  = KGCDataset(root=root, split='val',  cfg=cfg, _shared=shared)
    test_ds = KGCDataset(root=root, split='test', cfg=cfg, _shared=shared)

    # Precompute and persist subgraph caches in the main process, before DataLoader
    # workers are forked.  Workers inherit the full _subgraph_cache via copy-on-write
    # and never need to run k_hop_subgraph.
    for ds in (train_ds, val_ds, test_ds):
        ds.ensure_disk_cache()

    return train_ds, val_ds, test_ds
