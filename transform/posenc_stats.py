"""
Positional encoding statistics precomputation.
Adapted from graphgps/transform/posenc_stats.py.
Keeps only LapPE and EquivStableLapPE (needed for configs/Exphormer/).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected)


def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """
    Precompute positional encoding statistics for one graph.

    Supported pe_types: 'LapPE', 'EquivStableLapPE'

    Returns the extended data object.
    """
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes
    else:
        N = data.x.shape[0]

    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    evals, evects = None, None

    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        if 'LapPE' in pe_types:
            pecfg = cfg.posenc_LapPE
        else:
            pecfg = cfg.posenc_EquivStableLapPE

        laplacian_norm_type = pecfg.eigen.laplacian_norm.lower()
        if laplacian_norm_type == 'none':
            laplacian_norm_type = None

        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index,
                           normalization=laplacian_norm_type,
                           num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray())

        max_freqs = pecfg.eigen.max_freqs
        eigvec_norm = pecfg.eigen.eigvec_norm

        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)

    return data


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm):
    """
    Process Laplacian eigenpairs into tensors suitable for PE encoders.

    Returns:
        EigVals: (N, max_freqs, 1) — padded eigenvalues per node
        EigVecs: (N, max_freqs) — padded eigenvectors
    """
    N = evects.shape[0]
    idx = evals.argsort()
    evals, evects = evals[idx], np.real(evects[:, idx])

    # Keep up to max_freqs smallest eigenvalues
    evals = evals[:max_freqs]
    evects = evects[:, :max_freqs]

    # Pad if fewer than max_freqs eigenvectors exist
    if evals.shape[0] < max_freqs:
        pad_size = max_freqs - evals.shape[0]
        evals = np.concatenate([evals, np.full(pad_size, np.nan)])
        evects = np.concatenate([evects, np.full((N, pad_size), np.nan)], axis=1)

    evals = torch.from_numpy(evals).float()
    evects = torch.from_numpy(evects).float()

    # Normalize eigenvectors
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)

    # EigVals broadcast to (N, max_freqs, 1)
    EigVals = evals.unsqueeze(0).expand(N, -1).unsqueeze(2)  # (N, k, 1)
    EigVecs = evects  # (N, k)

    return EigVals, EigVecs


def eigvec_normalizer(EigVecs, EigVals, normalization='L2', eps=1e-12):
    """Normalise eigenvectors."""
    if normalization == 'L2':
        # L2 norm per eigenvector
        denom = EigVecs.norm(p=2, dim=0, keepdim=True).clamp_min(eps)
        EigVecs = EigVecs / denom
    elif normalization == 'abs-val':
        denom = EigVecs.abs().max(dim=0, keepdim=True)[0].clamp_min(eps)
        EigVecs = EigVecs / denom
    elif normalization == 'min-max':
        min_val = EigVecs.min(dim=0, keepdim=True)[0]
        max_val = EigVecs.max(dim=0, keepdim=True)[0]
        denom = (max_val - min_val).clamp_min(eps)
        EigVecs = (EigVecs - min_val) / denom
    elif normalization == 'none':
        pass
    else:
        raise ValueError(f"Unknown eigenvector normalization: {normalization}")
    return EigVecs
