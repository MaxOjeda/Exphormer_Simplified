"""
Expander edge generation.
Adapted from graphgps/transform/expander_edges.py — import path updated.
"""
import math
import numpy as np
import torch

from transform.dist_transforms import laplacian_eigenv


def generate_random_regular_graph1(num_nodes, degree, rng=None):
    """Generates a random 2d-regular graph using permutations algorithm."""
    if rng is None:
        rng = np.random.default_rng()
    senders = [*range(num_nodes)] * degree
    receivers = []
    for _ in range(degree):
        receivers.extend(rng.permutation(list(range(num_nodes))).tolist())
    senders, receivers = [*senders, *receivers], [*receivers, *senders]
    return np.array(senders), np.array(receivers)


def generate_random_regular_graph2(num_nodes, degree, rng=None):
    """Generates a random 2d-regular graph using simple permutation variant."""
    if rng is None:
        rng = np.random.default_rng()
    senders = [*range(num_nodes)] * degree
    receivers = rng.permutation(senders).tolist()
    senders, receivers = [*senders, *receivers], [*receivers, *senders]
    return senders, receivers


def generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng=None):
    """Generates a 2d-regular graph using d random Hamiltonian cycles."""
    if rng is None:
        rng = np.random.default_rng()
    senders = []
    receivers = []
    for _ in range(degree):
        permutation = rng.permutation(list(range(num_nodes))).tolist()
        for idx, v in enumerate(permutation):
            u = permutation[idx - 1]
            senders.extend([v, u])
            receivers.extend([u, v])
    return np.array(senders), np.array(receivers)


def generate_random_expander(data, degree, algorithm, rng=None,
                              max_num_iters=100, exp_index=0,
                              check_spectral=True):
    """
    Generates a random d-regular expander graph and attaches it to data.

    Args:
        check_spectral: If True (default), run the Alon-Boppana eigenvalue loop
            to select the best expander from up to max_num_iters candidates.
            If False, generate exactly one random d-regular graph without any
            spectral quality check — ~100× faster, suitable for per-subgraph
            generation in KGC where subgraphs change every epoch.
    """
    num_nodes = data.num_nodes

    if rng is None:
        rng = np.random.default_rng()

    eig_val = -1
    eig_val_lower_bound = max(0, 2 * degree - 2 * math.sqrt(2 * degree - 1) - 0.1)
    max_eig_val_so_far = -1
    max_senders = []
    max_receivers = []
    cur_iter = 1

    if num_nodes <= degree:
        degree = num_nodes - 1

    if num_nodes <= 10:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    max_senders.append(i)
                    max_receivers.append(j)
    else:
        if not check_spectral:
            # Skip eigenvalue check: generate one random d-regular graph directly.
            # This is ~100× faster than the spectral loop and acceptable for
            # per-subgraph KGC use (different subgraph every iteration anyway).
            if algorithm == 'Random-d':
                max_senders, max_receivers = generate_random_regular_graph1(
                    num_nodes, degree, rng)
            elif algorithm == 'Random-d-2':
                max_senders, max_receivers = generate_random_regular_graph2(
                    num_nodes, degree, rng)
            elif algorithm == 'Hamiltonian':
                max_senders, max_receivers = \
                    generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng)
            else:
                raise ValueError('prep.exp_algorithm must be one of: Random-d, Hamiltonian')
        else:
            while eig_val < eig_val_lower_bound and cur_iter <= max_num_iters:
                if algorithm == 'Random-d':
                    senders, receivers = generate_random_regular_graph1(num_nodes, degree, rng)
                elif algorithm == 'Random-d-2':
                    senders, receivers = generate_random_regular_graph2(num_nodes, degree, rng)
                elif algorithm == 'Hamiltonian':
                    senders, receivers = generate_random_graph_with_hamiltonian_cycles(
                        num_nodes, degree, rng)
                else:
                    raise ValueError('prep.exp_algorithm must be one of: Random-d, Hamiltonian')

                [eig_val, _] = laplacian_eigenv(senders, receivers, k=1, n=num_nodes)
                if len(eig_val) == 0:
                    eig_val = 0
                else:
                    eig_val = eig_val[0]

                if eig_val > max_eig_val_so_far:
                    max_eig_val_so_far = eig_val
                    max_senders = senders
                    max_receivers = receivers

                cur_iter += 1

    # Remove self-loops
    non_loops = [
        *filter(lambda i: max_senders[i] != max_receivers[i], range(len(max_senders)))
    ]

    max_senders = torch.tensor(max_senders, dtype=torch.long).view(-1, 1)
    max_receivers = torch.tensor(max_receivers, dtype=torch.long).view(-1, 1)

    if exp_index == 0:
        data.expander_edges = torch.cat([max_senders, max_receivers], dim=1)
    else:
        attrname = f"expander_edges{exp_index}"
        setattr(data, attrname, torch.cat([max_senders, max_receivers], dim=1))

    return data
