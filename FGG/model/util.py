from functools import wraps, partial
from typing import TypeVar

import torch
import numpy as np
import networkx as nx

TensorLike = TypeVar("TensorLike", np.ndarray, torch.Tensor)


def random_seed_consistent_rng(seed=None) -> np.random.RandomState:
    if seed is not None:
        torch.manual_seed(seed=seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(seed=seed)
        np.random.seed(seed=seed)
    return np.random.RandomState(seed=seed)


def no_grad(func):
    @wraps(func)
    def without_grad(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return without_grad


def connected_entries_idxs(adj, directed=False, relevant_rows=None):
    """
    Returns the indices of all nodes in the graph adjacency matrix (taking into account the relevant rows),
    that are connected. Indices are of shape (batch, 2), where the last dimension is row, column.
    :param relevant_rows:
    :param adj:
    :param directed:
    :return:
    """
    adj_ = adj.copy()  # Can maybe be removed
    if not directed:
        # Since the graph if undirected we only need the entries lower than the diagonal
        triagle_idx = np.triu_indices_from(adj_, k=0)
        adj_[triagle_idx] = 0.
    # We don't care for any nodes that are not relevant
    if relevant_rows is not None:
        irrelevant_indices = np.setxor1d(relevant_rows, np.arange(len(adj_)))
        adj_[irrelevant_indices] = 0.
        adj_[:, irrelevant_indices] = 0.
    return np.stack(np.nonzero(adj_)).T


def disconnected_entries_idxs( adj, directed=False, relevant_rows=None):
    """
    Returns the indices of all nodes in the graph adjacency matrix (taking into account the relevant rows),
    that are not connected. Indices are of shape (batch, 2), where the last dimension is row, column.
    :param relevant_rows:
    :param adj:
    :param directed:
    :return:
    """
    adj_ = adj.copy()
    adj_[np.abs(adj) > 0.] = 0.
    adj_[adj == 0.] = 1.

    if not directed:
        triangle_idx = np.triu_indices_from(adj, k=0)
        adj_[triangle_idx] = 0
    if relevant_rows is not None:
        irrelevant_indices = np.setxor1d(relevant_rows, np.arange(len(adj_)))
        adj_[irrelevant_indices] = 0.
        adj_[:, irrelevant_indices] = 0.
    return np.stack(np.nonzero(adj_)).T


def make_known_pair_selector(adj_same, adj_different, directed=False, relevant_rows=None):
    return partial(connected_pairs_from_graph_structure, relevant_rows=relevant_rows, adj_same=adj_same,
                   adj_different=adj_different, directed=directed)


def make_unknown_pair_selector(relevant_rows, adj_same, adj_different, directed=False):
    return partial(disconnected_pairs_from_graph_structure, relevant_rows=relevant_rows, adj_same=adj_same,
                   adj_different=adj_different, directed=directed)


def connected_pairs_from_graph_structure(logits, adj_same, adj_different, directed=False,
                                         relevant_rows=None,):
    """
    Select logits pairwise so that all pairs are either connected with a "same" or "different" edge.
    Undirected will only return each pair once.
    The pairs are built along the last dimension, i.e. result has shape (batch, features, pair)
    :param logits: The logits to select from.
    :param relevant_rows: Only care for these matrix rows
    :param adj_same: The adjacency matrix of same indices
    :param adj_different: The adjacency matrix of different indices.
    :param directed: Whether the adjacency matrices are directed.
    :return:
    """
    assert logits.shape[0] == adj_same.shape[0] == adj_different.shape[0], f"{logits.shape[0]}, {adj_same.shape[0]}"
    adj = np.abs(adj_same) + np.abs(adj_different)
    entries_idx = connected_entries_idxs(relevant_rows=relevant_rows, adj=adj, directed=directed)
    entries = logits[entries_idx.flatten()].reshape(-1, 2, logits.shape[-1]).permute(0, 2, 1)
    return entries


def disconnected_pairs_from_graph_structure(logits, adj_same, adj_different, directed=False,
                                            relevant_rows=None):
    """
    Select logits pairwise so that all pairs are NOT connected with any edge.
    Undirected will only return each pair once.
    The pairs are built along the last dimension, i.e. result has shape (batch, features, pair)
    :param logits: The logits to select from.
    :param relevant_rows: Only care for these matrix rows
    :param adj_same: The adjacency matrix of same indices
    :param adj_different: The adjacency matrix of different indices.
    :param directed: Whether the adjacency matrices are directed.
    :return:
    """
    disconnected_idx = disconnected_entries_idxs(relevant_rows=relevant_rows,
                                                 adj=np.abs(adj_same) + np.abs(adj_different),
                                                 directed=directed)
    paired_entries = logits[disconnected_idx.flatten()].reshape(-1, 2, logits.shape[-1]).permute(0, 2, 1)
    return paired_entries


def targets_from_graph_structure(adj_same, adj_different, directed=False, relevant_rows=None):
    # This is the other way around as everywhere else on purpose so that same gets labeled as 1
    adj = np.stack((adj_different, adj_same))
    adj = adj.argmax(axis=0)
    all_idx = connected_entries_idxs(relevant_rows=relevant_rows, adj=np.abs(adj_different) + np.abs(adj_same),
                                     directed=directed)
    targets = torch.from_numpy(adj[all_idx[..., 0], all_idx[..., 1]].flatten().astype(np.float32))
    return targets


def mean_neighbors(features, adjacency):
    """
    Compute the average features of all neighbors of each node in a graph.
    The features must have the same number of rows as the adjacency matrix.
    :param features: The features of each node, in the same order as the adjacency matrix rows.
    :param adjacency: The adjacency matrix of the graph
    :return: A tensor with the same shape as features, but each row is the mean of all neighbors of that node.
             Nodes without neighbors are set to 0.
    """
    bool_adj = (adjacency != 0.).float()
    counts = bool_adj.sum(dim=1, keepdim=True)

    neighbor_means = (bool_adj @ features) / torch.clamp(counts, min=1.)
    return neighbor_means


def mean_pool_tracks(adjacency: np.ndarray, *tensors: np.ndarray):
    """
    Mean pools all connected components given by an adjacency matrix over all input tensors.

    :param adjacency:
    :return:
    """
    assert len(set(tensor.shape[0] for tensor in tensors)) == 1, "All input tensors must have same length"
    adjacency[adjacency != 1] = 0  # Remove the weighted edges due to similarity
    graph = nx.from_numpy_array(adjacency )
    means = [[] for _ in range(len(tensors))]
    for component in nx.connected_components(graph):
        indices = list(component)
        for i, tensor in enumerate(tensors):
            means[i].append(tensor[indices].mean(axis=0))
    return map(np.stack, means)


def no_nans(tensor):
    return not torch.isnan(tensor).any()


def l2_norm(arr: TensorLike) -> TensorLike:
    if isinstance(arr, torch.Tensor):
        row_sums = torch.sqrt(torch.pow(arr, 2).sum(dim=1))
    else:
        row_sums = np.sqrt(np.square(arr).sum(axis=1))
    normed = arr / row_sums[:, None]
    return normed


def num_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

