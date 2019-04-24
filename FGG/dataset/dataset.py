import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
from typing import List
import scipy

from FGG.dataset.graph_builder import GraphBuilder,  EdgeTypes
from FGG.metrics.evaluation import GraphMetrics
from FGG.dataset.tracks import TrackCollection


class CompleteGraphEpisodeSampler(object):

    def __init__(self, episodes: List[TrackCollection], graph_builder: GraphBuilder,
                 split_disconnected_components=False):
        graphs = [component for episode in episodes
                  for component in graph_builder.constraints_to_graph(episode,
                                                                      split_disconnected_components=split_disconnected_components)]
        self.episode_graphs = graphs

    def num_samples(self) -> int:
        return len(self.episode_graphs)

    def __getitem__(self, idx):
        episode = self.episode_graphs[idx]
        return episode




class EpisodeGraphDataset(Dataset):

    def __init__(self, episodes: List[TrackCollection], graph_builder: GraphBuilder,
                 split_disconnected_components=False):
        super().__init__()
        self.sampler = CompleteGraphEpisodeSampler(episodes=episodes, graph_builder=graph_builder,
                                                   split_disconnected_components=split_disconnected_components)
        self.graph_builder = graph_builder

    def __len__(self):
        return self.sampler.num_samples()

    def __getitem__(self, idx):
        graph, tracks = self.sampler[idx]
        print(GraphMetrics(graph))
        features = tracks.pooled_features()

        # plot_episode_graph(graph)  # You can use this to display the graph in the browser
        # input("Press any key to continue")

        adjacency_matrix = self._dense_adjacency_matrix(graph=graph)
        adjacency_matrix = apply_renormalization_trick(adjacency=adjacency_matrix)
        adjacency_matrix = torch.from_numpy(adjacency_matrix)
        features = torch.from_numpy(features)
        return tracks, adjacency_matrix, features

    @staticmethod
    def _dense_adjacency_matrix(graph):
        node_order = sorted(graph.nodes)

        edge_type_matrix = np.array(nx.attr_matrix(graph, edge_attr="type", rc_order=node_order))
        typed_adjacency = np.stack(
            [np.zeros_like(edge_type_matrix, dtype=np.float32)] * len(EdgeTypes))

        weight_matrix = np.array(nx.attr_matrix(graph, edge_attr="weight", rc_order=node_order))

        for i, typ in enumerate(EdgeTypes):
            typed_adjacency[i] = edge_type_matrix == typ

        typed_adjacency *= weight_matrix[None, ...]
        return typed_adjacency


def apply_renormalization_trick(adjacency):
    adjs = []
    if len(adjacency.shape) == 2:
        adjacency = adjacency[None, ...]
    for i, adj in enumerate(adjacency):
        # Unfortunately we can't perform this in one go over all edge types
        adj[np.diag_indices_from(adj)] = 1
        node_degree_matrix = np.diag(np.sum(adj, axis=1))
        inverse_root_node_degree_matrix = scipy.linalg.sqrtm(np.linalg.inv(node_degree_matrix)).real
        adj = inverse_root_node_degree_matrix @ adj @ inverse_root_node_degree_matrix
        adjs.append(adj)
    return np.stack(adjs)
