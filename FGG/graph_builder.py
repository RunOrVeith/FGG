import enum
import numpy as np
import networkx as nx
from typing import Union, Optional
import itertools
import warnings
from scipy.spatial import distance

from FGG.tracks import TrackCollection
from FGG.split_strategy import SplitStrategy


@enum.unique
class EdgeTypes(enum.IntEnum):
    # Warning: Some other parts of the code base rely on the order!
    must_link = 1
    cannot_link = 2


class GraphBuilder(object):

    def __init__(self, split_strategy: SplitStrategy, pos_edge_dropout: float = None, neg_edge_dropout: float = None,
                 pair_sample_fraction=0.4, edge_between_top_fraction=0.03, weighted_edges=True,
                 isolates_similarity_only=False, add_wrong_edges=None,
                 rng: Union[Optional[int], np.random.RandomState] = None):
        self.split_strategy = split_strategy

        self._original_rng = rng
        self.rng = None
        self.reset()
        self.add_wrong_edges = add_wrong_edges
        self.pos_edge_dropout = pos_edge_dropout
        self.neg_edge_dropout = neg_edge_dropout
        self.pair_sample_fraction = pair_sample_fraction
        self.edge_between_top_fraction = edge_between_top_fraction
        self.weighted_edges = weighted_edges
        self.isolates_similarity_only = isolates_similarity_only

    def reset(self):
        if isinstance(self._original_rng, int) or self._original_rng is None:
            self.rng = np.random.RandomState(seed=self._original_rng)
        else:
            self.rng = self._original_rng
        self.split_strategy.reset()

    @staticmethod
    def unconnected_graph(tracks: TrackCollection):
        graph = nx.Graph()
        for track in tracks:
            graph.add_node(track)
        return graph

    @staticmethod
    def cannot_link_from_temporal_overlap(graph):

        for track_a, track_b in itertools.combinations(graph.nodes, 2):
            if track_a.overlaps(track_b):
                graph.add_edge(track_a, track_b, type=EdgeTypes.cannot_link, weight=1)
        return graph

    @staticmethod
    def must_link_from_tracker_label(graph):

        for track_a, track_b in itertools.combinations(graph.nodes, 2):
            if track_a.tracker_id == track_b.tracker_id:
                graph.add_edge(track_a, track_b, type=EdgeTypes.must_link, weight=1)
        return graph

    def both_types_from_sample_distance(self, graph: nx.Graph, tracks, distance_func=distance.cosine):
        if self.edge_between_top_fraction is None or self.pair_sample_fraction is None:
            return graph

        if self.isolates_similarity_only:
            sample_from = list(nx.isolates(graph))
        else:
            sample_from = graph.nodes

        graph_size = len(sample_from)
        if graph_size <= 1:
            return graph
        num_samples = int(self.pair_sample_fraction * graph_size)
        selected_nodes = self.rng.choice(sample_from, num_samples, replace=False)

        assert len(selected_nodes) == num_samples
        samples = list(itertools.combinations(selected_nodes, 2))
        assert len(samples) == num_samples * (num_samples - 1) / 2
        samples = [(track_a, track_b) for track_a, track_b in samples if not graph.has_edge(track_a, track_b)]
        distances = np.array([distance_func(tracks[track_a].mean(axis=0), tracks[track_b].mean(axis=0))
                              for track_a, track_b in samples])

        num_samples_to_connect = int(self.edge_between_top_fraction * len(samples) / 2)
        most_similar = np.argpartition(distances, num_samples_to_connect)[:num_samples_to_connect]
        least_similar = np.argpartition(-distances, num_samples_to_connect)[:num_samples_to_connect]

        for same_idx, different_idx in zip(most_similar, least_similar):
            # Use 1-distance in both cases because the negation is already present in the edge type
            pos_weight, neg_weight = 1, 1
            if self.weighted_edges:
                pos_weight, neg_weight = 1 - distances[same_idx], 1 - distances[different_idx]
            graph.add_edge(*samples[same_idx], type=EdgeTypes.must_link, weight=pos_weight)
            graph.add_edge(*samples[different_idx], type=EdgeTypes.cannot_link, weight=neg_weight)
        return graph

    @staticmethod
    def split(graph, split_strategy: SplitStrategy):
        marked_for_deletion = []
        split_graph = graph.copy(as_view=False)
        for track in graph.nodes:
            into = split_strategy(track)
            neighbors = list(nx.all_neighbors(split_graph, track))
            subtracks = track.split(into=into)

            for subtrack in subtracks:
                split_graph.add_node(subtrack)

                for neighbor in neighbors:
                    split_graph.add_edge(subtrack, neighbor,
                                         type=split_graph[track][neighbor]["type"],
                                         weight=split_graph[track][neighbor]["weight"])
            for subtrack_a, subtrack_b in itertools.combinations(subtracks, 2):
                split_graph.add_edge(subtrack_a, subtrack_b, type=EdgeTypes.must_link, weight=1)
            marked_for_deletion.append(track)
        split_graph.remove_nodes_from(marked_for_deletion)
        return split_graph

    @staticmethod
    def graph_to_track_collection(graph, tracks: TrackCollection):
        graph_tracks = sorted(graph.nodes)
        return TrackCollection(tracks=graph_tracks, features=tracks.features,
                               person_id_handler=tracks.person_id_handler)

    def edge_dropout(self, graph: nx.Graph, edge_type, p):
        drop_edges = [(u, v) for u, v, data in graph.edges(data=True)
                      if data["type"] == edge_type and self.rng.random_sample() <= p]
        graph.remove_edges_from(drop_edges)
        return graph

    def add_random_wrong_edges(self, graph):
        graph_size = nx.number_of_nodes(graph)
        num_samples = int(self.add_wrong_edges * graph_size)
        sample_from = graph.nodes
        selected_nodes = self.rng.choice(sample_from, num_samples, replace=False)

        for track_a, track_b in itertools.combinations(selected_nodes, 2):
            if graph.has_edge(track_a, track_b):
                continue
            elif track_a.label == track_b.label:
                graph.add_edge(track_a, track_b, type=EdgeTypes.cannot_link, weight=1)
            else:
                graph.add_edge(track_a, track_b, type=EdgeTypes.must_link, weight=1)

        return graph

    def constraints_to_graph(self, tracks: TrackCollection, split_disconnected_components=False):
        graph = self.unconnected_graph(tracks=tracks)
        graph = self.cannot_link_from_temporal_overlap(graph)
        graph = self.split(graph, split_strategy=self.split_strategy)
        graph = self.must_link_from_tracker_label(graph)
        if self.pos_edge_dropout is not None:
            graph = self.edge_dropout(graph=graph, edge_type=EdgeTypes.must_link, p=self.pos_edge_dropout)
        if self.neg_edge_dropout is not None:
            graph = self.edge_dropout(graph=graph, edge_type=EdgeTypes.cannot_link, p=self.neg_edge_dropout)

        if self.add_wrong_edges is not None:
            graph = self.add_random_wrong_edges(graph)
        graph = self.both_types_from_sample_distance(graph, tracks=tracks)
        if not split_disconnected_components:
            print(GraphMetrics(graph))
            yield graph, self.graph_to_track_collection(graph=graph, tracks=tracks)
        else:
            # Need to merge single node components into one because batch norm does not work otherwise
            single_node_components = []
            for component in nx.connected_components(graph):
                if len(component) == 1:
                    single_node_components.extend(component)
                    continue
                subgraph = graph.subgraph(component)
                print(GraphMetrics(subgraph))
                yield subgraph, self.graph_to_track_collection(graph=subgraph, tracks=tracks)
            if len(single_node_components) == 1:
                warnings.warn("Found one single-node component, skipping!")
            else:
                merged_single_nodes = graph.subgraph(single_node_components)
                print(GraphMetrics(merged_single_nodes))
                yield merged_single_nodes, self.graph_to_track_collection(graph=merged_single_nodes, tracks=tracks)


class GraphMetrics(object):

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def __str__(self):
        return f"Density: {self.density} #Nodes: {self.number_of_nodes} " \
               f"#Edges: {self.number_of_edges} Edges/Node: {self.edges_per_node}"

    @property
    def density(self):
        N = self.number_of_nodes
        if N <= 1:
            return 0
        num_edges = self.number_of_edges
        return 2 * num_edges / ((N - 1) * N)

    @property
    def edges_per_node(self):
        return self.number_of_edges / self.number_of_nodes

    @property
    def number_of_nodes(self):
        return nx.number_of_nodes(self.graph)

    @property
    def number_of_edges(self):
        return nx.number_of_edges(self.graph)

