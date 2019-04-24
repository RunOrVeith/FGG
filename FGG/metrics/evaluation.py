import contextlib
import inspect

import networkx as nx

from FGG.model.clustering import cluster_mode_prediction
from FGG.metrics.base import to_dense_numpy

from FGG.metrics.metric import Metric, ConfusionMatrix, ClusteringPurity, Accuracy, F1Score, \
    NormalizedMutualInformation, MeanAveragePrecision, BCubed


class MetricProvider(object):

    def __init__(self, *metrics: Metric):
        self.metrics = metrics
        self._required_arguments = dict()
        for metric in self.metrics:
            self._required_arguments[metric.name()] = inspect.signature(metric.submit).parameters.keys()

    def __call__(self, **kwargs):
        scores = {}
        for metric in self.metrics:
            arguments_for_metric = self._required_arguments[metric.name()]
            result = None
            if all(argument in kwargs for argument in arguments_for_metric):
                submission_arguments = {argument: kwargs[argument] for argument in arguments_for_metric}
                result = metric.submit(**submission_arguments)
            elif len(kwargs) == 0:
                result = metric.compute()
            if result is not None:
                provides = metric.provides()
                if len(provides) > 1:
                    for name, score in zip(provides, result):
                        scores[name] = score
                else:
                    scores[provides[0]] = result

        return scores

    def batched_metrics_only(self):
        stack = contextlib.ExitStack()
        for metric in self.metrics:
            stack.enter_context(metric.batch_mode())

        return stack


class ClusterMetricProvider(MetricProvider):

    def __init__(self, labels, cluster_assignment_function=cluster_mode_prediction):
        metrics_to_run = (
            F1Score(labels=labels, weighted=False), F1Score(labels=labels, weighted=True),
            Accuracy(), ConfusionMatrix(labels=labels), MeanAveragePrecision(),
            ClusteringPurity(labels=labels, weighted=True), ClusteringPurity(labels=labels, weighted=False),
            NormalizedMutualInformation(adjusted=False), NormalizedMutualInformation(adjusted=True),
            BCubed(),
        )
        super().__init__(*metrics_to_run)
        self.cluster_assignment_function = cluster_assignment_function
        self.additional_wcp = ClusteringPurity(labels=labels, weighted=True)

    def __call__(self, target, prediction):
        target, prediction = to_dense_numpy(target, prediction)
        cluster_assignment = prediction
        prediction = self.cluster_assignment_function(target=target, prediction=prediction)
        scores = super().__call__(target=target, cluster_assignment=cluster_assignment, prediction=prediction)
        return scores


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