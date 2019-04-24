import tempfile

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import numpy as np


def agglomerative_clustering(features, n_clusters: int, connectivity=None):
    """
    Perform agglomerative clustering.
    :param features: The features to cluster.
    :param n_clusters: The numer of clusters.
    :param connectivity: Compute KNN (K=this parameter) for potential improved clustering.
                         Defaults to unused.
    :return:
    """
    with tempfile.TemporaryDirectory() as folder:
        if isinstance(connectivity, int):
            connectivity = kneighbors_graph(features, connectivity, include_self=False)
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", memory=folder,
                                        connectivity=connectivity)
        return model.fit_predict(features)


def cluster_mode_prediction(target: np.ndarray, prediction: np.ndarray, cluster_labels_only=False):
    """
    Predicts labels for elements based on the most common element per cluster.
    :param target: The labels for each element.
    :param prediction: The predicted cluster assignments.
    :param cluster_labels_only: If True only return the label for each cluster
                                instead of a class prediction for each element.
    :return:
    """
    class_prediction = np.empty_like(prediction)
    unique = np.unique(prediction)
    modes = []
    for cluster_id in unique:
        assigned_to_this = prediction == cluster_id
        true_labels = target[assigned_to_this]
        true_labels_in_cluster, counts = np.unique(true_labels, return_counts=True)
        mode = true_labels_in_cluster[counts.argmax()]
        modes.append(mode)
        class_prediction[assigned_to_this] = mode

    return class_prediction if not cluster_labels_only else modes
