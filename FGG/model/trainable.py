from typing import Sequence
import torch
import torch.nn.functional as F

from FGG.model.layers import ResidualGraphConvolution, GraphConvolution


class GCN(torch.nn.Module):

    def __init__(self, feature_progression: Sequence[int], num_edge_types: int = 1, activation=F.elu,
                 use_residual=True, sparse_adjacency=True):
        super().__init__()
        assert len(feature_progression) >= 2
        self.activation = activation
        self._layers = torch.nn.ModuleList()
        self._bns = torch.nn.ModuleList()
        layer_type = ResidualGraphConvolution if use_residual else GraphConvolution

        for in_features, out_features in zip(feature_progression[:-1], feature_progression[1:]):
            gc = layer_type(in_features=in_features, out_features=out_features, num_edge_types=num_edge_types,
                            sparse_adjacency=sparse_adjacency)
            self._layers.append(gc)
        for i in range(1, len(feature_progression)):
            self._bns.append(torch.nn.BatchNorm1d(num_features=feature_progression[i],
                                                  track_running_stats=False))

    def forward(self, features, adjacency_matrix):
        result = [features]
        for i, layer in enumerate(self._layers):
            out = layer(features=result[-1], adjacency_matrix=adjacency_matrix)
            if i < len(self._layers) - 1:
                out = self._bns[i](out)
                out = self.activation(out)
            result.append(out)
        return result[1:]


class FGG(torch.nn.Module):

    def __init__(self, in_feature_size: int = 2048, downsample_feature_size: int = 1024,
                 gc_feature_sizes: Sequence[int] = (512, 256, 128),
                 num_edge_types: int = 2,
                 activation=F.elu, use_residual=True, sparse_adjacency=True):
        """

        :param in_feature_size: Number of input features
        :param downsample_feature_size: Output feature size after the first linear layer.
        :param gc_feature_sizes: Progression of feature sizes for resGC layers.
                                 Will create as many layers as there are numbers here.
        :param num_edge_types: The number of edge types. 2 should be fine.
        :param activation: Activation function to use.
        :param use_residual: If False use normal GC layers instead of residual layers.
        :param sparse_adjacency: Whether to multiply using sparse adjacency matrices.
                                 This saves some memory, but is still not very efficient.
        """
        super().__init__()
        self.activation = activation
        self.downsample = torch.nn.Linear(in_features=in_feature_size, out_features=downsample_feature_size, bias=True)
        self.gcn = GCN(feature_progression=(downsample_feature_size, *gc_feature_sizes), num_edge_types=num_edge_types,
                       activation=activation, use_residual=use_residual, sparse_adjacency=sparse_adjacency)

    def forward(self, features, adjacency_matrix):
        features = self.downsample(features)
        features = self.activation(features)
        # The GCN returns the output of all layers
        features = self.gcn(features=features, adjacency_matrix=adjacency_matrix)[-1]
        return features
