import torch


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, in_features: int, out_features: int, bias=True, num_edge_types=2, sparse_adjacency=False):
        """

        :param in_features: Input feature size
        :param out_features: Output feature size
        :param bias: If True add a bias term per edge type.
        :param num_edge_types: Number of expected edge types.
        :param sparse_adjacency: If True use a sparse representation of the adjacency matrix.
                                 This is not very fast and needs two multiplications instead of one.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_edge_types = num_edge_types
        assert self.num_edge_types > 0
        self.weight = torch.nn.Parameter(torch.Tensor(num_edge_types, in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(num_edge_types, 1, out_features))
        else:
            self.register_parameter('bias', None)

        self._sparse_adjacency = sparse_adjacency

        self.apply(self._initialize)

    def _initialize(self, m):
        """
        Initialize a set of parameters with the uniform He initialization.
        :param m:
        :return:
        """
        torch.nn.init.kaiming_uniform_(m.weight.data)
        torch.nn.init.kaiming_uniform_(m.bias.data)

    def _to_correct_shapes(self, features, adjacency_matrix):
        """
        Adapt the shapes of the inputs to the expected formats (i.e. edge type dimension).
        :param features:
        :param adjacency_matrix:
        :return:
        """
        assert features.size(-1) == self.in_features
        assert adjacency_matrix.size(-1) == adjacency_matrix.size(-2)
        if adjacency_matrix.dim() == 2:
            # There seems to be only one edge type
            adjacency_matrix = adjacency_matrix.unsqueeze(0)
        assert features.size(0) == adjacency_matrix.size(1), f"{features.shape}, {adjacency_matrix.shape}"
        assert adjacency_matrix.size(0) == self.num_edge_types, f"{adjacency_matrix.size(0)} vs.{self.num_edge_types}"
        return features, adjacency_matrix

    def forward(self, features, adjacency_matrix):
        """
        Perform a typed graph convolution of the features using the adjacency matrix.
        Warning: Assumes we have a torch tensor for the adjacency where the renormalization trick has already been applied
        :param features:
        :param adjacency_matrix:
        :return:
        """
        features, adjacency_matrix = self._to_correct_shapes(features=features, adjacency_matrix=adjacency_matrix)

        if not self._sparse_adjacency:
            # Compute multiple edge types in one multiplication in dense representation
            expanded_features = features.expand(self.num_edge_types, -1, -1)
            output = adjacency_matrix @ expanded_features @ self.weight
        else:
            # TODO: pytorch does not support sparse matrix multiplication with more than 2 dimensions unfortunately
            adj_by_feats = tuple([torch.sparse.mm(adj.to_sparse(), features) for adj in adjacency_matrix])
            adj_by_feats = torch.stack(adj_by_feats)
            output = adj_by_feats @ self.weight

        assert output.shape == (adjacency_matrix.size(0), adjacency_matrix.size(1), self.out_features)
        if self.bias is not None:
            output = output + self.bias

        # Need to reduce the features created by the different edge types into a single feature matrix
        # Multiple edge types from http://cbl.eng.cam.ac.uk/pub/Intranet/MLG/ReadingGroup/2018-11-14_GNNs.pdf page 12
        output = torch.sum(output, dim=(0,)) / self.num_edge_types
        return output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"


class ResidualGraphConvolution(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, num_edge_types=1, sparse_adjacency=False):
        """

        :param in_features:
        :param out_features:
        :param bias:
        :param num_edge_types:
        :param sparse_adjacency:
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gc1 = GraphConvolution(in_features=in_features, out_features=out_features, bias=bias,
                                    num_edge_types=num_edge_types, sparse_adjacency=sparse_adjacency)

        if in_features != out_features:
            self.shortcut_projection = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        else:
            self.shortcut_projection = None

    def forward(self, features, adjacency_matrix):
        shortcut = features
        y = self.gc1(features=features, adjacency_matrix=adjacency_matrix)
        if self.shortcut_projection is not None:
            shortcut = self.shortcut_projection(shortcut)
        output = y + shortcut
        return output


