import torch
import numpy as np

from FGG.graph_builder import EdgeTypes


class ContrastiveLoss(torch.nn.Module):

    # From http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf

    def __init__(self, margin=1., pos_weight=1., neg_weight=1., squared=True, pos_label=1):
        super().__init__()
        self.margin = margin
        self.pos_label = pos_label
        self.squared = squared
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, logits, targets):
        x1, x2 = logits[..., 0], logits[..., 1]
        distance = torch.sum((x1 - x2) ** 2, dim=-1)
        if not self.squared:
            distance = torch.sqrt(distance)
        margin_distance = torch.clamp(self.margin - distance, min=0.)
        loss = 0.5 * torch.where(targets == self.pos_label, self.pos_weight * distance, self.neg_weight * margin_distance)
        return loss.mean()


class GraphContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1., pos_weight=1., neg_weight=1., squared=True, pos_label=1,
                 directed=False):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(margin=margin, pos_weight=pos_weight, neg_weight=neg_weight,
                                                squared=squared, pos_label=pos_label)
        self.directed = directed

    def forward(self, adjacency_matrix, logits):
        # adj should have a 1 entry if the connection was "must-link"
        if EdgeTypes.must_link < EdgeTypes.cannot_link:
            adj = adjacency_matrix.argmin(dim=0)
        else:
            adj = adjacency_matrix.argmax(dim=0)
        # Get all connected nodes indices, regardless which type
        all_idx = self.connected_entries_idxs(adj=torch.abs(adjacency_matrix).sum(dim=(0,)))
        targets = adj[all_idx[..., 0], all_idx[..., 1]].flatten()
        pairs = logits[all_idx.flatten()].reshape(-1, 2, logits.shape[-1]).permute(0, 2, 1)
        assert pairs.shape[0] == targets.shape[0]
        loss = self.contrastive_loss(logits=pairs, targets=targets)
        return loss

    def connected_entries_idxs(self, adj, relevant_rows=None, copy=False):
        """
        Returns the indices of all nodes in the graph adjacency matrix (taking into account the relevant rows),
        that are connected. Indices are of shape (batch, 2), where the last dimension is row, column.
        :param relevant_rows: relevant rows in the adjacency matrix. Only entries from this list will be returned.
                             (i.e. nodes corresponding to rows not in this list are discarded).
                             Default is to use all connected components.
        :param adj: The adjacency matrix to find all connected components for. Does not handle typed adjacency matrices.
        :param copy: The adj tensor will be modified. Use this flag to copy first.
        :return:
        """
        assert len(adj.shape) == 2, "Can not handle multiple types here"
        if copy:
            adj_ = adj.copy()
        else:
            adj_ = adj
        if not self.directed:
            # Since the graph if undirected we only need the entries lower than the diagonal
            # otherwise we would get duplicates
            triangle_idx = np.triu_indices(adj.size(-1), k=0)
            adj_[triangle_idx] = 0.
        # We don't care for any nodes that are not relevant for the loss
        if relevant_rows is not None:
            irrelevant_indices = np.setxor1d(relevant_rows, np.arange(adj.size(-1)))
            adj_[irrelevant_indices] = 0.
            adj_[:, irrelevant_indices] = 0.
        return torch.nonzero(adj_)
