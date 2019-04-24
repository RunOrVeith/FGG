from typing import Union
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn import metrics as metrics

from FGG.util import insert_at


def _one_hot_to_dense_torch(one_hot: torch.Tensor):
    dense = torch.argmax(one_hot, dim=1)
    assert dense.shape[0] == one_hot.shape[0]
    return dense


def _one_hot_to_dense_np(one_hot: np.ndarray):
    return np.argmax(one_hot, axis=1)


def one_hot_to_dense(one_hot: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(one_hot, np.ndarray):
        return _one_hot_to_dense_np(one_hot=one_hot)
    else:
        return _one_hot_to_dense_torch(one_hot=one_hot)


def _maybe_one_hot_to_dense(tensor: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if len(tensor.shape) == 2:
        if tensor.shape[-1] == 1:
            return tensor.squeeze(-1)
        else:
            return one_hot_to_dense(one_hot=tensor)
    elif len(tensor.shape) == 1:
        return tensor
    else:
        raise ValueError("Can not convert to dense form!")


def _maybe_to_numpy(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().data.numpy()
    return tensor


def to_dense_numpy(*args):
    out = tuple(_maybe_to_numpy(_maybe_one_hot_to_dense(tensor=tensor)) for tensor in args)
    if len(out) == 1:
        return out[0]
    return out


def mean_f1(target: np.ndarray, prediction: np.ndarray, labels):
    labels = list(range(len(labels)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return metrics.f1_score(y_true=target, y_pred=prediction, labels=labels, average="macro")


def weighted_f1(target: np.ndarray, prediction: np.ndarray, labels):
    labels = list(range(len(labels)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return metrics.f1_score(y_true=target, y_pred=prediction, labels=labels, average="weighted")


def accuracy(target: np.ndarray, prediction: np.ndarray, ):
    return metrics.accuracy_score(y_true=target, y_pred=prediction)


def mean_average_precision(target: np.ndarray, prediction: np.ndarray):
    confusion = confusion_matrix(target=target, prediction=prediction)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.diag(confusion) / np.sum(confusion, axis=0)
    precision[np.isnan(precision)] = 0.
    return np.mean(precision)


def normalized_mutual_information(target: np.ndarray, prediction: np.ndarray):
    return metrics.normalized_mutual_info_score(labels_true=target, labels_pred=prediction,
                                                average_method="arithmetic")


def adjusted_mutual_information(target: np.ndarray, prediction: np.ndarray):
    return metrics.adjusted_mutual_info_score(labels_true=target, labels_pred=prediction,
                                              average_method="arithmetic")


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, labels=None):
    matrix = metrics.confusion_matrix(y_true=target, y_pred=prediction, )

    if labels is not None:
        present_labels = np.unique(np.concatenate((prediction, target)))
        present_labels.sort()
        all_labels = np.arange(len(labels))
        missing_idxs = np.setdiff1d(all_labels, present_labels, assume_unique=True)
        if len(missing_idxs) > 0:
            matrix = insert_at(matrix, output_size=(len(all_labels), len(all_labels)),
                               indices=(missing_idxs, missing_idxs))
        matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    return matrix


def purity_score(contingency_matrix, weighted=True):
    max_percentages = (contingency_matrix / np.sum(contingency_matrix, axis=1, keepdims=True)).max(axis=1)
    max_percentages[np.isnan(max_percentages)] = 0
    if weighted:
        weighted_percentages = max_percentages * np.sum(contingency_matrix, axis=1) / np.sum(contingency_matrix)
    else:
        weighted_percentages = max_percentages / contingency_matrix.shape[0]
    purity = np.sum(weighted_percentages)
    assert 0 <= purity <= 1, purity
    return purity

