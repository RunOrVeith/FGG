import abc
import contextlib
import re
from functools import partial
from typing import List

import numpy as np

from FGG.metrics.base import mean_f1, mean_average_precision, confusion_matrix, accuracy, purity_score, weighted_f1, \
    adjusted_mutual_information, normalized_mutual_information
from FGG.metrics.BCUBED.B3score.b3 import calc_b3


class Metric(object, metaclass=abc.ABCMeta):

    def __init__(self):
        self._real_submit = None
        self._real_compute = None

    def name(self) -> str:
        return re.sub(r"(\w)([A-Z])", r"\1 \2", self.__class__.__name__).lower()

    def provides(self) -> List[str]:
        return [self.name()]

    @contextlib.contextmanager
    def batch_mode(self):
        self._real_submit = self.submit
        self._real_compute = self.compute
        self.submit = self._dummy_submit
        self.compute = self._dummy_compute
        yield
        self.submit = self._real_submit
        self.compute = self._real_compute
        self._real_submit = None
        self._real_compute = None

    @abc.abstractmethod
    def submit(self, *args, **kwargs):
        raise NotImplementedError

    def _dummy_submit(self, *args, **kwargs):
        return None

    def _dummy_compute(self):
        return None

    @abc.abstractmethod
    def compute(self):
        raise NotImplementedError


class BatchableMetric(Metric, metaclass=abc.ABCMeta):

    def __init__(self):
        super().__init__()
        self._batch_mode = False

    @contextlib.contextmanager
    def batch_mode(self):
        self._batch_mode = True
        self._real_compute = self.compute
        self.compute = self._dummy_compute
        yield
        self.compute = self._real_compute
        self._real_compute = None
        self._batch_mode = False


class ConcatBatchableMetric(BatchableMetric):

    def __init__(self, func, **kwargs):
        super().__init__()
        self._targets = None
        self._predictions = None
        self._reset_batch_storage()

        if len(kwargs) > 0:
            curry = partial(func, **kwargs)
            curry.__name__ = func.__name__
            self.func = curry
        else:
            self.func = func

    def submit(self, target, prediction):
        self._targets.append(target)
        self._predictions.append(prediction)

        if not self._batch_mode:
            return self.compute()

    def compute(self):
        targets = np.concatenate(self._targets)
        predictions = np.concatenate(self._predictions)
        self._reset_batch_storage()
        return self.func(target=targets, prediction=predictions)

    def name(self):
        return self.func.__name__.replace("_", " ")

    def _reset_batch_storage(self):
        self._targets = []
        self._predictions = []


class ConfusionMatrix(BatchableMetric):

    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self._batched_confusion = None
        self._reset_batch_storage()

    def submit(self, target, prediction):
        confusion = confusion_matrix(target=target, prediction=prediction, labels=self.labels)
        if not self._batch_mode:
            self._batched_confusion = confusion
            return self.compute()
        else:
            self._batched_confusion += confusion

    def compute(self):
        confusion = self._batched_confusion
        self._reset_batch_storage()
        return confusion

    def _reset_batch_storage(self):
        self.batched_confusion = np.zeros(shape=(len(self.labels), len(self.labels)))


class ClusteringPurity(BatchableMetric):

    def __init__(self, labels, weighted=True):
        super().__init__()
        self.weighted = weighted
        self.labels = labels
        self.batched_contingency = None
        self._reset_batch_storage()

    def submit(self, target, cluster_assignment):
        t = confusion_matrix(target=target, prediction=cluster_assignment,
                             labels=self.labels if self._batch_mode else None).T
        if self._batch_mode:
            t = t.values
        if not self._batch_mode:
            self.batched_contingency = t
            return self.compute()
        else:
            self.batched_contingency += t

    def compute(self):
        purity = purity_score(contingency_matrix=self.batched_contingency, weighted=self.weighted)
        self._reset_batch_storage()
        return purity

    def name(self):
        return f"{'weighted' if self.weighted else 'unweighted'} {super().name()}"

    def _reset_batch_storage(self):
        self.batched_contingency = np.zeros(shape=(len(self.labels), len(self.labels)))


class Accuracy(ConcatBatchableMetric):

    def __init__(self):
        super().__init__(func=accuracy)


class F1Score(ConcatBatchableMetric):

    def __init__(self, labels, weighted=False):
        super().__init__(func=mean_f1 if not weighted else weighted_f1, labels=labels)


class NormalizedMutualInformation(ConcatBatchableMetric):

    def __init__(self, adjusted=False):
        if adjusted:
            super().__init__(func=adjusted_mutual_information)
        else:
            super().__init__(func=normalized_mutual_information)


class MeanAveragePrecision(ConcatBatchableMetric):

    def __init__(self):
        super().__init__(func=mean_average_precision)


class BCubed(ConcatBatchableMetric):

    def __init__(self, class_norm=False, beta=1.0):
        def wrap_b3(target, prediction):
            return calc_b3(L=target, K=prediction, class_norm=class_norm, beta=beta)

        wrap_b3.__name__ = "bcubed"
        super().__init__(func=wrap_b3)

    def provides(self):
        return [f"{self.name()} f-score", f"{self.name()} precision", f"{self.name()} recall"]