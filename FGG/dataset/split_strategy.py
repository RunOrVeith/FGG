import abc
from typing import Union, Optional

import numpy as np

from FGG.dataset.tracks import Track


class SplitStrategy(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, track: Track) -> Union[int, np.ndarray]:
        """
        :param track:
        :return: The number of sub-tracks to split the given track into,
                 or an array of indices (see numpy.split documentation).
        """
        pass

    def reset(self):
        pass


class HeuristicSplit(SplitStrategy):

    def __init__(self, no_split_under=50, max_split=10, step_size=10):
        """
        Heuristic split from the paper.
        :param no_split_under: X from the paper. Tracks below this length will not be split.
        :param max_split: B from the paper. Split at most into this many sub-tracks.
        :param step_size: Delta from the paper.
                          Increase the number of resulting sub-tracks after every additional `step_size` frames.
        """
        self.no_split_under = no_split_under
        self.max_split = max_split
        self.step_size = step_size

    def __call__(self, track: Track):
        into = max(1, min(self.max_split, 1 + (len(track) - self.no_split_under) // self.step_size))
        return into


class NoSplit(SplitStrategy):

    """
    Does not split the tracks at all.
    """
    def __call__(self, track: Track):
        return 1


class SplitEveryXFrames(SplitStrategy):

    def __init__(self, x=10):
        """
        :param x: For every x frames a node is created.
                  This may result in very large (-> slow/memory!) graphs.
        """
        self.x = x

    def __call__(self, track: Track):
        return np.arange(0, len(track), self.x)[1:]


class MaxSplit(SplitStrategy):

    def __call__(self, track: Track):
        """
        :return: Split each track the maximum number of times, i.e. a frame-level representation.
                 This may result in very large (-> slow/memory!) graphs.
        """
        return len(track)


class TrackLengthSplit(SplitStrategy):

    def __init__(self, increase_every=10):
        """
        Experimental second way to get more features.
        :param increase_every:
        """
        self.increase_every = increase_every

    def __call__(self, track: Track):
        return 1 + (len(track) // self.increase_every)


class RandomSplitStrategy(SplitStrategy):

    def __init__(self, rng: Union[Optional[int], np.random.RandomState] = None, max_split: int = None):
        """
        Split each track a random number of times.
        :param rng: Random number generator or seed.
        :param max_split: The maximum number of sub-tracks that can be chosen.
        """
        self._original_rng = rng
        self.rng = None
        self.reset()
        self.max_split = max_split
        super().__init__()

    def __call__(self, track: Track):
        if self.max_split is not None:
            return self.rng.randint(1, min(self.max_split, len(track)))
        else:
            return self.rng.randint(1, len(track))

    def reset(self):
        """
        Reset this split strategy so that we can train and test on the same graph.
        :return:
        """
        if isinstance(self._original_rng, int) or self._original_rng is None:
            self.rng = np.random.RandomState(seed=self._original_rng)
        else:
            self.rng = self._original_rng
