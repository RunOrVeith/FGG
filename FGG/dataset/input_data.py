import abc
from pathlib import Path
import warnings
from typing import Union, List, Tuple
import numpy as np
from functools import lru_cache

from FGG.dataset.person_id_handler import PersonIDHandler
from FGG.dataset.tracks import TrackCollection


def get_data_base_path() -> Path:
    """
    :return: Base Path to where the datasets are stored.
    """
    #return Path(__file__).parent.parent / "/datasets"
    return Path("/home/veith/data/masterarbeit-data/formatted_data")


class InputData(object, metaclass=abc.ABCMeta):

    def __init__(self, episode_index_train: Union[int, list, None], possible_episode_paths: List[Path],
                 center_crop_idx: int, episode_index_val: Union[int, List[int], None] = None,
                 episode_index_test: Union[int, List[int], None] = ...,
                 include_unknown: bool = False, **matfile_format):
        """
        Abstract base class for input datasets.
        Note that for all indices of episodes the offset to the episode number is 1, i.e.
        to select Episode 1 the index should be 0.
        :param episode_index_train: The index of the episode in the list of possible episodes to train on.
                                    If None, no episode will be loaded.
        :param possible_episode_paths: The list of paths to possible episodes to train on.
        :param center_crop_idx: If the dataset contains features for multiple crop indices,
                                you can select the one you want here.
        :param episode_index_val: The indices of the episodes to validate on after each training epoch.
                                  If None, no validation will occur.
                                  If Ellipsis (i.e. '...') the same as training episodes will be used.
        :param episode_index_test: The indices of the episodes to test on after training has completed.
                                   If None, no testing will occur.
                                   If Ellipsis (i.e. '...') the same as training episodes will be used.
        :param include_unknown: Whether to include a class of unknown characters, i.e. ones that are not listed in
                                the main_characters.
        :param matfile_format: Overwrite the default matfile format headers to load data in other formats.
        """

        self.person_id_handler = PersonIDHandler(main_characters=self.main_characters(),
                                                 include_unknown=include_unknown)
        self.episode_index_train = episode_index_train
        self.episode_index_val = episode_index_train if episode_index_val is Ellipsis else episode_index_val
        self.episodes = possible_episode_paths
        self.matfile_format = matfile_format
        self.center_crop_idx = center_crop_idx
        self.episode_index_test = episode_index_train if episode_index_test is Ellipsis else episode_index_test
        # Per instance LRU cache so that we don't have to reload the data for consecutive experiments
        self.load_episodes = lru_cache(maxsize=1)(self.load_episodes)

    def train_episodes(self) -> List[Path]:
        """
        :return: List of files of selected episodes to train on.
        """
        if self.episode_index_train is None:
            return []
        eps = np.array(self.episodes)[self.episode_index_train]
        if isinstance(eps, Path):
            return [eps]
        return eps.tolist()

    def test_episodes(self) -> List[Path]:
        """
        :return: List of files of selected episodes to test on.
        """
        if self.episode_index_test is None:
            return []
        eps = np.array(self.episodes)[self.episode_index_test]
        if isinstance(eps, Path):
            return [eps]
        return eps.tolist()

    def val_episodes(self) -> List[Path]:
        """
        :return: List of files of selected episodes to validate on.
        """
        if self.episode_index_val is None:
            return []
        eps = np.array(self.episodes)[self.episode_index_val]
        if isinstance(eps, Path):
            return [eps]
        return eps.tolist()

    @staticmethod
    @abc.abstractmethod
    def main_characters() -> List[str]:
        """
        Overwrite this method in each new subclass to select the characters.
        Should return the names as they appear in the labels.
        """
        pass

    def load_episodes(self) -> Tuple[List[TrackCollection], List[TrackCollection], List[TrackCollection]]:
        """
        Loads all selected episodes into memory as track collections.
        :return: Training episodes, validation episodes, test episodes
        """
        episodes = dict()
        train_episodes = self.train_episodes()
        val_episodes = self.val_episodes()
        test_episodes = self.test_episodes()

        for file in train_episodes + test_episodes + val_episodes:
            # Only load what we have not already loaded
            # (e.g. no need to load twice if we train and test on the same episode)
            if file not in episodes:
                episodes[file] = TrackCollection.from_hdf5_track_file(track_file=file,
                                                                      person_id_handler=self.person_id_handler,
                                                                      crop_idx=self.center_crop_idx,
                                                                      **self.matfile_format)

        train_episodes = [episodes[file] for file in train_episodes]
        val_episodes = [episodes[file] for file in val_episodes]
        test_episodes = [episodes[file] for file in test_episodes]

        print("Train data distribution:")
        self.label_distribution(train_episodes)
        print("Val data distribution:")
        self.label_distribution(val_episodes)
        print("Test data distribution:")
        self.label_distribution(test_episodes)

        return train_episodes, val_episodes, test_episodes

    def label_distribution(self, episodes) -> None:
        """
        Prints out an overview over the label distribution in the episodes.
        :param episodes: Episodes to print the distribution for.
        """
        if len(episodes) == 0:
            print("No data")
            return
        for episode in episodes:
            print(self.person_id_handler.get_distribution_of_persons(episode))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}" \
               f" {self.main_characters()} " \
               f"train {self.episode_index_train} val {self.episode_index_val}"


# TODO re-store VGG datasets in given folder.

class BigBangTheory(InputData):
    """
    Dataset for The Big Bang Theory.
    Available episodes are season 1 1-6.
    By default BBT0101 is trained and tested, and there is no validation step.
    """

    def __init__(self, episode_index_train: Union[int, slice, list, None] = 0, include_unknown=False,
                 episode_index_val=None, episode_index_test=...):
        possible_episodes = list(
            map(lambda x: get_data_base_path() / x, [f"bbt/supervised/bbt_s01e0{i}/tracks.mat" for i in range(1, 7)])
        )
        super().__init__(possible_episode_paths=possible_episodes, center_crop_idx=4,
                         episode_index_train=episode_index_train, include_unknown=include_unknown,
                         episode_index_val=episode_index_val, episode_index_test=episode_index_test)

    @staticmethod
    def main_characters():
        return ["penny", "howard", "leonard", "sheldon", "raj"]


class Buffy(InputData):

    """
    Dataset for Buffy - The Vampire Slayer.
    Available episodes are season 5 1-6.
    By default BF0502 is trained and tested, and there is no validation step.
    """

    def __init__(self, episode_index_train: Union[int, slice, list, None] = 1, include_unknown=False,
                 episode_index_val=None, episode_index_test=...):
        possible_episodes = list(
            map(lambda x: get_data_base_path() / x,
                [f"buffy/supervised/buffy_s05e0{i}/tracks.mat" for i in range(1, 7)])
        )
        super().__init__(possible_episode_paths=possible_episodes, center_crop_idx=4,
                         episode_index_train=episode_index_train, include_unknown=include_unknown,
                         episode_index_val=episode_index_val, episode_index_test=episode_index_test,)

    @staticmethod
    def main_characters():
        return ['xander', 'buffy', 'dawn', 'anya', 'willow', 'giles']


class Accio(InputData):

    """
    Dataset for ACCIO (Harry Potter 1).
    You can not select episodes here (but dis/enable validation works).

    WARNING:
        When instatiating Accio, the global class property that returns the number of clusters is overwritten
        with a value that can be chosen.
        This also effects OTHER datasets such as BigBangTheory due to the way properties work in python.
        Don't instatiate Accio and another dataset at the same time!
    """

    def __init__(self, episode_index_train: Union[None, int, slice] = 0, num_clusters=36, include_unknown=False,
                 episode_index_val=None, episode_index_test=...):
        possible_episodes = list(
            map(lambda x: get_data_base_path() / x, [f"hp/supervised/hp_s01e0{i}/tracks.mat" for i in range(1, 2)])
        )
        super().__init__(possible_episode_paths=possible_episodes, center_crop_idx=0,
                         episode_index_train=episode_index_train, include_unknown=include_unknown,
                         episode_index_val=episode_index_val, label_header="groundTruthIdentity",
                         episode_index_test=episode_index_test)

        # replace num_characters with num_characters + 1 because people messed up in accio
        # and used 36 clusters instead of 35 named characters

        def _fake_num_characters(_self):
            return num_clusters

        # Ignore pycharm, this works. But don't use load Accio and a different dataset at the same time
        # as it will change the property for all instances
        PersonIDHandler.num_characters = property(_fake_num_characters)
        assert self.person_id_handler.num_characters == num_clusters
        warnings.warn(f"Using Accio; Using fake cluster count {num_clusters} as num_characters for all loaded datasets")

    @staticmethod
    def main_characters():
        return ["albus_dumbledore", "angelina_johnson", "argus_filch", "aurora_sinistra", "dean_thomas",
                "draco_malfoy", "dudley_dursley", "fat_lady", "fred_weasley", "ginny_weasley", "gregory_goyle",
                "griphook", "harry_potter", "hermione_granger", "james_potter", "lee_jordan", "lily_potter",
                "minerva_mcgonagall", "molly_weasley", "neville_longbottom", "oliver_wood", "ollivander",
                "percy_weasley", "petunia_dursley", "quirinus_quirrell", "rolanda_hooch", "ron_weasley",
                "rubeus_hagrid", "seamus_finnigan", "severus_snape", "sir_nicholas", "tom",
                "vernon_dursley", "vincent_crabbe", "voldemort"]


