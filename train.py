#! /usr/bin/env python
import datetime
from pathlib import Path
from typing import Tuple

from FGG.dataset.input_data import Buffy, BigBangTheory, Accio
from FGG.config import FaceGroupingConfig


class Experiment(object):
    out_folder = Path(__file__).parent / "experiment_results"
    if not out_folder.is_dir():
        out_folder.mkdir(exist_ok=True, parents=True)

    def __init__(self, header=None, num_runs=1):
        """
        Base experiment for serializing multiple runs.
        Can be used for training.
        Will output a csv file with the resulting WCP of each experiment, including the dataset name.
        :param header: If None, no output experiment overview file will be produced.
                       If an empty tuple, the csv file will contain the columns
                       'dataset", "episode_train", "episode_val", "episode_test", "wcp".
                       You can add more info for experiments by setting more headers.
                       Be sure to supply values for the header in `modify_config`.
        :param num_runs: How many serialized runs there should be.
        """
        if header is None:
            self.file_name = "/dev/null"
            self.header = ""
        else:
            timestamp = str(datetime.datetime.now().timestamp()).split(".")[0]
            self.file_name = self.out_folder / f"{self.__class__.__name__}_{timestamp}.csv"
            self.header = header + ("dataset", "episode_train", "episode_val", "episode_test", "wcp",)

        self.num_runs = num_runs

    def next_experiment(self):
        """
        Generator for configs.
        Should be sent the result WCP of the previous experiment.
        :return:
        """
        with open(self.file_name, "w") as f:
            f.write(f"{','.join(self.header)}\n")
        for i in range(self.num_runs):
            config = FaceGroupingConfig()
            entries = list(map(str, self.modify_config(config, i)))
            config.model_name = self.create_model_name(config, i)
            config.finalize()
            result = yield config
            with open(self.file_name, "a") as f:
                f.write(f"{','.join(entries + [str(result), ])}\n")

    def create_model_name(self, config, i):
        prefix = Path(self.file_name).stem
        if prefix is None:
            prefix = "fgg"
        return f"{prefix}_{i}_idx{config.dataset.episode_index_train}"

    def modify_config(self, config: FaceGroupingConfig, i) -> Tuple:
        try:
            train_ep = str(config.dataset.episode_index_train + 1)
        except TypeError:
            train_ep = str(config.dataset.episode_index_train)
        try:
            val_ep = str(config.dataset.episode_index_val + 1)
        except TypeError:
            val_ep = str(config.dataset.episode_index_val)
        try:
            test_ep = str(config.dataset.episode_index_test + 1)
        except TypeError:
            test_ep = str(config.dataset.episode_index_test)

        return (config.dataset.__class__.__name__,
                train_ep, val_ep, test_ep)


class BF0502BBT0101Experiment(Experiment):

    def __init__(self, ):
        """
        Trains 5 runs on BF0502 and 5 runs on BBT0101.
        """
        self.bf = Buffy(episode_index_train=1, episode_index_val=None, episode_index_test=1)
        self.bbt = BigBangTheory(episode_index_train=0, episode_index_val=None, episode_index_test=0)
        super().__init__(header=(), num_runs=10)

    def modify_config(self, config: FaceGroupingConfig, i):
        if i < self.num_runs / 2:
            config.dataset = self.bf
        else:
            config.dataset = self.bbt

        return super().modify_config(config, i)


class BBTExperiment(Experiment):

    def __init__(self):
        self.bbt = BigBangTheory()
        super().__init__(header=(), num_runs=5)

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = self.bbt
        return super().create_model_name(config, i)


class BFExperiment(Experiment):

    def __init__(self):
        self.bf = Buffy()
        super().__init__(header=(), num_runs=5)

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = self.bf
        return super().create_model_name(config, i)


class AccioExperiment(Experiment):

    def __init__(self):
        """
        Train on Accio for 5 iterations.
        This will take a while.
        """
        self.num_clusters = 36
        self.accio = Accio(episode_index_val=None, episode_index_test=0, num_clusters=self.num_clusters)
        super().__init__(header=("num_clusters",), num_runs=5)

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = self.accio
        config.model_params["sparse_adjacency"] = True
        return (str(self.num_clusters), *super().modify_config(config, i))


class AllEpisodesExperiment(Experiment):

    def __init__(self):
        """
        Train on each episode of the Big Bang Theory for 5 times.
        """
        super().__init__(header=(), num_runs=5 * 6)

    def modify_config(self, config, i):
        config.dataset = BigBangTheory(episode_index_val=None)
        config.dataset.episode_index_train = i // 5
        config.dataset.episode_index_test = i // 5
        return super().modify_config(config, i)


if __name__ == '__main__':
    from FGG.runner import Runner
    from FGG.persistence.run_configuration import enable_auto_run_save

    enable_auto_run_save()

    # --- Train indivdual datasets (FGG 5 times) ---
    # If you want to train on BBT, uncomment:
    experiment_type = BBTExperiment()
    # If you want to train on BF, uncomment:
    #experiment_type = BFExperiment()
    # If you want to train on Accio, uncomment
    #experiment_type = AccioExperiment()

    # -----------Or train multiple together ----------------
    # You can also train both after each other automatically:
    # experiment_type = BF0502BBT0101Experiment()
    meta_experiment = experiment_type.next_experiment()
    wcp = None
    while True:
        try:
            config = meta_experiment.send(wcp)
        except StopIteration:
            break
        else:

            experiment = Runner.from_config(config, load_from=None)
            print(f"Starting to train model {config.model_name} at {datetime.datetime.now()}")
            wcp = experiment.train()
            print(f"Finished at {datetime.datetime.now()} at {wcp} wcp")
