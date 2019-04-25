#! /usr/bin/env python
import datetime
from pathlib import Path
from typing import Tuple

from FGG.dataset.input_data import Buffy, BigBangTheory, Accio
from FGG.config import FaceGroupingConfig




class Experiment(object):
    out_folder = Path(__file__).parent.parent / "experiment_results"
    if not out_folder.is_dir():
        out_folder.mkdir(exist_ok=True, parents=True)

    def __init__(self, header=None, num_runs=1):
        if header is None:
            self.file_name = "/dev/null"
            self.header = ""
        else:
            timestamp = str(datetime.datetime.now().timestamp()).split(".")[0]
            self.file_name = self.out_folder / f"{self.__class__.__name__}_{timestamp}.csv"
            self.header = header + ("dataset", "episode_train", "episode_val", "episode_test", "wcp",)

        self.num_runs = num_runs

    def next_experiment(self):
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
            test_ep = str(config.dataset.episode_index_val + 1)
        except TypeError:
            test_ep = str(config.dataset.episode_index_val)

        return (config.dataset.__class__.__name__,
                train_ep, val_ep, test_ep)


class BF0502BBT0101Experiment(Experiment):

    def __init__(self, header=("changes",), num_runs=10):
        self.bf = Buffy(episode_index_train=1, episode_index_val=None, episode_index_test=1)
        self.bbt = BigBangTheory(episode_index_train=0, episode_index_val=None, episode_index_test=0)
        super().__init__(header=header, num_runs=num_runs)

    def modify_config(self, config: FaceGroupingConfig, i):
        if i < self.num_runs / 2:
            config.dataset = self.bf
        else:
            config.dataset = self.bbt

        return ("shuffle", *super().modify_config(config, i))


class AccioExperiment(Experiment):

    def __init__(self):
        self.num_clusters = 36
        self.accio = Accio(episode_index_val=None, episode_index_test=0, num_clusters=self.num_clusters)
        super().__init__(header=("num_clusters",), num_runs=5)

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = self.accio
        config.model_params["sparse_adjacency"] = True
        config.graph_builder_params["pair_sample_fraction"] = 1
        config.graph_builder_params["edge_between_top_fraction"] = 0.03

        return (str(self.num_clusters), *super().modify_config(config, i))


class AllEpisodesExperiment(Experiment):

    def __init__(self):
        super().__init__(header=(), num_runs=5 * 6)

    def modify_config(self, config, i):
        config.dataset = BigBangTheory(episode_index_val=None)
        config.dataset.episode_index_train = i // 5
        config.dataset.episode_index_test = i // 5
        return super().modify_config(config, i)


class DebugExperiment(Experiment):

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = BigBangTheory(episode_index_train=0, episode_index_val=None, episode_index_test=None)
        config.train_epochs = 3
        return super().modify_config(config, i)


if __name__ == '__main__':
    from FGG.config import FaceGroupingConfig
    from FGG.runner import Runner
    from FGG.persistance.run_configuration import enable_auto_run_save

    enable_auto_run_save()
    experiment_type = DebugExperiment()
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
