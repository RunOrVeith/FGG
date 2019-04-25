#! /usr/bin/env python
import datetime
from FGG.dataset.input_data import Buffy, BigBangTheory, Accio
from FGG.config import FaceGroupingConfig

from FGG.runner import MetaExperiment, TestExperiment, InferenceExperiment


class BF0502BBT0101Experiment(MetaExperiment):

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


class AccioExperiment(MetaExperiment):

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


class AllEpisodesExperiment(MetaExperiment):

    def __init__(self):
        super().__init__(header=(), num_runs=5 * 6)

    def modify_config(self, config, i):
        config.dataset = BigBangTheory(episode_index_val=None)
        config.dataset.episode_index_train = i // 5
        config.dataset.episode_index_test = i // 5
        return super().modify_config(config, i)


class PooledExperiment(TestExperiment):

    def __init__(self):
        super().__init__(header=("split_frames",), num_runs=3)

    def modify_config(self, config: FaceGroupingConfig, i):
        from FGG.dataset.split_strategy import SplitEveryXFrames
        x = 10
        config.graph_builder_params["split_strategy"] = SplitEveryXFrames(x=x)
        config.pool_before_clustering = True
        if i == 0:
            config.model_load_file = "/cvhci/data/PLUMCOT/AVT_Veith/veith/best_models/bbt0101/checkpoint.tar"
            config.dataset = BigBangTheory(episode_index_val=None, episode_index_train=None,
                                           episode_index_test=0)
        elif i == 1:
            config.model_load_file = "/cvhci/data/PLUMCOT/AVT_Veith/veith/best_models/buffy0502/checkpoint.tar"
            config.dataset = Buffy(episode_index_val=None, episode_index_train=None,
                                   episode_index_test=1)
        else:
            config.model_load_file = "/cvhci/data/PLUMCOT/AVT_Veith/veith/best_models/accio/40/checkpoint.tar"
            config.model_params["sparse_adjacency"] = True
            config.dataset = Accio(episode_index_test=0, episode_index_val=None, episode_index_train=None)
        return (str(x), *super().modify_config(config, i))


class InferExperiment(InferenceExperiment):

    def __init__(self):
        self.dataset = Buffy()
        super().__init__(checkpoint_file="/tmp/bbt_checkpoint.tar")  # TODO Fill this out

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = self.dataset
        return super().modify_config(config, i)


class DebugExperiment(MetaExperiment):

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = BigBangTheory(episode_index_train=0, episode_index_val=None, episode_index_test=None)
        config.train_epochs = 3
        return super().modify_config(config, i)


if __name__ == '__main__':
    from FGG.config import FaceGroupingConfig
    from FGG.runner import Runner
    from FGG.persistance.run_configuration import enable_auto_run_save

    enable_auto_run_save()
    experiment_type = InferExperiment()
    meta_experiment = experiment_type.next_experiment()
    wcp = None
    while True:
        try:
            config = meta_experiment.send(wcp)
        except StopIteration:
            break
        else:

            if isinstance(experiment_type, InferenceExperiment):
                print("Running inference, assuming we don't know any labels!")
                experiment = Runner.from_config(config, load_from="last")
                wcp = experiment.infer()
            elif isinstance(experiment_type, TestExperiment):
                print("Running tests only!")
                experiment = Runner.from_config(config, load_from="last")
                wcp = experiment.test()
            else:
                experiment = Runner.from_config(config, load_from=None)
                print(f"Starting to train model {config.model_name} at {datetime.datetime.now()}")
                wcp = experiment.train()
            print(f"Finished at {datetime.datetime.now()} at {wcp} wcp")
