from pathlib import Path
from train import Experiment

from FGG.dataset.input_data import Accio, BigBangTheory, Buffy
from FGG.config import FaceGroupingConfig


class EvalExperiment(Experiment):

    def __init__(self, checkpoint_file: str = ..., header=(), num_runs=1):
        """
        Runs evaluation. Assumes that labels are known.
        If you dont' have labels, please use "infer.py" instead.

        :param checkpoint_file: Checkpoint file to load weights from.
                                The default is to look for pretrained weights in this repo in the "weights" folder
                                depending on the selected dataset.
        """
        self.checkpoint_file = checkpoint_file
        super().__init__(header=header, num_runs=num_runs)

    def modify_config(self, config: FaceGroupingConfig, i):
        out = super().modify_config(config, i)
        assert config.dataset.episode_index_test is not None
        config.dataset.episode_index_train = None
        config.dataset.episode_index_val = None
        if isinstance(config.dataset, BigBangTheory) and self.checkpoint_file is Ellipsis:
            config.model_load_file = "weights/bbt0101/checkpoint.tar"
        elif isinstance(config.dataset, Buffy) and self.checkpoint_file is Ellipsis:
            config.model_load_file = "weights/bf0502/checkpoint.tar"
        elif isinstance(config.dataset, Accio) and self.checkpoint_file is Ellipsis:
            config.model_load_file = "weights/accio/checkpoint.tar"
        else:
            config.model_load_file = self.checkpoint_file

        assert config.model_load_file is None or Path(config.model_load_file).is_file()
        return out

    def create_model_name(self, config, i):
        prefix = Path(self.file_name).stem
        try:
            return f"{prefix}_{i}_ep{config.dataset.episode_index_test + 1}"
        except TypeError:
            return f"{prefix}_{i}_idx{config.dataset.episode_index_test}"


class PooledExperiment(EvalExperiment):

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


class EvaluateBBT0101BF0502Experiment(EvalExperiment):

    def __init__(self):
        super().__init__(num_runs=2)

    def modify_config(self, config: FaceGroupingConfig, i):
        if i == 0:
            config.dataset = BigBangTheory(episode_index_test=0)
        else:
            config.dataset = Buffy(episode_index_test=1)

        return super().modify_config(config, i)


if __name__ == '__main__':
    import datetime
    from FGG.config import FaceGroupingConfig
    from FGG.runner import Runner
    from FGG.persistance.run_configuration import enable_auto_run_save

    enable_auto_run_save()
    experiment_type = EvaluateBBT0101BF0502Experiment()
    meta_experiment = experiment_type.next_experiment()
    wcp = None
    while True:
        try:
            config = meta_experiment.send(wcp)
        except StopIteration:
            break
        else:
            assert isinstance(experiment_type, EvalExperiment)
            print("Running evaluation only!")
            experiment = Runner.from_config(config, load_from="last")
            wcp = experiment.test()
            print(f"Finished at {datetime.datetime.now()} at {wcp} wcp")
