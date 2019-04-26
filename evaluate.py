from pathlib import Path
from train import Experiment
from typing import Union

from FGG.dataset.input_data import Accio, BigBangTheory, Buffy
from FGG.config import FaceGroupingConfig


class EvalExperiment(Experiment):

    def __init__(self, checkpoint_file: Union[None, str] = ..., header=(), num_runs=1):
        """
        Runs evaluation. Assumes that labels are known.
        If you dont' have labels, please use "infer.py" instead.

        :param checkpoint_file: Checkpoint file to load weights from.
                                The default is to look for pretrained weights in this repo in the "weights" folder
                                depending on the selected dataset.
                                If None evaluation will happen on randomly initialized weights.
                                Otherwise you can provide a checkpoint file path.
        """
        self.checkpoint_file = checkpoint_file
        super().__init__(header=header, num_runs=num_runs)
        self.load_from = "last"

    def modify_config(self, config: FaceGroupingConfig, i):
        out = super().modify_config(config, i)
        assert config.dataset.episode_index_test is not None
        config.dataset.episode_index_train = None
        config.dataset.episode_index_val = None

        # These are the paths for the pretrained weights.
        if isinstance(config.dataset.episode_index_test, int):
            if isinstance(config.dataset, BigBangTheory) and self.checkpoint_file is Ellipsis:
                config.model_load_file = f"weights/bbt010{config.dataset.episode_index_test + 1}_checkpoint.tar"
            elif isinstance(config.dataset, Buffy) and self.checkpoint_file is Ellipsis:
                config.model_load_file = f"weights/bf050{config.dataset.episode_index_test + 1}_checkpoint.tar"
            elif isinstance(config.dataset, Accio) and self.checkpoint_file is Ellipsis:
                config.model_load_file = "weights/accio_checkpoint.tar"
            elif config.model_load_file is None:
                config.model_load_file = self.checkpoint_file

        print("Selected checkpoint", config.model_load_file)
        assert config.model_load_file is None or Path(config.model_load_file).is_file(), config.model_load_file
        return out

    def create_model_name(self, config, i):
        prefix = Path(self.file_name).stem
        try:
            return f"{prefix}_{i}_ep{config.dataset.episode_index_test + 1}"
        except TypeError:
            return f"{prefix}_{i}_idx{config.dataset.episode_index_test}"


class EvaluateBBTBFExperiment(EvalExperiment):

    def __init__(self):
        """
        Perform one evaluation step on all episodes of BBT and BF using pretrained weights.
        """
        super().__init__(num_runs=12)

    def modify_config(self, config: FaceGroupingConfig, i):
        if i < 6:
            config.dataset = BigBangTheory(episode_index_test=i)
        else:
            config.dataset = Buffy(episode_index_test=i - 6)

        return super().modify_config(config, i)


class EvaluateBBTExperiment(EvalExperiment):

    def __init__(self):
        """
        Perform one evaluation step on all episodes of BBT using pretrained weights.
        """
        super().__init__(num_runs=6)

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = BigBangTheory(episode_index_test=i)
        return super().modify_config(config, i)


class EvaluateBFExperiment(EvalExperiment):

    def __init__(self):
        """
        Perform one evaluation step on all episodes of BF using pretrained weights.
        """
        super().__init__(num_runs=6)

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = Buffy(episode_index_test=i)
        return super().modify_config(config, i)


class EvaluateAccioExperiment(EvalExperiment):
    """
    Perform one evaluation step on ACCIO using pretrained weights.
    """

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = Accio()
        config.model_params["sparse_adjacency"] = True
        return super().modify_config(config, i)


class MoreFeaturesBBTExperiment(EvalExperiment):

    def __init__(self):
        """
        Warning: Experimental.
        Allows to retrieve more features per track by splitting every x frames.
        This creates large graphs that may result in a memory error or very long run times.
        """
        super().__init__(header=("split_frames",), num_runs=6,)

    def modify_config(self, config: FaceGroupingConfig, i):
        from FGG.dataset.split_strategy import SplitEveryXFrames
        x = 10
        config.graph_builder_params["split_strategy"] = SplitEveryXFrames(x=x)
        config.pool_before_clustering = True
        config.dataset = BigBangTheory(episode_index_val=None, episode_index_train=None,
                                       episode_index_test=i)
        return (str(x), *super().modify_config(config, i))


class MoreFeaturesBFExperiment(EvalExperiment):

    def __init__(self):
        """
        Warning: Experimental.
        Allows to retrieve more features per track by splitting every x frames.
        This creates large graphs that may result in a memory error or very long run times.
        """
        super().__init__(header=("split_frames",), num_runs=6)

    def modify_config(self, config: FaceGroupingConfig, i):
        from FGG.dataset.split_strategy import SplitEveryXFrames
        x = 10
        config.graph_builder_params["split_strategy"] = SplitEveryXFrames(x=x)
        config.pool_before_clustering = True
        config.dataset = Buffy(episode_index_val=None, episode_index_train=None, episode_index_test=i)
        return (str(x), *super().modify_config(config, i))


if __name__ == '__main__':
    import datetime
    from FGG.config import FaceGroupingConfig
    from FGG.runner import Runner
    from FGG.persistence.run_configuration import enable_auto_run_save

    enable_auto_run_save()

    # --- Single dataset evaluation ---
    # Evaluate on all episodes of BBT:
    experiment_type = EvaluateBBTExperiment()
    # Evaluate on all episodes of BF:
    # experiment_type = EvaluateBFExperiment()
    # Evaluate on all episodes of BBT and BF:
    #experiment_type = EvaluateBBTBFExperiment()
    # Evaluate on ACCIO:
    # experiment_type = EvaluateAccioExperiment()

    # --------------Extract more features per track-------------------
    # If you want to extract feature maps with a higher number of features per track, run this:
    # It will extract a feature for every ten frames.
    # Warning: This is computationally more expensive than the default
    # experiment_type = MoreFeaturesBBTExperiment()
    # or
    # experiment_type = MoreFeaturesBFExperiment()
    # Accio is too large for our memory, but you can implement it in the same way if you want to try.

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
            experiment = Runner.from_config(config, load_from=experiment_type.load_from)
            wcp = experiment.test()
            print(f"Finished at {datetime.datetime.now()} at {wcp} wcp")
