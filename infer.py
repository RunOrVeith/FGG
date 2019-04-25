from pathlib import Path

from FGG.dataset.input_data import Buffy, BigBangTheory, Accio
from FGG.config import FaceGroupingConfig
from evaluate import EvalExperiment


class InferenceExperiment(EvalExperiment):

    def __init__(self, checkpoint_file: str=..., num_runs=1):
        """
        Experiment type for running on datasets where the labels are not known.
        Runs evaluation only, no training.

        :param checkpoint_file: Checkpoint file to load weights from.
                                The default is to look for pretrained weights in this repo in the "weights" folder
                                depending on the selected dataset.
        :param num_runs: Number of runs. Useful for inferring multiple datasets at once.
        """
        super().__init__(checkpoint_file=checkpoint_file, header=None, num_runs=num_runs)

    def modify_config(self, config: FaceGroupingConfig, i):
        out = super().modify_config(config, i)
        # Disable trying to read labels from the feature matrices
        config.dataset.matfile_format["label_header"] = None
        return out

    def create_model_name(self, config, i):
        prefix = Path(config.model_load_file).parent.stem + "-weights"
        try:
            return f"{prefix}_{i}_ep{config.dataset.episode_index_test + 1}"
        except TypeError:
            return f"{prefix}_{i}_idx{config.dataset.episode_index_test}"


class InferBBT0101Experiment(InferenceExperiment):

    def __init__(self):
        """
        This is just an example class.
        It runs inference on BBT0101, but assumes we don't know the labels.
        Implement a class similar to this to run on new datasets.
        """
        self.dataset = BigBangTheory(episode_index_test=0)
        super().__init__(checkpoint_file=...)

    def modify_config(self, config: FaceGroupingConfig, i):
        config.dataset = self.dataset
        return super().modify_config(config, i)


if __name__ == '__main__':
    import datetime
    from FGG.runner import Runner
    #from FGG.persistance.run_configuration import enable_auto_run_save

    #enable_auto_run_save()
    experiment_type = InferBBT0101Experiment()
    meta_experiment = experiment_type.next_experiment()
    wcp = None
    while True:
        try:
            config = meta_experiment.send(wcp)
        except StopIteration:
            break
        else:
            assert isinstance(experiment_type, InferenceExperiment)
            print("Running inference, assuming we don't know any labels!")
            experiment = Runner.from_config(config, load_from="last")
            wcp = experiment.infer()
            print(f"Finished at {datetime.datetime.now()} at {wcp} wcp")
