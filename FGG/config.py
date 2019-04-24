import os
from pathlib import Path
import json

import torch
import torch.nn.functional as F

from FGG.dataset.dataset import CompleteGraphEpisodeSampler
from FGG.dataset.input_data import BigBangTheory
from FGG.dataset.split_strategy import HeuristicSplit
from FGG.persistance.run_configuration import AutoConfig
from FGG.model.trainable import FGG
from FGG.model.loss import GraphContrastiveLoss
from FGG.model.clustering import cluster_mode_prediction


class FaceGroupingConfig(AutoConfig):

    def __init__(self):
        """
        This is the central configuration file for FGG.
        You can change most settings here without touching the code.
        All data that is stored in this class is automatically serialized so you know what you did.

        The parameters here can be set via command line flags that will be read automatically.
        The flags' names correspond to the attributes defined here.
        """
        super().__init__()

        with self.argument_group("model-load-store"):
            self.model_name = f"{self.timestamp}_{self.git_hexsha[:5]}"
            self.output_folder = None
            self.statistics_file = None
            self.replay_log_file = None
            self.run_info_file = None
            self.model_save_file = None
            self.model_load_file = None
            self.strict = True
            self.performance_indicator = "test-weighted clustering purity"

        with self.argument_group("task-data"):
            self.include_unknown = False
            self.dataset = BigBangTheory(episode_index_train=0, include_unknown=self.include_unknown)
            self.wcp_version = cluster_mode_prediction

        with self.argument_group("execution"):
            self.device = "cuda"
            self.seed = None
            self.store_base_path = Path(__file__).parent.parent / "runs"
            self.store_features_to = None

        with self.argument_group("model-parameters"):
            self.model_type = FGG
            self.model_params = dict(in_feature_size=2048,
                                     downsample_feature_size=1024,
                                     gc_feature_sizes=(512, 256, 128),
                                     use_residual=True, activation=F.elu,
                                     num_edge_types=2, sparse_adjacency=False)
            self.loss_type = GraphContrastiveLoss
            self.loss_params = dict(margin=1.)
            self.graph_builder_params = dict(pos_edge_dropout=None,
                                             neg_edge_dropout=None,
                                             split_strategy=HeuristicSplit(),
                                             pair_sample_fraction=1,
                                             edge_between_top_fraction=0.03,
                                             isolates_similarity_only=True,
                                             weighted_edges=True, )
            self.pool_before_clustering = False

        with self.argument_group("runtime-duration"):
            self.train_epochs = 30
            self.split_disconnected_components = False

        with self.argument_group("optimizer-parameters"):
            self.optimizer_type = torch.optim.Adam
            self.optimizer_params = dict(lr=1e-4, weight_decay=0)

    def finalize(self):
        self.output_folder = self.store_base_path / self.dataset.__class__.__name__ / self.model_name
        self.statistics_file = self.output_folder / "statistics.h5"
        self.replay_log_file = self.output_folder / "replay_log.json"
        self.run_info_file = self.output_folder / "run_info.json"
        self.model_save_file = self.output_folder / "checkpoint.tar"
        self.store_features_to = self.output_folder / "features.h5"
        self.output_folder.mkdir(exist_ok=True, parents=True)

        if self.seed is None:
            import numpy as np
            self.seed = np.random.randint(0, 99999999)
        print(f"Using seed {self.seed}")

    def serialize(self, data):
        with self.run_info_file.open("w") as f:
            json.dump(data, f)
