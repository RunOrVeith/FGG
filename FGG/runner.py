import datetime
import sys
import warnings
from functools import partial, wraps
from pathlib import Path
from typing import Tuple

import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader

from FGG.model.util import no_grad, random_seed_consistent_rng, num_trainable_parameters, l2_norm, mean_pool_tracks
from FGG.model.clustering import agglomerative_clustering
from FGG.dataset.dataset import EpisodeGraphDataset
from FGG.persistance.statistics import Statistics
from FGG.persistance.model_storage import ModelStorage
from FGG.metrics.visdom_base import VisdomViz
from FGG.metrics.evaluation import ClusterMetricProvider
from FGG.dataset.graph_builder import GraphBuilder
from FGG.config import FaceGroupingConfig


class Runner(object):

    def __init__(self, loss_function, model, optimizer, metric_provider,
                 device, statistics, model_storage, train_loader, val_loader=None, test_loader=None,
                 train_epochs=30, store_features_to=None, pool_before_clustering=False,
                 vis=None, ):

        self.device = device
        self.loss_function = loss_function.to(device)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.metric_provider = metric_provider
        self.statistics = statistics
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_epochs = train_epochs
        self.vis = vis
        self.model_storage = model_storage
        self.store_features_to = store_features_to
        self.pool_before_clustering = pool_before_clustering

    @staticmethod
    def from_config(config, load_from=None) -> "Runner":
        assert load_from in ("best", "last", None) or isinstance(load_from, int)

        # Create the Loss
        loss = config.loss_type(**config.loss_params)

        # Create the Model
        model = config.model_type(**config.model_params)
        print("Num parameters:", num_trainable_parameters(model=model))

        optimizer = config.optimizer_type(model.parameters(), **config.optimizer_params)

        # General utilities
        random_seed_consistent_rng(seed=config.seed)
        device = torch.device(config.device)
        model_storage = ModelStorage(config.model_save_file, model=model, optimizer=optimizer,
                                     load_from_old_path=config.model_load_file, strict=config.strict)

        if load_from == "best":
            model, optimizer = model_storage.load_best(determine=np.argmax)
        elif load_from == "last":
            model, optimizer = model_storage.load_most_recent()
        elif isinstance(load_from, int):
            model, optimizer = model_storage.load_from_epoch(epoch=load_from)

        # Statistics tracking and visualization
        metric_provider = ClusterMetricProvider(labels=config.dataset.person_id_handler.main_characters,
                                                cluster_assignment_function=config.wcp_version)
        statistics = Statistics(run_name=config.model_name, load_from=None)
        vis = VisdomViz(statistics=statistics, replay_log=str(config.replay_log_file), )
        if config.statistics_file.is_file():
            # Loading now instead of at creation time allows the visdom plotting callbacks to be called
            statistics.load_from_file(file=config.statistics_file)

        # Enable saving without passing arguments of the current performance:
        def call_with_performance(func):

            @wraps(func)
            def save_model_with_performance():
                try:
                    performance_indicator = statistics["scalar"][config.performance_indicator][-1]
                except (IndexError, KeyError):
                    warnings.warn(f"Performance indicator {config.performance_indicator} not found.")
                    performance_indicator = None

                return func(performance_indicator=performance_indicator)

            return save_model_with_performance

        model_storage.save = call_with_performance(model_storage.save)
        statistics.to_hdf5 = partial(statistics.to_hdf5, file=config.statistics_file)

        # Data
        train_episodes, val_episodes, test_episodes = config.dataset.load_episodes()
        train_loader = DataLoader(EpisodeGraphDataset(episodes=train_episodes,
                                                      split_disconnected_components=config.split_disconnected_components,
                                                      graph_builder=GraphBuilder(**config.graph_builder_params,
                                                                                 rng=config.seed)),
                                  shuffle=False, batch_size=1, num_workers=1,
                                  collate_fn=lambda samples: samples[0]) if len(train_episodes) > 0 else None
        val_loader = DataLoader(EpisodeGraphDataset(episodes=val_episodes,
                                                    split_disconnected_components=config.split_disconnected_components,
                                                    graph_builder=GraphBuilder(**config.graph_builder_params,
                                                                               rng=config.seed)),
                                shuffle=False, batch_size=1, num_workers=1,
                                collate_fn=lambda samples: samples[0]) if len(val_episodes) > 0 else None
        test_loader = DataLoader(EpisodeGraphDataset(episodes=test_episodes,
                                                     split_disconnected_components=config.split_disconnected_components,
                                                     graph_builder=GraphBuilder(**config.graph_builder_params,
                                                                                rng=config.seed)),
                                 shuffle=False, batch_size=1, num_workers=1,
                                 collate_fn=lambda samples: samples[0]) if len(test_episodes) > 0 else None

        experiment = Runner(loss_function=loss, device=device,
                            model=model, optimizer=optimizer,
                            metric_provider=metric_provider,
                            statistics=statistics,
                            train_loader=train_loader, val_loader=val_loader,
                            test_loader=test_loader,
                            train_epochs=config.train_epochs,
                            model_storage=model_storage,
                            store_features_to=config.store_features_to,
                            pool_before_clustering=config.pool_before_clustering,
                            vis=vis)

        return experiment

    def train(self, test_untrained=False):
        performance = "unknown"
        if self.val_loader is not None and test_untrained:
            self._val_epoch(epoch=-1)
        for i in range(self.train_epochs):
            self.model_storage.next_epoch()
            self.model.train()
            self.train_epoch(epoch=i)
            # The arguments for this are set in from_config
            if self.val_loader is not None:
                performance = self._val_epoch(epoch=i)
            self.statistics.to_hdf5()
        performance = self.test() or performance
        self.model_storage.save()
        return performance

    @no_grad
    def infer(self):
        assert self.test_loader is not None
        return self.infer_epoch(self.test_loader)

    def test(self):
        if self.test_loader is not None:
            return self._test(data_loader=self.test_loader, epoch="final", mode="test")

    def _val_epoch(self, epoch: int):
        return self._test(data_loader=self.val_loader, epoch=epoch, mode="val")

    @no_grad
    def _test(self, data_loader, epoch, mode):
        self.model.eval()
        if len(data_loader) == 1:
            return self.test_epoch(data_loader=data_loader, epoch=epoch, mode=mode)
        return self.test_multiple_episodes(data_loader=data_loader, epoch=epoch, mode=mode)

    @staticmethod
    def predict(features, n_cluster: int):
        predicted_clusters = agglomerative_clustering(features=l2_norm(features), n_clusters=n_cluster,
                                                      connectivity=int(0.1 * len(features)))
        return predicted_clusters

    def train_epoch(self, epoch: int):

        for _, adjacency, features in tqdm.tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            # We do not need the person targets because we train strictly on the graph structure
            adjacency, features = adjacency.to(self.device), features.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model.forward(features=features, adjacency_matrix=adjacency)
            train_loss = self.loss_function(adjacency_matrix=adjacency, logits=logits)
            train_loss.backward()
            self.optimizer.step()
            self.statistics.log_scalar("train loss", train_loss.detach().item())

    def test_epoch(self, data_loader, epoch, mode):

        for tracks, adjacency, features in tqdm.tqdm(data_loader, desc=f"{mode} Epoch {epoch}"):
            # Don't be confused by the for loop, we're assuming you are testing a single graph only.
            adjacency, features = adjacency.to(self.device), features.to(self.device)
            all_targets = np.array([tracks.person_id_handler.main_characters.index(track.label) for track in tracks])
            logits = self.model.forward(features=features, adjacency_matrix=adjacency)
            val_loss = self.loss_function(adjacency_matrix=adjacency, logits=logits)

            embedding = logits.cpu().numpy()
            if self.pool_before_clustering:
                embedding, all_targets = mean_pool_tracks(adjacency[0].cpu().numpy(), embedding, all_targets)
            prediction = self.predict(features=embedding, n_cluster=tracks.person_id_handler.num_characters)

            tracks.set_prediction_output(feature_embedding=embedding,
                                         predicted_labels=self.metric_provider.cluster_assignment_function(
                                             target=all_targets,
                                             prediction=prediction))
            self.statistics.log_scalar(f"{mode} loss", val_loss.item())
            metrics = self.metric_provider(target=all_targets, prediction=prediction)
            self.statistics.log_metrics(name=mode, data=metrics)

            # self.vis.view_projection(features=embedding, target=all_targets,
            #                          name=f"{mode} projection", which="PCA",
            #                          labels=tracks.person_id_handler.main_characters,
            #                          prediction=prediction,
            #                          assign_clusters=self.metric_provider.cluster_assignment_function)

            if self.store_features_to is not None:
                tracks.output_features(self.store_features_to)
            return metrics["weighted clustering purity"]

    def test_multiple_episodes(self, data_loader, epoch, mode):
        total_targets = []
        total_embeddings = []
        total_loss = []
        n_cluster = None
        for tracks, adjacency, features in tqdm.tqdm(data_loader, desc=f"{mode} Epoch {epoch}"):
            adjacency, features = adjacency.to(self.device), features.to(self.device)
            all_targets = np.array([tracks.person_id_handler.main_characters.index(track.label) for track in tracks])
            logits = self.model.forward(features=features, adjacency_matrix=adjacency)
            val_loss = self.loss_function(adjacency_matrix=adjacency, logits=logits)

            embedding = logits.cpu().numpy()
            total_targets.append(all_targets)
            total_embeddings.append(embedding)
            total_loss.append(val_loss.item())
            if n_cluster is not None and n_cluster != tracks.person_id_handler.num_characters:
                raise NotImplementedError("Can not have mixed-series tests.")
            n_cluster = tracks.person_id_handler.num_characters

        total_embeddings = np.concatenate(total_embeddings)
        total_targets = np.concatenate(total_targets)

        prediction = self.predict(features=total_embeddings, n_cluster=n_cluster)

        self.statistics.log_scalar(f"{mode} loss", np.mean(total_loss))
        metrics = self.metric_provider(target=total_targets, prediction=prediction)
        self.statistics.log_metrics(name=mode, data=metrics)

        return metrics["weighted clustering purity"]

    def infer_epoch(self, data_loader):
        mode = "infer"
        for tracks, adjacency, features in tqdm.tqdm(data_loader, desc=f"{mode}"):
            # Don't be confused by the for loop, we're assuming you are testing a single graph only.
            adjacency, features = adjacency.to(self.device), features.to(self.device)
            logits = self.model.forward(features=features, adjacency_matrix=adjacency)
            embedding = logits.cpu().numpy()
            if self.pool_before_clustering:
                embedding, all_targets = mean_pool_tracks(adjacency[0].cpu().numpy(), embedding)
            prediction = self.predict(features=embedding, n_cluster=tracks.person_id_handler.num_characters)

            tracks.set_prediction_output(feature_embedding=embedding,
                                         predicted_labels=prediction)  # These are only the cluster assignments

            if self.store_features_to is not None:
                tracks.output_features(self.store_features_to)
        return "unknown"


class MetaExperiment(object):
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


class TestExperiment(MetaExperiment):

    def create_model_name(self, config, i):
        try:
            return f"{Path(self.file_name).stem}_{i}_ep{config.dataset.episode_index_test + 1}"
        except TypeError:
            return f"{Path(self.file_name).stem}_{i}_idx{config.dataset.episode_index_test}"


class InferenceExperiment(TestExperiment):

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        super().__init__()

    def modify_config(self, config: FaceGroupingConfig, i):
        out = super().modify_config(config, i)
        config.dataset.matfile_format["label_header"] = None  # Disables trying to read labels
        config.model_load_file = self.checkpoint_file
        return out

