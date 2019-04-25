from pathlib import Path
import os

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from bokeh.palettes import Viridis, Category10, Viridis256, Inferno256
import visdom
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import requests
import numpy as np
import pandas as pd

from FGG.persistance.statistics import Statistics
from FGG.metrics.base import to_dense_numpy


def project(features, out_dimension=2, which="PCA"):
    # We have a fixed seed because we want to see progress over time
    if which == "PCA":
        return PCA(n_components=out_dimension, random_state=0).fit_transform(features)
    elif which == "TSNE":
        return TSNE(n_components=out_dimension, random_state=0).fit_transform(features)
    else:
        raise ValueError(f"Unknown projection mechanism {which}")


def hex_to_rgb_colormap(color_map):
    return np.array([tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for color in color_map])


def get_color_map(num_targets, unique_targets, palette):
    if num_targets < len(palette):
        if isinstance(palette, dict):
            colors = hex_to_rgb_colormap(palette[max(3, num_targets)])[:num_targets]
        elif isinstance(palette, list):
            step = len(palette) // num_targets
            colors = hex_to_rgb_colormap(palette[::step][:num_targets])
        else:
            raise ValueError("Unknown palette format")
    else:
        colors = np.round(255 * unique_targets / num_targets).astype(np.uint8).repeat(3).reshape(-1, 3)
    return colors


def matplotlib_scatter(X, Y, name, folder, color_map, labels):
    base_path = Path(f"{os.environ['HOME']}/Pictures/{folder}/{name}")
    if not base_path.is_dir():
        numel = 0
        base_path.mkdir(parents=True)
    else:
        numel = len(list(base_path.iterdir()))

    n_found_labels = len(np.unique(Y))

    fig, ax = plt.subplots()
    ax.set_title(name)
    for group in np.unique(Y):
        idx = Y == group
        x, y = X[:, 0][idx], X[:, 1][idx]
        if len(x) == 0:
            continue
        ax.scatter(x=x, y=y, c=color_map[labels[group - 1]], label=labels[group - 1])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       fancybox=True, shadow=True, ncol=n_found_labels)
    fig.savefig(str(base_path / f"{numel}.png"), bbox_extra_artists=(legend,))
    plt.close(fig)


class VisdomViz(object):

    def __init__(self, statistics: Statistics, replay_log: str = None):
        self.statistics = statistics
        self.enabled = False
        try:
            self.vis = visdom.Visdom(log_to_filename=replay_log)
            self.enabled = self.vis.check_connection()
        except (requests.exceptions.ConnectionError, ConnectionRefusedError):
            self.enabled = False
        if not self.enabled:
            print("Visdom disabled! Start a visdom server to use it!")
        if replay_log is not None and Path(replay_log).is_file() and self.enabled:
            self.vis.replay_log(replay_log)
        self.register_updates()

    def view_projection(self, features, target, name="projection", prediction=None, labels=None,
                        assign_clusters=None, which="PCA"):

        if not self.enabled:
            return

        projection = project(features=features, out_dimension=2, which=which)

        t = to_dense_numpy(target)
        unique_targets, inverse = np.unique(t, return_inverse=True)
        targ = inverse + 1

        if len(labels) <= 11:
            palette = Viridis
        else:
            palette = Viridis256
        colors = get_color_map(num_targets=len(labels), unique_targets=unique_targets, palette=palette)

        if labels is not None:
            legend_items = np.array(labels)[unique_targets].tolist()
        else:
            legend_items = unique_targets.tolist()
        self.vis.scatter(X=projection, Y=targ, win=f"{name} {which} ground truth",
                         env=self.statistics.run_name,
                         opts=dict(title=f"{name} {which} ground truth",
                                   xlabel="x", ylabel="y", markercolor=colors, webgl=True,
                                   legend=legend_items))
        if projection.shape[-1] == 2:
            col = np.array(colors) / 255
            color_map = {label: col[np.newaxis, labels.index(label)] for label in labels}
            matplotlib_scatter(X=projection, Y=targ, name=f"ground truth {which}", folder=self.statistics.run_name,
                               color_map=color_map,
                               labels=legend_items)

        if prediction is not None:
            p = to_dense_numpy(prediction)
            unique_pred, inverse = np.unique(prediction, return_inverse=True)
            pred = inverse + 1
            num_pred = len(unique_pred)
            if len(labels) <= 11:
                palette = Viridis
            else:
                palette = Viridis256
            colors = get_color_map(num_targets=num_pred, unique_targets=unique_pred, palette=palette)
            if labels is not None and assign_clusters is not None:
                modes = assign_clusters(target=t, prediction=p, cluster_labels_only=True)
                legend_items = np.array(labels)[modes].tolist()
            else:
                legend_items = unique_pred.tolist()
            self.vis.scatter(X=projection, Y=pred, win=f"{name} {which} predicted",
                             env=self.statistics.run_name,
                             opts=dict(title=f"{name} {which} predicted",
                                       xlabel="x", ylabel="y", markercolor=colors,
                                       webgl=True, legend=legend_items))
            if projection.shape[-1] == 2:
                col = np.array(colors) / 255
                color_map = {label: col[np.newaxis, labels.index(label)] for label in labels}
                matplotlib_scatter(X=projection, Y=pred, name=f"predicted {which}", folder=self.statistics.run_name,
                                   color_map=color_map, labels=legend_items)

    def register_updates(self):
        self.statistics.log_scalar = self._register_on_update(self.statistics.log_scalar,
                                                              callback=self.on_scalar_update)
        self.statistics.log_matrix = self._register_on_update(self.statistics.log_matrix,
                                                              callback=self.on_matrix_update)

    def on_scalar_update(self, name, *args):
        if not self.enabled:
            return
        last_x = len(self.statistics["scalar"][name])
        if len(args) > 0:
            self.vis.line(Y=np.array(args), X=np.arange(last_x, last_x + len(args)),
                          win=name, env=self.statistics.run_name,
                          opts=dict(title=name, xlabel="Step"),
                          update="append")

    def on_matrix_update(self, name, *matrices):
        if not self.enabled:
            return
        for matrix in matrices:
            opts = dict(title=name)

            if isinstance(matrix, pd.DataFrame):
                columns = matrix.index.tolist()
                rows = matrix.index.tolist()
                opts["columnnames"] = columns
                opts["rownames"] = rows
                matrix = matrix.values

            self.vis.heatmap(X=matrix, win=name, env=self.statistics.run_name, opts=opts)

    @staticmethod
    def _register_on_update(func, callback):
        def _pass_arguments(name, *data, ):
            callback(name, *data)
            update_log = func(name, *data)
            return update_log

        return _pass_arguments


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    stats = Statistics("testrun")
    viz = VisdomViz(statistics=stats, replay_log="/tmp/testlog")

    stats.log_scalar("loss", *range(0, 10))
    names = ["a", "b", "c", "d", "e", "f", "g"]
    x = pd.DataFrame(np.random.randint(low=0, high=25, size=(7, 7)), index=names, columns=names)
    stats.log_matrix("confusion", x, x + 10, x + 20)
