from collections import defaultdict
import datetime
import pprint
from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd


class Statistics(object):
    _supported_types = ("scalar", "matrix")
    _version = "1.0.0"
    _supported_version = "1.0.0"

    def __init__(self, run_name: str, load_from: Path = None):
        self.run_name = run_name
        self.data = defaultdict(lambda: defaultdict(list))
        if load_from is not None and load_from.is_file():
            self.load_from_file(file=load_from)
        elif load_from is not None:
            print(f"Could not load {load_from}")
        self.dirty = False

    def log_scalar(self, name, *scalar):
        self.dirty = True
        self.data["scalar"][name].extend(scalar)

    def log_matrix(self, name, *matrix):
        self.dirty = True
        self.data["matrix"][name].extend(matrix)

    def log_metrics(self, name, data):
        for key, value in data.items():
            if isinstance(value, (pd.DataFrame, np.ndarray)):
                self.log_matrix(f"{name}-{key}", value)
            else:
                self.log_scalar(f"{name}-{key}", value)
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        result = {}
        for attribute, values in self.data["scalar"].items():
            result[attribute] = {"step": len(values), "last": values[-1], "min": min(values), "max": max(values)}
        for attribute, values in self.data["matrix"].items():
            result[attribute] = {"step": len(values)}
        return pprint.pformat(result)

    def load_from_file(self, file):
        print(f"Loading statistics from {str(file)}")
        with h5py.File(file, "r") as f:
            try:
                run_group = f[self.run_name]
            except KeyError:
                # This run is not present
                warnings.warn(f"Run '{self.run_name}' not found in {file}")
                return
            for log_type in self._supported_types:
                try:
                    log_type_group = run_group[log_type]
                except KeyError:
                    continue

                if log_type == "scalar":
                    for log_name, log_dataset in log_type_group.items():
                        self.log_scalar(log_name, *log_dataset.value.tolist())
                elif log_type == "matrix":
                    for log_name, log_group in log_type_group.items():
                        col_names = []
                        values = []
                        dtypes = []
                        for col_name, col_dataset in log_group.items():
                            values.append(col_dataset.value)
                            col_names.append(col_name)
                            dtypes.append(col_dataset.dtype)
                        if len(values) > 0:
                            all_matrices = np.array(values).transpose((1, 2, 0))
                            for matrix in all_matrices:
                                datum = pd.DataFrame(data=matrix, columns=col_names).set_index("index").infer_objects()
                                # TODO apparently the column order is sometimes messed up. Why?
                                datum = datum.reindex(datum.index.tolist(), axis=1)
                                self.log_matrix(log_name, datum)
        self.dirty = False

    def to_hdf5(self, file):

        if not self.dirty:
            return

        def _to_h5py_dtype(dtype=None):
            if dtype == np.object or dtype is None:
                return h5py.special_dtype(vlen=str)
            return dtype

        with h5py.File(file, "a") as f:
            run_group = f.require_group(name=self.run_name)
            run_group.attrs.create(name="version", data=self._version, dtype=_to_h5py_dtype())
            run_group.attrs.create(name="timestamp", data=datetime.datetime.now().timestamp())
            for log_type, logs in self.data.items():
                log_type_group = run_group.require_group(name=log_type)
                for log_name, log_data in logs.items():
                    try:
                        del log_type_group[log_name]
                    except KeyError:
                        pass
                    if log_type == "scalar":
                        log_type_group.create_dataset(name=log_name, data=np.array(log_data))
                    elif log_type == "matrix":
                        all_matrices = np.stack([matrix.reset_index(drop=False).values for matrix in log_data])
                        dummy_matrix = log_data[0].reset_index(drop=False)
                        matrix_group = log_type_group.create_group(name=log_name)
                        for i, column in enumerate(all_matrices.transpose((2, 0, 1))):
                            col_name = dummy_matrix.columns[i]
                            col_dtype = _to_h5py_dtype(dummy_matrix[col_name].dtype)
                            matrix_group.create_dataset(name=col_name, data=column.astype(col_dtype), )
        self.dirty = False


if __name__ == '__main__':
    stats = Statistics("testrun")
    stats.log_scalar("loss", *range(0, 10))
    names = ["a", "b", "c", "d", "e", "f", "g"]
    x = pd.DataFrame(np.random.randint(low=0, high=25, size=(7, 7)), index=names, columns=names)
    print(x.index)
    stats.log_matrix("confusion", x, x + 10, x + 20)
    stats.to_hdf5("/tmp/test.h5")
    print(stats.data)
    input("Press anything to reload from file")
    stats = Statistics("testrun", load_from="/tmp/test.h5")
    print(stats.data)
