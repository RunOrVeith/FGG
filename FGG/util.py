import warnings
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from datetime import timedelta
from contextlib import contextmanager
import gzip

import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import torch




def load_typed_frame(file_path, column_to_output):
    frame = pd.read_csv(file_path)
    frame = frame.dropna()
    for col, func in column_to_output.items():
        frame[col] = frame[col].map(func)
    return frame


def parallel_for(iterable, func, mp=True, unpack=True):
    cores = cpu_count()
    Executor = ProcessPoolExecutor if mp else ThreadPoolExecutor
    with Executor(max_workers=cores) as executor:
        futures = [executor.submit(func, *stuff if unpack else stuff) for stuff in iterable]
        for future in as_completed(futures):
            yield future.result()


def from_mat_file(file_path):
    mats = sio.loadmat(str(file_path))
    keys = [key for key in mats.keys() if not key.startswith("_")]
    if len(keys) < 1:
        raise ValueError(f"Can not extract mat file, unknown main key: {keys}")

    def data_from_mat(mat):
        for row in mat.T:
            data_dict = {}
            if len(row.shape) == 1:
                row = row[:, None]
            for idx in mat.dtype.names or range(row.shape[-1]):
                print(row.shape, idx)
                values = row[idx]
                if isinstance(values, (list, tuple)):
                    values = np.concatenate(values).flatten().tolist()
                data_dict[idx] = values
            max_len = max(len(data) for data in data_dict.values())
            for name, value in data_dict.items():
                if len(value) == 1:
                    if isinstance(value[0], np.ndarray):
                        if len(value) == 1:
                            value = value[0].flatten().tolist()
                        else:
                            raise ValueError("Unknown column dimension.")
                    data_dict[name] = value * max_len
            frame = pd.DataFrame(data_dict)
            yield frame

    result = {}
    for key in keys:
        mat = mats[key]
        result.update({key: list(data_from_mat(mat))})
    return result


def timedelta_range(end, start=None, hz=1.):
    if start is None:
        start = timedelta()
    diff = timedelta(seconds=1 / hz)
    range_start = int(start / diff)
    range_end = int(end / diff)

    for i in range(range_start, range_end):
        yield timedelta(seconds=i * 1 / hz)


def save_np_as_gzip(array, output_file):
    with gzip.GzipFile(output_file, "w") as f:
        np.save(file=f, arr=array)


def load_np_from_gzip(input_file):
    with gzip.GzipFile(input_file, "r") as f:
        return np.load(f)


def load_from_hdf(hdf_file, dtype, *keys):
    # This should only be used for nested references,
    # if the array contains values directly, use f.get("name").value
    with h5py.File(hdf_file, 'r') as f:

        def recurse(ref):
            if isinstance(ref, (np.ndarray, h5py.Dataset)):
                for r in ref:
                    yield list(recurse(r))
            elif isinstance(ref, h5py.Reference):
                yield list(recurse(f[ref]))
            else:
                yield ref

        data = f
        for key in keys:
            data = data[key]

        for refs in data:
            datum = np.array(list(recurse(refs))).flatten()
            if dtype == str:
                yield np.array(["".join(chr(i) for i in datum)])[None, ...]
            else:
                yield datum.astype(dtype)[None, ...]


@contextmanager
def measure_time(title: str = None):
    if title is not None:
        print(f"Starting to {title}...")
    start_time = time.time()
    yield
    end_time = time.time() - start_time
    print(f"Took {end_time} seconds.")


def as_string(*args, delimiter=" "):
    return delimiter.join(map(str, args))


def insert_at(arr, output_size, indices):
    """
    Insert zeros at specific indices over whole dimensions, e.g. rows and/or columns and/or channels.
    You need to specify indices for each dimension, or leave a dimension untouched by specifying
    `...` for it. The following assertion should hold:

        `assert len(output_size) == len(indices) == len(arr.shape)`
    :param arr: The array to insert zeros into
    :param output_size: The size of the array after insertion is completed
    :param indices: The indices where zeros should be inserted, per dimension. For each dimension, you can
                    specify: - an int
                             - a tuple of ints
                             - a generator yielding ints (such as `range`)
                             - Ellipsis (=...)
    :return: An array of shape `output_size` with the content of arr and zeros inserted at the given indices.
    """
    assert len(arr.shape) == len(output_size) == len(indices)

    result = np.zeros(output_size)

    existing_indices = [np.setdiff1d(np.arange(axis_size), axis_indices, assume_unique=True)
                        for axis_size, axis_indices in zip(output_size, indices)]
    result[np.ix_(*existing_indices)] = arr
    return result


def combination_matrix(arr):
    idxs = np.arange(len(arr))
    mesh = np.stack(np.meshgrid(idxs, idxs))

    def np_combination_matrix():
        output = np.zeros((len(arr), len(arr), 2, *arr.shape[1:]), dtype=arr.dtype)
        num_dims = len(output.shape)
        idx = np.ix_(idxs, idxs)
        output[idx] = arr[mesh].transpose((2, 1, 0, *np.arange(3, num_dims)))
        return output

    def torch_combination_matrix():
        output_shape = (2, len(arr), len(arr), *arr.shape[1:])  # Note that this is different to numpy!
        return arr[mesh.flatten()].reshape(output_shape).permute(2, 1, 0, *range(3, len(output_shape)))

    if isinstance(arr, np.ndarray):
        return np_combination_matrix()
    elif isinstance(arr, torch.Tensor):
        return torch_combination_matrix()


def graph_empty(adjacency):
    if adjacency.shape == (0, 0):
        warnings.warn(message="Empty graph received", category=RuntimeWarning, stacklevel=2)
        return True
    return False