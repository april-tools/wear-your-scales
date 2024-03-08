import typing as t

import h5py
import numpy as np


def append(ds, value):
    """Append value to a H5 dataset"""
    ds.resize((ds.shape[0] + value.shape[0]), axis=0)
    ds[-value.shape[0] :] = value


def write(filename, content: t.Dict[str, np.ndarray], overwrite: bool = False):
    """Write or append content to H5 file"""
    assert type(content) == dict
    with h5py.File(filename, mode="w" if overwrite else "a") as file:
        for k, v in content.items():
            if k in file:
                append(file[k], v)
            else:
                file.create_dataset(
                    k,
                    shape=v.shape,
                    dtype=v.dtype,
                    data=v,
                    chunks=True,
                    maxshape=(None,) + v.shape[1:],
                )


def get(filename, name: str):
    """Return the dataset with the given name"""
    with h5py.File(filename, mode="r") as file:
        if name not in file.keys():
            raise KeyError("{} cannot be found".format(name))
        return file[name][:]


def get_dataset_length(filename, name):
    with h5py.File(filename, mode="r") as file:
        dataset = file[name]
        length = dataset.len()
    return length


def contains(filename: str, name: str):
    """Return True if filename contains key name"""
    with h5py.File(filename, mode="r") as file:
        keys = list(file.keys())
    return name in keys
