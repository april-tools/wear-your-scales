import os
import typing as t
from zipfile import ZipFile

import numpy as np


def shuffle(x: np.ndarray, y: np.ndarray):
    """Shuffle the 0 index of x and y, jointly."""
    indexes = np.arange(x.shape[0])
    indexes = np.random.permutation(indexes)
    return x[indexes], y[indexes]


def unzip(filename: str, unzip_dir: str):
    """Unzip filename to unzip_dir with the same basename"""
    with ZipFile(filename, mode="r") as f:
        f.extractall(
            os.path.join(unzip_dir, os.path.basename(filename).replace(".zip", ""))
        )


def unzip_session(data_dir: str, session_id: int) -> str:
    """Return the path to the unzipped recording directory for session ID"""
    unzip_dir = os.path.join(data_dir, "unzip")
    recording_dir = os.path.join(unzip_dir, str(session_id))
    # unzip recording to recording folder not found.
    if not os.path.isdir(recording_dir):
        zip_filename = os.path.join(data_dir, f"{session_id}.zip")
        if not os.path.exists(zip_filename):
            raise FileNotFoundError(f"session {zip_filename} not found.")
        unzip(os.path.join(data_dir, f"{session_id}.zip"), unzip_dir=unzip_dir)
    return recording_dir


def get_channel_names(channel_data: t.Dict[str, np.ndarray]) -> t.List[str]:
    """Return printable channel names"""
    channel_names = []
    for channel in channel_data.keys():
        channel = channel.upper()
        if channel.startswith("HRV"):
            txt = channel.split("_")
            b = txt[1] if len(txt) == 2 else f"{txt[1]} {txt[2]}"
            channel_names.append(r"${a}_{b}$".format(a=txt[0], b=r"{" + b + r"}"))
        else:
            channel_names.append(rf"${channel}$")
    return channel_names


def normalize(
    x: np.ndarray, x_min: t.Union[float, np.ndarray], x_max: t.Union[float, np.ndarray]
):
    """Normalize x to [0, 1]"""
    return (x - x_min) / ((x_max - x_min) + 1e-6)


def standardize(
    x: np.ndarray, x_mean: t.Union[float, np.ndarray], x_std: t.Union[float, np.ndarray]
):
    return (x - x_mean) / x_std
