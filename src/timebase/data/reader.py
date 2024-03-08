import os
import pickle
import typing as t
from copy import deepcopy
from math import ceil

import numpy as np
import pandas as pd
import sklearn.utils
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from timebase.data import resample
from timebase.data.static import *
from timebase.utils import h5, plots, tensorboard


def compute_statistics(
    args,
    data: t.Dict[str, t.Any],
    ds_name: str = "x_train",
    buffer: int = 100000,
):
    """
    Compute the min, max, mean and standard deviation of the training
    """
    cache = os.path.join(args.dataset, "stats.pkl")
    if args.reuse_stats and os.path.exists(cache):
        with open(cache, "rb") as file:
            stats = pickle.load(file)
    else:
        if args.verbose:
            print("Compute dataset statistics...")
        channels = list(args.input_shapes.keys())
        stats = {
            ds_name: {
                channel: {s: [] for s in ["min", "max", "mean", "second_moment"]}
                for channel in channels
            }
        }
        for tranche in np.split(
            data[ds_name],
            indices_or_sections=np.linspace(
                start=buffer,
                stop=len(data[ds_name]),
                num=len(data[ds_name]) // buffer if buffer < len(data[ds_name]) else 0,
                endpoint=False,
                dtype=int,
            ),
        ):
            tranche_collector = {
                k: np.empty(shape=[len(tranche), v[0]])
                for k, v in args.input_shapes.items()
            }
            for i, filename in enumerate(tranche):
                for channel in channels:
                    tranche_collector[channel][i] = h5.get(filename, name=channel)
            for channel, v in tranche_collector.items():
                stats[ds_name][channel]["min"].append(np.min(v))
                stats[ds_name][channel]["max"].append(np.max(v))
                stats[ds_name][channel]["mean"].append(np.mean(v))
                stats[ds_name][channel]["second_moment"].append(np.mean(np.power(v, 2)))
        for channel in channels:
            stats[ds_name][channel]["min"] = np.min(stats[ds_name][channel]["min"])
            stats[ds_name][channel]["max"] = np.max(stats[ds_name][channel]["max"])
            stats[ds_name][channel]["mean"] = np.mean(stats[ds_name][channel]["mean"])
            # std = var**(1/2) var = E[X**2] - (E[X])**2
            stats[ds_name][channel]["std"] = np.sqrt(
                np.mean(stats[ds_name][channel]["second_moment"])
                - np.power(stats[ds_name][channel]["mean"], 2)
            )
            del stats[ds_name][channel]["second_moment"]
        with open(cache, "wb") as file:
            pickle.dump(stats, file)
    return stats


def select_items(args):
    """select which item(s) should be set as target(s)"""
    assert (0 not in args.ymrs) or (
        0 not in args.hdrs
    ), "At least one item should be selected as target"
    if len(set(args.ymrs).difference(set(range(0, 12)))):
        raise Exception(f"{args.ymrs} not in [0,11]")
    if len(set(args.hdrs).difference(set(range(0, 18)))):
        raise Exception(f"{args.ymrs} not in [0,17]")
    if 0 in args.ymrs:
        ymrs_selected = []
    else:
        ymrs_selected = [f"YMRS{i}" for i in args.ymrs]
    if 0 in args.hdrs:
        hdrs_selected = []
    else:
        hdrs_selected = [f"HDRS{i}" for i in args.hdrs]
    args.selected_items = ymrs_selected + hdrs_selected


def within_session_split(args, y, session_ids):
    """split each session into 70:15:15 along the temporal dimension"""
    train_idx, val_idx, test_idx = (
        [],
        [],
        [],
    )
    excluded_sessions, session_last_timestamp = [], {}
    # if args.hours2keep > 0, only n=hours2keep hours of recorded selected,
    # if a session is shorter drop it
    time_cut_point = (60 * 60) * args.hours2keep
    segments_cut_point = round(time_cut_point / args.ds_info["segment_length"])
    for session_id in session_ids:
        indexes = np.where(y[:, LABEL_COLS.index("Session_Code")] == session_id)[0]
        no_segments = len(indexes)
        if no_segments < 4 or no_segments < segments_cut_point:
            excluded_sessions.append(int(session_id))
            if args.verbose > 2:
                print(
                    f"Session {int(session_id)} dropped since it has fewer "
                    f"than {np.maximum(4, segments_cut_point)} segments"
                )
        else:
            if segments_cut_point:
                indexes = indexes[:segments_cut_point]
                no_segments = len(indexes)
                session_last_timestamp[session_id] = np.max(
                    y[indexes, LABEL_COLS.index("unix_segment_t0")]
                )
            if args.split_mode == 2:
                indexes = sklearn.utils.shuffle(indexes, random_state=args.seed)
            session_idx_train, session_idx_val, session_idx_test = np.split(
                indexes,
                [int(no_segments * 0.70), int(no_segments * 0.85)],
            )
            train_idx.extend(list(np.random.permutation(session_idx_train)))
            val_idx.extend(list(session_idx_val))
            test_idx.extend(list(session_idx_test))
    args.excluded_sessions = excluded_sessions
    args.included_sessions = [
        int(i)
        for i in list(
            set(session_ids.astype(np.int32)).difference(set(excluded_sessions))
        )
    ]
    if segments_cut_point:
        args.session_last_timestamp = {
            int(k): np.uint32(v) for k, v in session_last_timestamp.items()
        }
    if args.verbose == 2:
        print(f"Sessions excluded from analysis as too short:" f" {excluded_sessions}")
        qc = []
        for session_id in args.included_sessions:
            dropped_upon_qc = args.ds_info["sessions_info"][str(session_id)][
                "filtered_out_percentage"
            ]
            qc.append(dropped_upon_qc)
        print(
            f"Percentage of recorded seconds removed upon quality control from "
            f"included sessions: mean={np.mean(qc)}, sd={np.std(qc)}"
        )
    return (train_idx, val_idx, test_idx)


def oos_split(args, y, t_sub_id_dict):
    splits = {s: [] for s in ["train", "val", "test"]}
    for k, v in t_sub_id_dict.items():
        t_mask = np.array(pd.Series(y[:, LABEL_COLS.index("NHC")]).isin(v))
        sub_id_t = pd.DataFrame(y[t_mask], columns=LABEL_COLS).drop_duplicates(
            subset="NHC"
        )
        sub_id_t.iloc[:, LABEL_COLS.index("age")] = pd.cut(
            sub_id_t.iloc[:, LABEL_COLS.index("age")],
            bins=[
                18,
                35,
                np.inf,
            ],
            labels=False,
        )
        try:
            train, test = train_test_split(
                sub_id_t,
                stratify=sub_id_t.iloc[
                    :, [LABEL_COLS.index("age"), LABEL_COLS.index("sex")]
                ],
                train_size=0.7,
                random_state=args.seed,
            )
        except:
            try:
                train, test = train_test_split(
                    sub_id_t,
                    stratify=sub_id_t.iloc[:, LABEL_COLS.index("age")],
                    train_size=0.7,
                    random_state=args.seed,
                )
            except:
                print(
                    f"at no. session(s) {k}: too few instaces per class under "
                    f"the variable(s) used to stratify, random assignment was "
                    f"used instead for train/holdout split"
                )
                sub_id_t = sub_id_t.sample(frac=1)
                cutpoint = ceil(len(sub_id_t) * 0.5)
                train, test = sub_id_t.iloc[:cutpoint, :], sub_id_t.iloc[cutpoint:, :]
        try:
            val, test = train_test_split(
                test,
                stratify=test.iloc[
                    :, [LABEL_COLS.index("age"), LABEL_COLS.index("sex")]
                ],
                train_size=0.5,
                random_state=args.seed,
            )
        except:
            try:
                val, test = train_test_split(
                    test,
                    stratify=test.iloc[:, LABEL_COLS.index("age")],
                    train_size=0.5,
                    random_state=args.seed,
                )
            except:
                print(
                    f"at no. session(s) {k}: too few instaces per class under "
                    f"the variable(s) used to stratify, random assignment was "
                    f"used instead for val/test split"
                )
                test = test.sample(frac=1)
                cutpoint = ceil(len(test) * 0.5)
                val, test = test.iloc[:cutpoint, :], test.iloc[cutpoint:, :]
        splits["train"].extend(list(train["NHC"]))
        splits["val"].extend(list(val["NHC"]))
        splits["test"].extend(list(test["NHC"]))

    time_cut_point = (60 * 60) * args.hours2keep
    segments_cut_point = round(time_cut_point / args.ds_info["segment_length"])
    splits_idx = {k: [] for k in splits.keys()}
    for k, v in splits.items():
        for sub_id in v:
            sub_id_mask = y[:, LABEL_COLS.index("NHC")] == sub_id
            for session_id in np.unique(
                y[sub_id_mask, LABEL_COLS.index("Session_Code")]
            ):
                indexes = np.where(
                    y[:, LABEL_COLS.index("Session_Code")] == session_id
                )[0]
                no_segments = len(indexes)
                if no_segments < 4 or no_segments < segments_cut_point:
                    pass
                else:
                    if segments_cut_point:
                        indexes = indexes[:segments_cut_point]
                    splits_idx[k].extend(indexes)

    assert (
        (
            not bool(
                set(np.unique(y[splits_idx["train"], LABEL_COLS.index("NHC")]))
                & set(np.unique(y[splits_idx["val"], LABEL_COLS.index("NHC")]))
            )
        )
        and (
            not bool(
                set(np.unique(y[splits_idx["train"], LABEL_COLS.index("NHC")]))
                & set(np.unique(y[splits_idx["test"], LABEL_COLS.index("NHC")]))
            )
        )
        and (
            not bool(
                set(np.unique(y[splits_idx["val"], LABEL_COLS.index("NHC")]))
                & set(np.unique(y[splits_idx["test"], LABEL_COLS.index("NHC")]))
            )
        )
    ), f"at least one subject_id (NHC) appears across different splits"

    return splits_idx["train"], splits_idx["val"], splits_idx["test"]


def split_into_sets(args, y: np.ndarray):
    if args.split_mode == 0:
        # random splits with no stratification
        idx = np.arange((len(y)))
        train_idx, test_idx = train_test_split(idx, train_size=0.7, random_state=123)
        val_idx, test_idx = train_test_split(test_idx, train_size=0.5, random_state=123)
    else:
        match args.status_selection:
            case "exclude_hc":
                mask = y[:, LABEL_COLS.index("status")] != DICT_STATE["HC"]
            case "mood_disorders":
                mask = np.array(
                    pd.Series(y[:, LABEL_COLS.index("status")]).isin(
                        [
                            v
                            for k, v in DICT_STATE.items()
                            if k
                            in [
                                "MDE_BD",
                                "MDE_MDD",
                                "ME",
                                "MX",
                                "Eu_BD",
                                "Eu_MDD",
                            ]
                        ]
                    )
                )
            case "ongoing_mood_disorders":
                mask = np.array(
                    pd.Series(y[:, LABEL_COLS.index("status")]).isin(
                        [
                            v
                            for k, v in DICT_STATE.items()
                            if k
                            in [
                                "MDE_BD",
                                "MDE_MDD",
                                "ME",
                                "MX",
                            ]
                        ]
                    )
                )
            case "unfiltered":
                mask = np.arange(len(y))
            case _:
                raise NotImplementedError(
                    f"status_selection {args.status_selection} not implemented."
                )

        match args.split_mode:
            case 1 | 2:
                # split each session into 70:15:15 along the temporal dimension

                # unique session_ids from individuals whose status passes the
                # args.status_selection filter
                session_ids = np.unique(y[mask, LABEL_COLS.index("Session_Code")])
                train_idx, val_idx, test_idx = within_session_split(
                    args, y=y, session_ids=session_ids
                )
            case 3:
                # stratify on demographics (i.e. age, sex) such that segments from
                # a given individual are assigned to one split only
                # i.e. individuals do not overlap across train/val/test but each
                # subject appears in one set only; this is used for OOS
                # generalisation estimation

                # unique subject_ids of individuals whose status passes the
                # args.status_selection filter
                sub_ids = np.unique(
                    y[
                        mask,
                        LABEL_COLS.index("NHC"),
                    ]
                )
                # t_sub_id_dict -> keys are 1, ..., 4; values are lists of
                # subject_ids that have a corresponding number (either 1, ..., or 4)
                # of recorded sessions available. Train:validation:test split is
                # performed independently in such groups of individuals to avoid
                # having, for example, one set containing individuals with only
                # one session available and one set containing only individuals
                # with four sessions available
                t_sub_id_dict = {
                    t: []
                    for t in np.unique(y[mask, LABEL_COLS.index("time")]).astype(int)
                    + 1
                }

                # if args.hours2keep > 0, sessions shorter than args.hours2keep are dropped
                time_cut_point = (60 * 60) * args.hours2keep
                segments_cut_point = round(
                    time_cut_point / args.ds_info["segment_length"]
                )
                discarded_sessions = []
                for sub_id in sub_ids:
                    sub_id_mask = y[:, LABEL_COLS.index("NHC")] == sub_id
                    i = 0
                    for session_id in np.unique(
                        y[sub_id_mask, LABEL_COLS.index("Session_Code")]
                    ):
                        indexes = np.where(
                            y[:, LABEL_COLS.index("Session_Code")] == session_id
                        )[0]
                        no_segments = len(indexes)
                        if no_segments < 4 or no_segments < segments_cut_point:
                            discarded_sessions.append(int(session_id))
                            if args.verbose > 2:
                                print(
                                    f"Session {int(session_id)} dropped since it "
                                    f"has fewer than {np.maximum(4, segments_cut_point)} segments"
                                )
                        else:
                            i += 1
                    if i:
                        t_sub_id_dict[i].append(sub_id)
                args.discarded_sessions = discarded_sessions
                # split such that a given subjection appears in one set only;
                # split is based on demographics and is done for subjects with 1
                # session, then for subjects with 2 sessions, etc...
                train_idx, val_idx, test_idx = oos_split(
                    args, y=y, t_sub_id_dict=t_sub_id_dict
                )
            case _:
                raise NotImplementedError(
                    f"split_mode {args.split_mode} not implemented."
                )

    return {
        "train": np.array(train_idx),
        "val": np.array(val_idx),
        "test": np.array(test_idx),
    }


def construct_dataset(args, data: t.Dict):
    """Construct feature-label pairs for the specified classification mode

    task mode:
        0 - cross-entropy loss
        1 - weighted (quadratic) kappa loss
        2 - ONTRAM

    Args:
        data: t.Dict[str, t.Any], with the follow key-value
            - data_paths: np.ndarray, paths to each segment
            - labels: np.ndarray, labels from clinical information
            - ds_info: dict, dataset information
            - clinical_info: pd.DataFrame, clinical information
            - invalid_sessions: t.List[int], invalid session IDs
    """
    assert args.task_mode in (0, 1, 2, 3, 4)
    idx = split_into_sets(args, y=data["labels"])
    # convert labels to dictionary
    labels = {k: data["labels"][:, i] for i, k in enumerate(LABEL_COLS)}

    data["x_train"] = data["data_paths"][idx["train"]]
    data["y_train"] = {k: v[idx["train"]] for k, v in labels.items()}

    data["x_val"] = data["data_paths"][idx["val"]]
    data["y_val"] = {k: v[idx["val"]] for k, v in labels.items()}

    data["x_test"] = data["data_paths"][idx["test"]]
    data["y_test"] = {k: v[idx["test"]] for k, v in labels.items()}
    del data["data_paths"], data["labels"]


def get_subject_ids(args, nhc: np.ndarray):
    unique_ids = np.unique(nhc)
    ids_renaming_dict = dict(zip(unique_ids, np.arange(len(unique_ids))))
    refactored_ids = np.array(pd.Series(nhc).replace(ids_renaming_dict), dtype=int)
    if not hasattr(args, "num_train_subjects"):
        args.num_train_subjects = len(unique_ids)
    return refactored_ids


def compute_rank_frequency(args, data: t.Dict[str, t.Any]):
    item_frequency = {}
    eps = float(np.finfo(np.float32).eps)
    for item in args.selected_items:
        if args.verbose > 2:
            train_rank = np.unique(data["y_train"][item])
            val_rank = np.unique(data["y_val"][item])
            test_rank = np.unique(data["y_test"][item])
            if set(train_rank) != set(val_rank) or set(train_rank) != set(test_rank):
                print(f"ranks are not the same across train/val/test for item {item}")
        frequency = {rank: eps for rank in range(ITEM_RANKS[item])}
        ranks, counts = np.unique(data["y_train"][item], return_counts=True)
        ranks, item_size = ranks.astype(int), np.sum(counts)
        for i, rank in enumerate(range(len(ranks))):
            frequency[rank] = float(counts[i] / item_size)
        item_frequency[item] = {
            k: v / sum(frequency.values()) for k, v in frequency.items()
        }
    return item_frequency


def resample_train(args, data: t.Dict[str, t.Any]):
    if args.imb_mode != 3:
        return None
    return resample.combine_ros_rus(args, data=data)


class SegmentDataset(Dataset):
    def __init__(
        self,
        args,
        filenames: np.ndarray,
        labels: t.Dict[str, np.ndarray],
        item_frequency: t.Dict[str, t.Dict[str, float]],
        stats: t.Dict[str, t.Dict[str, float]],
        segment_weights: np.array = None,
        subject_ids: np.array = None,
    ):
        self.filenames = filenames
        self.labels = labels
        self.item_frequency = item_frequency
        self.segment_weights = segment_weights
        self.subject_ids = subject_ids
        self.session_codes = labels["Session_Code"].astype(int)
        self.selected_items = args.selected_items
        self.stats = stats
        self.channels = list(args.input_shapes.keys())
        self.task_mode = args.task_mode
        assert args.scaling_mode in (0, 1, 2)
        self.scaling_mode = args.scaling_mode
        self.eps = np.finfo(np.float32).eps

    def __len__(self):
        return len(self.filenames)

    def process_features(self, data: t.Dict[str, np.ndarray]):
        """Process data based on args.scaling_mode
        0 - no scaling
        1 - normalize features by the overall min and max values from the
            training set
        2 - standardize features by the overall mean and standard deviation
            from the training set
        """
        match self.scaling_mode:
            case 0:
                pass
            case 1:
                for k, v in data.items():
                    data[k] = (v - self.stats[k]["min"]) / (
                        self.stats[k]["max"] - self.stats[k]["min"] + self.eps
                    )
            case 2:
                for k, v in data.items():
                    data[k] = (v - self.stats[k]["mean"]) / (
                        self.stats[k]["std"] + self.eps
                    )
            case _:
                raise NotImplementedError(
                    f"scaling_mode {self.scaling_mode} not implemented."
                )
        return data

    def process_label(self, label: t.Dict[str, np.ndarray]):
        selected = {}
        for item in self.selected_items:
            selected[item] = label[item]
            if self.task_mode in (0, 1, 2, 3):
                # one unit step between consecutive ranks across all items
                selected[item] = int(selected[item] / RANK_NORMALIZER[item])
        return selected

    @staticmethod
    def segment_id(filename: str):
        return int(os.path.basename(filename).replace(".h5", ""))

    @staticmethod
    def session_id(filename: str):
        return int(os.path.basename(os.path.dirname(filename)))

    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Dict[str, t.Any]:
        filename = self.filenames[idx]
        data = {c: h5.get(filename, name=c).astype(np.float32) for c in self.channels}
        data = self.process_features(data)
        label = self.process_label({k: v[idx] for k, v in self.labels.items()})
        metadata = {
            "session_id": self.session_id(filename),
            "segment_id": self.segment_id(filename),
            # len(unique(recording_id)) <= len(unique(session_id))
            "recording_id": self.session_codes[idx],
        }
        sample = {"data": data, "label": label, "metadata": metadata}
        if self.segment_weights is not None:
            sample["segment_weight"] = self.segment_weights[idx]
        if self.subject_ids is not None:
            sample["subject_id"] = self.subject_ids[idx]
        return sample


def drop_channel(args):
    channels = list(args.ds_info["channel_freq"].keys())
    for channel in channels:
        if args.channel2drop in channel:
            del args.ds_info["channel_freq"][channel]


def get_baseline_model_data(args, data: t.Dict):
    partitions = ["train", "val", "test"]
    features_container = dict(zip(partitions, [[], [], []]))
    for partition in partitions:
        for path in data[f"x_{partition}"]:
            features = np.expand_dims(h5.get(path, "FLIRT"), axis=0)
            features_container[partition].append(features)
    datasets = {}
    for partition in partitions:
        datasets[f"x_{partition}"] = pd.DataFrame(
            data=np.concatenate(features_container[partition], axis=0),
            columns=FLIRT_EDA + FLIRT_ACC + FLIRT_HRV,
        )
        datasets[f"y_{partition}"] = pd.DataFrame(
            data=np.concatenate(
                [
                    np.expand_dims(data[f"y_{partition}"][item], axis=1)
                    for item in args.selected_items
                ],
                axis=1,
            ),
            columns=args.selected_items,
        )
    return datasets


def get_datasets(args, summary: tensorboard.Summary = None):
    assert args.task_mode in (0, 1, 2, 3, 4)

    filename = os.path.join(args.dataset, "info.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find info.pkl in {args.dataset}.")
    with open(filename, "rb") as file:
        data = pickle.load(file)
    args.ds_info = data["ds_info"]
    if args.channel2drop is not None:
        drop_channel(args)
    args.input_shapes = {
        k: h5.get(data["data_paths"][0], k).shape
        for k in args.ds_info["channel_freq"].keys()
    }
    select_items(args)
    construct_dataset(args, data=data)
    if args.task_mode == 4:
        datasets = get_baseline_model_data(args, data=data)
        return datasets
    args.ds_info["stats"] = compute_statistics(args, data=data)
    item_frequency = compute_rank_frequency(args, data=data)
    segment_weights = resample_train(args, data=data)
    if (summary is not None) and (args.plot_mode in (1, 3)):
        plots.eda_plots(args, data=data, summary=summary)

    # settings for DataLoader
    dataloader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    if args.device.type in ("cuda", "mps"):
        gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)

    train_ds = DataLoader(
        SegmentDataset(
            args,
            filenames=data["x_train"],
            labels=data["y_train"],
            item_frequency=item_frequency,
            stats=args.ds_info["stats"]["x_train"],
            segment_weights=segment_weights,
            subject_ids=get_subject_ids(args, nhc=data["y_train"]["NHC"]),
        ),
        shuffle=True,
        **dataloader_kwargs,
    )
    val_ds = DataLoader(
        SegmentDataset(
            args,
            filenames=data["x_val"],
            labels=data["y_val"],
            item_frequency=item_frequency,
            stats=args.ds_info["stats"]["x_train"],
            subject_ids=get_subject_ids(args, nhc=data["y_val"]["NHC"]),
        ),
        **dataloader_kwargs,
    )
    test_ds = DataLoader(
        SegmentDataset(
            args,
            filenames=data["x_test"],
            labels=data["y_test"],
            item_frequency=item_frequency,
            stats=args.ds_info["stats"]["x_train"],
            subject_ids=get_subject_ids(args, nhc=data["y_test"]["NHC"]),
        ),
        **dataloader_kwargs,
    )

    if args.use_wandb:
        wandb.config.update({"segment_length": data["ds_info"]["segment_length"]})

    return train_ds, val_ds, test_ds
