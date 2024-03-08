import argparse
import itertools
import math
import pickle
import platform
import typing as t
from copy import deepcopy
from datetime import datetime, timedelta

import matplotlib
import matplotlib.offsetbox
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import pearsonr, ttest_ind
from sklearn.metrics import f1_score, mean_squared_error
from torch.utils.data import DataLoader
from torchmetrics import functional as F
from tqdm import tqdm

import train
from timebase import criterions, metrics
from timebase.data.reader import SegmentDataset, get_datasets
from timebase.data.static import *
from timebase.models.models import Classifier, get_models
from timebase.utils import tensorboard, utils, yaml
from timebase.utils.plots import compute_set_level_metrics
from timebase.utils.scheduler import Scheduler

########################### Plotting functions #################################
################################# START ########################################


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """size: length of bar in data units
    extent : height of bar ends in axes units"""

    def __init__(
        self,
        size=1,
        extent=0.03,
        label="",
        fontsize: int = 10,
        loc=2,
        ax=None,
        pad=0.4,
        borderpad=0.5,
        ppad=0,
        sep=2,
        prop=None,
        frameon: bool = True,
        linekw: dict = {},
        **kwargs,
    ):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, size], [0, 0], **linekw)
        vline1 = Line2D([0, 0], [-extent / 2.0, extent / 2.0], **linekw)
        vline2 = Line2D([size, size], [-extent / 2.0, extent / 2.0], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, textprops={"fontsize": fontsize})
        self.vpac = matplotlib.offsetbox.VPacker(
            children=[size_bar, txt], align="center", pad=ppad, sep=sep
        )
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=self.vpac,
            prop=prop,
            frameon=frameon,
            **kwargs,
        )


def plot_residuals(args, test: t.Dict[str, t.Any], filename: str = None):
    hdrs = [f"HDRS{i}" for i in range(1, 18)]
    ymrs = [f"YMRS{i}" for i in range(1, 12)]

    residuals, items = [], []
    for k, v in test["residuals"].items():
        items.extend([k] * len(v))
        residuals.extend(v)
    df = pd.DataFrame(data={"residuals": residuals, "items": items})
    df = df.astype(dtype={"residuals": "int32", "items": "object"})
    color_code = {"HDRS": "#0077B6", "YMRS": "#DC2F02"}
    tick_fontsize, label_fontsize, title_fontsize = 8, 10, 12

    figure = plt.figure(figsize=(8, 3.5), dpi=120)
    sub_figures = figure.subfigures(nrows=2, ncols=1, hspace=0)

    y_ticks = np.linspace(-4, 4, 5).astype(int)

    # plot HDRS
    hdrs_scale = df[df["items"].str.contains("HDRS")]
    axes = sub_figures[0].subplots(nrows=1, ncols=len(hdrs))
    for i, item in enumerate(hdrs):
        y, x = np.unique(
            hdrs_scale.loc[hdrs_scale["items"] == item, "residuals"], return_counts=True
        )
        axes[i].barh(
            y=y, width=x, color=color_code["HDRS"], edgecolor="black", linewidth=0.5
        )
        axes[i].set_ylim((-4.5, 4.5))
        axes[i].set_xlabel(item.replace("DRS", ""), loc="left", fontsize=label_fontsize)
        axes[i].set_xticks([])
        if i:
            axes[i].set_yticks([])
        else:
            tensorboard.set_yticks(
                axis=axes[i],
                ticks=y_ticks,
                ticks_loc=y_ticks,
                tick_fontsize=tick_fontsize,
            )
            tensorboard.set_ticks_params(axis=axes[i])
            axes[i].set_ylabel("Residual", labelpad=0.5, fontsize=label_fontsize)
        sns.despine(
            ax=axes[i],
            top=True,
            right=True,
            left=True if i else False,
            bottom=True,
            offset=3,
            trim=True,
        )

    # add scale bar
    ob = AnchoredHScaleBar(
        size=5000,
        label="5000 samples",
        fontsize=label_fontsize,
        loc="upper right",
        frameon=False,
        pad=0.8,
        sep=-14,
        linekw=dict(color="black", linewidth=0.8),
    )
    axes[-1].add_artist(ob)

    # plot YMRS
    ymrs_scale = df[df["items"].str.contains("YMRS")]
    axes = sub_figures[1].subplots(nrows=1, ncols=len(ymrs))
    for i, item in enumerate(ymrs):
        y, x = np.unique(
            ymrs_scale.loc[ymrs_scale["items"] == item, "residuals"], return_counts=True
        )
        axes[i].barh(
            y=y, width=x, color=color_code["YMRS"], edgecolor="black", linewidth=0.5
        )
        axes[i].set_ylim((-4.5, 4.5))
        axes[i].set_xlabel(item.replace("MRS", ""), loc="left", fontsize=label_fontsize)
        axes[i].set_xticks([])
        if i:
            axes[i].set_yticks([])
        else:
            tensorboard.set_yticks(
                axis=axes[i],
                ticks=y_ticks,
                ticks_loc=y_ticks,
                tick_fontsize=tick_fontsize,
            )
            tensorboard.set_ticks_params(axis=axes[i])
            axes[i].set_ylabel("Residual", labelpad=0.5, fontsize=label_fontsize)
        sns.despine(
            ax=axes[i],
            top=True,
            right=True,
            left=True if i else False,
            bottom=True,
            offset=3,
            trim=True,
        )

    if filename:
        print(f"saved plot to {filename}.")
        tensorboard.save_figure(figure, filename=filename, dpi=args.dpi)


def plot_demographics(args):
    filename = os.path.join(FILE_DIRECTORY, "TIMEBASE_database_reshaped.xlsx")
    data = pd.read_excel(filename)
    df = (
        data[data["Session_Code"].isin(args.included_sessions)]
        .groupby(["NHC", "time"])
        .agg({"status": "first"})
        .reset_index()
    )
    res = {
        t: {s: 0 for s in np.unique(df.status)}
        for t in range(1, len(np.unique(df["time"])) + 1, 1)
    }
    for nhc in np.unique(df["NHC"]):
        num_recs = len(df[df["NHC"] == nhc]["time"])
        status = list(df[df["NHC"] == nhc]["status"].values)[0]
        for r in range(1, num_recs + 1, 1):
            res[r][status] += 1

    counts = {k: list(v.values()) for k, v in res.items()}
    tick_fontsize, label_fontsize, title_fontsize = 8, 9, 10

    color_dict = {
        "Eu_BD": "#EF476F",
        "Eu_MDD": "#F78C6B",
        "MDE_BD": "#FFD166",
        "MDE_MDD": "#06D6A0",
        "ME": "#118AB2",
        "MX": "#073B4C",
    }
    palette = list(color_dict.values())
    figure, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(8, 2.5),
        gridspec_kw={"wspace": 0.15, "hspace": 0.2},
        dpi=120,
        facecolor="white",
    )
    x = list(range(6))
    for i, ax in enumerate(axes.flatten()):
        values = counts[i + 1]
        bars = ax.bar(
            x=x,
            height=values,
            color=palette,
            label=color_dict.keys(),
            edgecolor=None,
        )
        ax.bar_label(bars, fontsize=tick_fontsize, padding=1)
        ax.set_title(
            f"{i + 1} Rec (N = {sum(values)})",
            fontsize=title_fontsize,
        )
        ax.set_ylim(0, 30)
        tensorboard.set_yticks(
            axis=ax,
            ticks_loc=np.linspace(0, 30, 3),
            ticks=np.linspace(0, 30, 3, dtype=int) if i == 0 else [],
            label="Count" if i == 0 else "",
            tick_fontsize=tick_fontsize,
            label_fontsize=label_fontsize,
        )
        ax.set_xticks([])
        sns.despine(ax=ax)
    axes[0].legend(
        frameon=False,
        ncols=len(color_dict.keys()),
        handletextpad=0.35,
        handlelength=0.6,
        markerscale=1,
        columnspacing=0.6,
        loc=(0.8, -0.15),
    )
    plt.show()
    tensorboard.save_figure(
        figure, filename="plots/demographic.pdf", dpi=args.dpi, close=False
    )
    plt.close(figure)


def score_by_subject(args):
    filename = os.path.join(FILE_DIRECTORY, "TIMEBASE_database_reshaped.xlsx")
    data = pd.read_excel(filename)
    path2res = os.path.join(args.output_dir, "res.pkl")
    with open(path2res, "rb") as file:
        res = pickle.load(file)

    sessions = list(
        set(np.unique(data["Session_Code"])).intersection(
            set(np.unique(res["test"]["metadata"]["session_id"]))
        )
    )
    df = data[data["Session_Code"].isin(sessions)]

    sub_to_sessions = {
        s_id: list(data.loc[data["NHC"] == s_id, "Session_Code"])
        for s_id in np.unique(df["NHC"])
    }

    sub_res = {}
    for s_id, ses_ids in sub_to_sessions.items():
        # find the indexes of the session ids mapping to a given subject
        idx = np.where(np.isin(res["test"]["metadata"]["session_id"], ses_ids))[0]
        sub_average_f1_macro = np.mean(
            [
                f1_score(
                    y_true=res["test"]["labels"][item][idx],
                    y_pred=res["test"]["predictions"][item][idx],
                    average="macro",
                )
                for item in args.selected_items
            ]
        )
        sub_res[s_id] = sub_average_f1_macro

    points = list(sub_res.values())
    points.sort(reverse=True)

    figure, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=args.dpi)
    tick_fontsize, label_fontsize, title_fontsize = 9, 12, 15
    ax.scatter(np.arange(len(points)), points, color="red", marker="x", s=10)
    ax.set_title("Performance across subjects", size=title_fontsize)
    tensorboard.set_ticks_params(ax)
    yticks = np.arange(58, 63) * 0.01
    tensorboard.set_yticks(
        ax,
        ticks_loc=yticks,
        ticks=np.round(yticks, 2),
        label="Item Average $F1^M$",
        tick_fontsize=tick_fontsize,
        label_fontsize=label_fontsize,
    )
    xticks = np.linspace(0, 75, 6, dtype=np.int16)
    tensorboard.set_xticks(
        ax,
        ticks_loc=xticks,
        ticks=xticks,
        label="Subject ID",
        tick_fontsize=tick_fontsize,
        label_fontsize=label_fontsize,
    )
    ax.set_xlim(-1, 76)
    ax.set_ylim(0.58, 0.62)
    sns.despine(ax=ax)
    plt.show()
    tensorboard.save_figure(
        figure, filename="plots/subject_performance.pdf", dpi=args.dpi, close=False
    )
    plt.close(figure)


def plot_across_time_performance(args, summary: pd.DataFrame):
    # sns.set_theme(style="ticks")
    tensorboard.set_font()

    label_fontsize, title_fontsize, tick_fontsize = 10, 9, 8
    assert summary.shape[1] == 28
    # Ranks distribution over sessions
    nrows, ncols = 6, 5
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        gridspec_kw={
            "wspace": 0.1,
            "hspace": 0.45,
        },
        figsize=(8, 10),
        dpi=120,
        facecolor="white",
    )

    ymrs = [f"YMRS{i}" for i in range(1, 12)]
    hdrs = [f"HDRS{i}" for i in range(1, 18)]

    ymin, ymax = 0, 0.8
    y_ticks = np.linspace(ymin, ymax, 4)
    x_ticks = np.linspace(0, 50, 4)

    for idx, (item, ax) in enumerate(zip(ymrs + hdrs, axes.flatten())):
        ax.scatter(
            x=np.arange(len(summary.loc[:, item])),
            y=summary.loc[:, item],
            s=10,
            marker="o",
            color="#DC2F02" if item.startswith("YMRS") else "#0077B6",
            zorder=2,
        )
        ax.plot(
            np.arange(len(summary.loc[:, item])),
            summary.loc[:, item],
            color="#5A5A5A",
            linestyle="-",
            linewidth=1,
            zorder=1,
        )

        ax.set_ylim(ymin, ymax)
        left_column = idx % ncols == 0
        if left_column:
            tensorboard.set_yticks(
                axis=ax,
                ticks_loc=y_ticks,
                ticks=np.round(y_ticks, 1),
                label="",
                tick_fontsize=tick_fontsize,
                label_fontsize=label_fontsize,
            )
        else:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        tensorboard.set_xticks(
            axis=ax,
            ticks_loc=x_ticks,
            ticks=x_ticks.astype(int),
            label="time-slot" if idx > 32 else "",
            tick_fontsize=tick_fontsize,
            label_fontsize=label_fontsize,
        )

        ax.grid(axis="y")
        sns.despine(ax=ax, left=not left_column, trim=True)

        ax.set_title(
            item.replace("DRS", "")
            if item.startswith("HDRS")
            else item.replace("MRS", ""),
            fontsize=title_fontsize,
        )

    # delete unused axes
    figure.delaxes(axes[-1, -2])
    figure.delaxes(axes[-1, -1])

    # add x and y labels
    pos = axes[-1][0].get_position()
    figure.text(
        x=pos.x0 - 0.05,
        y=0.5,
        s="QCK",
        fontsize=label_fontsize,
        va="center",
        rotation="vertical",
    )
    figure.text(
        x=0.5,
        y=pos.y0 - 0.03,
        s="Future time points (30mins window)",
        fontsize=label_fontsize,
        ha="center",
    )

    tensorboard.save_figure(
        figure,
        filename=os.path.join("plots", "shifted_tests.pdf"),
        dpi=args.dpi,
        close=False,
    )
    plt.show()
    plt.close(figure)


def plot_performance_difference(args, loo_dict: t.Dict, filename: str = None):
    channels, items, performance_baseline, performance_loo, performance_delta = (
        [],
        [],
        [],
        [],
        [],
    )
    h_y_items = [i for i in args.selected_items if "HDRS" in i] + [
        i for i in args.selected_items if "YMRS" in i
    ]
    for item in h_y_items:
        qck_baseline = loo_dict["all"]["test"]["metrics"][item]["ck_quadratic"]
        performance_baseline.append(qck_baseline)
        for channel in CSV_CHANNELS[:-1]:
            qck_loo = loo_dict[channel]["test"]["metrics"][item]["ck_quadratic"]
            qck_delta = qck_baseline - qck_loo
            channels.append(channel)
            items.append(item)
            performance_loo.append(qck_loo)
            performance_delta.append(qck_delta)

    df = pd.DataFrame(
        {
            "Channels": channels,
            "Items": items,
            "Values": performance_loo,
            "Delta": performance_delta,
        }
    )
    df.to_csv(
        os.path.join(args.output_dir, "loo_performance_difference.csv"),
    )

    # divide DataFrame into 2
    items = df["Items"].unique()
    df_top = df.loc[df["Items"].isin(items[: len(items) // 2])]
    df_bottom = df.loc[df["Items"].isin(items[len(items) // 2 :])]
    size = len(items) // 2
    item_names = [name.replace("DRS", "").replace("MRS", "") for name in items]

    sns.set_theme(style="whitegrid")
    tensorboard.set_font()

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"hspace": 0.3},
        figsize=(8, 3),
        dpi=120,
    )

    tick_fontsize, label_fontsize = 8, 9
    with plt.style.context(["science"]):
        palette = sns.color_palette(n_colors=5)
    yticks = np.linspace(0, 0.8, 5)

    dot_pads = 0.318

    # plot first row
    sns.barplot(
        x="Items",
        y="Values",
        hue="Channels",
        data=df_top,
        order=df_top.Items.unique().tolist(),
        palette=palette,
        ax=axes[0],
    )
    plt.setp(axes[0].patches, linewidth=0)
    tensorboard.set_yticks(
        axis=axes[0],
        ticks=np.round(yticks, 1),
        ticks_loc=yticks,
        label="QCK",
        tick_fontsize=tick_fontsize,
        label_fontsize=label_fontsize,
    )
    tensorboard.set_xticks(
        axis=axes[0],
        ticks=item_names[:size],
        ticks_loc=np.arange(size),
        tick_fontsize=label_fontsize,
    )
    axes[0].set_xlabel(None)
    for i in range(len(items) // 2):
        axes[0].plot(
            [i - dot_pads, i + dot_pads],
            [performance_baseline[i], performance_baseline[i]],
            color="black",
            linestyle=":",
            linewidth=1.2,
        )
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.225),
        ncols=5,
        frameon=False,
        handletextpad=0.35,
        handlelength=0.6,
        markerscale=0.8,
        columnspacing=0.85,
        fontsize=label_fontsize,
    )
    # plot second row
    sns.barplot(
        x="Items",
        y="Values",
        hue="Channels",
        data=df_bottom,
        order=df_bottom.Items.unique().tolist(),
        palette=palette,
        ax=axes[1],
    )
    axes[1].legend_.remove()
    plt.setp(axes[1].patches, linewidth=0)
    tensorboard.set_yticks(
        axis=axes[1],
        ticks=np.round(yticks, 1),
        ticks_loc=yticks,
        label="QCK",
        tick_fontsize=tick_fontsize,
        label_fontsize=label_fontsize,
    )
    tensorboard.set_xticks(
        axis=axes[1],
        ticks=item_names[size:],
        ticks_loc=np.arange(size),
        tick_fontsize=label_fontsize,
    )
    axes[1].set_xlabel("")
    for i in range(size):
        axes[1].plot(
            [i - dot_pads, i + dot_pads],
            [performance_baseline[size + i], performance_baseline[size + i]],
            color="black",
            linestyle=":",
            linewidth=1.2,
        )

    for ax in axes:
        ax.set_xlim(left=-0.5, right=(len(items) / 2) - 0.5)
        ax.set_ylim(bottom=0, top=0.81)
        sns.despine(ax=ax, top=True, left=True, right=True)
        tensorboard.set_ticks_params(axis=ax, pad=2)

    if filename:
        tensorboard.save_figure(figure, filename=filename, dpi=args.dpi)
        print(f"saved figure to {filename}.")


########################### Plotting functions #################################
################################# END ########################################


def derive_metrics(args, test: t.Dict[str, t.Any]):
    test["metrics"] = {}
    for item in args.selected_items:
        metric_values, metric_names, _ = compute_set_level_metrics(
            y_true=test["labels"], y_pred=test["predictions"], item=item
        )
        test["metrics"][item] = dict(zip(metric_names, metric_values))


def compute_residuals(test: t.Dict[str, t.Any]):
    """
    RESIDUALS = difference between the estimated and the observed value of
    quantity of interest
    """
    test["residuals"], test["imb_rate"] = {}, {}
    for (item, y_true), y_pred in zip(
        test["labels"].items(), test["predictions"].values()
    ):
        # for easier comparison, ranks are rescaled such that all have a
        # step size of 1 between consecutive ranks
        test["residuals"][item] = (y_pred / RANK_NORMALIZER[item]) - (
            y_true / RANK_NORMALIZER[item]
        )
        _, counts = np.unique(y_true, return_counts=True)
        test["imb_rate"][item] = np.max(counts) / np.min(counts)


def save_residuals(args, test: t.Dict[str, t.Any]):
    ### SAVE DATA FOR GRAPHICAL MODEL(s) ###

    # Residuals, co-variates
    with open(os.path.join(args.dataset, "info.pkl"), "rb") as file:
        info = pickle.load(file)
    clinical_info = info["clinical_info"]
    t = type(clinical_info["Session_Code"][0])
    indexes = [
        clinical_info.index[clinical_info["Session_Code"] == session_code.astype(t)][0]
        for session_code in test["metadata"]["recording_id"]
    ]
    df4graph = test["residuals"].copy()
    df4graph["age"] = clinical_info.loc[indexes, "age"].to_numpy().astype(float)
    df4graph["sex"] = clinical_info.loc[indexes, "sex"].to_numpy().astype(int)
    filename = os.path.join(args.output_dir, "residuals.csv")
    pd.DataFrame(df4graph).to_csv(filename, index=False)

    # Predictions
    df4graph = pd.DataFrame(
        data=np.concatenate(
            [
                np.expand_dims(v / RANK_NORMALIZER[k], axis=1)
                for k, v in test["predictions"].items()
            ],
            axis=1,
        ).astype("int"),
        columns=test["predictions"].keys(),
    )
    filename = os.path.join(args.output_dir, "predictions.csv")
    pd.DataFrame(df4graph).to_csv(filename, index=False)

    # Labels
    df4graph = pd.DataFrame(
        data=np.concatenate(
            [
                np.expand_dims(v / RANK_NORMALIZER[k], axis=1)
                for k, v in test["labels"].items()
            ],
            axis=1,
        ).astype("int"),
        columns=test["labels"].keys(),
    )
    filename = os.path.join(args.output_dir, "labels.csv")
    pd.DataFrame(df4graph).to_csv(filename, index=False)


def get_data(args):
    filename = os.path.join(args.dataset, "info.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find info.pkl in {args.dataset}.")
    with open(filename, "rb") as file:
        data = pickle.load(file)
    if args.verbose == 2:
        print(f" Sessions removed during pre-process " f" {data['invalid_sessions']}")
    return data


def get_shifted_test_sets(args, data):
    if args.shift_mode == 0:
        # time-slots every 30 minutes
        timestamps_unix = data["labels"][:, LABEL_COLS.index("unix_segment_t0")]
        # 1 unix timestamp = 1 second
        slots = np.arange(0, 48) * 30 * 60
        shifted_test_idx, incomplete = {s: {} for s in slots}, {s: [] for s in slots}
        for session_id in args.included_sessions:
            session_in_sample_end = (
                args.session_last_timestamp[session_id] + args.ds_info["segment_length"]
            )
            for s in slots:
                shifted_test_idx[s][session_id] = []
                indexes = np.where(
                    (
                        data["labels"][:, LABEL_COLS.index("Session_Code")]
                        == float(session_id)
                    )
                    & (timestamps_unix >= session_in_sample_end + s)
                    & (timestamps_unix < session_in_sample_end + s + (30 * 60))
                )[0]
                if len(indexes):
                    shifted_test_idx[s][session_id].extend(indexes)
                else:
                    incomplete[s].append(session_id)
        yaml.save(
            filename=os.path.join(
                args.output_dir, f"mode{args.shift_mode}_shifted_tests_incomplete.yaml"
            ),
            data={int(k): v for k, v in incomplete.items()},
        )
        return shifted_test_idx

    if args.shift_mode == 1:
        # specific time-slots from the day after the recording starts
        slots = ["02:00:00", "10:00:00", "15:00:00", "19:00:00"]
        shifted_test_idx, incomplete = {s: {} for s in slots}, {s: [] for s in slots}
        recordings_start = []
        timestamps_unix = data["labels"][:, LABEL_COLS.index("unix_segment_t0")]
        timestamps_daytime = pd.Series(timestamps_unix).apply(
            lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
        )
        for session_id in args.included_sessions:
            t0 = data["labels"][
                data["labels"][:, LABEL_COLS.index("Session_Code")] == float(session_id)
            ][:, LABEL_COLS.index("unix_segment_t0")].min()
            recordings_start.append(datetime.utcfromtimestamp(t0).strftime("%H:%M:%S"))
            next_day = (datetime.utcfromtimestamp(t0) + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            for window_start in shifted_test_idx.keys():
                shifted_test_idx[window_start][session_id] = []
                window_end = (
                    datetime.strptime(window_start, "%H:%M:%S") + timedelta(hours=3)
                ).strftime("%H:%M:%S")
                start = next_day + " " + window_start
                end = next_day + " " + window_end
                indexes = np.where(
                    (
                        data["labels"][:, LABEL_COLS.index("Session_Code")]
                        == float(session_id)
                    )
                    & (timestamps_daytime > start)
                    & (timestamps_daytime < end)
                )[0]
                if len(indexes):
                    shifted_test_idx[window_start][session_id].extend(indexes)
                else:
                    incomplete[window_start].append(session_id)
        if np.sum([len(v) for k, v in incomplete.items()]):
            yaml.save(
                filename=os.path.join(
                    args.output_dir,
                    f"mode{args.shift_mode}_shifted_tests_incomplete.yaml",
                ),
                data=incomplete,
            )
        recordings_start = sorted(recordings_start)
        print(
            f"Earliest start time: {recordings_start[0]}\t Latest start time:"
            f" {recordings_start[-1]}"
        )

        figure, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(12, 6),
            gridspec_kw={"wspace": 0.01, "hspace": 0.01},
            dpi=args.dpi,
        )
        ax.plot_date(
            x=recordings_start, y=np.array([0] * len(recordings_start)), markersize=0.1
        )
        for idx, xtick in enumerate(ax.get_xticklabels()):
            if not idx % 30 == 0:
                xtick.set_visible(False)
            if idx == len(recordings_start) - 2:
                xtick.set_visible(True)
            xtick.set_rotation(45)
        ax.set_yticklabels([])
        ax.set_yticks([])
        filename = os.path.join(args.output_dir, "t0_distribution.svg")
        tensorboard.save_figure(figure, filename=filename, dpi=240)
        print(f"saved figure to {filename}.")

    else:
        raise NotImplementedError(f"shift_mode {args.shift_mode} not implemented.")

    return shifted_test_idx


@torch.no_grad()
def test_step(
    batch: t.Dict[str, t.Any],
    classifier: Classifier,
    criterion_classifier: criterions.ClassifierCriterion,
    device: torch.device,
):
    classifier.to(device)
    inputs = train.load(batch["data"], device=device)
    labels = train.load(batch["label"], device=device)
    classifier.train(False)

    outputs_classifier, _ = classifier(inputs)
    outputs, labels = metrics.postprocess4metrics(
        labels=labels,
        outputs=outputs_classifier,
        coral=classifier.item_predictor._get_name() == "CoralPredictor",
        item_frequency=criterion_classifier.item_frequency
        if criterion_classifier.outputs_thresholding
        else None,
    )
    return {"outputs": outputs, "targets": labels}


def inference_shifted_test_sets(args, data, shifted_test_idx):
    args.reuse_stats = True
    utils.get_device(args)
    train_ds, _, _ = get_datasets(args, summary=None)
    classifier, critic = get_models(args, summary=None)
    criterion_classifier, _, _ = criterions.get_criterion(
        args,
        output_shapes=classifier.output_shapes,
        item_frequency=train_ds.dataset.item_frequency,
    )
    scheduler_classifier = Scheduler(
        args,
        model=classifier,
        checkpoint_dir=os.path.join(args.output_dir, "ckpt_classifier"),
        mode="max",
        save_optimizer=False,
    )
    scheduler_classifier.restore(force=True)
    # settings for DataLoader
    dataloader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    if args.device.type in ("cuda", "mps"):
        gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)
    labels = {k: data["labels"][:, i] for i, k in enumerate(LABEL_COLS)}
    kappas = {i: [] for i in args.selected_items}
    for s, v in shifted_test_idx.items():
        indexes = list(itertools.chain(*[i for i in v.values()]))
        x_test = data["data_paths"][indexes]
        y_test = {k: v[indexes] for k, v in labels.items()}
        test_ds = DataLoader(
            SegmentDataset(
                args,
                filenames=x_test,
                labels=y_test,
                item_frequency=train_ds.dataset.item_frequency,
                stats=args.ds_info["stats"]["x_train"],
            ),
            shuffle=False,
            **dataloader_kwargs,
        )
        outputs = {"outputs": {}, "targets": {}}
        for batch in tqdm(test_ds, desc=f"Test (shift={s})", disable=args.verbose <= 1):
            output = test_step(
                batch,
                classifier=classifier,
                criterion_classifier=criterion_classifier,
                device=args.device,
            )
            utils.update_dict(target=outputs["outputs"], source=output["outputs"])
            utils.update_dict(target=outputs["targets"], source=output["targets"])
        for i in args.selected_items:
            kappa_item = F.cohen_kappa(
                preds=torch.concat(outputs["outputs"][i], dim=0),
                target=torch.concat(outputs["targets"][i]),
                task="multiclass",
                weights="quadratic",
                num_classes=ITEM_RANKS[i],
            )
            kappas[i].append(kappa_item.cpu().numpy().item())
    summary = pd.DataFrame(
        data=np.concatenate(
            [np.expand_dims(np.array(kappas[i]), axis=1) for i in args.selected_items],
            axis=1,
        ),
        columns=args.selected_items,
    )
    summary["average"] = summary.mean(axis=1)
    summary["YMRS_average"] = summary[
        [col for col in args.selected_items if col.startswith("YMRS")]
    ].mean(axis=1)
    summary["HDRS_average"] = summary[
        [col for col in args.selected_items if col.startswith("HDRS")]
    ].mean(axis=1)
    summary.to_csv(
        os.path.join(args.output_dir, f"mode{args.shift_mode}_shifted_tests_kappa.csv"),
        index=False,
    )
    return summary


def item_level_performance(args, test: t.Dict):
    qck_items = {k: v["ck_quadratic"] for k, v in test["metrics"].items()}
    imb_rate_items = {k: v for k, v in test["imb_rate"].items()}
    entropy_items = {
        k: stats.entropy(pk=np.unique(v, return_counts=True)[1] / len(v), base=math.e)
        for k, v in test["labels"].items()
    }
    res = stats.pearsonr(list(qck_items.values()), list(imb_rate_items.values()))
    print(
        f"Pearson R between \u03C1 and item QCK: "
        f"statistic {res[0]}, p-value {res[1]}\n"
    )
    res = stats.pearsonr(list(qck_items.values()), list(entropy_items.values()))
    print(
        f"Pearson R between \u03C1 and item H: "
        f"statistic {res[0]}, p-value {res[1]}\n"
    )
    sorted_dict = dict(sorted(qck_items.items(), reverse=True, key=lambda x: x[1]))
    yaml.save(
        filename=os.path.join(args.output_dir, "item_level_performance.yaml"),
        data=sorted_dict,
    )


def associations_with_demographics(args):
    filename = os.path.join(FILE_DIRECTORY, "TIMEBASE_database_reshaped.xlsx")
    data = pd.read_excel(filename)
    path2res = os.path.join(args.output_dir, "res.pkl")
    with open(path2res, "rb") as file:
        res = pickle.load(file)
    d = {
        ses_id: data[data["Session_Code"] == ses_id]["NHC"].values[0]
        for ses_id in np.unique(data["Session_Code"])
    }
    res["test"]["metadata"]["sub_id"] = np.array(
        [d[ses_id] for ses_id in res["test"]["metadata"]["session_id"]]
    )
    df = {"score": [], "age": [], "sex": [], "hdrs_sum": [], "ymrs_sum": []}
    for rec_id in np.unique(res["test"]["metadata"]["recording_id"]):
        idx = np.where(res["test"]["metadata"]["recording_id"] == rec_id)[0]
        avg = np.mean(
            [
                f1_score(
                    y_true=res["test"]["labels"][i][idx],
                    y_pred=res["test"]["predictions"][i][idx],
                    average="macro",
                )
                for i in res["test"]["labels"].keys()
            ]
        )
        df["score"].append(avg)
        df["age"].append(data[data["Session_Code"] == rec_id]["age"].values[0])
        df["sex"].append(data[data["Session_Code"] == rec_id]["sex"].values[0])
        df["hdrs_sum"].append(
            data[data["Session_Code"] == rec_id]["HDRS_SUM"].values[0]
        )
        df["ymrs_sum"].append(
            data[data["Session_Code"] == rec_id]["YMRS_SUM"].values[0]
        )
    df = pd.DataFrame(df, index=np.unique(res["test"]["metadata"]["recording_id"]))
    stat_val, p_val = pearsonr(x=df["score"], y=df["age"])
    print(f"age-scores association: p_val={stat_val}, pear_r={p_val}")
    stat_val, p_val = pearsonr(x=df["score"], y=df["hdrs_sum"])
    print(f"hdrs_tot-scores association: p_val={stat_val}, pear_r={p_val}")
    stat_val, p_val = pearsonr(x=df["score"], y=df["ymrs_sum"])
    print(f"ymrs_tot-scores association: p_val={stat_val}, pear_r={p_val}")
    stat_val, p_val = ttest_ind(
        a=df[df["sex"] == 0]["score"], b=df[df["sex"] == 1]["score"]
    )
    print(f"sex-scores association: p_val={stat_val}, t_stat={p_val}")

    hdrs_tot_score_true = np.concatenate(
        [
            np.expand_dims(v, axis=1)
            for k, v in res["test"]["labels"].items()
            if "HDRS" in k
        ],
        axis=1,
    )
    hdrs_tot_score_pred = np.concatenate(
        [
            np.expand_dims(v, axis=1)
            for k, v in res["test"]["predictions"].items()
            if "HDRS" in k
        ],
        axis=1,
    )
    mse_tot = mean_squared_error(
        y_true=np.sum(hdrs_tot_score_true, axis=1),
        y_pred=np.sum(hdrs_tot_score_pred, axis=1),
    )
    print(f"MSE on HDRS total score={mse_tot}")

    ymrs_tot_score_true = np.concatenate(
        [
            np.expand_dims(v, axis=1)
            for k, v in res["test"]["labels"].items()
            if "YMRS" in k
        ],
        axis=1,
    )
    ymrs_tot_score_pred = np.concatenate(
        [
            np.expand_dims(v, axis=1)
            for k, v in res["test"]["predictions"].items()
            if "YMRS" in k
        ],
        axis=1,
    )
    mse_tot = mean_squared_error(
        y_true=np.sum(ymrs_tot_score_true, axis=1),
        y_pred=np.sum(ymrs_tot_score_pred, axis=1),
    )
    print(f"MSE on YMRS total score={mse_tot}")


def main(args):
    if platform.system() == "Darwin" and args.verbose:
        matplotlib.use("TkAgg")
    args.plot_dir = os.path.join(args.output_dir, "plots")
    if not os.path.isdir(args.plot_dir):
        os.makedirs(args.plot_dir)
    tensorboard.set_font()

    # Load results from best model configuration
    path2res = os.path.join(args.output_dir, "res.pkl")
    assert os.path.exists(path2res)
    loo_dict = {"all": {}}
    with open(path2res, "rb") as file:
        loo_dict["all"] = pickle.load(file)
    utils.load_args(args)
    args.use_wandb = False
    args.clear_output_dir = False

    compute_residuals(test=loo_dict["all"]["test"])
    derive_metrics(args, test=loo_dict["all"]["test"])
    item_level_performance(args, test=loo_dict["all"]["test"])
    save_residuals(args, test=loo_dict["all"]["test"])
    plot_residuals(
        args,
        test=loo_dict["all"]["test"],
        filename=os.path.join(args.plot_dir, "residual.pdf"),
    )

    if args.split_mode == 2:
        data = get_data(args)
        shifted_test_idx = get_shifted_test_sets(args, data)
        summary = inference_shifted_test_sets(args, data, shifted_test_idx)
        plot_across_time_performance(args, summary=summary)
    del args.shift_mode

    for channel in tqdm(CSV_CHANNELS[:-1], desc="Leave-one-out channel"):
        c_args = deepcopy(args)
        c_args.verbose, c_args.plot_mode = 0, 0
        c_args.save_plots, c_args.save_predictions = False, True
        c_args.channel2drop = channel
        c_args.output_dir = os.path.join(args.output_dir, channel)
        if not os.path.exists(c_args.output_dir):
            args.device = "mps"
            train.main(c_args)
        res = os.path.join(c_args.output_dir, "res.pkl")
        with open(res, "rb") as file:
            loo_dict[channel] = pickle.load(file)
        compute_residuals(test=loo_dict[channel]["test"])
        derive_metrics(c_args, test=loo_dict[channel]["test"])
    plot_performance_difference(
        args,
        loo_dict=loo_dict,
        filename=os.path.join(args.plot_dir, "loo_performance_difference.pdf"),
    )
    plot_demographics(args)
    associations_with_demographics(args)
    score_by_subject(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--shift_mode",
        type=int,
        default=0,
        choices=[0, 1],
        help="task mode: "
        "0) time-slots every n hours"
        "1) specific time-slots from the day after the recording starts",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument("--verbose", default=0, choices=[0, 1], type=int)
    main(parser.parse_args())
