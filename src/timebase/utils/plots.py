import itertools
import pickle
import random
import typing as t

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from timebase.data.static import *
from timebase.utils import h5, tensorboard, yaml
from timebase.utils.utils import create_young_hamilton_labels

PARAMS_PAD = 1
PARAMS_LENGTH = 2

plt.rcParams.update(
    {
        "mathtext.default": "regular",
        "xtick.major.pad": PARAMS_PAD,
        "ytick.major.pad": PARAMS_PAD,
        "xtick.major.size": PARAMS_LENGTH,
        "ytick.major.size": PARAMS_LENGTH,
        "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
        "axes.facecolor": (0.0, 0.0, 0.0, 0.0),
        "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
    }
)

TICKER_FORMAT = matplotlib.ticker.FormatStrFormatter("%.2f")

JET = cm.get_cmap("jet")
GRAY = cm.get_cmap("gray")
TURBO = cm.get_cmap("turbo")
COLORMAP = TURBO
GRAY2RGB = COLORMAP(np.arange(256))[:, :3]
tick_fontsize, label_fontsize, title_fontsize = 9, 12, 15


def set_ticks_params(
    axis: matplotlib.axes.Axes, length: int = PARAMS_LENGTH, pad: int = PARAMS_PAD
):
    axis.tick_params(axis="both", which="both", length=length, pad=pad, colors="black")


def compute_set_level_metrics(y_true: t.Dict, y_pred: t.Dict, item: str):
    item_k = [
        cohen_kappa_score(
            y1=y_true[item] / RANK_NORMALIZER[item],
            y2=y_pred[item] / RANK_NORMALIZER[item],
            labels=np.arange(ITEM_RANKS[item]),
            weights=weightage,
        )
        for weightage in [None, "linear", "quadratic"]
    ]
    f1_macro = f1_score(
        y_true=y_true[item] / RANK_NORMALIZER[item],
        y_pred=y_pred[item] / RANK_NORMALIZER[item],
        average="macro",
    )
    precision_macro = precision_score(
        y_true=y_true[item] / RANK_NORMALIZER[item],
        y_pred=y_pred[item] / RANK_NORMALIZER[item],
        average="macro",
        zero_division=0,
    )
    recall_macro = recall_score(
        y_true=y_true[item] / RANK_NORMALIZER[item],
        y_pred=y_pred[item] / RANK_NORMALIZER[item],
        average="macro",
        zero_division=0,
    )
    ranks, counts = np.unique(y_true[item], return_counts=True)
    mse_micro = mean_squared_error(y_true[item], y_pred[item], squared=True)
    mse_rank = [
        mean_squared_error(
            y_true[item][np.where(y_true[item] == r)[0]],
            y_pred[item][np.where(y_true[item] == r)[0]],
            squared=True,
        )
        for r in ranks
    ]
    mse_macro = np.mean(mse_rank)
    mse_weighted = np.dot(mse_rank, counts / np.sum(counts))
    rmse_micro = mean_squared_error(y_true[item], y_pred[item], squared=False)
    rmse_rank = [
        mean_squared_error(
            y_true[item][np.where(y_true[item] == r)[0]],
            y_pred[item][np.where(y_true[item] == r)[0]],
            squared=False,
        )
        for r in ranks
    ]
    rmse_macro = np.mean(rmse_rank)
    rmse_weighted = np.dot(rmse_rank, counts / np.sum(counts))
    mae_micro = mean_absolute_error(y_true[item], y_pred[item])
    mae_rank = [
        mean_absolute_error(
            y_true[item][np.where(y_true[item] == r)[0]],
            y_pred[item][np.where(y_true[item] == r)[0]],
        )
        for r in ranks
    ]
    mae_macro = np.mean(mae_rank)
    mae_weighted = np.dot(mae_rank, counts / np.sum(counts))
    acc = accuracy_score(y_true[item], y_pred[item])
    mcc = matthews_corrcoef(y_true[item], y_pred[item])

    return (
        item_k
        + [
            f1_macro,
            precision_macro,
            recall_macro,
            mse_micro,
            mse_macro,
            mse_weighted,
            mae_micro,
            mae_macro,
            mae_weighted,
            rmse_micro,
            rmse_macro,
            rmse_weighted,
            acc,
            mcc,
        ],
        [
            "ck_unweighted",
            "ck_linear",
            "ck_quadratic",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "mse_micro",
            "mse_macro",
            "mse_weighted",
            "mae_micro",
            "mae_macro",
            "mae_weighted",
            "rmse_micro",
            "rmse_macro",
            "rmse_weighted",
            "acc",
            "mcc",
        ],
        {
            "ranks": ranks,
            "mse_rank": mse_rank,
            "mae_rank": mae_rank,
            "rmse_rank": rmse_rank,
        },
    )


def pass_preds_through_gmm(
    args,
    summary: tensorboard.Summary,
    y_true: t.Dict[str, np.ndarray],
    y_pred: t.Dict[str, np.ndarray],
    step: int = 0,
    mode: int = 0,
):
    if len(args.selected_items) == 28:
        with open(os.path.join(args.output_dir, "sklearn_utils.pkl"), "rb") as file:
            sklearn_models = pickle.load(file)
        true = np.concatenate(
            [np.expand_dims(v, axis=1) for k, v in y_true.items()], axis=1
        )
        pred = np.concatenate(
            [np.expand_dims(v, axis=1) for k, v in y_pred.items()], axis=1
        )
        scores = sklearn_models["scaler"].transform(true)
        scores_pred = sklearn_models["scaler"].transform(pred)

        membership = sklearn_models["gmm"].predict(scores)
        membership_pred = sklearn_models["gmm"].predict(scores_pred)

        log_likelihood = sklearn_models["gmm"].score_samples(scores)
        log_likelihood_pred = sklearn_models["gmm"].score_samples(scores_pred)

        pc1_coord = sklearn_models["pca"].transform(scores_pred)[:, 0]
        pc2_coord = sklearn_models["pca"].transform(scores_pred)[:, 1]

        figure, axs = plt.subplots(1, 3, figsize=(18, 7), dpi=args.dpi)

        ### ax0

        axs0 = axs[0].scatter(
            x=pc1_coord,
            y=pc2_coord,
            marker="x",
            c=log_likelihood_pred,
            s=8,
            cmap="viridis",
            alpha=0.6,
        )
        outliers_idx = np.nonzero(log_likelihood_pred < np.min(log_likelihood))[0]
        if len(outliers_idx):
            axs[0].scatter(
                x=pc1_coord[outliers_idx],
                y=pc2_coord[outliers_idx],
                marker="o",
                facecolor="none",
                edgecolor="r",
                s=8,
            )
        norm = plt.Normalize(
            np.minimum(log_likelihood.min(), log_likelihood_pred.min()),
            np.maximum(log_likelihood.max(), log_likelihood_pred.max()),
        )
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        figure.colorbar(axs0, ax=axs[0])
        axs[0].set_xlabel("PC1", fontsize=label_fontsize)
        axs[0].set_ylabel("PC2", fontsize=label_fontsize)
        axs[0].set_title(
            f"PC projection - log_likelihood (outliers={(len(outliers_idx)/len(log_likelihood_pred)) * 100:.02f}%)",
            fontsize=title_fontsize,
        )

        ### axs1

        labels = list(np.unique(membership))
        cm = confusion_matrix(membership, membership_pred, labels=labels)
        cm_plot = sns.heatmap(
            cm / cm.sum(),
            vmin=0,
            vmax=1,
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            annot=cm.astype(str),
            fmt="",
            linewidths=0.01,
            cbar=False,
            ax=axs[1],
        )
        cm_plot.set_xlabel(f"Predictions", fontsize=label_fontsize)
        cm_plot.set_ylabel("Targets", fontsize=label_fontsize)
        acc_micro = accuracy_score(membership, membership_pred)
        acc_macro = balanced_accuracy_score(membership, membership_pred)
        mcc = matthews_corrcoef(membership, membership_pred)
        cm_plot.set_title(
            f"ACC_m={acc_micro:.02f}%, ACC_M={acc_macro:.02f}%, MCC={mcc:.02f}%",
            fontsize=title_fontsize,
        )
        ticklabels = [int(i) for i in labels]
        cm_plot.set_xticklabels(
            ticklabels, va="top", ha="center", rotation=90, fontsize=tick_fontsize
        )
        cm_plot.set_yticklabels(
            ticklabels, va="center", ha="right", rotation=0, fontsize=tick_fontsize
        )
        set_ticks_params(axis=cm_plot, length=0, pad=PARAMS_PAD + 2)

        ### axs2

        log_likelihood_distr = pd.DataFrame(
            {
                "log_likelihood": np.squeeze(
                    np.concatenate(
                        (
                            np.expand_dims(log_likelihood, axis=1),
                            np.expand_dims(log_likelihood_pred, axis=1),
                        ),
                        axis=0,
                    )
                ),
                "label": ["true"] * len(log_likelihood)
                + ["pred"] * len(log_likelihood_pred),
            }
        )
        axs2 = sns.histplot(
            data=log_likelihood_distr,
            x="log_likelihood",
            hue="label",
            multiple="dodge",
            stat="count",
            shrink=0.8,
            binwidth=5,
            ax=axs[2],
        )
        axs2.set_xlabel("log_likelihood", fontsize=label_fontsize)
        axs2.set_ylabel("count", fontsize=label_fontsize)

        summary.figure(tag="GMM - Predictions", figure=figure, step=step, mode=mode)
    else:
        pass


def items_cm(
    args,
    y_true: t.Dict[str, np.ndarray],
    y_pred: t.Dict[str, np.ndarray],
    metadata: t.Dict,
    representations: np.ndarray,
    summary: tensorboard.Summary,
    clinical: t.Dict[str, np.ndarray],
    step: int = 0,
    mode: int = 0,
):
    """
    confusion matrices by item
    """
    figure, axs = plt.subplots(
        nrows=1,
        ncols=len(args.selected_items),
        figsize=(6.5 * len(args.selected_items), 6),
        gridspec_kw={"wspace": 0.2, "hspace": 0.3},
        dpi=args.dpi,
    )
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    for idx, item in enumerate(y_true.keys()):
        labels = list(
            np.arange(ITEM_RANKS[item]) * RANK_NORMALIZER[item]
        )  # items 5,6,8,9 of YMRS are scored [0, 2, 4, 6, 8]
        cm = confusion_matrix(y_true[item], y_pred[item], labels=labels)
        mcc = matthews_corrcoef(y_true[item], y_pred[item])
        cm_plot = sns.heatmap(
            cm / cm.sum(),
            vmin=0,
            vmax=1,
            cmap=cmap,
            annot=cm.astype(str),
            fmt="",
            linewidths=0.01,
            cbar=False,
            ax=axs[idx] if len(y_true.keys()) > 1 else axs,
        )
        cm_plot.set_xlabel(f"Predictions", fontsize=label_fontsize)
        cm_plot.set_ylabel("Targets", fontsize=label_fontsize)
        cm_plot.set_title(
            f"{item} - ACC={100 * (np.trace(cm) / np.sum(cm)):.02f}%, MCC={mcc:.02f}",
            fontsize=title_fontsize,
        )

        ticklabels = [int(i) for i in labels]
        cm_plot.set_xticklabels(
            ticklabels, va="top", ha="center", rotation=90, fontsize=tick_fontsize
        )
        cm_plot.set_yticklabels(
            ticklabels, va="center", ha="right", rotation=0, fontsize=tick_fontsize
        )

        set_ticks_params(axis=cm_plot, length=0, pad=PARAMS_PAD + 2)

    summary.figure(tag="items confusion matrix", figure=figure, step=step, mode=mode)


def item_and_rank_performance(
    args,
    y_true: t.Dict[str, np.ndarray],
    y_pred: t.Dict[str, np.ndarray],
    metadata: t.Dict,
    representations: np.ndarray,
    summary: tensorboard.Summary,
    clinical: t.Dict[str, np.ndarray],
    step: int = 0,
    mode: int = 0,
):
    """
    for each item show, ranks distribution, predictions distribution over ranks,
    residual distribution, performance aggregated and by rank
    """
    figure, axs = plt.subplots(
        nrows=len(args.selected_items),
        ncols=5,
        figsize=(5 * 7, len(args.selected_items) * 6),
        gridspec_kw={"wspace": 0.5, "hspace": 0.5},
        dpi=args.dpi,
    )

    for idx, item in enumerate(y_true.keys()):
        # axs[item, 0]
        true_ranks_hist = sns.histplot(
            x=y_true[item],
            stat="count",
            binwidth=1,
            discrete=True,
            ax=axs[idx, 0] if len(args.selected_items) > 1 else axs[0],
        )
        true_ranks_hist.set_xlabel("true", fontsize=label_fontsize)
        true_ranks_hist.set_xticks(
            list(np.arange(y_true[item].min(), y_true[item].max() + 1))
        )
        true_ranks_hist.set_xticklabels(
            list(np.arange(y_true[item].min(), y_true[item].max() + 1, dtype=int)),
            rotation=90,
            fontsize=tick_fontsize,
        )
        true_ranks_hist.set_ylabel("count", fontsize=label_fontsize)
        ranks, counts = np.unique(y_true[item].astype(int), return_counts=True)
        true_ranks_hist.set_title(
            f"{item} -  \u03C1={np.max(counts) / np.min(counts):.02f}",
            fontsize=title_fontsize,
        )
        # axs[item, 1]
        pred_ranks_hist = sns.histplot(
            x=y_pred[item],
            stat="count",
            binwidth=1,
            discrete=True,
            ax=axs[idx, 1] if len(args.selected_items) > 1 else axs[1],
        )
        pred_ranks_hist.set_xlabel("pred", fontsize=label_fontsize)
        pred_ranks_hist.set_xticks(
            list(
                np.arange(
                    y_pred[item].min(),
                    y_pred[item].max() + 1,
                )
            )
        )
        pred_ranks_hist.set_xticklabels(
            list(
                np.arange(
                    y_pred[item].min(),
                    y_pred[item].max() + 1,
                    dtype=int,
                )
            ),
            rotation=90,
            fontsize=tick_fontsize,
        )
        pred_ranks_hist.set_ylabel("count", fontsize=label_fontsize)
        pred_ranks_hist.set_title(
            item,
            fontsize=title_fontsize,
        )

        # axs[item, 2]
        residuals = y_pred[item] - y_true[item]
        residua_hist = sns.histplot(
            x=residuals,
            stat="count",
            binwidth=1,
            discrete=True,
            ax=axs[idx, 2] if len(args.selected_items) > 1 else axs[2],
        )
        residua_hist.set_xlabel("pred - true", fontsize=label_fontsize)
        residua_hist.set_xticks(list(np.arange(residuals.min(), residuals.max() + 1)))
        residua_hist.set_xticklabels(
            list(np.arange(residuals.min(), residuals.max() + 1, dtype=int)),
            rotation=90,
            fontsize=tick_fontsize,
        )
        residua_hist.set_ylabel("count", fontsize=label_fontsize)
        residua_hist.set_title(
            item,
            fontsize=title_fontsize,
        )

        # axs[item, 3]
        metric_values, metric_names, rank_level_dict = compute_set_level_metrics(
            y_true=y_true, y_pred=y_pred, item=item
        )
        item_level = sns.barplot(
            y=metric_names,
            x=metric_values,
            ax=axs[idx, 3] if len(args.selected_items) > 1 else axs[3],
            orient="h",
        )
        for i in item_level.containers:
            item_level.bar_label(
                i,
            )
        for label in item_level.get_xticklabels():
            label.set_rotation(90)
        item_level.set_ylabel("metrics", fontsize=label_fontsize)
        item_level.set_title(
            item,
            fontsize=title_fontsize,
        )
        # axs[item, 4]
        rank_level = sns.barplot(
            x=list(rank_level_dict["ranks"].astype(int)) * 3,
            y=rank_level_dict["mse_rank"]
            + rank_level_dict["mae_rank"]
            + rank_level_dict["rmse_rank"],
            hue=["mse"] * len(rank_level_dict["mse_rank"])
            + ["mae"] * len(rank_level_dict["mae_rank"])
            + ["rmse"] * len(rank_level_dict["rmse_rank"]),
            ax=axs[idx, 4] if len(args.selected_items) > 1 else axs[4],
        )
        for label in rank_level.get_xticklabels():
            label.set_rotation(90)
        rank_level.set_ylabel("metrics by rank", fontsize=label_fontsize)
        rank_level.set_title(
            item,
            fontsize=title_fontsize,
        )
        rank_level.legend(loc="best", edgecolor="white", facecolor="white")

    summary.figure(tag="item and rank performance", figure=figure, step=step, mode=mode)


def young_and_hamilton_performance(
    args,
    y_true: t.Dict[str, np.ndarray],
    y_pred: t.Dict[str, np.ndarray],
    metadata: t.Dict,
    representations: np.ndarray,
    clinical: t.Dict[str, np.ndarray],
    summary: tensorboard.Summary,
    step: int = 0,
    mode: int = 0,
):
    young_hamilton_true, young_true, hamilton_true = create_young_hamilton_labels(
        args, y=y_true
    )
    young_hamilton_pred, young_pred, hamilton_pred = create_young_hamilton_labels(
        args, y=y_pred
    )
    y_true_yh = [young_hamilton_true, young_true, hamilton_true]
    y_pred_yh = [young_hamilton_pred, young_pred, hamilton_pred]
    figure, axs = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(35, 15),
        gridspec_kw={"wspace": 0.2, "hspace": 1.2},
        dpi=args.dpi,
    )

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    titles = ["YOUNG_HAMILTON", "YOUNG", "HAMILTON"]
    for col, (true, pred) in enumerate(zip(y_true_yh, y_pred_yh)):
        # axs[0, col]
        labels, counts = np.unique(pred, return_counts=True)
        if col == 0:
            labels_names = pd.Series(labels).replace(
                {v: k for k, v in YOUNG_HAMILTON_DICT.items()}
            )
        elif col == 1:
            labels_names = [f"young_{l}" for l in labels]
        else:
            labels_names = [f"hamilton_{l}" for l in labels]
        axs0 = sns.barplot(
            x=labels_names,
            y=counts,
            ax=axs[0, col],
        )
        axs[0, col].set_ylabel("count", fontsize=label_fontsize)
        for level in axs0.get_xticklabels():
            level.set_rotation(90)
        axs[0, col].set_ylabel("predictions (counts)", fontsize=label_fontsize)
        axs[0, col].set_title(
            titles[col],
            fontsize=title_fontsize,
        )

        # axs[1, col]
        if col == 0:
            all_labels = np.array(list(YOUNG_HAMILTON_DICT.values()))
        else:
            all_labels = np.arange(5)
        cm = confusion_matrix(true, pred, labels=all_labels)
        mcc = matthews_corrcoef(true, pred)
        sns.heatmap(
            cm / cm.sum(),
            vmin=0,
            vmax=1,
            cmap=cmap,
            annot=cm.astype(str),
            fmt="",
            linewidths=0.01,
            cbar=False,
            ax=axs[1, col],
        )
        axs[1, col].set_xlabel(f"Predictions", fontsize=label_fontsize)
        axs[1, col].set_ylabel("Targets", fontsize=label_fontsize)
        axs[1, col].set_title(
            f"{titles[col]} - ACC={100 * (np.trace(cm) / np.sum(cm)):.02f}%, "
            f"MCC={mcc:.02f}",
            fontsize=title_fontsize,
        )

        if col == 0:
            ticklabels = list(YOUNG_HAMILTON_DICT.keys())
        else:
            ticklabels = [int(i) for i in all_labels]
        axs[1, col].set_xticks([i for i in range(len(ticklabels))])
        axs[1, col].set_xticklabels(
            ticklabels, va="top", ha="center", rotation=90, fontsize=tick_fontsize
        )
        axs[1, col].set_yticks([i for i in range(len(ticklabels))])
        axs[1, col].set_yticklabels(
            ticklabels, va="center", ha="right", rotation=0, fontsize=tick_fontsize
        )

        set_ticks_params(axis=axs[1, col], length=0, pad=PARAMS_PAD + 2)

        # axs[2, col]
        item_k = [
            cohen_kappa_score(
                y1=true,
                y2=pred,
                labels=all_labels,
                weights=weightage,
            )
            for weightage in [None, "linear", "quadratic"]
        ]
        ranks, counts = np.unique(true, return_counts=True)
        mse_micro = mean_squared_error(true, pred, squared=True)
        mse_rank = [
            mean_squared_error(
                true[np.where(true == r)[0]],
                pred[np.where(true == r)[0]],
                squared=True,
            )
            for r in ranks
        ]
        mse_macro = np.mean(mse_rank)
        mse_weighted = np.dot(mse_rank, counts / np.sum(counts))
        rmse_micro = mean_squared_error(true, pred, squared=False)
        rmse_rank = [
            mean_squared_error(
                true[np.where(true == r)[0]],
                pred[np.where(true == r)[0]],
                squared=False,
            )
            for r in ranks
        ]
        rmse_macro = np.mean(rmse_rank)
        rmse_weighted = np.dot(rmse_rank, counts / np.sum(counts))
        mae_micro = mean_absolute_error(true, pred)
        mae_rank = [
            mean_absolute_error(
                true[np.where(true == r)[0]],
                pred[np.where(true == r)[0]],
            )
            for r in ranks
        ]
        mae_macro = np.mean(mae_rank)
        mae_weighted = np.dot(mae_rank, counts / np.sum(counts))

        item_level = sns.barplot(
            x=[
                "ck_unweighted",
                "ck_linear",
                "ck_quadratic",
                "mse_micro",
                "mse_macro",
                "mse_weighted",
                "mae_micro",
                "mae_macro",
                "mae_weighted",
                "rmse_micro",
                "rmse_macro",
                "rmse_weighted",
            ],
            y=item_k
            + [
                mse_micro,
                mse_macro,
                mse_weighted,
                mae_micro,
                mae_macro,
                mae_weighted,
                rmse_micro,
                rmse_macro,
                rmse_weighted,
            ],
            ax=axs[2, col],
        )
        for l in item_level.get_xticklabels():
            l.set_rotation(90)
        axs[2, col].set_ylabel("metrics", fontsize=label_fontsize)
        axs[2, col].set_title(
            titles[col],
            fontsize=title_fontsize,
        )

        # axs[3, col]
        rank_level = sns.barplot(
            x=list(ranks.astype(int)) * 3,
            y=mse_rank + mae_rank + rmse_rank,
            hue=["mse"] * len(mse_rank)
            + ["mae"] * len(mae_rank)
            + ["rmse"] * len(rmse_rank),
            ax=axs[3, col],
        )
        for l in rank_level.get_xticklabels():
            l.set_rotation(90)
        axs[3, col].set_ylabel("metrics by rank", fontsize=label_fontsize)
        axs[3, col].set_title(
            titles[col],
            fontsize=title_fontsize,
        )
        rank_level.legend(loc="best", edgecolor="white", facecolor="white")

    summary.figure(
        tag="young and hamilton performance", figure=figure, step=step, mode=mode
    )


def scores_within_session(
    args,
    y_true: t.Dict[str, np.ndarray],
    y_pred: t.Dict[str, np.ndarray],
    metadata: t.Dict,
    summary: tensorboard.Summary,
    representations: np.ndarray,
    clinical: t.Dict[str, np.ndarray],
    step: int = 0,
    mode: int = 0,
):
    """
    for each recording session, show ground truth and predictions by item
    """
    df_list = []
    true = np.concatenate(
        [np.expand_dims(v, axis=1) for k, v in y_true.items()], axis=1
    )
    pred = np.concatenate(
        [np.expand_dims(v, axis=1) for k, v in y_pred.items()], axis=1
    )
    for session in np.unique(metadata["recording_id"]):
        session_idx = np.where(metadata["recording_id"] == session)[0]
        session_pred = pred[session_idx]
        session_true = true[session_idx[0]]
        df_list.append(np.expand_dims(session_true, axis=0))
        df_list.append(np.mean(session_pred, axis=0, keepdims=True))
        df_list.append(np.std(session_pred, axis=0, keepdims=True))

    indexes = [
        np.repeat(np.unique(metadata["recording_id"]).astype(np.int32), 3),
        np.array(
            ["true", "pred_mean", "pred_std"] * len(np.unique(metadata["recording_id"]))
        ),
    ]

    df = pd.DataFrame(
        np.concatenate(df_list, axis=0), index=indexes, columns=args.selected_items
    )
    df.index.names = ["session", "summary"]
    df.to_csv(
        os.path.join(
            args.output_dir,
            "plots",
            f"scores_within_sessions_mode{mode}_step{step}.csv",
        ),
    )


def sessions_across_splits(
    args,
    data: t.Dict[str, t.Dict[str, np.ndarray]],
    summary: tensorboard.Summary,
):
    """
    For each recording session, shows how it is distributed across
    train/val/test
    """
    sessions_across_splits = {}
    for ds_idx, ds_split in enumerate(["y_train", "y_val", "y_test"]):
        sessions, counts = np.unique(data[ds_split]["Session_Code"], return_counts=True)
        sessions_across_splits[f"{ds_idx}"] = {
            "sessions": sessions,
            "counts": counts,
        }
    all_sessions = list(
        np.unique([s for k, v in sessions_across_splits.items() for s in v["sessions"]])
    )
    count_dict = {k: [] for k in sessions_across_splits.keys()}
    for s in all_sessions:
        for k, v in sessions_across_splits.items():
            if s in v["sessions"]:
                count_idx = np.where(v["sessions"] == s)[0][0]
                count_dict[k].append(v["counts"][count_idx])
            else:
                count_dict[k].append(0)
    count = pd.DataFrame(count_dict).T.values
    percentage_across_splits = count / np.sum(count, axis=0)
    percentage_within_split = count / np.sum(count, axis=1, keepdims=True)
    splits = ["train", "val", "test"]
    var = ["no segments", "% across splits", "% within split"]
    df = np.empty(shape=[len(splits) * len(var), len(all_sessions)], dtype=np.float32)
    df[0::3], df[1::3], df[2::3] = (
        count,
        percentage_across_splits,
        percentage_within_split,
    )
    indexes = [
        np.repeat(splits, 3),
        np.array(var * len(var)),
    ]
    cols = np.array(all_sessions).astype("int")
    df = pd.DataFrame(df, index=indexes, columns=cols)
    df.to_csv(
        os.path.join(
            args.output_dir,
            "plots",
            f"segment_session_count.csv",
        ),
    )


def plot_sample_segment(
    args,
    data: t.Dict[str, t.Dict[str, np.ndarray]],
    summary: tensorboard.Summary,
):
    """Plot a random sample segment from train, validation,
    test when epoch == 0"""
    for ds_idx, ds_split in enumerate(["x_train", "x_val", "x_test"]):
        idx = random.randint(0, len(data[ds_split]) - 1)
        sample_segment = {
            c: h5.get(data[ds_split][idx], name=c).astype(np.float32)
            for c in args.ds_info["channel_freq"].keys()
        }

        figure, axs = plt.subplots(
            nrows=len(sample_segment),
            ncols=1,
            figsize=(9, 5 * len(sample_segment)),
            gridspec_kw={"wspace": 0.01, "hspace": 0.2},
            dpi=args.dpi,
        )
        color_dict = {
            "BVP": "tab:blue",
            "EDA": "tab:orange",
            "HR": "tab:green",
            "TEMP": "tab:red",
            "ACC_x": "tab:purple",
            "ACC_y": "tab:purple",
            "ACC_z": "tab:purple",
        }
        for i, (k, v) in enumerate(sample_segment.items()):
            axs[i].plot(v, linestyle="dotted", color=color_dict[k])
            axs[i].set_ylabel(k, fontsize=label_fontsize)

        summary.figure(tag=f"sample_segment", figure=figure, step=0, mode=ds_idx)


def cases_detection(
    args,
    y_true: t.Dict[str, np.ndarray],
    y_pred: t.Dict[str, np.ndarray],
    metadata: t.Dict,
    representations: np.ndarray,
    clinical: t.Dict[str, np.ndarray],
    summary: tensorboard.Summary,
    step: int = 0,
    mode: int = 0,
):
    """
    cases are defined as subject with a depression and mania severity score
    band >= subsyndromal (total HDRS > 7 and/ortotal YMRS > 7), controls as
    subject with both depression and mania severity < subsyndromal
    """
    if len(args.selected_items):
        ymrs_scores_pred = np.concatenate(
            [np.expand_dims(v, axis=1) for k, v in y_pred.items() if "YMRS" in k],
            axis=1,
        )
        ymrs_scores_pred_binary = np.where(np.sum(ymrs_scores_pred, axis=1) > 7, 1, 0)
        ymrs_scores_true = np.concatenate(
            [np.expand_dims(v, axis=1) for k, v in y_true.items() if "YMRS" in k],
            axis=1,
        )
        ymrs_scores_true_binary = np.where(np.sum(ymrs_scores_true, axis=1) > 7, 1, 0)

        hdrs_scores_pred = np.concatenate(
            [np.expand_dims(v, axis=1) for k, v in y_pred.items() if "HDRS" in k],
            axis=1,
        )
        hdrs_scores_pred_binary = np.where(np.sum(hdrs_scores_pred, axis=1) > 7, 1, 0)
        hdrs_scores_true = np.concatenate(
            [np.expand_dims(v, axis=1) for k, v in y_true.items() if "HDRS" in k],
            axis=1,
        )
        hdrs_scores_true_binary = np.where(np.sum(hdrs_scores_true, axis=1) > 7, 1, 0)

        mood_scores_true_binary = np.where(
            (ymrs_scores_true_binary + hdrs_scores_true_binary) >= 1, 1, 0
        )
        mood_scores_pred_binary = np.where(
            (ymrs_scores_pred_binary + hdrs_scores_pred_binary) >= 1, 1, 0
        )

        figure, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(5, 5),
            gridspec_kw={"wspace": 0.01, "hspace": 0.01},
            dpi=args.dpi,
        )

        cm = confusion_matrix(mood_scores_true_binary, mood_scores_pred_binary)
        cm_plot = sns.heatmap(
            cm,
            vmin=0,
            vmax=1,
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            annot=cm.astype(str),
            fmt="",
            linewidths=0.01,
            cbar=False,
            ax=ax,
        )
        cm_plot.set_xlabel(f"Predictions", fontsize=label_fontsize)
        cm_plot.set_ylabel("Targets", fontsize=label_fontsize)
        tn, fp, fn, tp = confusion_matrix(
            mood_scores_true_binary, mood_scores_pred_binary
        ).ravel()
        if (fn + tp) > 0:
            miss_rate = round(fn / (fn + tp), 3)
        else:
            miss_rate = "nan"
        cm_plot.set_title(
            f"Miss Rate = {miss_rate}",
            fontsize=title_fontsize,
        )
        labels = ["Case", "Control"]
        cm_plot.set_xticklabels(
            labels, va="top", ha="center", rotation=90, fontsize=tick_fontsize
        )
        cm_plot.set_yticklabels(
            labels, va="center", ha="right", rotation=0, fontsize=tick_fontsize
        )
        set_ticks_params(axis=cm_plot, length=0, pad=PARAMS_PAD + 2)

        summary.figure(tag="miss_rate", figure=figure, step=step, mode=mode)
    else:
        pass


def pca_gmm(
    args, data: t.Dict[str, t.Dict[str, np.ndarray]], summary: tensorboard.Summary
):
    """
    run PCA on HDRS-YMRS, then run GMM on PCs
    """

    true = np.concatenate(
        [np.expand_dims(data["y_train"][item], axis=1) for item in args.selected_items],
        axis=1,
    )
    sessions = np.unique(data["y_train"]["Session_Code"])
    if (len(args.selected_items) == 28) and (len(sessions) > len(args.selected_items)):
        indexes = [
            np.where(data["y_train"]["Session_Code"] == s)[0][0]
            for s in np.unique(sessions)
        ]
        session_scores = true[indexes]
        scaler = StandardScaler().fit(session_scores)
        X = scaler.transform(session_scores)
        # fit models with 1-10 components
        N = np.arange(1, 11)
        models = [None for i in range(len(N))]
        for i in range(len(N)):
            models[i] = GaussianMixture(
                n_components=N[i], n_init=5, random_state=args.seed
            ).fit(X)
        BIC = [m.bic(X) for m in models]
        gmm = GaussianMixture(
            n_components=np.argmin(BIC) + 1, n_init=5, random_state=args.seed
        ).fit(X)
        pca = PCA().fit(X)

        membership = gmm.predict(X)
        log_likelihood = gmm.score_samples(X)
        pc1_coord = pca.transform(X)[:, 0]
        pc2_coord = pca.transform(X)[:, 1]

        figure, axs = plt.subplots(5, 1, figsize=(9, 7 * 5), dpi=args.dpi)

        ### ax0

        exp_var = pca.explained_variance_ratio_
        loadings = pca.components_
        num_pc = pca.n_features_in_
        pc_list = ["PC" + str(i) for i in list(range(1, num_pc + 1))]
        loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
        loadings_df["scale_item"] = args.selected_items
        loadings_df = loadings_df.set_index("scale_item")

        axs[0].bar(pc_list, exp_var)
        axs[0].set_ylabel("Proportion of explained variance", fontsize=label_fontsize)
        axs[0].set_title("Scree plot", fontsize=title_fontsize)
        axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axs[0].set_xticks(list(np.arange(len(pc_list))))
        axs[0].set_xticklabels(pc_list, rotation=90)

        ### ax1

        cmap = sns.color_palette("rocket_r", as_cmap=True)
        sns.heatmap(
            loadings_df.iloc[:, :5],
            annot=np.round(loadings_df.iloc[:, :5], 2).astype(str),
            cmap=cmap,
            fmt="",
            ax=axs[1],
        )
        axs[1].set_title("Correlation matrix for loadings", fontsize=title_fontsize)
        axs[1].set_ylabel("Items", fontsize=label_fontsize)

        ### ax2
        coord_d = pd.DataFrame(
            {"PC1": pc1_coord, "PC2": pc2_coord, "membership": membership}
        )
        membership_plot = sns.scatterplot(
            x="PC1",
            y="PC2",
            data=coord_d,
            hue="membership",
            ax=axs[2],
            s=15,
            palette="deep",
        )
        plt.setp(axs[2].get_legend().get_texts(), fontsize="14")
        plt.setp(axs[2].get_legend().get_title(), fontsize="16")
        axs[2].set_xlabel("PC1", fontsize=label_fontsize)
        axs[2].set_ylabel("PC2", fontsize=label_fontsize)
        axs[2].set_title("PC projection - GMM membership", fontsize=title_fontsize)
        membership_plot.legend(loc="best", edgecolor="white", facecolor="white")

        ### ax3

        axs3 = axs[3].scatter(
            x=pc1_coord, y=pc2_coord, c=log_likelihood, s=15, cmap="viridis"
        )
        norm = plt.Normalize(log_likelihood.min(), log_likelihood.max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        figure.colorbar(axs3, ax=axs[3])
        axs[3].set_xlabel("PC1", fontsize=label_fontsize)
        axs[3].set_ylabel("PC2", fontsize=label_fontsize)
        axs[3].set_title("PC projection - log_likelihood", fontsize=title_fontsize)

        ### axs4

        means = pd.DataFrame(
            data=gmm.means_,
            index=np.unique(membership),
            columns=args.selected_items,
        )
        means.index.names = ["Gaussian_no"]
        sns.heatmap(
            means.T,
            annot=np.round(means.T, 2).astype(str),
            cmap=cmap,
            fmt="",
            ax=axs[4],
        )
        axs[4].set_title("Guassian distributions' mean", fontsize=title_fontsize)
        axs[4].set_ylabel("Items", fontsize=label_fontsize)

        summary.figure(tag="pca_gmm", figure=figure, step=0, mode=0)


def clinical_demographic_table(
    args, indexes: t.List, d: t.Dict[str, np.ndarray], mode: int
):
    """
    clinical-demographic characteristics table
    """
    if len(args.selected_items) == 28:
        # AGE
        info_names = list(
            itertools.product(["Age (years)"], ["mean", "sd", "median", "iqr"])
        )
        info_values = [
            np.mean(d["age"][indexes]),
            np.std(d["age"][indexes]),
            np.median(d["age"][indexes]),
            stats.iqr(d["age"][indexes]),
        ]
        # SEX
        info_names.extend(
            list(
                itertools.product(
                    ["Sex"], ["males (N)", "males (%)", "females (N)", "females (%)"]
                )
            )
        )
        info_values.extend(
            [
                np.sum(d["sex"][indexes] == 0),
                np.mean(d["sex"][indexes] == 0),
                np.sum(d["sex"][indexes] == 1),
                np.mean(d["sex"][indexes] == 1),
            ]
        )
        # YMRS
        indexes_sessions = list(
            pd.DataFrame(
                np.concatenate(
                    (
                        np.expand_dims(d["NHC"], axis=1),
                        np.expand_dims(d["time"], axis=1),
                    ),
                    axis=1,
                )
            )
            .drop_duplicates(keep="first")
            .index
        )
        ymrs_tot_score = np.sum(
            np.concatenate(
                [
                    np.expand_dims(d[item][indexes_sessions], axis=1)
                    for item in args.selected_items
                    if item.startswith("YMRS")
                ],
                axis=1,
            ),
            axis=1,
        )
        info_names.extend(
            list(
                itertools.product(
                    ["YMRS (total score)"], ["mean", "sd", "median", "iqr"]
                )
            )
        )
        info_values.extend(
            [
                np.mean(ymrs_tot_score),
                np.std(ymrs_tot_score),
                np.median(ymrs_tot_score),
                stats.iqr(ymrs_tot_score),
            ]
        )
        # HDRS
        hdrs_tot_score = np.sum(
            np.concatenate(
                [
                    np.expand_dims(d[item][indexes_sessions], axis=1)
                    for item in args.selected_items
                    if item.startswith("HDRS")
                ],
                axis=1,
            ),
            axis=1,
        )
        info_names.extend(
            list(
                itertools.product(
                    ["HDRS (total score)"], ["mean", "sd", "median", "iqr"]
                )
            )
        )
        info_values.extend(
            [
                np.mean(hdrs_tot_score),
                np.std(hdrs_tot_score),
                np.median(hdrs_tot_score),
                stats.iqr(hdrs_tot_score),
            ]
        )
        # DIAGNOSIS
        stati = [
            k for k, v in DICT_STATE.items() if v in np.unique(d["status"][indexes])
        ]
        stati_info = list(
            (" ".join(e) for e in itertools.product(stati, ["(N)", "(%)"]))
        )
        info_names.extend(list(itertools.product(["Diagnosis"], stati_info)))
        for s in stati:
            info_values.extend([np.sum(d["status"][indexes] == DICT_STATE[s])])
            info_values.extend([np.mean(d["status"][indexes] == DICT_STATE[s])])

        index = pd.MultiIndex.from_tuples(info_names, names=["var", "stat"])
        df = pd.Series(info_values, index=index)
        df.to_csv(
            os.path.join(
                args.output_dir,
                "plots",
                f"clinical_demographics_table_mode_{mode}.csv",
            ),
        )


def clinical_demographics(
    args,
    data: t.Dict[str, t.Dict[str, np.ndarray]],
    summary: tensorboard.Summary,
):
    """ "
    age, sex, clinical status, Ts distribution
    """
    for ds_idx, ds_split in enumerate(["y_train", "y_val", "y_test"]):
        # get index where a given subject_id (i.e. NHC) first appears
        indexes = []
        for subject in np.unique(data[ds_split]["NHC"]):
            subject_idx = np.where(data[ds_split]["NHC"] == subject)[0][0]
            indexes.append(subject_idx)
        # print clinical-demographic characteristics summary table
        clinical_demographic_table(args, indexes=indexes, d=data[ds_split], mode=ds_idx)
        figure, axs = plt.subplots(
            nrows=1,
            ncols=4,
            figsize=(5 * 4, 6),
            gridspec_kw={"wspace": 0.15, "hspace": 0.01},
            dpi=args.dpi,
        )

        ax0 = sns.histplot(
            x=data[ds_split]["age"][indexes],
            hue=data[ds_split]["sex"][indexes].astype(int),
            palette={0: "fuchsia", 1: "aqua"},
            multiple="stack",
            stat="count",
            shrink=0.8,
            binwidth=2,
            ax=axs[0],
        )
        ax0.set_xlabel("Age (Years)", fontsize=label_fontsize)
        ax0.set_ylabel("count", fontsize=label_fontsize)
        legend = ax0.get_legend()
        handles = legend.legendHandles
        legend.remove()
        ax0.legend(
            handles,
            ["Female", "Male"],
            title="Sex",
            loc="best",
            edgecolor="white",
            facecolor="white",
        )

        variables = ["sex", "status", "time"]
        dictionaries = [
            {0: "Males", 1: "Females"},
            {v: k for k, v in DICT_STATE.items()},
            {v: k for k, v in DICT_TIME.items()},
        ]
        for i, (v, d) in enumerate(zip(variables, dictionaries), 1):
            if v == "time":
                counted_col = (
                    pd.DataFrame(
                        np.concatenate(
                            (
                                np.expand_dims(data[ds_split]["NHC"], axis=1),
                                np.expand_dims(data[ds_split][v], axis=1),
                            ),
                            axis=1,
                        )
                    )
                    .drop_duplicates(keep="first")
                    .iloc[:, 1]
                    .astype(int)
                    .map(d)
                )

            else:
                counted_col = pd.Series(data[ds_split][v][indexes].astype(int)).map(d)
            ax = sns.countplot(
                x=counted_col,
                ax=axs[i],
            )
            ax.set_ylabel("count", fontsize=label_fontsize)
            ax.set_xlabel(v.capitalize(), fontsize=label_fontsize)
            for level in ax.get_xticklabels():
                level.set_rotation(45)
            ax.bar_label(ax.containers[0])

        summary.figure(tag="clinical_demographics", figure=figure, step=0, mode=ds_idx)


def plot_num_sessions(
    args, data: t.Dict[str, t.Dict[str, np.ndarray]], summary: tensorboard.Summary
):
    """
    plot subjects with 1, 2, 3, or 4 recording sessions and color by status
    """

    palette = sns.color_palette("deep", n_colors=len(DICT_STATE))
    stati_color_dictionary = dict(zip(DICT_STATE.values(), palette))
    legend_fontsize = 10
    title_fontsize = 12
    for ds_idx, ds_split in enumerate(["y_train", "y_val", "y_test"]):
        figure, axs = plt.subplots(
            nrows=1,
            ncols=4,
            figsize=(18, 10),
            gridspec_kw={"wspace": 0.24, "hspace": 0.1},
            dpi=120,
        )
        Ts = {t: [] for t in range(1, 5)}
        for subject in np.unique(data[ds_split]["NHC"]):
            subject_idx = np.where(data[ds_split]["NHC"] == subject)[0]
            observations = data[ds_split]["time"][subject_idx]
            for i in range(0, len(np.unique(observations)), 1):
                Ts[i + 1].append(np.unique(data[ds_split]["status"][subject_idx])[0])

        for col_idx, t in enumerate(Ts.keys()):
            stati = Ts[t]
            conditions, counts = np.unique(stati, return_counts=True)
            palette = {
                k: v for k, v in stati_color_dictionary.items() if k in conditions
            }
            patches, texts = axs[col_idx - 1].pie(
                counts,
                startangle=90,
                colors=[palette[key] for key in conditions],
                explode=tuple([0.1] * len(counts)),
                radius=1,
                wedgeprops={"antialiased": True},
            )
            labels = [
                f"{i} - {j}"
                for i, j in zip(
                    [
                        {v: k for k, v in DICT_STATE.items()}[condition]
                        for condition in conditions
                    ],
                    counts,
                )
            ]

            patches, labels, dummy = zip(
                *sorted(zip(patches, labels, counts), key=lambda x: x[2], reverse=True)
            )

            axs[col_idx].legend(
                patches,
                labels,
                loc="upper left",
                bbox_to_anchor=(-0.25, 0.97),
                fontsize=legend_fontsize,
                frameon=False,
                facecolor="white",
                handletextpad=0.35,
                handlelength=0.8,
                markerscale=1,
            )
            axs[col_idx].set_title(
                f"Subjects with {t} recording session(s) (N = {counts.sum()})",
                fontsize=title_fontsize,
            )
        summary.figure(tag="count_num_sessions", figure=figure, step=0, mode=ds_idx)


def plot_extracted_features(
    args,
    y_true: t.Dict[str, np.ndarray],
    y_pred: t.Dict[str, np.ndarray],
    metadata: t.Dict,
    representations: np.ndarray,
    clinical: t.Dict[str, np.ndarray],
    summary: tensorboard.Summary,
    step: int = 0,
    mode: int = 0,
):
    representations = StandardScaler().fit_transform(representations)
    t_sne = TSNE(
        n_components=2,
        perplexity=30,
        init="random",
        n_iter=250,
        random_state=args.seed,
    )
    embedding = t_sne.fit_transform(representations)

    figure, axs = plt.subplots(
        nrows=1,
        ncols=len(args.selected_items),
        figsize=(6.5 * len(args.selected_items), 6),
        gridspec_kw={"wspace": 0.2, "hspace": 0.3},
        dpi=args.dpi,
    )
    colors = ["red", "blue", "green", "purple", "yellow"]
    for idx, (item, v) in enumerate(y_true.items()):
        ranks = np.unique(v.astype("int"))
        color_dict = {r: colors[idx] for idx, r in enumerate(ranks)}
        emb_plot = sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=v.astype("int"),
            s=6,
            alpha=0.3,
            palette=color_dict,
            ax=axs[idx] if len(args.selected_items) > 1 else axs,
        )
        emb_plot.legend(loc="best", edgecolor="white", facecolor="white")
        emb_plot.set_xlabel("emb 01", fontsize=label_fontsize)
        emb_plot.set_ylabel("emb 02", fontsize=label_fontsize)
        emb_plot.set_title(
            item,
            fontsize=title_fontsize,
        )

    summary.figure(tag="embedded_features_by_rank", figure=figure, step=step, mode=mode)


def eda_plots(args, data: t.Dict, summary: tensorboard.Summary):
    fun_dict = {
        "clinical_demographics": clinical_demographics,
        "plot_sample_segment": plot_sample_segment,
        "pca_gmm": pca_gmm,
        "sessions_across_splits": sessions_across_splits,
        "plot_num_sessions": plot_num_sessions,
    }
    for fun_name, fun in tqdm.tqdm(
        fun_dict.items(), desc="EDA plotting", disable=args.verbose == 0
    ):
        fun(args, data=data, summary=summary)
        if args.verbose > 2:
            print(f"\n{fun_name} plotted")


def training_loop_plots(
    args,
    summary: tensorboard.Summary,
    y_true: t.Dict[str, np.ndarray],
    y_pred: t.Dict[str, np.ndarray],
    metadata: t.Dict,
    representations: np.ndarray,
    clinical: t.Dict[str, np.ndarray],
    step: int = 0,
    mode: int = 0,
):
    fun_dict = {
        "items_cm": items_cm,
        "item_and_rank_performance": item_and_rank_performance,
        # "young_and_hamilton_performance": young_and_hamilton_performance,
        "scores_within_session": scores_within_session,
        "cases_detection": cases_detection,
        "plot_extracted_features": plot_extracted_features,
    }
    for fun_name, fun in tqdm.tqdm(
        fun_dict.items(), desc="Plotting predictions", disable=args.verbose <= 1
    ):
        fun(
            args,
            y_true=y_true,
            y_pred=y_pred,
            metadata=metadata,
            representations=representations,
            clinical=clinical,
            summary=summary,
            step=step,
            mode=mode,
        )
        if args.verbose > 2:
            print(f"\n{fun_name} plotted")
