import typing as t

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from timebase.data.static import *
from timebase.utils.utils import create_young_hamilton_labels


def create_resampled_train(indices_resampled: np.ndarray, data: t.Dict):
    data["x_train"] = np.concatenate(
        [data["x_train"][i] for i in indices_resampled], axis=0
    )
    data["y_train"] = {
        k: np.concatenate([v[i] for i in indices_resampled], axis=0)
        for k, v in data["y_train"].items()
    }


def combine_ros_rus(args, data: t.Dict):
    """
    Segments are resampled such that each class has exactly n = tot_number_segments / C (where C is number of classes)
    segments. Thus, classes whose original number of segments is greater (smaller) than n are undersampled (oversampled)
    """
    y = data["y_train"]
    if not args.imb_mode_item_focus:
        assert (
            len(args.selected_items) == 28
        ), "Resampling on aggregated YMRS-HDRS is only meaningful when all 28 items are used"
        young_hamilton, _, _ = create_young_hamilton_labels(args, y=y)
        var_to_resample_over = young_hamilton
    else:
        assert args.imb_mode_item_focus in args.selected_items
        var_to_resample_over = y[args.imb_mode_item_focus]
    bin_labels, bin_counts = np.unique(var_to_resample_over, return_counts=True)
    bin_weights = [c / bin_counts.sum() for c in bin_counts]
    heavy_bins = [
        bin_labels[i] for i, w in enumerate(bin_weights) if w > 1 / len(bin_weights)
    ]
    # given B number of bins, each should have an equal number of instances,
    # i.e. total num instances divided by B
    no2keep = round((1 / len(bin_weights)) * (bin_counts.sum()))
    rus = RandomUnderSampler(
        random_state=args.imb_mode_seed,
        sampling_strategy={k: no2keep for k in heavy_bins},
    )
    indices_downsampled, labels_downsampled = rus.fit_resample(
        np.arange(len(var_to_resample_over))[..., np.newaxis], var_to_resample_over
    )
    ros = RandomOverSampler(random_state=args.imb_mode_seed)
    indices_resampled, labels_resampled = ros.fit_resample(
        indices_downsampled, labels_downsampled
    )
    create_resampled_train(indices_resampled=indices_resampled, data=data)
    # segment's loss is multiplied by the corresponding class resampling ratio
    resampling_ratios = (
        np.unique(var_to_resample_over, return_counts=True)[1]
        / np.unique(labels_resampled, return_counts=True)[1]
    )
    bin_loss_scaling_factor = {
        bin_labels[i]: resampling_ratios[i] / sum(resampling_ratios)
        for i in range(len(resampling_ratios))
    }
    segment_weights = pd.Series(labels_resampled).replace(bin_loss_scaling_factor)
    return segment_weights.to_numpy(dtype=np.float32)
