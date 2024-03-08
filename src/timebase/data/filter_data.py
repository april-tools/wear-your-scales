import typing as t

import numpy as np
import pandas as pd
import sklearn.utils

from timebase.data.static import *


def quality_control(
    channel_data: t.Dict[str, np.ndarray],
    sampling_rates: t.Dict[str, int],
    verbose: int = 1,
):
    """
    Apply Kleckner et al. 2018 quality control to EDA and TEMP plus new rule on HR
    see Figure 1 in https://pubmed.ncbi.nlm.nih.gov/28976309/
    Returns:
        features: np.ndarray, the filtered recordings where 0 index is one continuous recording
    """
    eda_slope = np.gradient(channel_data["EDA"])

    # Rule 1: remove EDA not within 0.05 - 60 muS
    rule1 = (channel_data["EDA"] < 0.05) + (channel_data["EDA"] > 60)
    if verbose > 1:
        print(
            f"Kleckner al. 2018 quality control:\t"
            f"{rule1.sum()/len(rule1):.02f}% measurements removed "
            f"by Rule 1"
        )

    # Rule 2: remove EDA slope not within -10 - 10 muS/sec
    rule2 = (eda_slope < -10) + (eda_slope > 10)
    if verbose > 1:
        print(f"{rule2.sum()/len(rule2):.02f}% measurements removed by Rule 2")

    # Rule 3: remove TEMP not within 30 - 40 °C
    rule3 = (channel_data["TEMP"] < 30) + (channel_data["TEMP"] > 40)
    if verbose > 1:
        print(f"{rule3.sum()/len(rule3):.02f}% measurements removed by Rule 3")

    # Rule 4: EDA surrounded (within 5 sec) by invalid data according to Rule 1-3
    assert (
        len(channel_data["EDA"]) > 5 * sampling_rates["EDA"]
    ), "recording is shorter than 5 seconds, cannot apply Rule 4."
    rule4 = np.correlate(
        rule1 + rule2 + rule3,
        np.ones(shape=(5 * sampling_rates["EDA"]), dtype=np.int8),
        mode="same",
    ).astype(bool)
    if verbose > 1:
        print(
            f"{(rule4.sum() - (rule1.sum() + rule2.sum() + rule3.sum()))/len(rule4):.02f}% "
            f"measurements removed by Rule 4\t"
        )

    # Rule 5: remove HR that are not within 25 - 250 bpm
    # Note: this is not from Kleckner et al. 2018
    rule5 = (channel_data["HR"] < 25) + (channel_data["HR"] > 250)
    if verbose > 1:
        print(f"{rule5.sum()/len(rule5):.02f}% measurements removed by rule 5")

    # HR is in 1Hz, EDA and TEMP are in 4Hz
    # We need to downsampling Rule 1-4 masks to 1Hz and join with Rule 5
    total_mask = rule1 + rule2 + rule3 + rule4
    total_mask = np.reshape(total_mask, newshape=(-1, sampling_rates["EDA"]))
    # set False to rows with False and total_mask is now 1Hz
    total_mask = np.min(total_mask, axis=-1)
    total_mask = total_mask + rule5
    for channel in channel_data.keys():
        if channel != "IBI":
            # convert 1Hz mask to channel sampling rates
            mask = np.repeat(total_mask, repeats=sampling_rates[channel], axis=0)
            channel_data[channel][mask] = np.nan

    # percentage of (wall time) seconds removed from recording
    filtered_out_percentage = total_mask.sum() / len(total_mask)
    if verbose > 1:
        print(
            f"{filtered_out_percentage:.02f}% of recordings removed "
            f"upon Quality Control."
        )
    return filtered_out_percentage


def set_unique_recording_id(labels: np.ndarray):
    """
    It is possible that a single session (i.e. a given NHC at a given time T)
    has multiple recordings (i.e. the watch starts and stops recording multiple
    times), we therefore create unique recording IDs for these sub-recordings
    which belong to a single session.

    Due to issues with either compliance or the device, multiple recordings
    were sometimes registered for a single session. When this is the case set
    the first session_id throughout the session
    """
    session_idx = LABEL_COLS.index("Session_Code")
    nhc_col = labels[:, LABEL_COLS.index("NHC")]
    time_col = labels[:, LABEL_COLS.index("time")]

    for nhc in np.unique(nhc_col):
        for time in np.unique(time_col[np.where(nhc_col == nhc)[0]]):
            condition = np.where((nhc_col == nhc) & (time_col == time))[0]
            unique_recordings = np.unique(labels[:, session_idx][condition])
            if len(unique_recordings) > 1:
                labels[:, session_idx][condition] = unique_recordings[0]
