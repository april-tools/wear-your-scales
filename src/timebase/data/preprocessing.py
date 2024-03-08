"""
Helper functions to preprocess CSV files
Reference on data export and formatting of Empatica E4 wristband
https://support.empatica.com/hc/en-us/articles/201608896-Data-export-and-formatting-from-E4-connect-
"""

import datetime
import math
import typing as t
import warnings
from copy import deepcopy
from math import ceil, floor

import biosppy
import flirt
import hrvanalysis
import mne
import pandas as pd

from timebase.data import filter_data, utils
from timebase.data.static import *
from timebase.utils.utils import update_dict

warnings.simplefilter("error", RuntimeWarning)


def read_clinical_info(filename: str):
    """Read clinical EXCEL file"""
    assert os.path.isfile(filename), f"clinical file {filename} does not exists."
    xls = pd.ExcelFile(filename)
    info = pd.read_excel(xls, sheet_name=None)  # read all sheets
    return pd.concat(info)


def low_pass_filter(recording: np.ndarray, sampling_rate: int):
    return mne.filter.filter_data(
        data=recording.astype(np.float64),
        sfreq=sampling_rate,
        l_freq=0,
        h_freq=0.35,
        filter_length=257,
        verbose=False,
    ).astype(np.float32)


def split_acceleration(
    channel_data: t.Dict[str, np.ndarray],
    sampling_rates: t.Dict[str, int],
):
    """Split 3D ACC into ACC_x, ACC_y and ACC_z"""
    channel_data["ACC_x"] = channel_data["ACC"][:, 0]
    channel_data["ACC_y"] = channel_data["ACC"][:, 1]
    channel_data["ACC_z"] = channel_data["ACC"][:, 2]
    del channel_data["ACC"]
    sampling_rates["ACC_x"] = sampling_rates["ACC"]
    sampling_rates["ACC_y"] = sampling_rates["ACC"]
    sampling_rates["ACC_z"] = sampling_rates["ACC"]
    del sampling_rates["ACC"]


def segmentation(
    args,
    session_data: t.Dict[str, np.ndarray],
    channel_freq: t.Dict[str, int],
    unix_t0: t.Dict,
) -> (t.Dict[str, np.ndarray], int):
    """
    Segment preprocessed features along the temporal dimension into
    N non-overlapping segments where each segment has size args.segment_length
    Return:
        data: t.Dict[str, np.ndarray]
                dictionary of np.ndarray, where the keys are the channels
                and each np.ndarray are in shape (num. segment, window size)
        size: int, number of extracted segments
    """
    assert (segment_length := args.segment_length) > 0
    channels = list(session_data.keys())
    ibi = session_data["IBI"]
    channels.remove("IBI")
    channel_segments = {c: [] for c in channels}

    # segment each channel using a sliding window that space out equally
    for channel in channels:
        window_size = segment_length * channel_freq[channel]
        recording = session_data[channel]
        num_segments = floor(len(recording) / window_size)
        indexes = np.linspace(
            start=0,
            stop=len(recording) - window_size,
            num=num_segments,
            dtype=int,
        )
        channel_segments[channel].extend(
            [recording[i : i + window_size, ...] for i in indexes]
        )
        if channel == "HR":
            unix_time = (np.arange(len(recording)) + unix_t0[channel]).astype(np.int32)

            unix_time = np.where(
                np.isnan(recording), np.nan, unix_time.astype(np.float64)
            )
            channel_segments["unix_time"] = [
                unix_time[i : i + window_size, ...] for i in indexes
            ]
    channels.extend(["unix_time"])
    num_channel_segments = [len(s) for s in channel_segments.values()]
    assert (
        len(set(num_channel_segments)) == 1
    ), "all channels must have equal length after segmentation"
    # dictionary of list of np.ndarray, where channel are the keys.
    data = {c: [] for c in channels}
    for i in range(num_channel_segments[0]):
        segment, drop = {}, False
        for channel in channels:
            recording = channel_segments[channel][i]
            # drop segment with NaN values
            if np.isnan(recording).any():
                drop = True
                break
            segment[channel] = recording
        if not drop:
            for channel in channels:
                data[channel].append(segment[channel])
    # ensure each channel has equal number of segments
    sizes = [len(data[c]) for c in channels]
    assert len(set(sizes)) == 1, "unequal number of extracted segments."
    data = {c: np.asarray(r) for c, r in data.items()}
    data["IBI"] = ibi
    return data, sizes[0]


def load_channel(recording_dir: str, channel: str):
    """Load channel CSV data from file
    Returns
      unix_t0: int, the start time of the recording in UNIX time
      sampling_rate: int, sampling rate of the recording (if exists)
      data: np.ndarray, the raw recording data
    """
    assert channel in CSV_CHANNELS, f"unknown channel {channel}"
    raw_data = pd.read_csv(
        os.path.join(recording_dir, f"{channel}.csv"), delimiter=",", header=None
    ).values

    unix_t0, sampling_rate, data = None, -1.0, None
    if channel == "IBI":
        unix_t0 = raw_data[0, 0]
        data = raw_data[1:]
    else:
        unix_t0 = raw_data[0] if raw_data.ndim == 1 else raw_data[0, 0]
        sampling_rate = raw_data[1] if raw_data.ndim == 1 else raw_data[1, 0]
        data = raw_data[2:]
    assert sampling_rate.is_integer(), "sampling rate must be an integer"
    data = np.squeeze(data)
    return int(unix_t0), int(sampling_rate), data.astype(np.float32)


def pad(args, data: np.ndarray, sampling_rate: int):
    """
    Upsample channel whose sampling rate is lower than args.time_alignment
    """

    # trim additional recordings that does not make up a second.
    data = data[: data.shape[0] - (data.shape[0] % sampling_rate)]

    s_shape = [data.shape[0] // sampling_rate, sampling_rate]
    p_shape = [s_shape[0], args.time_alignment]  # padded shape
    o_shape = [s_shape[0] * args.time_alignment]  # output shape
    if len(data.shape) > 1:
        s_shape.extend(data.shape[1:])
        p_shape.extend(data.shape[1:])
        o_shape.extend(data.shape[1:])
    # reshape data s.t. the 1st dimension corresponds to one second
    s_data = np.reshape(data, newshape=s_shape)

    # calculate the padding value
    if args.padding_mode == "zero":
        pad_value = 0
    elif args.padding_mode == "last":
        pad_value = s_data[:, -1, ...]
        pad_value = np.expand_dims(pad_value, axis=1)
    elif args.padding_mode == "average":
        pad_value = np.mean(s_data, axis=1, keepdims=True)
    elif args.padding_mode == "median":
        pad_value = np.median(s_data, axis=1, keepdims=True)
    else:
        raise NotImplementedError(
            f"padding_mode {args.padding_mode} has not been implemented."
        )

    padded_data = np.full(shape=p_shape, fill_value=pad_value, dtype=np.float32)
    padded_data[:, :sampling_rate, ...] = s_data
    padded_data = np.reshape(padded_data, newshape=o_shape)
    return padded_data


def pool(args, data: np.ndarray, sampling_rate: int):
    """
    Downsample channel whose sampling rate is greater than args.time_alignment
    """
    size = data.shape[0] - (data.shape[0] % sampling_rate)
    shape = (
        size // int(sampling_rate / args.time_alignment),
        int(sampling_rate / args.time_alignment),
    )
    if data.ndim > 1:
        shape += (data.shape[-1],)
    data = data[:size]
    new_data = np.reshape(data, newshape=shape)
    # apply pooling on the axis=1
    if args.downsampling == "average":
        new_data = np.mean(new_data, axis=1)
    elif args.downsampling == "max":
        new_data = np.max(new_data, axis=1)
    else:
        raise NotImplementedError(f"unknown downsampling method {args.downsampling}.")
    return new_data


def trim(data: np.ndarray, sampling_rate: int):
    """
    Trim, if necessary, tail of channel whose sampling rate is equal to
    args.time_alignment
    """
    size = data.shape[0] - (data.shape[0] % sampling_rate)
    return data[:size]


def resample(args, data: np.ndarray, sampling_rate: int):
    """
    Resample data so that channels are time aligned based on the required no of
    cycles per second (args.time_alignment)
    """
    ratio = args.time_alignment / sampling_rate
    if ratio > 1:
        new_data = pad(args, data, sampling_rate)
    elif ratio < 1:
        new_data = pool(args, data, sampling_rate)
    else:
        new_data = trim(data, sampling_rate)
    return new_data


def resample_channels(
    args,
    channel_data: t.Dict[str, np.ndarray],
    sampling_rates: t.Dict[str, int],
):
    data_freq = deepcopy(sampling_rates)
    if args.time_alignment:
        for channel, recording in channel_data.items():
            channel_data[channel] = resample(
                args, data=recording, sampling_rate=sampling_rates[channel]
            )
            data_freq[channel] = args.time_alignment
    return data_freq


def preprocess_channel(recording_dir: str, channel: str):
    """
    Load and downsample channel using args.downsampling s.t. each time-step
    corresponds to one second in wall-time
    """
    assert channel in CSV_CHANNELS
    unix_t0, sampling_rate, data = load_channel(
        recording_dir=recording_dir, channel=channel
    )
    # transform to g for acceleration
    if channel == "ACC":
        data = data * 2 / 128
    # despike, apply filter on EDA and TEMP data
    # note: kleckner2018 uses a length of 2057 for a signal sampled at 32Hz,
    # EDA from Empatica E4 is sampled at 4Hz (1/8)
    if channel == "EDA" or channel == "TEMP":
        data = low_pass_filter(recording=data, sampling_rate=sampling_rate)
    if channel not in ("HR", "IBI"):
        # HR begins at t0 + 10s, remove first 10s from channels other than HR
        data = data[sampling_rate * HR_OFFSET :]
    return data, sampling_rate, unix_t0


def find_ibi_gaps(ibi: np.ndarray):
    """
    Find gaps in IBI channel and assign the first column in data to indicate if
    a gap exists
    """
    times, intervals = ibi[:, 0], ibi[:, 1]
    # manually compute the delta between the time of consecutive inter-beat
    deltas = np.zeros_like(intervals)
    deltas[1:] = np.diff(times)
    # compare the manually computed deltas against the recorded intervals
    gaps = np.isclose(deltas, intervals)
    # assign 1 to indicate there is a gap and 0 otherwise
    gaps = np.logical_not(gaps).astype(np.float32)
    ibi = np.concatenate((gaps[:, np.newaxis], ibi), axis=1)
    return ibi


def clean_and_interpolate_ibi(args, timestamps: np.ndarray, durations: np.ndarray):
    """Fill NaN values in data with args.interpolation method"""

    # trim nan values if they appear at the edges (no extrapolation for IBI)
    is_nan = np.where(np.isnan(durations) == 0)
    idx_start, idx_end = is_nan[0][0], is_nan[0][-1] + 1
    timestamps = timestamps[idx_start:idx_end]
    durations = durations[idx_start:idx_end]

    # transform durations from seconds to milliseconds
    d = list(durations * 1000)

    # outliers from signal: remove RR intervals if not 300ms <= RR_duration <= 2000ms
    d_without_outliers = hrvanalysis.remove_outliers(
        rr_intervals=d, low_rri=300, high_rri=2000, verbose=False
    )

    # interpolate durations
    d_interpolated = hrvanalysis.interpolate_nan_values(
        rr_intervals=d_without_outliers, interpolation_method=args.ibi_interpolation
    )

    # remove ectopic beats from signal
    nn_intervals_list = hrvanalysis.remove_ectopic_beats(
        rr_intervals=d_interpolated, method="malik", verbose=False
    )

    # replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = hrvanalysis.interpolate_nan_values(
        rr_intervals=nn_intervals_list
    )

    output = np.zeros(shape=(len(interpolated_nn_intervals), 2), dtype=np.float32)
    output[:, 0] = timestamps
    output[:, 1] = interpolated_nn_intervals
    return output


def fill_ibi_gaps(args, ibi: np.ndarray, t_start: int, t_end: int):
    """
    Remove outliers and ectopic beats, interpolate IBI data (if missing
    values at the edges, these are cropped and corresponding seconds removed
    form all channels) and fill gaps using method specified in
    args.ibi_interpolation
    """
    # crop values that exceed t_end
    if ibi[-1, 1] > t_end:
        ibi = ibi[: np.where(ibi[..., 1] > t_end)[0][0]]

    timestamps, durations = [], []
    # append NaN values if gap exists otherwise the recorded value
    for i in range(ibi.shape[0]):
        if ibi[i, 0]:
            start = floor(timestamps[-1]) + 1 if i else t_start
            end = floor(ibi[i, 1])
            timestamps.extend(list(range(start, end)))
            durations.extend([np.nan] * (end - start))
        else:
            timestamps.append(ibi[i, 1])
            durations.append(ibi[i, 2])
    # append NaN to the end of the IBI recording to match the other channels
    if ceil(timestamps[-1]) < t_end:
        start, end = floor(timestamps[-1]) + 1, t_end + 1
        timestamps.extend(list(range(start, end)))
        durations.extend([np.nan] * (end - start))

    assert len(timestamps) == len(durations)
    assert ceil(timestamps[-1]) == t_end
    timestamps = np.array(timestamps, dtype=np.float32)
    durations = np.array(durations, dtype=np.float32)
    ibi = clean_and_interpolate_ibi(args, timestamps=timestamps, durations=durations)

    return ibi


def compute_hrv_features(args, ibi: np.ndarray):
    """
    Compute heart rate variability from IBI
    reference: https://www.biopac.com/application/ecg-cardiology/advanced-feature/rmssd-for-hrv-analysis/
    """
    assert 1 <= args.hrv_length < args.segment_length
    # HR_OFFSET = first_second if no missing values at the beginning of IBI,
    # otherwise first_second > HR_OFFSET
    start = 0
    times, recordings = ibi[:, 0], ibi[:, 1]
    window_end = max(HR_OFFSET + args.hrv_length, floor(times[0]) + args.hrv_length)

    count, hrv_data = 0, {}
    while times[start] + args.hrv_length <= ceil(times[-1]):
        end = np.where(times > window_end)[0][0]
        features = hrvanalysis.get_time_domain_features(recordings[start:end].tolist())
        update_dict(hrv_data, features)
        window_end += args.hrv_length
        start = end
        count += 1

    # get the remaining time domain features
    features = hrvanalysis.get_time_domain_features(recordings[start:].tolist())
    update_dict(hrv_data, features)
    # compute the remaining size needed
    remaining_length = (ceil(times[-1]) - floor(times[0])) - (count * args.hrv_length)

    repeat = lambda a, reps: np.reshape(
        np.tile(np.expand_dims(a, axis=-1), reps=(1, reps)), newshape=(-1)
    )

    for k, v in hrv_data.items():
        v = np.expand_dims(v, axis=-1)
        hrv_data[k] = np.concatenate(
            [
                repeat(v[:-1], args.hrv_length),
                repeat(v[-1], remaining_length),
            ],
            axis=0,
            dtype=np.float32,
        )
    return hrv_data


def ibi2hrv(args, recording_dir: str, t_start: int, t_end: int):
    """Convert IBI to HRV"""
    _, _, ibi = load_channel(recording_dir=recording_dir, channel="IBI")
    ibi = find_ibi_gaps(ibi)
    ibi = fill_ibi_gaps(args, ibi=ibi, t_start=t_start, t_end=t_end)
    hrv_data = compute_hrv_features(args, ibi=ibi)
    return hrv_data, floor(ibi[0, 0]), ceil(ibi[-1, 0])


def remove_zeros(recordings: np.ndarray, threshold: int = 5) -> t.List[np.ndarray]:
    """
    Remove recordings where all channels contain 0s for longer than threshold
    time-steps
    Args:
      recordings: np.ndarray
      threshold: int, the threshold (in time-steps) where channels can contain 0s
    Return:
      features: np.ndarray, filtered recordings where 0 index one continuous recording
    """
    assert 0 < threshold < recordings.shape[0]
    sums = np.sum(np.abs(recordings), axis=-1)
    features = []
    start, end = 0, 0
    while end < sums.shape[0]:
        if sums[end] == 0:
            current = end
            while end < sums.shape[0] and sums[end] == 0:
                end += 1
            if end - current >= threshold:
                features.append(recordings[start:current, ...])
                start = end
        end += 1
    if start + 1 < end:
        features.append(recordings[start:end, ...])
    return features


def process_ibi(
    args,
    recording_dir: str,
    channel_data: t.Dict[str, np.ndarray],
    min_duration: int,
):
    # HR_OFFSET was cropped from start of channels != IBI so in order to
    # get the last second in real time HR_OFFSET should be added to min_length
    hrv_data, first_ibi_second, last_ibi_second = ibi2hrv(
        args,
        recording_dir=recording_dir,
        t_start=HR_OFFSET,
        t_end=min_duration + HR_OFFSET,
    )

    if args.hrv_features == "all":
        args.hrv_features = HRV_FEATURES
    for hrv_features in args.hrv_features:
        channel_data[f"HRV_{hrv_features}"] = np.repeat(
            hrv_data[hrv_features], repeats=args.time_alignment, axis=0
        )

    # IBI missing values at the edges (if any) are not extrapolated
    # the corresponding seconds should therefore be removed from channels != IBI
    idx_start = 0
    if first_ibi_second > HR_OFFSET:
        non_ibi_leading_seconds_to_crop = int(first_ibi_second - HR_OFFSET)
        idx_start = non_ibi_leading_seconds_to_crop * args.time_alignment

    idx_end = int(min_duration * args.time_alignment)
    if min_duration + HR_OFFSET > last_ibi_second:
        non_ibi_trailing_seconds_to_crop = int(
            (min_duration + HR_OFFSET) - last_ibi_second
        )
        trailing_row_to_drop = non_ibi_trailing_seconds_to_crop * args.time_alignment
        idx_end = int((min_duration * args.time_alignment) - trailing_row_to_drop)

    for c in channel_data.keys():
        if not c.startswith("HRV"):
            channel_data[c] = channel_data[c][idx_start:idx_end]


def normalize(channel_data: t.Dict[str, np.ndarray], session_info: dict):
    for channel, recording in channel_data.items():
        channel_data[channel] = utils.normalize(
            recording,
            x_min=session_info["min"][channel],
            x_max=session_info["max"][channel],
        )


def preprocess_dir(args, recording_dir: str, session_id: int):
    """
    Preprocess channels in recording_dir and return the preprocessed features
    and corresponding label obtained from spreadsheet.
    Returns:
      features: np.ndarray, preprocessed channels in SAVE_CHANNELS format
    """
    durations, channel_data, sampling_rates, unix_t0s = [], {}, {}, {}
    # load and preprocess all channels except IBI
    for channel in CSV_CHANNELS:
        if channel != "IBI":
            channel_data[channel], sampling_rate, unix_t0 = preprocess_channel(
                recording_dir=recording_dir, channel=channel
            )
            durations.append(len(channel_data[channel]) // sampling_rate)
            sampling_rates[channel] = sampling_rate
            unix_t0s[channel] = unix_t0
        else:
            channel_data[channel], _, unix_t0 = preprocess_channel(
                recording_dir=recording_dir, channel=channel
            )
            unix_t0s[channel] = unix_t0
    # all channels should have the same durations, but as a failsafe, crop
    # each channel to the shortest duration
    min_duration = min(durations)
    for channel, recording in channel_data.items():
        if channel != "IBI":
            channel_data[channel] = recording[: min_duration * sampling_rates[channel]]

    split_acceleration(channel_data=channel_data, sampling_rates=sampling_rates)

    # quality control recording
    if args.qc_mode == 0:
        pass
    elif args.qc_mode == 1:
        filtered_out_percentage = filter_data.quality_control(
            channel_data=channel_data,
            sampling_rates=sampling_rates,
            verbose=args.verbose,
        )
    else:
        raise NotImplementedError(f"QC mode {args.qc_mode} has not been implemented.")

    channel_freq = resample_channels(
        args, channel_data=channel_data, sampling_rates=sampling_rates
    )

    session_info = {
        "channel_names": utils.get_channel_names(channel_data),
        "sampling_rates": sampling_rates,
        "channel_freq": channel_freq,
        "unix_t0": unix_t0s,
    }

    try:
        session_info.update(
            {
                "min": {k: np.nanmin(v, axis=0) for k, v in channel_data.items()},
                "max": {k: np.nanmax(v, axis=0) for k, v in channel_data.items()},
                "mean": {k: np.nanmean(v, axis=0) for k, v in channel_data.items()},
                "std": {k: np.nanstd(v, axis=0) for k, v in channel_data.items()},
            }
        )
        if args.qc_mode == 1:
            session_info["filtered_out_percentage"] = filtered_out_percentage
    except RuntimeWarning as e:
        print(f"Session {session_id} warning: {e}")

    return channel_data, session_info


def extract_features(args, session_data: t.Dict, num_segments: int, unix_t0: t.Dict):
    timestamps_beats = pd.to_datetime(
        session_data["IBI"][:, 0] + unix_t0["IBI"], unit="s", origin="unix"
    )
    features_container = []
    warnings.filterwarnings(action="ignore", category=UserWarning)

    for i in range(num_segments):
        # EDA
        eda = pd.DataFrame(
            data=session_data["EDA"][i],
            columns=["eda"],
            dtype=np.float64,
        )
        eda_timestamps = pd.to_datetime(
            session_data["unix_time"][i][0], unit="s", origin="unix"
        ) + np.arange(len(eda)) * datetime.timedelta(seconds=CHANNELS_FREQ["EDA"] ** -1)
        eda = eda.set_index(pd.DatetimeIndex(data=eda_timestamps, name="datetime"))
        try:
            eda_features = flirt.eda.get_eda_features(
                data=eda["eda"],
                data_frequency=CHANNELS_FREQ["EDA"],
                window_length=math.ceil((eda.index[-1] - eda.index[0]).total_seconds()),
                window_step_size=math.ceil(
                    (eda.index[-1] - eda.index[0]).total_seconds()
                ),
            )
            if not eda_features.shape[-1] == len(FLIRT_EDA):
                eda_features = np.empty(shape=[1, len(FLIRT_HRV)])
                eda_features.fill(np.nan)
                eda_features = pd.DataFrame(data=eda_features, columns=FLIRT_EDA)
        except:
            eda_features = np.empty(shape=[1, len(FLIRT_HRV)])
            eda_features.fill(np.nan)
            eda_features = pd.DataFrame(data=eda_features, columns=FLIRT_EDA)

        # ACC
        acc = pd.DataFrame(
            data=np.concatenate(
                [
                    np.expand_dims(axis, axis=1)
                    for axis in [
                        session_data["ACC_x"][i],
                        session_data["ACC_y"][i],
                        session_data["ACC_z"][i],
                    ]
                ],
                axis=1,
            ),
            columns=["acc_x", "acc_y", "acc_z"],
            dtype=np.float64,
        )
        # reverse transformation to g values
        acc = (acc * 128) / 2
        acc_timestamps = pd.to_datetime(
            session_data["unix_time"][i][0], unit="s", origin="unix"
        ) + np.arange(len(acc)) * datetime.timedelta(
            seconds=CHANNELS_FREQ["ACC_x"] ** -1
        )
        acc = acc.set_index(pd.DatetimeIndex(data=acc_timestamps, name="datetime"))
        try:
            acc_features = flirt.acc.get_acc_features(
                data=acc,
                data_frequency=CHANNELS_FREQ["ACC_x"],
                window_length=math.ceil((acc.index[-1] - acc.index[0]).total_seconds()),
                window_step_size=math.ceil(
                    (acc.index[-1] - acc.index[0]).total_seconds()
                ),
            )
            if not acc_features.shape[-1] == len(FLIRT_ACC):
                acc_features = np.empty(shape=[1, len(FLIRT_ACC)])
                acc_features.fill(np.nan)
                acc_features = pd.DataFrame(data=acc_features, columns=FLIRT_ACC)
        except:
            acc_features = np.empty(shape=[1, len(FLIRT_ACC)])
            acc_features.fill(np.nan)
            acc_features = pd.DataFrame(data=acc_features, columns=FLIRT_ACC)

        # HRV
        if args.from_bvp2ibi_mode == 0:
            segment_start = datetime.datetime.fromtimestamp(
                session_data["unix_time"][i][0]
            )
            segment_end = datetime.datetime.fromtimestamp(
                session_data["unix_time"][i][-1]
            )
            ibi_segment = timestamps_beats[
                (timestamps_beats >= segment_start) & (timestamps_beats <= segment_end)
            ]
            ibi = np.around(np.diff(ibi_segment).astype(np.int64) / 10**6, decimals=3)
            df_ibi = pd.DataFrame(data=ibi, columns=["ibi"]).set_index(
                pd.DatetimeIndex(data=ibi_segment[1:], tz="UTC", name="datetime")
            )
        else:
            # Signal time axis reference (seconds):
            # https://biosppy.readthedocs.io/en/stable/biosppy.signals.html#biosppy-signals-bvp
            ts, filtered, onsets, heart_rate_ts, heart_rate = biosppy.signals.bvp.bvp(
                signal=session_data["BVP"][i],
                sampling_rate=CHANNELS_FREQ["BVP"],
                show=False,
            )
            # interpulse interval, pulse rate variability:
            # https://www.kubios.com/hrv-time-series/
            ipi = np.diff(ts[onsets]) * 1000
            ipi_timestamps = pd.to_datetime(
                session_data["unix_time"][i][0], unit="s", origin="unix"
            ) + np.array([datetime.timedelta(milliseconds=ms) for ms in ipi])
            df_ibi = pd.DataFrame(data=ipi, columns=["ibi"])
            df_ibi = df_ibi.set_index(
                pd.DatetimeIndex(data=ipi_timestamps, name="datetime")
            )
        try:
            hrv_features = flirt.hrv.get_hrv_features(
                data=df_ibi["ibi"],
                window_length=args.segment_length,
                window_step_size=args.segment_length,
                domains=["td", "fd", "nl", "stat"],
            )
            if not hrv_features.shape[-1] == len(FLIRT_HRV):
                hrv_features = np.empty(shape=[1, len(FLIRT_HRV)])
                hrv_features.fill(np.nan)
                hrv_features = pd.DataFrame(data=hrv_features, columns=FLIRT_HRV)
        except:
            hrv_features = np.empty(shape=[1, len(FLIRT_HRV)])
            hrv_features.fill(np.nan)
            hrv_features = pd.DataFrame(data=hrv_features, columns=FLIRT_HRV)

        features_container.append(
            np.concatenate(
                (eda_features.values, acc_features.values, hrv_features.values), axis=1
            )
        )

    session_data["FLIRT"] = np.concatenate(features_container, axis=0)
