import argparse
import datetime
import os
import pickle
import time
import typing as t
from functools import partial
from shutil import rmtree

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import concurrent

from timebase.data import filter_data, preprocessing, spreadsheet, utils
from timebase.data.static import *
from timebase.utils import h5
from timebase.utils.utils import set_random_seed


def get_session_label(clinical_info: pd.DataFrame, session_id: int):
    session = clinical_info[clinical_info.Session_Code == session_id]
    return None if session.empty else session.values[0].astype(np.float32)


def preprocess_session(args, session_id: int, clinical_info: pd.DataFrame):
    recording_dir = utils.unzip_session(args.data_dir, session_id=session_id)
    session_label = get_session_label(clinical_info, session_id=session_id)
    if session_label is None:
        raise ValueError(f"Cannot find session {session_id} in spreadsheet.")

    session_data, session_info = preprocessing.preprocess_dir(
        args, recording_dir=recording_dir, session_id=session_id
    )

    session_data, num_segments = preprocessing.segmentation(
        args,
        session_data=session_data,
        channel_freq=session_info["channel_freq"],
        unix_t0=session_info["unix_t0"],
    )

    if not num_segments:
        raise ValueError(f"Session {session_id} has no valid segments.")

    # TODO put back
    # preprocessing.extract_features(
    #     args,
    #     session_data=session_data,
    #     num_segments=num_segments,
    #     unix_t0=session_info["unix_t0"],
    # )

    session_output_dir = os.path.join(args.output_dir, str(session_id))
    if not os.path.isdir(session_output_dir):
        os.makedirs(session_output_dir)

    del session_data["IBI"]
    unix_t0_segments = (session_data["unix_time"][:, 0]).astype("uint32")
    del session_data["unix_time"]
    session_paths = []
    for n in range(num_segments):
        filename = os.path.join(session_output_dir, f"{n}.h5")
        segment = {k: v[n] for k, v in session_data.items()}
        h5.write(filename=filename, content=segment, overwrite=True)
        session_paths.append(filename)

    session_paths = np.array(session_paths, dtype=str)
    session_labels = np.concatenate(
        (
            np.tile(session_label, reps=(num_segments, 1)),
            unix_t0_segments[..., np.newaxis],
        ),
        axis=1,
    )
    return {"paths": session_paths, "labels": session_labels, "info": session_info}


def preprocess_wrapper(session_id: int, args, clinical_info: pd.DataFrame):
    try:
        results = preprocess_session(
            args, session_id=session_id, clinical_info=clinical_info
        )
    except ValueError as e:
        print(e)
        return None
    return results


def main(args):
    starting_time = time.time()
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir {args.data_dir} not found.")
    if os.path.isdir(args.output_dir):
        if args.overwrite:
            rmtree(args.output_dir)
        else:
            raise FileExistsError(
                f"output_dir {args.output_dir} already exists. Add --overwrite "
                f" flag to overwrite the existing preprocessed data."
            )
    os.makedirs(args.output_dir)

    set_random_seed(args.seed)

    clinical_info = spreadsheet.read(args)
    args.session_codes = list(clinical_info["Session_Code"])

    print(f"\nPreprocessing data from {args.data_dir}...")

    clinical_info.replace({"status": DICT_STATE}, inplace=True)
    clinical_info.replace({"time": DICT_TIME}, inplace=True)

    ds_info = {
        "time_alignment": args.time_alignment,
        "downsampling": args.downsampling,
        "padding_mode": args.padding_mode,
        "qc_mode": args.qc_mode,
        "ibi_interpolation": args.ibi_interpolation,
        "hrv_features": args.hrv_features,
        "hrv_length": args.hrv_length,
        "segment_length": args.segment_length,
    }

    results = concurrent.process_map(
        partial(preprocess_wrapper, args=args, clinical_info=clinical_info),
        args.session_codes,
        max_workers=args.num_workers,
        desc="Preprocessing",
    )

    sessions_paths, sessions_labels, invalid_sessions = [], [], []
    sessions_info = {}
    for i, session_id in enumerate(args.session_codes):
        result = results[i]
        # result = preprocess_session(
        #     args, session_id=session_id, clinical_info=clinical_info
        # )
        if result is None:
            invalid_sessions.append(session_id)
            continue
        sessions_paths.append(result["paths"])
        sessions_labels.append(result["labels"])
        for info_name in ["channel_names", "channel_freq", "sampling_rates"]:
            if info_name not in ds_info:
                ds_info[info_name] = result["info"][info_name]
            del result["info"][info_name]
        sessions_info[session_id] = result["info"]

    # joint features and labels from all sessions
    sessions_paths = np.concatenate(sessions_paths, axis=0)
    sessions_labels = np.concatenate(sessions_labels, axis=0)

    # define recording IDs in sessions with multiple recordings
    filter_data.set_unique_recording_id(sessions_labels)

    ds_info["sessions_info"] = sessions_info
    if hasattr(args, "extracted_features_names"):
        ds_info["extracted_features_names"] = args.extracted_features_names

    with open(os.path.join(args.output_dir, "info.pkl"), "wb") as file:
        pickle.dump(
            {
                "data_paths": sessions_paths,
                "labels": sessions_labels,
                "ds_info": ds_info,
                "clinical_info": clinical_info,
                "invalid_sessions": invalid_sessions,
            },
            file,
        )

    print(f"Saved processed data to {args.output_dir}")
    runtime = round(
        datetime.timedelta(seconds=time.time() - starting_time).total_seconds()
    )
    print(f"Runtime: {runtime} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw_data",
        help="path to directory with raw data in zip files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to directory to store dataset",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing preprocessed directory",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[1, 2])

    # preprocessing configuration
    parser.add_argument(
        "--downsampling",
        type=str,
        default="average",
        choices=["average", "max"],
        help="downsampling method to use",
    )
    parser.add_argument(
        "--time_alignment",
        type=int,
        required=True,
        choices=[0, 1, 2, 4, 8, 16, 32, 64],
        help="number of samples per second (Hz) for time-alignment, "
        "set 0 to train embedding layers instead.",
    )
    parser.add_argument(
        "--padding_mode",
        type=str,
        default="average",
        choices=["zero", "last", "average", "median"],
        help="padding mode for channels samples at a lower frequency",
    )
    parser.add_argument(
        "--qc_mode",
        type=int,
        default=1,
        choices=[0, 1],
        help="quality control mode:"
        "0 - no QC"
        "1 - Kleckner et al. 2018 - https://pubmed.ncbi.nlm.nih.gov/28976309/",
    )
    parser.add_argument(
        "--ibi_interpolation",
        type=str,
        default="quadratic",
        choices=["linear", "quadratic"],
        help="interpolation method to use in IBI channel",
    )
    parser.add_argument(
        "--hrv_features",
        nargs="+",
        default=[],
        help="choose which HRV features should be extracted from IBI",
    )
    parser.add_argument(
        "--hrv_length",
        type=int,
        default=60 * 5,
        help="window length for computing HRV from IBI",
    )
    parser.add_argument(
        "--from_bvp2ibi_mode",
        type=int,
        default=0,
        choices=[0, 1],
        help=""
        "0) Use Empatica IBI (provided as part of the E4 output and "
        "derived through a propriety algorithm. "
        "1) Compute IBI from BVP with bioppsy open-source package",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=2**9,
        help="segmentation window length in seconds",
    )
    parser.add_argument(
        "--downsample_mode",
        type=int,
        default=1,
        choices=[0, 1],
        help="0) no downsampling, 1) downsample segments from majority class",
    )

    parser.add_argument("--num_workers", type=int, default=6)

    main(parser.parse_args())
