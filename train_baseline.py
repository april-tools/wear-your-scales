import argparse
import shutil
import typing as t
from time import time

from imblearn.over_sampling import RandomOverSampler
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from timebase.data.reader import get_datasets
from timebase.data.static import *
from timebase.utils import utils, yaml
from timebase.utils.utils import get_sklearn_classifier


def random_search(
    args,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    search_space: t.Dict,
):
    x_train = x_train.copy()
    y_train = y_train.copy()
    x_val = x_val.copy()
    y_val: y_val.copy()
    # Find columns that contain all np.nan values
    all_nan_columns = np.where(np.all(np.isnan(x_train), axis=0) == True)[0]
    # Drop columns with all np.nan values
    x_train = x_train[:, ~all_nan_columns]
    x_val = x_val[:, ~all_nan_columns]
    # Set non-finite values to np.nan
    x_train = np.where(np.isinf(x_train), np.nan, x_train)
    x_val = np.where(np.isinf(x_val), np.nan, x_val)
    # Mean value imputation
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    x_train = imp.fit_transform(x_train)
    x_val = imp.transform(x_val)
    # Random Over-sampling
    ros = RandomOverSampler(random_state=args.seed)
    x_train, y_train = ros.fit_resample(x_train, y_train)

    validation_scores = []
    model_settings = []
    for _ in range(args.n_iter_search):
        # randomly sample model setting
        model_setting = {}
        for k, v in search_space.items():
            if isinstance(v, list):
                model_setting[k] = np.random.choice(v)
            else:
                model_setting[k] = v.rvs()

        model = get_sklearn_classifier(args, model_name, model_setting)
        model.fit(X=x_train, y=y_train)
        y_pred = model.predict(x_val)
        qck = cohen_kappa_score(y1=y_val, y2=y_pred, weights="quadratic")
        validation_scores.append(qck)
        model_settings.append(model_setting)

    # run best model on test set
    best_index = np.argmax(validation_scores)
    validation_score = validation_scores[best_index]
    best_setting = model_settings[best_index]
    best_model = get_sklearn_classifier(args, model=model_name, setting=best_setting)

    return validation_score, best_model


def test(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    best_model,
):
    x_train = x_train.copy()
    y_train = y_train.copy()
    x_val = x_val.copy()
    y_val: y_val.copy()
    x_test = x_test.copy()
    y_test = y_test.copy()
    # Find columns that contain all np.nan values
    all_nan_columns = np.where(np.all(np.isnan(x_train), axis=0) == True)[0]
    # Drop columns with all np.nan values
    x_train = x_train[:, ~all_nan_columns]
    x_val = x_val[:, ~all_nan_columns]
    x_test = x_test[:, ~all_nan_columns]
    # Set non-finite values to np.nan
    x_train = np.where(np.isinf(x_train), np.nan, x_train)
    x_val = np.where(np.isinf(x_val), np.nan, x_val)
    x_test = np.where(np.isinf(x_test), np.nan, x_test)
    # Mean value imputation
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    x_train = imp.fit_transform(x_train)
    x_val = imp.transform(x_val)
    x_test = imp.transform(x_test)

    # Fit best model selected from validation set on train/validation set
    x_train, y_train = np.concatenate((x_train, x_val), axis=0), np.concatenate(
        (y_train, y_val), axis=0
    )
    best_model.fit(x_train, y_train)

    y_pred = best_model.predict(x_test)
    test_score = cohen_kappa_score(y1=y_test, y2=y_pred, weights="quadratic")

    return test_score


def main(args, wandb_sweep: bool = False):
    utils.set_random_seed(args.seed, verbose=args.verbose)

    if args.clear_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    segment_lengths = [2**n for n in np.arange(3, 11)]
    segment_lengths = [64]
    # define search space specific to sklearn models
    random_forest = {
        "n_estimators": stats.randint(10, 1000),
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"],
    }
    enet = {"l1_ratio": stats.uniform(0, 1), "alpha": stats.loguniform(1e-4, 1)}

    search_spaces = {"random_forest": random_forest, "enet": enet}

    start = time()
    res, models_by_sl_and_item = {}, {}
    for segment_length in tqdm(
        segment_lengths, desc="Segment length:  ", disable=args.verbose == 0
    ):
        if args.verbose:
            print(f"{segment_length} seconds")
        args.dataset = os.path.join(args.path2preprocessed, f"ta0_sl{segment_length}")
        assert os.path.isdir(args.dataset), f"{args.dataset} not found."
        datasets = get_datasets(args)
        res[segment_length] = []
        models_by_sl_and_item[segment_length] = {}
        if args.verbose:
            print(f"segment length {segment_length}")
        for item in tqdm(args.selected_items, desc="Item ", disable=args.verbose == 0):
            if args.verbose:
                print(item)
            validation_score, best_model = random_search(
                args,
                x_train=datasets["x_train"].values,
                y_train=datasets["y_train"][item].values,
                x_val=datasets["x_val"].values,
                y_val=datasets["y_val"][item].values,
                model_name=args.model,
                search_space=search_spaces[args.model],
            )
            res[segment_length].append(validation_score)
            models_by_sl_and_item[segment_length][item] = best_model
            if args.verbose == 2:
                print(f"validation QCK = {validation_score:.03f} on {item}")
        if args.verbose:
            print(
                f"Average validation QCK for segment length {segment_length}:"
                f" {np.mean(res[segment_length]):.03f}"
            )

    # find segment length associated with the highest validation average QCK
    highest_mean_qck = float("-inf")
    highest_mean_qck_segment_length = None
    for key, value in res.items():
        mean = np.mean(value)
        if mean > highest_mean_qck:
            highest_mean_qck = mean
            highest_mean_qck_segment_length = key

    # load dataset
    args.dataset = os.path.join(
        args.path2preprocessed, f"ta0_sl{highest_mean_qck_segment_length}"
    )
    datasets = get_datasets(args)
    # test
    test_res = {}
    for item in args.selected_items:
        test_score = test(
            x_train=datasets["x_train"].values,
            y_train=datasets["y_train"][item].values,
            x_val=datasets["x_val"].values,
            y_val=datasets["y_val"][item].values,
            x_test=datasets["x_test"].values,
            y_test=datasets["y_test"][item].values,
            best_model=models_by_sl_and_item[highest_mean_qck_segment_length][item],
        )
        test_res[item] = test_score

    yaml.save(
        filename=os.path.join(f"{args.output_dir}", "results.yaml"), data=test_res
    )
    print(f"Test item average QCK: {np.mean(list(test_res.values())):.03f}")

    end = time()
    print(f"Elapse: {(end - start) // 60:.02f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training configuration
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--num_workers", type=int, default=2, help="number of workers for DataLoader"
    )
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument(
        "--channel2drop",
        type=str,
        default=None,
        help="set channel to drop, defaults to None (used for leave-one-out"
        "channel importance)",
    )

    # dataset configuration
    parser.add_argument(
        "--path2preprocessed",
        type=str,
        required=True,
        help="path to directory where preprocessed datasets are stored",
    )
    parser.add_argument(
        "--hours2keep",
        type=float,
        default=5,
        help="if > 0 only the first hours2keep hours of a session are kept for "
        "analyses and when a session has fewer than hours2keep it is dropped, "
        "otherwise all recorded time is used.",
    )
    parser.add_argument(
        "--status_selection",
        type=str,
        default="mood_disorders",
        choices=[
            "unfiltered",
            "exclude_hc",
            "mood_disorders",
            "ongoing_mood_disorders",
        ],
        help="filter data based on status"
        "unfiltered: all recordings are used irrespective of status"
        "exclude_hc: HCs are excluded from analyses"
        "mood_disorders: only patients with a mood disorder diagnosis are used "
        "in analyses"
        "ongoing_mood_disorders: only patients with an ongoing mood episode "
        "(i.e. no euthymia) are used in analyses",
    )
    parser.add_argument(
        "--task_mode",
        type=int,
        default=4,
        choices=[4],
        help="task mode: baseline",
    )
    parser.add_argument(
        "--imb_mode_seed",
        type=int,
        default=123,
        help="seed for data-level imbalanced learning",
    )
    parser.add_argument(
        "--hdrs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        help="HDRS items: "
        "0 drop all HDRS items from target"
        "[1:17] item(s) to be included in target",
    )
    parser.add_argument(
        "--ymrs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        help="YMRS items: "
        "0 drop all YMRS items from target"
        "[1:11] item(s) to be included in target",
    )
    parser.add_argument(
        "--split_mode",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="criterion for train/val/test split:"
        "0) partition data at random into 70:15:15 train:validation:test"
        "1) split each session into 70:15:15 train:validation:test along the "
        "temporal dimension -> in-sample"
        "2) split each session into 70:15:15 train:validation:test after "
        "shuffling segments (so that  (unlike 1) temporal order between "
        "consecutive segments is broken)"
        "3) 70:15:15 train:validation:test splits are created in such a way "
        "that each subject is contained in one set only -> out-of-sample",
    )

    # matplotlib
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument(
        "--format", type=str, default="svg", choices=["pdf", "png", "svg"]
    )
    parser.add_argument("--dpi", type=int, default=120)
    # misc
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--clear_output_dir", action="store_true")

    # model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="enet",
        help="choose which sklearn model should be used",
    )
    parser.add_argument(
        "--n_iter_search",
        type=int,
        default=20,
        help="no. algorithm specific hp to search",
    )

    main(parser.parse_args())
