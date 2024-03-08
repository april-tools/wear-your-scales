import collections
import copy
import csv
import random
import subprocess
import typing as t
from copy import deepcopy

import pandas as pd
import torch
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from torch import nn

from timebase.data.static import *
from timebase.utils import yaml


def set_random_seed(seed: int, deterministic: bool = False, verbose: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if verbose > 2:
        print(f"set random seed: {seed}")


def get_device(args):
    """Get the appropriate torch.device from args.device argument"""
    device = args.device
    if not device:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            # allow TensorFloat32 computation
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = "mps"
    args.device = torch.device(device)


def wandb_init(args, wandb_sweep: bool):
    """initialize wandb and strip information from args"""
    os.environ["WANDB_SILENT"] = "true"
    if not wandb_sweep:
        try:
            config = deepcopy(args.__dict__)
            config.pop("ds_info", None)
            config.pop("channel2drop", None)
            config.pop("discarded_sessions", None)
            config.pop("input_shapes", None)
            config.pop("selected_items", None)
            config.pop("hdrs", None)
            config.pop("ymrs", None)
            config.pop("imb_mode_item_focus", None)
            config.pop("output_dir", None)
            config.pop("device", None)
            config.pop("format", None)
            config.pop("dpi", None)
            config.pop("save_plots", None)
            config.pop("plot_mode", None)
            config.pop("save_predictions", None)
            config.pop("verbose", None)
            config.pop("use_wandb", None)
            config.pop("wandb_group", None)
            config.pop("reuse_stats", None)
            config.pop("clear_output_dir", None)
            wandb.init(
                config=config,
                dir=os.path.join(args.output_dir, "wandb"),
                project="timebase",
                entity="timebase",
                group=args.wandb_group,
                name=os.path.basename(args.output_dir),
            )
        except AssertionError as e:
            print(f"wandb.init error: {e}\n")
            args.use_wandb = False


def update_dict(target: t.Dict, source: t.Dict, replace: bool = False):
    """add or update items in source to target"""
    for key, value in source.items():
        if replace:
            target[key] = value
        else:
            if key not in target:
                target[key] = []
            target[key].append(value)


def check_output(command: list):
    """Execute command in subprocess and return output as string"""
    return subprocess.check_output(command).strip().decode()


def save_args(args):
    """Save args object as dictionary to output_dir/args.yaml"""
    """Save args object as dictionary to args.output_dir/args.json"""
    arguments = copy.deepcopy(args.__dict__)
    try:
        arguments["git_hash"] = check_output(["git", "describe", "--always"])
        arguments["hostname"] = check_output(["hostname"])
    except subprocess.CalledProcessError as e:
        if args.verbose:
            print(f"Unable to call subprocess: {e}")
    yaml.save(filename=os.path.join(args.output_dir, "args.yaml"), data=arguments)


def load_args(args, replace: bool = False):
    """Load args from output_dir/args.yaml"""
    filename = os.path.join(args.output_dir, "args.yaml")
    assert os.path.exists(filename)
    arguments = yaml.load(filename=filename)
    for key, value in arguments.items():
        if (replace == True) or (not hasattr(args, key)):
            setattr(args, key, value)


def load_args_oos(args):
    """Load args from output_dir/args.yaml"""
    filename = os.path.join(args.output_dir, "args.yaml")
    assert os.path.exists(filename)
    arguments = yaml.load(filename=filename)
    for key, value in arguments.items():
        if not hasattr(args, key) and key not in [
            "class2name",
            "class2session",
            "session2class",
            "train_steps",
            "val_steps",
            "test_steps",
            "ds_info",
        ]:
            setattr(args, key, value)


def write_csv(output_dir, content: list):
    with open(os.path.join(output_dir, "results.csv"), "a") as file:
        writer = csv.writer(file)
        writer.writerow(content)


def to_numpy(a: t.Union[torch.Tensor, np.ndarray]):
    return a.cpu().numpy() if torch.is_tensor(a) else a


class BufferDict(nn.Module):
    """Holds buffers in a dictionary.

    Reference: https://botorch.org/api/utils.html#botorch.utils.torch.BufferDict

    BufferDict can be indexed like a regular Python dictionary, but buffers it
    contains are properly registered, and will be visible by all Module methods.

    :class:`~torch.nn.BufferDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.BufferDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.BufferDict` (the argument to
      :meth:`~torch.nn.BufferDict.update`).

    Note that :meth:`~torch.nn.BufferDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Args:
        buffers (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.Tensor`) or an iterable of key-value pairs
            of type (string, :class:`~torch.Tensor`)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.buffers = nn.BufferDict({
                        'left': torch.randn(5, 10),
                        'right': torch.randn(5, 10)
                })

            def forward(self, x, choice):
                x = self.buffers[choice].mm(x)
                return x
    """

    def __init__(self, buffers=None):
        r"""
        Args:
            buffers: A mapping (dictionary) from string to :class:`~torch.Tensor`, or
                an iterable of key-value pairs of type (string, :class:`~torch.Tensor`).
        """
        super(BufferDict, self).__init__()
        if buffers is not None:
            self.update(buffers)

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, buffer):
        self.register_buffer(key, buffer)

    def __delitem__(self, key):
        del self._buffers[key]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        """Remove all items from the BufferDict."""
        self._buffers.clear()

    def pop(self, key):
        r"""Remove key from the BufferDict and return its buffer.

        Args:
            key (string): key to pop from the BufferDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the BufferDict keys."""
        return self._buffers.keys()

    def items(self):
        r"""Return an iterable of the BufferDict key/value pairs."""
        return self._buffers.items()

    def values(self):
        r"""Return an iterable of the BufferDict values."""
        return self._buffers.values()

    def update(self, buffers):
        r"""Update the :class:`~torch.nn.BufferDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`buffers` is an ``OrderedDict``, a :class:`~torch.nn.BufferDict`,
            or an iterable of key-value pairs, the order of new elements in it is
            preserved.

        Args:
            buffers (iterable): a mapping (dictionary) from string to
                :class:`~torch.Tensor`, or an iterable of
                key-value pairs of type (string, :class:`~torch.Tensor`)
        """
        if not isinstance(buffers, collections.abc.Iterable):
            raise TypeError(
                "BuffersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(buffers).__name__
            )

        if isinstance(buffers, collections.abc.Mapping):
            if isinstance(buffers, (collections.OrderedDict, BufferDict)):
                for key, buffer in buffers.items():
                    self[key] = buffer
            else:
                for key, buffer in sorted(buffers.items()):
                    self[key] = buffer
        else:
            for j, p in enumerate(buffers):
                if not isinstance(p, collections.abc.Iterable):
                    raise TypeError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._buffers.items():
            size_str = "x".join(str(size) for size in p.size())
            device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
            parastr = "Buffer containing: [{} of size {}{}]".format(
                torch.typename(p), size_str, device_str
            )
            child_lines.append("  (" + k + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError("BufferDict should not be called.")


def create_young_hamilton_labels(args, y: t.Dict[str, np.ndarray]):
    ymrs_sum_binned = pd.cut(
        np.sum(
            np.concatenate(
                [
                    np.expand_dims(y[col], axis=1)
                    for col in args.selected_items
                    if "YMRS" in col
                ],
                axis=1,
            ),
            axis=1,
        ),
        # https://pubmed.ncbi.nlm.nih.gov/19624385/
        bins=[
            0,
            7,
            14,
            25,
            60,
        ],
        include_lowest=True,
        labels=False,
    )
    hdrs_sum_binned = pd.cut(
        np.sum(
            np.concatenate(
                [
                    np.expand_dims(y[col], axis=1)
                    for col in args.selected_items
                    if "HDRS" in col
                ],
                axis=1,
            ),
            axis=1,
        ),
        # https://pubmed.ncbi.nlm.nih.gov/19624385/
        bins=[
            0,
            7,
            14,
            23,
            52,
        ],  # [0, 7, 14, 23, 52] <- https://en.wikipedia.org/wiki/Hamilton_Rating_Scale_for_Depression,
        include_lowest=True,
        labels=False,
    )
    return (
        np.array(
            pd.Series(
                [
                    f"young{str(young)}_hamilton{str(ham)}"
                    for young, ham in zip(ymrs_sum_binned, hdrs_sum_binned)
                ]
            ).replace(YOUNG_HAMILTON_DICT)
        ),
        ymrs_sum_binned,
        hdrs_sum_binned,
    )


def get_sklearn_classifier(args, model: str, setting: dict):
    """Initialize and return classifier for the given case"""
    model = model.lower()
    match model:
        case "enet":
            return SGDClassifier(**setting, loss="log_loss", penalty="elasticnet")
        case "random_forest":
            return RandomForestClassifier(**setting, random_state=args.seed)
        case _:
            raise NotImplementedError(
                f"model {model} not implemented, please choose one of "
                f"[enet, random_forest]"
            )
