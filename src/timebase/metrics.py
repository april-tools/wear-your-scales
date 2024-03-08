import typing as t

import torch
from torchmetrics import functional as F

from timebase.data.static import *
from timebase.utils.utils import BufferDict

distance_metric_func = {
    "mae": F.mean_absolute_error,
    "mse": F.mean_squared_error,
}
_DIST_TYPES = t.Literal["mae", "mse"]
_AVG_TYPES = t.Literal["micro", "macro", "weighted"]


def mean_tensor(l: t.List[torch.Tensor], device: torch.device):
    return torch.mean(
        torch.tensor(l, device=device, dtype=torch.float32), dtype=torch.float32
    )


def compute_quadratic_cohen_kappa(
    outputs: t.Dict[str, t.List[torch.Tensor]],
    labels: t.Dict[str, t.List[torch.Tensor]],
):
    """
    Compute per item Quadratically Weighted Cohen's Kappa, requires all set not just batches from it
    """
    kappas = {}
    for k in outputs.keys():
        kappas[k] = F.cohen_kappa(
            preds=torch.concat(outputs[k], dim=0),
            target=torch.concat(labels[k]),
            task="multiclass",
            weights="quadratic",
            num_classes=ITEM_RANKS[k],
        )
    device = list(outputs.values())[0][0].device
    kappa_dict = {}
    kappa_dict["metrics/overall/kappa"] = mean_tensor(
        l=[v for k, v in kappas.items()], device=device
    )

    if len(labels) == 28:
        kappa_dict["metrics/HDRS/kappa"] = mean_tensor(
            l=[v for k, v in kappas.items() if "HDRS" in k], device=device
        )
        kappa_dict["metrics/YMRS/kappa"] = mean_tensor(
            l=[v for k, v in kappas.items() if "YMRS" in k], device=device
        )
    return kappa_dict


def compute_matthew_correlation_coefficient(
    outputs: t.Dict[str, torch.Tensor], labels: t.Dict[str, torch.Tensor]
):
    """
    Compute per item matthew correlation coefficient
    """
    items_mcc = {}
    for k in outputs.keys():
        items_mcc[k] = F.matthews_corrcoef(
            preds=outputs[k],
            target=labels[k],
            task="multiclass",
            num_classes=ITEM_RANKS[k],
        )
    return items_mcc


def compute_accuracy(outputs, labels):
    """
    Compute per item accuracy
    """
    items_acc = {}
    for k in outputs.keys():
        items_acc[k] = F.accuracy(
            preds=outputs[k],
            target=labels[k],
            task="multiclass",
            num_classes=ITEM_RANKS[k],
        )
    return items_acc


def compute_distance_metrics(
    outputs: t.Dict[str, torch.Tensor],
    labels: t.Dict[str, torch.Tensor],
    device: torch.device,
    metric_name: _DIST_TYPES = "mae",
    average: _AVG_TYPES = "macro",
):
    """
    Compute per item mae or mse
    """
    options = t.get_args(_DIST_TYPES)
    assert metric_name in options, f"'{metric_name}' is not in {options}"
    options = t.get_args(_AVG_TYPES)
    assert average in options, f"'{average}' is not in {options}"
    distance_metric = distance_metric_func[metric_name]
    items_distance_metric = {}
    for k in outputs.keys():
        item_step_size = RANK_NORMALIZER[k]
        y_pred = torch.argmax(input=outputs[k], dim=1, keepdim=False) * item_step_size
        y_true = labels[k] * item_step_size
        if average == "micro":
            items_distance_metric[k] = distance_metric(preds=y_pred, target=y_true)
        else:
            ranks, counts = torch.unique(y_true, return_counts=True)
            errors = torch.tensor(
                [
                    distance_metric(
                        preds=y_pred[torch.where(y_true == rank)],
                        target=y_true[torch.where(y_true == rank)],
                    )
                    for rank in ranks
                ],
                dtype=torch.float32,
                device=device,
            )
            if average == "macro":
                # Calculate statistics for each label and average them
                items_distance_metric[k] = torch.mean(errors)
            else:
                # Calculates statistics for each label and computes weighted average using their support
                weights = counts / torch.sum(counts)
                items_distance_metric[k] = torch.dot(errors, weights)
    return items_distance_metric


def cumprobs_to_softmax(cumprobs: torch.Tensor) -> torch.Tensor:
    # https://github.com/ck37/coral-ordinal/blob/e42038b66705bcd6fb52152cf28ce8278a16912c/coral_ordinal/activations.py#L23
    """Turns ordinal probabilities into label probabilities (softmax)."""

    # Number of columns is the number of classes - 1
    num_classes = cumprobs.shape[1] + 1

    # Create a list of tensors.
    probs = []

    # First, get probability predictions out of the cumulative logits.
    # Column 0 is Probability that y > 0, so Pr(y = 0) = 1 - Pr(y > 0)
    # Pr(Y = 0) = 1 - s(logit for column 0)
    probs.append(1.0 - cumprobs[:, 0])

    # For the other columns, the probability is:
    # Pr(y = k) = Pr(y > k) - Pr(y > k - 1)
    if num_classes > 2:
        for val in range(1, num_classes - 1):
            probs.append(cumprobs[:, val - 1] - cumprobs[:, val])

    # Special handling of the maximum label value.
    probs.append(cumprobs[:, num_classes - 2])

    # Combine as columns into a new tensor.
    probs_tensor = torch.concat([torch.unsqueeze(t, dim=1) for t in probs], dim=1)

    return probs_tensor


def outputs_postprocess(
    outputs: t.Dict[str, torch.Tensor],
    coral: bool = False,
    item_frequency: t.Union[t.Dict[str, t.Dict[str, float]], BufferDict] = None,
):
    """
    Classifier outputs are post-processed for metric computation, specifically: 1) CORAL does not output rank
    probabilities but ordinal logits, thus ordinal logits are converted to probabilities for output_thresholing
    (when --imb_mode 2) and for metrics computation (which relies on having probabilities from which the predicted rank
    is derived with argmax); 2) if --imb_mode 2, probabilities are ; 3) otherwise nothing is done.
    """
    if coral:
        outputs = {
            k: cumprobs_to_softmax(cumprobs=torch.sigmoid(v))
            for k, v in outputs.items()
        }
    if item_frequency is not None:
        if type(item_frequency) is dict:
            device = list(outputs.values())[0].device
            item_frequency = {
                item: torch.tensor(
                    list(frequency.values()), dtype=torch.float32, device=device
                )
                for item, frequency in item_frequency.items()
            }
        for k, v in outputs.items():
            thresholded_probs = v / item_frequency[k]
            outputs[k] = thresholded_probs / torch.sum(
                thresholded_probs, dim=-1, keepdim=True
            )
    return outputs


def postprocess4metrics(
    outputs: t.Dict[str, torch.Tensor],
    labels: t.Dict[str, torch.Tensor],
    detach: bool = True,
    coral: bool = False,
    item_frequency: t.Union[t.Dict[str, t.Dict[str, float]], BufferDict] = None,
):
    if detach:
        _detach = lambda d: {k: v.detach() for k, v in d.items()}
        outputs, labels = _detach(outputs), _detach(labels)
    outputs = outputs_postprocess(outputs, coral=coral, item_frequency=item_frequency)
    return outputs, labels


@torch.no_grad()
def compute_metrics(
    outputs: t.Dict[str, torch.Tensor], labels: t.Dict[str, torch.Tensor]
):
    """
    Compute metrics
    Args:
        outputs: t.Dict[str, torch.Tensor]
        labels: t.Dict[str, torch.Tensor]
        detach: bool, detach tensors in dictionary from computational graph.
    """
    device = list(outputs.values())[0].device

    mccs = compute_matthew_correlation_coefficient(outputs=outputs, labels=labels)

    mae_micro = compute_distance_metrics(
        outputs=outputs,
        labels=labels,
        device=device,
        metric_name="mae",
        average="micro",
    )
    mae_macro = compute_distance_metrics(
        outputs=outputs,
        labels=labels,
        device=device,
        metric_name="mae",
        average="macro",
    )
    mae_weighted = compute_distance_metrics(
        outputs=outputs,
        labels=labels,
        device=device,
        metric_name="mae",
        average="weighted",
    )

    metrics_dictionary = {
        "metrics/overall/mcc": mean_tensor(
            l=[v for k, v in mccs.items()], device=device
        ),
        "metrics/overall/mae_micro": mean_tensor(
            l=[v for k, v in mae_micro.items()], device=device
        ),
        "metrics/overall/mae_macro": mean_tensor(
            l=[v for k, v in mae_macro.items()], device=device
        ),
        "metrics/overall/mae_weighted": mean_tensor(
            l=[v for k, v in mae_weighted.items()], device=device
        ),
    }

    if len(labels) == 28:
        metrics_dictionary.update(
            {
                "metrics/HDRS/mcc": mean_tensor(
                    l=[v for k, v in mccs.items() if "HDRS" in k], device=device
                ),
                "metrics/HDRS/mae_micro": mean_tensor(
                    l=[v for k, v in mae_micro.items() if "HDRS" in k], device=device
                ),
                "metrics/HDRS/mae_macro": mean_tensor(
                    l=[v for k, v in mae_macro.items() if "HDRS" in k], device=device
                ),
                "metrics/HDRS/mae_weighted": mean_tensor(
                    l=[v for k, v in mae_weighted.items() if "HDRS" in k], device=device
                ),
                "metrics/YMRS/mcc": mean_tensor(
                    l=[v for k, v in mccs.items() if "YMRS" in k], device=device
                ),
                "metrics/YMRS/mae_micro": mean_tensor(
                    l=[v for k, v in mae_micro.items() if "YMRS" in k], device=device
                ),
                "metrics/YMRS/mae_macro": mean_tensor(
                    l=[v for k, v in mae_macro.items() if "YMRS" in k], device=device
                ),
                "metrics/YMRS/mae_weighted": mean_tensor(
                    l=[v for k, v in mae_weighted.items() if "YMRS" in k], device=device
                ),
            }
        )

    return metrics_dictionary
