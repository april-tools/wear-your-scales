import typing as t

import numpy as np
import torch
import torch.nn.functional as F
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss
from torch.nn.modules.loss import _Loss

from timebase.data.static import ITEM_RANKS, RANK_NORMALIZER
from timebase.utils.utils import BufferDict

EPS = torch.finfo(torch.float32).eps


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = False,
    reduction: t.Literal["none", "sum", "mean"] = "none",
    eps: t.Union[float, torch.tensor] = EPS,
):
    """cross entropy
    reference: https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/backend.py#L5544
    """
    if from_logits:
        input = F.softmax(input=input, dim=-1)
    p_true_class = torch.clamp(
        input[range(input.shape[0]), target], min=eps, max=1.0 - eps
    )
    loss = -torch.log(p_true_class)
    match reduction:
        case "none":
            pass
        case "mean":
            loss = torch.mean(loss)
        case "sum":
            loss = torch.sum(loss)
        case _:
            raise NotImplementedError(f"loss reduction {reduction} not implemented.")
    return loss


class ClassifierCriterion(_Loss):
    """
    Base criterion class for classifier

    Imbalanced learning mode (--imb_mode)
    - 0) pass
    - 1) Focal loss (for gamma equal 0, the loss is only scaled by the item
        inverse frequency)
    - 2) Probability thresholding
    - 3) Combine RUS and ROS with resampling weights
    """

    def __init__(
        self,
        args,
        output_shapes: t.Dict[str, tuple] = None,
        item_frequency: t.Dict[str, t.Dict[str, float]] = None,
    ):
        super(ClassifierCriterion, self).__init__()
        self.register_buffer("eps", torch.tensor(EPS))
        rank_step_weight = np.array(
            [RANK_NORMALIZER[item] for item in args.selected_items], dtype=np.float32
        )
        self.register_buffer(
            "rank_step_weight",
            torch.tensor(rank_step_weight / np.sum(rank_step_weight)),
        )
        if item_frequency is not None:
            self.item_frequency = BufferDict(
                {
                    item: torch.tensor(list(frequency.values()), dtype=torch.float32)
                    for item, frequency in item_frequency.items()
                }
            )
        self.rank_normalizer = RANK_NORMALIZER
        self.item_ranks = ITEM_RANKS
        self.imb_mode = args.imb_mode
        self.output_shapes = output_shapes
        self.outputs_thresholding = False

    def normalize(self, loss: torch.Tensor):
        return torch.tensordot(a=loss, b=self.rank_step_weight, dims=1)

    def forward(
        self,
        y_true: t.Union[t.Dict[str, torch.Tensor], torch.Tensor] = None,
        y_pred: t.Union[t.Dict[str, torch.Tensor], torch.Tensor] = None,
        weights: torch.Tensor = None,
        training: bool = True,
    ):
        raise NotImplementedError("forward function has not been implemented.")


class WeightedKappaLoss(ClassifierCriterion):
    def __init__(
        self,
        args,
        output_shapes: t.Dict[str, tuple],
        item_frequency: t.Dict[str, t.Dict[str, float]],
        weightage: t.Literal["linear", "quadratic"] = "quadratic",
    ):
        super(WeightedKappaLoss, self).__init__(
            args, output_shapes=output_shapes, item_frequency=item_frequency
        )
        self.focal_loss = args.imb_mode == 1
        self.outputs_thresholding = args.imb_mode == 2
        self.weighted_loss = args.imb_mode == 3
        self.weightage = weightage

    def kappa_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        batch_size, num_classes, device = y_pred.size(0), y_pred.size(1), y_pred.device
        if y_true.ndim == 1:
            y_true = F.one_hot(y_true, num_classes=num_classes).float()
        label_vec = torch.arange(num_classes, dtype=torch.float32, device=device)
        row_label_vec = torch.reshape(label_vec, [1, num_classes])
        col_label_vec = torch.reshape(label_vec, [num_classes, 1])
        col_mat = torch.tile(col_label_vec, [1, num_classes])
        row_mat = torch.tile(row_label_vec, [num_classes, 1])
        if self.weightage == "linear":
            weight_mat = torch.abs(col_mat - row_mat)
        else:
            weight_mat = (col_mat - row_mat) ** 2
        cat_labels = torch.matmul(y_true, col_label_vec)
        cat_label_mat = torch.tile(cat_labels, [1, num_classes])
        row_label_mat = torch.tile(row_label_vec, [batch_size, 1])
        if self.weightage == "linear":
            weight = torch.abs(cat_label_mat - row_label_mat)
        else:
            weight = (cat_label_mat - row_label_mat) ** 2
        numerator = torch.sum(weight * y_pred)
        label_dist = torch.sum(y_true, dim=0, keepdim=True)
        pred_dist = torch.sum(y_pred, dim=0, keepdim=True)
        w_pred_dist = torch.matmul(
            weight_mat, torch.transpose(pred_dist, dim0=0, dim1=1)
        )
        denominator = torch.sum(torch.matmul(label_dist, w_pred_dist))
        denominator /= batch_size
        loss = torch.nan_to_num(numerator / denominator, nan=0.0)
        loss = torch.log(loss + self.eps)
        return loss

    def forward(
        self,
        y_true: t.Union[t.Dict[str, torch.Tensor], torch.Tensor] = None,
        y_pred: t.Union[t.Dict[str, torch.Tensor], torch.Tensor] = None,
        weights: torch.Tensor = None,
        training: bool = True,
    ):
        total_loss = []
        for item in y_true.keys():
            if not training and self.outputs_thresholding:
                item_frequency = self.item_frequency[item]
                thresholded_probs = y_pred[item] / item_frequency
                y_pred[item] = thresholded_probs / torch.sum(
                    thresholded_probs, dim=-1, keepdim=True
                )
            loss = self.kappa_loss(y_true=y_true[item], y_pred=y_pred[item])
            loss.requires_grad_(True)
            total_loss.append(loss)
        total_loss = self.normalize(loss=torch.stack(total_loss))
        return total_loss


class CrossEntropy(ClassifierCriterion):
    def __init__(
        self,
        args,
        output_shapes: t.Dict[str, tuple],
        item_frequency: t.Dict[str, t.Dict[str, float]],
    ):
        super(CrossEntropy, self).__init__(
            args, output_shapes=output_shapes, item_frequency=item_frequency
        )
        self.focal_loss = args.imb_mode == 1
        self.outputs_thresholding = args.imb_mode == 2
        self.weighted_loss = args.imb_mode == 3
        self.register_buffer("focal_loss_gamma", torch.tensor(args.focal_loss_gamma))

    def forward(
        self,
        y_true: t.Union[t.Dict[str, torch.Tensor], torch.Tensor] = None,
        y_pred: t.Union[t.Dict[str, torch.Tensor], torch.Tensor] = None,
        weights: torch.Tensor = None,
        training: bool = True,
    ):
        total_loss = []
        for i, item in enumerate(y_true.keys()):
            item_frequency = self.item_frequency[item]
            if not training and self.outputs_thresholding:
                thresholded_probs = y_pred[item] / item_frequency
                y_pred[item] = thresholded_probs / torch.sum(
                    thresholded_probs, dim=-1, keepdim=True
                )
            loss = cross_entropy(
                input=y_pred[item],
                target=y_true[item],
                from_logits=False,
                reduction="none",
                eps=self.eps,
            )
            if training and self.focal_loss:
                one_hot_target = F.one_hot(
                    y_true[item], num_classes=self.item_ranks[item]
                )
                p_true_rank = torch.max(y_pred[item] * one_hot_target, dim=-1)[0]
                modulating_factor = torch.pow(
                    1 - p_true_rank, exponent=self.focal_loss_gamma
                )
                # map target to item frequency
                inverse_normalized_item_frequency = (item_frequency**-1) / torch.sum(
                    item_frequency**-1
                )
                # alpha in [0, 1] and sum to 1
                alpha = torch.max(
                    one_hot_target * inverse_normalized_item_frequency, dim=-1
                )[0]
                loss = alpha * loss * modulating_factor
            if training and self.weighted_loss and weights is not None:
                loss *= weights
            loss = torch.mean(loss)
            total_loss.append(loss)
        total_loss = self.normalize(loss=torch.stack(total_loss))
        return total_loss


class CORALLoss(ClassifierCriterion):
    def __init__(
        self,
        args,
        output_shapes: t.Dict[str, tuple],
        item_frequency: t.Dict[str, t.Dict[str, float]],
    ):
        super(CORALLoss, self).__init__(
            args, output_shapes=output_shapes, item_frequency=item_frequency
        )
        self.outputs_thresholding = args.imb_mode == 2
        self.weighted_loss = args.imb_mode == 3

    def forward(
        self,
        y_true: t.Union[t.Dict[str, torch.Tensor], torch.Tensor] = None,
        y_pred: t.Union[t.Dict[str, torch.Tensor], torch.Tensor] = None,
        weights: torch.Tensor = None,
        training: bool = True,
    ):
        total_loss = []
        device = list(y_pred.values())[0].device
        for i, item in enumerate(y_true.keys()):
            loss = coral_loss(
                y_pred[item],
                levels_from_labelbatch(
                    y_true[item].int(),
                    num_classes=self.item_ranks[item],
                ).to(device),
                reduction=None,
            )
            if training and self.weighted_loss and weights is not None:
                loss *= weights
            loss = torch.mean(loss)
            total_loss.append(loss)
        total_loss = self.normalize(loss=torch.stack(total_loss))
        return total_loss


class CriticScore(_Loss):
    """
    Critic score

    Further to trying to predict psychometric scale items, the classifier,
    when self.representation_loss_lambda > 0, will try to learn a
    shared-between-tasks representation that will make it harder for the
    critic to tell subjects apart, thus it will try to minimize the probability
    placed on the correct subject for a given segment
    """

    def __init__(self, args):
        super(CriticScore, self).__init__()
        self.register_buffer("coefficient", torch.tensor(args.critic_score_lambda))
        self.register_buffer("one", torch.tensor(1.0))
        self.register_buffer("eps", torch.tensor(EPS))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        indexes = torch.arange(y_pred.shape[0], device=y_pred.device)
        y_true_prob = y_pred[indexes, y_true]
        loss = torch.mean(-torch.log(self.one - y_true_prob + self.eps))
        return self.coefficient * loss


class CriticLoss(_Loss):
    """
    Cross entropy loss to train critic
    the critic will strive to tell subjects apart from the shared-between-tasks
    representation learned from the classifier's feature_encoder, thus it will
    try to minimize the cross_entropy below
    """

    def __init__(self):
        super(CriticLoss, self).__init__()
        self.register_buffer("eps", torch.tensor(EPS))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        loss = cross_entropy(
            input=y_pred,
            target=y_true,
            from_logits=False,
            reduction="mean",
            eps=self.eps,
        )
        return loss


def get_criterion(
    args,
    output_shapes: t.Dict[str, tuple],
    item_frequency: t.Dict[str, t.Dict[str, float]],
):
    match args.task_mode:
        case 1:
            criterion = WeightedKappaLoss
        case 0 | 2:
            criterion = CrossEntropy
        case 3:
            criterion = CORALLoss
        case _:
            raise NotImplementedError(
                f"task_mode {args.task_mode} " f"not implemented."
            )
    classifier_criterion = criterion(
        args, output_shapes=output_shapes, item_frequency=item_frequency
    )
    classifier_criterion.to(args.device)

    critic_score = CriticScore(args)
    critic_score.to(args.device)

    criterion_critic = CriticLoss()
    criterion_critic.to(args.device)
    return classifier_criterion, critic_score, criterion_critic
