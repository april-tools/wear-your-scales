import typing as t

import torch
import torch.nn.functional as F
from coral_pytorch.layers import CoralLayer
from einops import rearrange
from torch import nn
from torch.distributions import (
    AffineTransform,
    SigmoidTransform,
    TransformedDistribution,
    Uniform,
)

from timebase.data.static import ITEM_RANKS


class CoralPredictor(nn.Module):
    def __init__(self, args, input_shape: t.Tuple[int]):
        super(CoralPredictor, self).__init__()
        assert args.task_mode == 3
        self.input_shape = input_shape

        self.item_predictors = nn.ModuleDict(
            {
                item: CoralLayer(
                    size_in=input_shape[0],
                    num_classes=ITEM_RANKS[item],
                    preinit_bias=True,
                )
                for item in args.selected_items
            }
        )
        self.output_shapes = {item: (ITEM_RANKS[item],) for item in args.selected_items}

    def forward(self, inputs: torch.Tensor) -> t.Dict[str, torch.Tensor]:
        outputs = {}
        for item in self.item_predictors.keys():
            output = self.item_predictors[item](inputs)
            outputs[item] = output
        return outputs


class NominalPredictor(nn.Module):
    def __init__(self, args, input_shape: t.Tuple[int]):
        super(NominalPredictor, self).__init__()
        assert args.task_mode in (0, 1)
        self.input_shape = input_shape

        self.item_predictors = nn.ModuleDict(
            {
                item: nn.Linear(
                    in_features=input_shape[0], out_features=ITEM_RANKS[item]
                )
                for item in args.selected_items
            }
        )

        self.output_shapes = {item: (ITEM_RANKS[item],) for item in args.selected_items}

    def forward(
        self, inputs: torch.Tensor, activate: bool = True
    ) -> t.Dict[str, torch.Tensor]:
        outputs = {}
        for item in self.item_predictors.keys():
            output = self.item_predictors[item](inputs)
            if activate:
                output = F.softmax(output, dim=-1)
            outputs[item] = output
        return outputs


class ONTRAMLayer(nn.Module):
    """
    Ordinal Neural Network Transformation Models (ONTRAM)

    Reference
    - https://arxiv.org/abs/2010.08376
    """

    def __init__(
        self,
        input_shape: t.Tuple[int],
        num_classes: int,
        activation: t.Callable = nn.GELU,
    ):
        super(ONTRAMLayer, self).__init__()
        self.input_shape = input_shape

        self.intercepts = nn.Linear(
            in_features=input_shape[0], out_features=num_classes - 1
        )
        self.shift = nn.Linear(in_features=input_shape[0], out_features=1)
        self.activation = activation()
        self.distribution = self.LogisticDistribution(loc=0, scale=1)

        self.output_shapes = (num_classes,)

    @staticmethod
    def LogisticDistribution(loc: int, scale: int):
        return TransformedDistribution(
            Uniform(0, 1),
            [SigmoidTransform().inv, AffineTransform(loc, scale)],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        device, dtype = inputs.device, inputs.dtype
        batch_size = inputs.size(0)

        intercepts = self.intercepts(inputs)
        intercepts = self.activation(intercepts)

        shift = self.shift(inputs)
        shift = self.activation(shift)

        shape = (batch_size, 1)
        theta_0 = torch.full(shape, fill_value=-torch.inf, dtype=dtype, device=device)
        theta_K = torch.full(shape, fill_value=torch.inf, dtype=dtype, device=device)

        theta_1 = rearrange(intercepts[:, 0], "b -> b 1")
        theta_k = torch.exp(intercepts[:, 1:])
        theta_k = torch.cumsum(theta_k, dim=-1)

        thetas = torch.cat((theta_0, theta_1, theta_1 + theta_k, theta_K), dim=1)

        h = thetas - shift
        cumulative = self.distribution.cdf(h)

        outputs = cumulative[:, 1:] - cumulative[:, :-1]
        return outputs


class ONTRAMPredictor(nn.Module):
    def __init__(self, args, input_shape: t.Tuple[int]):
        super(ONTRAMPredictor, self).__init__()
        assert args.task_mode == 2
        self.input_shape = input_shape

        self.item_predictors = nn.ModuleDict(
            {
                item: ONTRAMLayer(
                    input_shape=input_shape,
                    num_classes=ITEM_RANKS[item],
                    activation=nn.GELU,
                )
                for item in args.selected_items
            }
        )

        self.output_shapes = {item: (ITEM_RANKS[item],) for item in args.selected_items}

    def forward(self, inputs: torch.Tensor) -> t.Dict[str, torch.Tensor]:
        outputs = {
            item: self.item_predictors[item](inputs)
            for item in self.item_predictors.keys()
        }
        return outputs


def get_item_predictor(args, input_shape: t.Tuple):
    match args.task_mode:
        case 0 | 1:
            predictor = NominalPredictor
        case 2:
            predictor = ONTRAMPredictor
        case 3:
            predictor = CoralPredictor
        case _:
            raise NotImplementedError(f"task_mode {args.task_mode} not implemented.")
    item_predictor = predictor(args, input_shape=input_shape)
    return item_predictor
