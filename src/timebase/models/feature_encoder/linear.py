import typing as t

import numpy as np
import torch
from torch import nn

from .encoder import FeatureEncoder, register


@register("linear")
class Linear(FeatureEncoder):
    def __init__(self, args, input_shape: t.Tuple[int, int]):
        super(Linear, self).__init__(args, input_shape=input_shape)
        self.output_shape = (args.num_units,)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=int(np.prod(input_shape)),
            out_features=args.num_units,
        )

    def forward(self, inputs: torch.Tensor):
        outputs = self.flatten(inputs)
        outputs = self.linear(outputs)
        return outputs
