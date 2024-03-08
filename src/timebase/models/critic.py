import typing as t

import torch
import torch.nn.functional as F
from torch import nn


class CriticMLP(nn.Module):
    def __init__(self, args, input_shape: t.Tuple[int]):
        super(CriticMLP, self).__init__()
        self.input_shape = input_shape
        self.output_shapes = args.num_train_subjects
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.input_shape[0], out_features=self.input_shape[0] // 2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.input_shape[0] // 2, out_features=self.output_shapes
            ),
        )

    def forward(self, inputs: torch.Tensor, activate: bool = True) -> torch.Tensor:
        outputs = self.mlp(inputs)
        if activate:
            outputs = F.softmax(outputs, dim=-1)
        return outputs
