import typing as t

import torch
from einops import rearrange
from torch import nn

from .encoder import FeatureEncoder, register


@register("bilstm")
class BiLSTM(FeatureEncoder):
    def __init__(self, args, input_shape: t.Tuple[int, int]):
        super(BiLSTM, self).__init__(args, input_shape=input_shape)
        num_layers = 1
        self.output_shape = (args.num_units * (2 * num_layers),)

        self.lstm = nn.LSTM(
            input_size=input_shape[-1],
            hidden_size=args.num_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, inputs: torch.Tensor):
        # return the concatenated final output state
        _, (h_n, c_n) = self.lstm(inputs)
        outputs = rearrange(h_n, "d b h -> b (d h)")
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        return outputs
