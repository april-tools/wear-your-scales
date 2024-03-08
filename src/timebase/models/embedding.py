import typing as t

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


class ChannelEmbedding(nn.Module):
    """Channel embedding base class"""

    def __init__(self, input_size: int, emb_dim: int, activation: t.Callable):
        super(ChannelEmbedding, self).__init__()

    def forward(self, inputs: torch.Tensor):
        raise NotImplementedError("Forward function has not been implemented.")


class MLPEmbedding(ChannelEmbedding):
    """MLP embedding layer for a channel
    Expected input shape: (batch size, num. steps)
    Output shape: (batch size, embedding dim, 1)
    """

    def __init__(self, input_size: int, emb_dim: int, activation: t.Callable):
        super(MLPEmbedding, self).__init__(
            input_size=input_size, emb_dim=emb_dim, activation=activation
        )
        self.layer = nn.Linear(in_features=input_size, out_features=emb_dim)
        self.activation = activation()
        self.input_shape = (input_size,)
        self.output_shape = (emb_dim, 1)

    def forward(self, inputs: torch.Tensor):
        outputs = self.layer(inputs)
        outputs = rearrange(outputs, "b l -> b l 1")
        outputs = self.activation(outputs)
        return outputs


class GRUEmbedding(ChannelEmbedding):
    """GRU embedding layer for a channel
    Expected input shape: (batch size, num. steps)
    Output shape: (batch size, embedding dim, 1)
    """

    def __init__(self, input_size: int, emb_dim: int, activation: t.Callable):
        super(GRUEmbedding, self).__init__(
            input_size=input_size, emb_dim=emb_dim, activation=activation
        )
        self.layer = nn.GRU(
            input_size=1,
            hidden_size=emb_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.activation = activation()
        self.input_shape = (input_size,)
        self.output_shape = (emb_dim, 1)

    def forward(self, inputs: torch.Tensor):
        outputs = rearrange(inputs, "b l -> b l 1")
        _, outputs = self.layer(outputs)
        outputs = rearrange(outputs, "1 b d -> b d 1")
        outputs = self.activation(outputs)
        return outputs


class Time2VecEmbedding(ChannelEmbedding):
    """Time2Vec embedding layer for a channel
    Reference:
    - https://arxiv.org/abs/1907.05321
    - https://ojus1.github.io/posts/time2vec/
    Expected input shape: (batch size, num. steps)
    Output shape: (batch size, embedding dim, 1)
    """

    def __init__(self, input_size: int, emb_dim: int, activation: t.Callable):
        super(Time2VecEmbedding, self).__init__(
            input_size=input_size, emb_dim=emb_dim, activation=activation
        )
        self.w0 = nn.parameter.Parameter(torch.randn(input_size, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(input_size, emb_dim - 1))
        self.b = nn.parameter.Parameter(torch.randn(emb_dim - 1))
        self.f = torch.sin

    def forward(self, inputs: torch.Tensor):
        # k-1 periodic features
        v1 = torch.sin(torch.matmul(inputs, self.w) + self.b)
        # One Non-periodic feature
        v2 = torch.matmul(inputs, self.w0) + self.b0
        outputs = torch.cat([v1, v2], 1)
        outputs = rearrange(outputs, "b l -> b l 1")
        return outputs


class ChannelEncoder(nn.Module):
    def __init__(self, args):
        super(ChannelEncoder, self).__init__()
        self.emb_dim = args.emb_dim
        self.input_shapes = args.input_shapes
        self.channel_names = sorted(self.input_shapes.keys())
        self.time_alignment = args.ds_info["time_alignment"]

        if self.time_alignment == 0:
            activation = nn.GELU

            match args.emb_type:
                case 0:
                    embedding = MLPEmbedding
                case 1:
                    embedding = GRUEmbedding
                case 2:
                    embedding = Time2VecEmbedding
                case _:
                    raise NotImplementedError(
                        f"emb_type {args.emb_type} not implemented."
                    )

            encoder = {
                channel: embedding(
                    input_size=self.input_shapes[channel][0],
                    emb_dim=self.emb_dim,
                    activation=activation,
                )
                for channel, input_shape in self.input_shapes.items()
            }
        else:
            # pass through layer
            encoder = {channel: nn.Identity() for channel in self.input_shapes.keys()}
        self.encoder = nn.ModuleDict(encoder)

        self.output_shape = (
            self.emb_dim
            if self.time_alignment == 0
            else args.ds_info["segment_length"],
            len(self.channel_names),
        )

    def forward(self, inputs: t.Dict[str, torch.Tensor]):
        """output shape: (batch_size, emb_dim, num. channels)"""
        outputs = []
        for channel in self.channel_names:
            outputs.append(self.encoder[channel](inputs[channel]))
        if self.time_alignment == 0:
            outputs = torch.cat(outputs, dim=-1)
        else:
            outputs = torch.cat(
                [torch.unsqueeze(input=output, dim=-1) for output in outputs], dim=-1
            )
        return outputs
