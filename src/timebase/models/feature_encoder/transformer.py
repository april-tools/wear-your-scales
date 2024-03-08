import typing as t

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from torch import nn

from .encoder import FeatureEncoder, register


class DropPath(nn.Module):
    """
    Stochastic depth for regularization https://arxiv.org/abs/1603.09382
    Reference:
    - https://github.com/aanna0701/SPT_LSA_ViT/blob/main/utils/drop_path.py
    - https://github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dropout: float = 0.0):
        super(DropPath, self).__init__()
        assert 0 <= dropout <= 1
        self.register_buffer("keep_prop", torch.tensor(1 - dropout))

    def forward(self, inputs: torch.Tensor):
        if self.keep_prop == 1 or not self.training:
            return inputs
        shape = (inputs.size(0),) + (1,) * (inputs.ndim - 1)
        random_tensor = torch.rand(shape, dtype=inputs.dtype, device=inputs.device)
        random_tensor = torch.floor(self.keep_prop + random_tensor)
        outputs = (inputs / self.keep_prop) * random_tensor
        return outputs


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = None,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super(MLP, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.model = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=use_bias),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)


class Attention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super(Attention, self).__init__()
        inner_dim = emb_dim * num_heads

        self.num_heads = num_heads

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.to_qkv = nn.Linear(
            in_features=emb_dim, out_features=inner_dim * 3, bias=False
        )

        self.projection = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=emb_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )
        self.layer_norm = nn.LayerNorm(emb_dim)

        self.register_buffer("scale", torch.tensor(emb_dim**-0.5))

    def forward(self, inputs: torch.Tensor):
        inputs = self.layer_norm(inputs)
        qkv = self.to_qkv(inputs).chunk(3, dim=-1)
        q, k, v = map(
            lambda a: rearrange(a, "b n (h d) -> b h n d", h=self.num_heads),
            qkv,
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        outputs = einsum(attn, v, "b h n i, b h i d -> b h n d")
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.projection(outputs)
        return outputs


@register("transformer")
class Transformer(FeatureEncoder):
    def __init__(self, args, input_shape: t.Tuple[int, int]):
        super(Transformer, self).__init__(args, input_shape=input_shape)
        self.use_bias = not args.disable_bias

        # project input to have same dimension as num_units
        self.projection = nn.Linear(
            in_features=input_shape[-1],
            out_features=args.num_units,
            bias=self.use_bias,
        )

        drop_paths = np.linspace(0, args.drop_path, args.num_blocks)
        self.blocks = nn.ModuleList([])
        for i in range(args.num_blocks):
            block = nn.ModuleDict(
                {
                    "mha": Attention(
                        emb_dim=args.num_units,
                        num_heads=args.num_heads,
                        dropout=args.a_dropout,
                        use_bias=self.use_bias,
                    ),
                    "mlp": MLP(
                        in_dim=args.num_units,
                        hidden_dim=args.mlp_dim,
                        dropout=args.m_dropout,
                        use_bias=self.use_bias,
                    ),
                    "drop_path": DropPath(dropout=float(drop_paths[i])),
                }
            )
            self.blocks.append(block)

        self.apply(self.init_weight)

        self.output_shape = (args.num_units,)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs: torch.Tensor):
        outputs = self.projection(inputs)
        for block in self.blocks:
            drop_path = block["drop_path"]
            outputs = drop_path(block["mha"](outputs)) + outputs
            outputs = drop_path(block["mlp"](outputs)) + outputs
        outputs = torch.mean(outputs, dim=1)  # global average pooling
        return outputs
