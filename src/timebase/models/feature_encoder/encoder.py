import typing as t

import torch
from torch import nn

_MODELS = dict()


def register(name: str):
    def add_to_dict(fn: t.Callable):
        global _MODELS
        _MODELS[name] = fn
        return fn

    return add_to_dict


class FeatureEncoder(nn.Module):
    def __init__(self, args, input_shape: t.Tuple[int, int]):
        super(FeatureEncoder, self).__init__()
        self.input_shape = input_shape

    def regularizer(self):
        raise NotImplementedError("Regularizer function has not been implemented.")


def get_feature_encoder(args, input_shape: t.Tuple[int, int]):
    if not args.model in _MODELS.keys():
        raise NotImplementedError(
            f"FeatureEncoder {args.model} has not been implemented."
        )
    return _MODELS[args.model](args, input_shape=input_shape)
