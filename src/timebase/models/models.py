import os
import typing as t

import torch
import torchinfo
import wandb
from torch import nn

from timebase.models.critic import CriticMLP
from timebase.models.embedding import ChannelEncoder
from timebase.models.feature_encoder import get_feature_encoder
from timebase.models.item_predictor import get_item_predictor
from timebase.utils import tensorboard


def get_model_info(
    model: nn.Module,
    input_data: t.Union[torch.Tensor, t.Sequence[t.Any], t.Mapping[str, t.Any]],
    filename: str = None,
    tag: str = "model/trainable_parameters",
    summary: tensorboard.Summary = None,
    device: torch.device = torch.device("cpu"),
):
    model_info = torchinfo.summary(
        model,
        input_data=input_data,
        depth=5,
        device=device,
        verbose=0,
    )
    if filename is not None:
        with open(filename, "w") as file:
            file.write(str(model_info))
    if summary is not None:
        summary.scalar(tag, model_info.trainable_params)
    return model_info


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.channel_encoder = ChannelEncoder(args)
        self.feature_encoder = get_feature_encoder(
            args, input_shape=self.channel_encoder.output_shape
        )
        self.item_predictor = get_item_predictor(
            args,
            input_shape=self.feature_encoder.output_shape,
        )
        self.output_shapes = self.item_predictor.output_shapes

    def forward(
        self, inputs: t.Dict[str, torch.Tensor]
    ) -> (t.Dict[str, torch.Tensor], torch.Tensor):
        outputs = self.channel_encoder(inputs)
        representation = self.feature_encoder(outputs)
        outputs = self.item_predictor(representation)
        return outputs, representation


class Critic(nn.Module):
    def __init__(self, args, input_shape: t.Tuple[int]):
        super(Critic, self).__init__()
        self.critic = CriticMLP(args, input_shape=input_shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.critic(inputs)
        return outputs


def get_models(args, summary: tensorboard.Summary = None) -> (Classifier, Critic):
    classifier = Classifier(args)
    critic = Critic(args, input_shape=classifier.feature_encoder.output_shape)

    classifier_info = get_model_info(
        model=classifier,
        input_data=[
            {
                channel: torch.randn(args.batch_size, *input_shape)
                for channel, input_shape in args.input_shapes.items()
            }
        ],
        filename=os.path.join(args.output_dir, "classifier.txt"),
        summary=summary,
    )
    if args.verbose > 2:
        print(str(classifier_info))

    critic_info = get_model_info(
        model=critic,
        input_data=torch.randn(
            args.batch_size, *classifier.feature_encoder.output_shape
        ),
        filename=os.path.join(args.output_dir, "critic.txt"),
        summary=summary,
    )
    if args.verbose > 2:
        print(str(critic_info))

    if args.use_wandb:
        wandb.log(
            {
                "classifier_size": classifier_info.trainable_params,
                "critic_size": critic_info.trainable_params,
            },
            step=0,
        )

    classifier.to(args.device)
    critic.to(args.device)

    return classifier, critic
