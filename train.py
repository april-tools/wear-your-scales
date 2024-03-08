import argparse
import pickle
import shutil
import typing as t
from time import time

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from timebase import criterions, metrics
from timebase.data.reader import get_datasets
from timebase.data.static import *
from timebase.models.models import Classifier, Critic, get_models
from timebase.utils import plots, tensorboard, utils, yaml
from timebase.utils.scheduler import Scheduler


def load(d: t.Dict[str, torch.Tensor], device: torch.device):
    """Load values in dictionary d to device"""
    return {k: v.to(device) for k, v in d.items()}


@torch.no_grad()
def get_true_and_pred(
    args,
    ds: DataLoader,
    classifier: Classifier,
    criterion_classifier: criterions.ClassifierCriterion,
    verbose: int = 1,
):
    device = args.device
    y_true, y_pred_probs, metadata, representations = {}, {}, {}, []
    classifier.to(device)
    classifier.train(False)
    for batch in tqdm(ds, disable=verbose == 0):
        inputs = load(batch["data"], device=device)
        labels = load(batch["label"], device=device)
        outputs_classifier, representation = classifier(inputs)
        outputs, labels = metrics.postprocess4metrics(
            labels=labels,
            outputs=outputs_classifier,
            coral=classifier.item_predictor._get_name() == "CoralPredictor",
            item_frequency=criterion_classifier.item_frequency
            if criterion_classifier.outputs_thresholding
            else None,
        )
        utils.update_dict(target=y_true, source=labels)
        utils.update_dict(target=y_pred_probs, source=outputs)
        utils.update_dict(target=metadata, source=batch["metadata"])
        representations.append(representation)

    y_true = {
        k: torch.cat(v, dim=0).cpu().numpy() * RANK_NORMALIZER[k]
        for k, v in y_true.items()
    }
    y_pred_probs = {
        k: torch.cat(v, dim=0).cpu().numpy() for k, v in y_pred_probs.items()
    }
    y_pred = {
        k: np.argmax(v, axis=1) * RANK_NORMALIZER[k] for k, v in y_pred_probs.items()
    }
    metadata = {k: torch.cat(v, dim=0).cpu().numpy() for k, v in metadata.items()}
    return {
        "labels": y_true,
        "pred_probs": y_pred_probs,
        "predictions": y_pred,
        "metadata": metadata,
        "representations": torch.concat(representations, dim=0).cpu().numpy(),
    }


def make_plots(
    args,
    ds: DataLoader,
    classifier: Classifier,
    criterion_classifier: criterions.ClassifierCriterion,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
):
    res = get_true_and_pred(
        args, ds=ds, classifier=classifier, criterion_classifier=criterion_classifier
    )
    plots.training_loop_plots(
        args,
        summary=summary,
        y_true=res["labels"],
        y_pred=res["predictions"],
        metadata=res["metadata"],
        representations=res["representations"],
        clinical=ds.dataset.labels,
        step=epoch,
        mode=mode,
    )


def train_step(
    batch: t.Dict[str, t.Any],
    classifier: Classifier,
    critic: Critic,
    optimizer_classifier: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer,
    criterion_classifier: criterions.ClassifierCriterion,
    criterion_critic: criterions.CriticLoss,
    critic_score: criterions.CriticScore,
    device: torch.device,
):
    result = {}
    classifier.to(device)
    critic.to(device)
    inputs = load(batch["data"], device=device)
    labels = load(batch["label"], device=device)
    segment_weights = None
    if "segment_weight" in batch:
        segment_weights = batch["segment_weight"].to(device)
    subject_ids = batch["subject_id"].to(device)

    # train classifier
    classifier.train(True)
    critic.train(False)
    outputs_classifier, representation = classifier(inputs)
    classifier_loss = criterion_classifier(
        y_true=labels, y_pred=outputs_classifier, weights=segment_weights
    )
    outputs_critic = critic(representation)
    representation_loss = critic_score(y_true=subject_ids, y_pred=outputs_critic)
    classifier_total_loss = classifier_loss + representation_loss
    classifier_total_loss.backward()
    optimizer_classifier.step()
    optimizer_classifier.zero_grad()
    result.update(
        {
            "loss/classifier": classifier_loss.detach(),
            "loss/representation": representation_loss.detach(),
            "loss/total": classifier_total_loss.detach(),
        }
    )

    # train critic
    representation = representation.detach()
    critic.train(True)
    outputs_critic = critic(representation)
    critic_loss = criterion_critic(y_true=subject_ids, y_pred=outputs_critic)
    critic_loss.backward()
    optimizer_critic.step()
    optimizer_critic.zero_grad()
    result.update({"loss/critic": critic_loss.detach()})

    return result


def train(
    args,
    ds: DataLoader,
    classifier: Classifier,
    critic: Critic,
    optimizer_classifier: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer,
    criterion_classifier: criterions.ClassifierCriterion,
    criterion_critic: criterions.CriticLoss,
    critic_score: criterions.CriticScore,
    summary: tensorboard.Summary,
    epoch: int,
):
    results = {}
    # true_and_pred_container = {"labels": {}, "predictions": {}}
    for batch in tqdm(ds, desc="Train", disable=args.verbose <= 1):
        result = train_step(
            batch=batch,
            classifier=classifier,
            critic=critic,
            optimizer_classifier=optimizer_classifier,
            optimizer_critic=optimizer_critic,
            criterion_classifier=criterion_classifier,
            criterion_critic=criterion_critic,
            critic_score=critic_score,
            device=args.device,
        )
        utils.update_dict(target=results, source=result)
    for k, v in results.items():
        results[k] = torch.mean(torch.stack(v)).item()
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    return results


@torch.no_grad()
def validation_step(
    batch: t.Dict[str, t.Any],
    classifier: Classifier,
    critic: Critic,
    criterion_classifier: criterions.ClassifierCriterion,
    criterion_critic: criterions.CriticLoss,
    critic_score: criterions.CriticScore,
    device: torch.device,
):
    result = {}
    classifier.to(device)
    critic.to(device)
    inputs = load(batch["data"], device=device)
    labels = load(batch["label"], device=device)
    segment_weights = None
    if "segment_weight" in batch:
        segment_weights = batch["segment_weight"].to(device)
    subject_ids = batch["subject_id"].to(device)

    classifier.train(False)
    critic.train(False)

    # validate classifier
    outputs_classifier, representation = classifier(inputs)
    outputs_critics = critic(representation)
    classifier_loss = criterion_classifier(
        y_true=labels,
        y_pred=outputs_classifier,
        weights=segment_weights,
        training=False,
    )
    representation_loss = critic_score(y_true=subject_ids, y_pred=outputs_critics)
    classifier_total_loss = classifier_loss + representation_loss
    result.update(
        {
            "loss/classifier": classifier_loss,
            "loss/representation": representation_loss,
            "loss/total": classifier_total_loss,
        }
    )

    # validate critic
    outputs_critic = critic(representation)
    critic_loss = criterion_critic(y_true=subject_ids, y_pred=outputs_critic)
    result.update({"loss/critic": critic_loss})
    # outputs thresholding is not performed in train_step,
    # even when --imb_mode 2
    outputs, labels = metrics.postprocess4metrics(
        labels=labels,
        outputs=outputs_classifier,
        coral=classifier.item_predictor._get_name() == "CoralPredictor",
        item_frequency=criterion_classifier.item_frequency
        if criterion_classifier.outputs_thresholding
        else None,
    )
    result.update(metrics.compute_metrics(outputs=outputs, labels=labels))
    return result, {"outputs": outputs, "targets": labels}


def validate(
    args,
    ds: DataLoader,
    classifier: Classifier,
    critic: Critic,
    criterion_classifier: criterions.ClassifierCriterion,
    criterion_critic: criterions.CriticLoss,
    critic_score: criterions.CriticScore,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
):
    results, outputs = {}, {"outputs": {}, "targets": {}}
    for batch in tqdm(ds, desc="Validate", disable=args.verbose <= 1):
        result, output = validation_step(
            batch,
            classifier=classifier,
            critic=critic,
            criterion_classifier=criterion_classifier,
            criterion_critic=criterion_critic,
            critic_score=critic_score,
            device=args.device,
        )
        utils.update_dict(target=results, source=result)
        utils.update_dict(target=outputs["outputs"], source=output["outputs"])
        utils.update_dict(target=outputs["targets"], source=output["targets"])
    for k, v in results.items():
        results[k] = torch.mean(torch.stack(v)).item()
        summary.scalar(k, value=results[k], step=epoch, mode=mode)
    kappa_dict = metrics.compute_quadratic_cohen_kappa(
        outputs=outputs["outputs"],
        labels=outputs["targets"],
    )
    for k, v in kappa_dict.items():
        results[k] = v.item()
        summary.scalar(k, value=results[k], step=epoch, mode=mode)
    return results


def main(args, wandb_sweep: bool = False):
    utils.set_random_seed(args.seed, verbose=args.verbose)

    if args.clear_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.task_mode == 1 and args.imb_mode == 1:
        print(
            "Warning: focal loss is only applicable to CrossEntropyLoss and "
            "has no effect on WeightedKappaLoss; set imb_mode 0."
        )
        args.imb_mode = 0
    if args.task_mode == 1 and args.imb_mode != 3:
        print("Warning: WeightedKappaLoss must use imb_mode 3")
        args.imb_mode = 3

    if args.use_wandb:
        utils.wandb_init(args, wandb_sweep=wandb_sweep)

    utils.get_device(args)
    summary = tensorboard.Summary(args)
    train_ds, val_ds, test_ds = get_datasets(args, summary=summary)

    classifier, critic = get_models(args, summary=summary)

    optimizer_classifier = torch.optim.AdamW(
        params=[{"params": classifier.parameters(), "name": "classifier"}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer_critic = torch.optim.AdamW(
        params=[{"params": critic.parameters(), "name": "critic"}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion_classifier, critic_score, criterion_critic = criterions.get_criterion(
        args,
        output_shapes=classifier.output_shapes,
        item_frequency=train_ds.dataset.item_frequency,
    )
    scheduler_classifier = Scheduler(
        args,
        model=classifier,
        checkpoint_dir=os.path.join(args.output_dir, "ckpt_classifier"),
        mode="max",
        optimizer=optimizer_classifier,
        lr_patience=args.lr_patience,
        min_epochs=args.min_epochs,
    )
    scheduler_critic = Scheduler(
        args,
        model=critic,
        checkpoint_dir=os.path.join(args.output_dir, "ckpt_critic"),
        mode="min",
        optimizer=optimizer_critic,
        lr_patience=args.lr_patience,
        min_epochs=args.min_epochs,
    )

    utils.save_args(args)

    epoch = scheduler_classifier.restore(load_optimizer=True, load_scheduler=True)
    _ = scheduler_critic.restore(load_optimizer=True, load_scheduler=True)

    plot = args.plot_mode in (2, 3)
    if plot and epoch == 0:
        # do not plot if a checkpoint is restored from any epoch > 0
        for mode, ds in tqdm(
            enumerate([train_ds, val_ds, test_ds]),
            desc="Plots",
            disable=args.verbose <= 1,
        ):
            make_plots(
                args,
                ds=ds,
                classifier=classifier,
                criterion_classifier=criterion_classifier,
                summary=summary,
                epoch=epoch,
                mode=mode,
            )

    results = {k: {} for k in ["train", "validation", "test"]}
    while (epoch := epoch + 1) < args.epochs + 1:
        if args.verbose:
            print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_results = train(
            args,
            ds=train_ds,
            classifier=classifier,
            critic=critic,
            optimizer_classifier=optimizer_classifier,
            optimizer_critic=optimizer_critic,
            criterion_classifier=criterion_classifier,
            criterion_critic=criterion_critic,
            critic_score=critic_score,
            summary=summary,
            epoch=epoch,
        )
        val_results = validate(
            args,
            ds=val_ds,
            classifier=classifier,
            critic=critic,
            criterion_classifier=criterion_classifier,
            criterion_critic=criterion_critic,
            critic_score=critic_score,
            summary=summary,
            epoch=epoch,
        )
        elapse = time() - start

        summary.scalar("elapse", value=elapse, step=epoch, mode=0)
        summary.scalar(
            f"model/classifier/lr",
            value=optimizer_classifier.param_groups[0]["lr"],
            step=epoch,
        )
        summary.scalar(
            f"model/critic/lr", value=optimizer_critic.param_groups[0]["lr"], step=epoch
        )
        utils.update_dict(target=results["train"], source=train_results)
        utils.update_dict(target=results["validation"], source=val_results)
        if args.verbose:
            print(
                f'Train\t\ttotal loss: {train_results["loss/total"]:.04f}\t'
                f'critic loss: {train_results["loss/critic"]:.04f}\n'
                f'Validate\ttotal loss: {val_results["loss/total"]:.04f}\t'
                f'critic loss: {train_results["loss/critic"]:.04f}\t'
                f'MAE_M: {val_results["metrics/overall/mae_macro"]:.02f}\t'
                f'quad_kappa: {val_results["metrics/overall/kappa"]:.02f}\n'
                f"Elapse: {elapse:.02f}s\n"
            )
        if (epoch % 25 == 0 or epoch == args.epochs) and plot:
            if args.verbose:
                print(f"\nPlotting training set epoch {epoch}")
            make_plots(
                args,
                ds=train_ds,
                classifier=classifier,
                criterion_classifier=criterion_classifier,
                summary=summary,
                epoch=epoch,
                mode=0,
            )
            if args.verbose:
                print(f"\nPlotting validation set epoch {epoch}")
            make_plots(
                args,
                ds=val_ds,
                classifier=classifier,
                criterion_classifier=criterion_classifier,
                summary=summary,
                epoch=epoch,
                mode=1,
            )
        scheduler_critic.step(val_results["loss/critic"], epoch=epoch)
        early_stop = scheduler_classifier.step(
            val_results["metrics/overall/kappa"], epoch=epoch
        )
        if args.use_wandb:
            wandb.log(
                {
                    "train_classifier_loss": train_results["loss/total"],
                    "train_critic_loss": train_results["loss/critic"],
                    "val_classifier_loss": val_results["loss/total"],
                    "val_critic_loss": val_results["loss/critic"],
                    "val_mae_marco": val_results["metrics/overall/mae_macro"],
                    "val_quad_kappa": val_results["metrics/overall/kappa"],
                    "best_quad_kappa": scheduler_classifier.best_value,
                    "elapse": elapse,
                },
                step=epoch,
            )
        if early_stop:
            break
        if np.isnan(train_results["loss/total"]) or np.isnan(val_results["loss/total"]):
            if args.use_wandb:
                wandb.finish(exit_code=1)  # mark run as failed
            exit("\nNaN loss detected, determinate training.")

    _ = scheduler_critic.restore()
    _ = scheduler_classifier.restore()

    if args.test_time:
        test_results = validate(
            args,
            ds=test_ds,
            classifier=classifier,
            critic=critic,
            criterion_classifier=criterion_classifier,
            criterion_critic=criterion_critic,
            critic_score=critic_score,
            summary=summary,
            epoch=epoch,
            mode=2,
        )
        if args.verbose:
            print(
                f"Test\t"
                f'MAE_M: {test_results["metrics/overall/mae_macro"]:.04f}\t'
                f'quad_kappa: {test_results["metrics/overall/kappa"]:.04f}\n'
            )
        if args.use_wandb:
            wandb.log(
                {
                    "test_mae_macro": test_results["metrics/overall/mae_macro"],
                    "test_quad_kappa": test_results["metrics/overall/kappa"],
                },
                step=epoch,
            )

        results["test"] = test_results
        if plot:
            for idx, ds in tqdm(
                enumerate([train_ds, val_ds, test_ds]),
                desc=f"Plotting epoch {scheduler_classifier.best_epoch}",
                disable=args.verbose == 0,
            ):
                make_plots(
                    args,
                    ds=ds,
                    classifier=classifier,
                    criterion_classifier=criterion_classifier,
                    summary=summary,
                    epoch=scheduler_classifier.best_epoch,
                    mode=idx,
                )
    yaml.save(filename=os.path.join(args.output_dir, "results.yaml"), data=results)
    if args.save_predictions and args.test_time:
        with open(os.path.join(args.output_dir, "res.pkl"), "wb") as file:
            pickle.dump(
                {
                    "train": get_true_and_pred(
                        args,
                        ds=train_ds,
                        classifier=classifier,
                        criterion_classifier=criterion_classifier,
                    ),
                    "val": get_true_and_pred(
                        args,
                        ds=val_ds,
                        classifier=classifier,
                        criterion_classifier=criterion_classifier,
                    ),
                    "test": get_true_and_pred(
                        args,
                        ds=test_ds,
                        classifier=classifier,
                        criterion_classifier=criterion_classifier,
                    ),
                },
                file,
            )
    if args.verbose:
        print(f"Results saved to {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training configuration
    parser.add_argument(
        "--test_time", action="store_true", help="assess test set performance"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--num_workers", type=int, default=2, help="number of workers for DataLoader"
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=20,
        help="number of epochs to train before enforcing in early stopping",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=10,
        help="number of epochs to wait before reducing lr.",
    )
    parser.add_argument(
        "--critic_score_lambda",
        type=float,
        default=0,
        help="when > 0, during training, the classifier model pays a price for "
        "encoding into h (i.e. the shared-between-tasks representation learned "
        "with feature_encoder) information that makes it easier for the critic "
        "model to tell subjects apart",
    )
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument(
        "--channel2drop",
        type=str,
        default=None,
        help="set channel to drop, defaults to None (used for leave-one-out"
        "channel importance)",
    )

    # dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to directory where preprocessed data are stored",
    )
    parser.add_argument(
        "--hours2keep",
        type=float,
        default=5,
        help="if > 0 only the first hours2keep hours of a session are kept for "
        "analyses and when a session has fewer than hours2keep it is dropped, "
        "otherwise all recorded time is used.",
    )
    parser.add_argument(
        "--status_selection",
        type=str,
        default="mood_disorders",
        choices=[
            "unfiltered",
            "exclude_hc",
            "mood_disorders",
            "ongoing_mood_disorders",
        ],
        help="filter data based on status"
        "unfiltered: all recordings are used irrespective of status"
        "exclude_hc: HCs are excluded from analyses"
        "mood_disorders: only patients with a mood disorder diagnosis are used "
        "in analyses"
        "ongoing_mood_disorders: only patients with an ongoing mood episode "
        "(i.e. no euthymia) are used in analyses",
    )
    parser.add_argument(
        "--task_mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="task mode: "
        "0) cross-entropy loss,"
        "1) weighted (quadratic) kappa loss,"
        "2) ONTRAM"
        "3) CORAL",
    )
    parser.add_argument(
        "--imb_mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="imbalanced learning mode:"
        "0) pass"
        "1) Focal loss (for gamma equal 0, the loss is only scaled by the item "
        "inverse frequency)"
        "2) Probability thresholding"
        "3) Combine RUS and ROS with resampling weights",
    )
    parser.add_argument(
        "--focal_loss_gamma",
        type=float,
        default=0,
        help="gamma is the exponent of the (1 - p_{t})^{gamma} term "
        "in the focal loss",
    )
    parser.add_argument(
        "--imb_mode_seed",
        type=int,
        default=123,
        help="seed for data-level imbalanced learning",
    )
    parser.add_argument(
        "--imb_mode_item_focus",
        type=str,
        default=None,
        help="choose which item (if any) should be balanced through resampling "
        "(this is used when experimenting with regression a single item)",
    )
    parser.add_argument(
        "--hdrs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        help="HDRS items: "
        "0 drop all HDRS items from target"
        "[1:17] item(s) to be included in target",
    )
    parser.add_argument(
        "--ymrs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        help="YMRS items: "
        "0 drop all YMRS items from target"
        "[1:11] item(s) to be included in target",
    )
    parser.add_argument(
        "--scaling_mode",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="normalize features: "
        "0) no scaling "
        "1) normalize features by the overall min and max values from the "
        "training set"
        "2) standardize features by the overall mean and standard deviation "
        "from the training set",
    )
    parser.add_argument(
        "--split_mode",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="criterion for train/val/test split:"
        "0) partition data at random into 70:15:15 train:validation:test"
        "1) split each session into 70:15:15 train:validation:test along the "
        "temporal dimension -> in-sample"
        "2) split each session into 70:15:15 train:validation:test after "
        "shuffling segments (so that  (unlike 1) temporal order between "
        "consecutive segments is broken)"
        "3) 70:15:15 train:validation:test splits are created in such a way "
        "that each subject is contained in one set only -> out-of-sample",
    )
    parser.add_argument(
        "--reuse_stats",
        action="store_true",
        help="reuse previously computed stats from training set for features "
        "scaling",
    )

    # embedding configuration
    parser.add_argument(
        "--emb_type",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="embedding to be used when args.time_alignment == 0"
        "0) MLP layer"
        "1) GRU layer"
        "2) Time2Vec layer",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=128,
        help="embedding dimension for each channel when " "args.time_alignment == 0",
    )

    # optimizer configuration
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay L2 in AdamW optimizer",
    )

    # matplotlib
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument(
        "--format", type=str, default="svg", choices=["pdf", "png", "svg"]
    )
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--plot_mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="control which plots are printed"
        "0) no plots"
        "1) eda plots"
        "2) training loop plots"
        "3) both eda and training loop plots",
    )

    # misc
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--clear_output_dir", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_group", type=str, default="")

    # model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model architecture for the feature encoder.",
    )

    # model specific hyper-parameters
    temp_args = parser.parse_known_args()[0]
    match temp_args.model:
        case "linear":
            parser.add_argument("--num_units", type=int, default=128)
        case "bilstm":
            parser.add_argument("--num_units", type=int, default=128)
            parser.add_argument("--dropout", type=float, default=0.0)
        case "transformer":
            parser.add_argument(
                "--num_blocks", type=int, default=3, help="number of MHA blocks"
            )
            parser.add_argument(
                "--num_heads", type=int, default=3, help="number of attention heads"
            )
            parser.add_argument(
                "--num_units",
                type=int,
                default=64,
                help="number of hidden units, or embed_dim in MHA",
            )
            parser.add_argument(
                "--mlp_dim",
                type=int,
                default=64,
                help="hidden size in Transformer MLP",
            )
            parser.add_argument(
                "--a_dropout",
                type=float,
                default=0.0,
                help="dropout rate of MHA",
            )
            parser.add_argument(
                "--m_dropout", type=float, default=0.0, help="dropout rate of MLP"
            )
            parser.add_argument(
                "--drop_path",
                type=float,
                default=0.0,
                help="dropout rate of stochastic depth",
            )
            parser.add_argument(
                "--disable_bias",
                action="store_true",
                help="disable bias term in Transformer",
            )
        case _:
            raise NotImplementedError(f"model {temp_args.model} not implemented.")

    del temp_args
    main(parser.parse_args())
