import argparse
import multiprocessing as mp
import os
from datetime import datetime
from functools import partial

import wandb

import train as trainer


def forked(fn):
    """
    Does not work on Windows (except WSL2), since the fork syscall is not supported here.
    fork creates a new process which inherits all the memory without it being copied.
    Memory is copied on write instead, meaning it is very cheap to create a new process
    Reference: https://gist.github.com/schlamar/2311116?permalink_comment_id=3932763#gistcomment-3932763
    """

    def call(*args, **kwargs):
        ctx = mp.get_context("fork")
        q = ctx.Queue(1)
        is_error = ctx.Value("b", False)

        def target():
            try:
                q.put(fn(*args, **kwargs))
            except BaseException as e:
                is_error.value = True
                q.put(e)

        ctx.Process(target=target).start()
        result = q.get()
        if is_error.value:
            raise result
        return result

    return call


class Args:
    def __init__(
        self,
        id: str,
        config: wandb.Config,
        output_dir: str,
        num_workers: int = 2,
        verbose: int = 1,
        test_time: bool = False,
    ):
        self.output_dir = os.path.join(
            output_dir, f"{datetime.now():%Y%m%d-%Hh%Mm}-{id}"
        )
        self.epochs = 400
        self.device = None
        self.seed = 1234
        self.num_workers = num_workers
        self.min_epochs = 50
        self.lr_patience = 10
        self.save_predictions = False
        self.channel2drop = None
        self.imb_mode_item_focus = None
        self.hdrs = list(range(1, 18))
        self.ymrs = list(range(1, 12))
        self.reuse_stats = True
        self.save_plots = False
        self.format = "svg"
        self.dpi = 120
        self.plot_mode = 0
        self.verbose = verbose
        self.clear_output_dir = False
        self.use_wandb = True
        self.test_time = test_time
        for key, value in config.items():
            if not hasattr(self, key):
                setattr(self, key, value)


def main(
    output_dir: str,
    wandb_group: str,
    num_workers: int = 2,
    verbose: int = 1,
    test_time: bool = False,
):
    run = wandb.init(group=wandb_group)
    config = run.config
    run.name = run.id
    args = Args(
        id=run.id,
        config=config,
        output_dir=output_dir,
        num_workers=num_workers,
        verbose=verbose,
        test_time=test_time,
    )
    trainer.main(args, wandb_sweep=True)


@forked
def agent(params):
    wandb.agent(
        sweep_id=f"timebase/timebase/{params.sweep_id}",
        function=partial(
            main,
            output_dir=params.output_dir,
            wandb_group=params.wandb_group,
            num_workers=params.num_workers,
            verbose=params.verbose,
            test_time=params.test_time,
        ),
        count=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--wandb_group", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="number of trials to run with this agent",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument(
        "--test_time", action="store_true", help="assess test set performance"
    )
    params = parser.parse_args()

    for _ in range(params.num_trials):
        agent(params)
