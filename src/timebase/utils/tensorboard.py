import io
import platform
import typing as t

import matplotlib
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from timebase.data.static import *

PARAMS_PAD = 1
PARAMS_LENGTH = 2

plt.rcParams.update(
    {
        "mathtext.default": "regular",
        "xtick.major.pad": PARAMS_PAD,
        "ytick.major.pad": PARAMS_PAD,
        "xtick.major.size": PARAMS_LENGTH,
        "ytick.major.size": PARAMS_LENGTH,
        "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
        "axes.facecolor": (0.0, 0.0, 0.0, 0.0),
        "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
    }
)

TICKER_FORMAT = matplotlib.ticker.FormatStrFormatter("%.2f")

JET = cm.get_cmap("jet")
GRAY = cm.get_cmap("gray")
TURBO = cm.get_cmap("turbo")
COLORMAP = TURBO
GRAY2RGB = COLORMAP(np.arange(256))[:, :3]


def set_font():
    """set custom font if exists"""
    font_path = "/Users/bryanlimy/Git/font-lexend/Lexend-Regular.ttf"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(path=font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams.update(
            {"font.family": "sans-serif", "font.sans-serif": prop.get_name()}
        )


def remove_spines(axis: matplotlib.axes.Axes):
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)


def remove_top_right_spines(axis: matplotlib.axes.Axes):
    """Remove the ticks and spines of the top and right axis"""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def set_right_label(axis: matplotlib.axes.Axes, label: str, fontsize: int = None):
    """Set y-axis label on the right-hand side"""
    right_axis = axis.twinx()
    kwargs = {"rotation": 270, "va": "center", "labelpad": 3}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    right_axis.set_ylabel(label, **kwargs)
    right_axis.set_yticks([])
    remove_top_right_spines(right_axis)


def set_xticks(
    axis: matplotlib.axes.Axes,
    ticks_loc: t.Union[np.ndarray, list],
    ticks: t.Union[np.ndarray, list],
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
    rotation: int = 0,
):
    axis.set_xticks(ticks_loc)
    axis.set_xticklabels(ticks, fontsize=tick_fontsize, rotation=rotation)
    if label:
        axis.set_xlabel(label, fontsize=label_fontsize)


def set_yticks(
    axis: matplotlib.axes.Axes,
    ticks_loc: t.Union[np.ndarray, list],
    ticks: t.Union[np.ndarray, list],
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
):
    axis.set_yticks(ticks_loc)
    axis.set_yticklabels(ticks, fontsize=tick_fontsize)
    if label:
        axis.set_ylabel(label, fontsize=label_fontsize)


def set_ticks_params(
    axis: matplotlib.axes.Axes, length: int = PARAMS_LENGTH, pad: int = PARAMS_PAD
):
    axis.tick_params(axis="both", which="both", length=length, pad=pad, colors="black")


def save_figure(
    figure: plt.Figure,
    filename: str,
    dpi: int = 120,
    transparent: bool = True,
    pad_inches: float = 0.01,
    close: bool = True,
):
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    figure.savefig(
        filename,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
        transparent=transparent,
    )
    if close:
        plt.close(figure)


class Summary(object):
    """Helper class to write TensorBoard summaries"""

    def __init__(self, args):
        self.dpi = args.dpi
        self.format = args.format
        self.dataset = args.dataset
        self.save_plots = args.save_plots

        # create SummaryWriter for train, validation and test set
        self.writers = [
            SummaryWriter(args.output_dir),
            SummaryWriter(os.path.join(args.output_dir, "val")),
            SummaryWriter(os.path.join(args.output_dir, "test")),
        ]

        self.plots_dir = os.path.join(args.output_dir, "plots")
        if not os.path.isdir(self.plots_dir):
            os.makedirs(self.plots_dir)

        if platform.system() == "Darwin" and args.verbose > 2:
            matplotlib.use("TkAgg")

    def get_writer(self, mode: int = 0):
        """Get SummaryWriter
        Args:
            mode: int, the SummaryWriter to get
                0 - train set
                1 - validation set
                2 - test set
        """
        return self.writers[mode]

    def close(self):
        for writer in self.writers:
            writer.close()

    def scalar(self, tag: str, value: t.Any, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        writer.add_scalar(tag, scalar_value=value, global_step=step)

    def histogram(self, tag: str, values: t.Any, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        writer.add_histogram(tag, values=values, global_step=step)

    def image(self, tag: str, values: t.Any, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        writer.add_image(tag, img_tensor=values, global_step=step, dataformats="CHW")

    def figure(
        self,
        tag: str,
        figure: plt.Figure,
        step: int = 0,
        close: bool = True,
        mode: int = 0,
    ):
        """Write matplotlib figure to summary
        Args:
          tag: str, data identifier
          figure: plt.Figure, matplotlib figure or a list of figures
          step: int, global step value to record
          close: bool, close figure if True
          mode: int, indicate which summary writers to use
        """
        if self.save_plots:
            save_figure(
                figure,
                filename=os.path.join(
                    self.plots_dir, f"epoch_{step:03d}", f"{tag}.{self.format}"
                ),
                dpi=self.dpi,
                close=False,
            )
        buffer = io.BytesIO()
        figure.savefig(
            buffer, dpi=self.dpi, format="png", bbox_inches="tight", pad_inches=0.02
        )
        buffer.seek(0)
        image = Image.open(buffer)
        image = transforms.ToTensor()(image)
        self.image(tag, image, step=step, mode=mode)
        if close:
            plt.close(figure)
