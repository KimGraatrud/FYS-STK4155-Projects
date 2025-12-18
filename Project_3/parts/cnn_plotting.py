import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from src import utils
from .cnn_training import init_model, _model_params, init_models_iter
from src.Dataset import GalaxyDataset
from torch.utils.data import DataLoader


def _plot_channels(channels, fig, nrows=4):
    """
    Helper function for small_demo()
    """
    ncols = int(np.ceil(channels.shape[0] / nrows))
    subgs = GridSpec(nrows=nrows, ncols=ncols, figure=fig)
    for i, channel in enumerate(channels):
        row = int(np.floor(i / ncols))
        col = i % ncols
        ax = fig.add_subplot(subgs[row, col])
        ax.set_axis_off()
        ax.imshow(channel)


def small_demo():
    model = init_model("d1")
    state = torch.load(model.filepath(), weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    ds = GalaxyDataset(mode="train")
    ex_image, _ = ds[424]

    x = torch.tensor(ex_image).unsqueeze(0)

    fig = plt.figure(figsize=(1.8 * utils.APS_COL_W, 0.7 * utils.APS_COL_W))
    gs = GridSpec(
        nrows=1,
        ncols=4,
        figure=fig,
        width_ratios=[1, 2, 4, 4],
        wspace=0.1,
    )

    # Original image
    subfig = fig.add_subfigure(gs[0, 0])
    subfig.suptitle(r"Original")
    _plot_channels(ex_image, subfig, nrows=5)

    # turn the network into an iterable to
    # go through it layer by layer
    iter_network = iter(model.network)

    # First layer
    x = next(iter_network)(x)  # conv.
    x = next(iter_network)(x)  # actvn. func.
    subfig = fig.add_subfigure(gs[0, 1])
    subfig.set_facecolor("0.95")
    subfig.suptitle(r"$\mathrm{C}_1$")
    _plot_channels(x.detach()[0], subfig)

    # Second layer
    x = next(iter_network)(x)  # conv.
    x = next(iter_network)(x)  # actvn. func.
    subfig = fig.add_subfigure(gs[0, 2])
    subfig.suptitle(r"$\mathrm{C}_2$")
    _plot_channels(x.detach()[0], subfig)

    # pool
    x = next(iter_network)(x)  # pool
    subfig = fig.add_subfigure(gs[0, 3])
    subfig.set_facecolor("0.95")
    subfig.suptitle(r"$\mathrm{P}_1$")
    _plot_channels(x.detach()[0], subfig)

    fig.savefig(os.path.join(utils.FIGURES_URL, "small_demo"))

    plt.close(fig)


def eval_rmse_r2(target, prediction):

    ss_res = np.sum((target - prediction) ** 2)

    rmse = np.sqrt(ss_res / len(prediction))

    ss_tot = np.sum((target - np.mean(target)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    return rmse, r2


def plot_evaluation():
    models = list(init_models_iter())
    params = [utils.trainable_params(m) for m in models]

    dset = GalaxyDataset(mode="validate")

    # Hardcoded from the most recent logs
    # with more time this would have been programatic
    training_times = np.array(
        [  # minutes
            2.4367866079012552,
            2.9513913949330646,
            4.6105883320172625,
            5.689929437637329,
            9.468382430076598,
            13.644561878840129,
            2.0632637977600097,
            2.2450461665789287,
            3.6567726890246073,
            7.336251902580261,
            23.2779647231102,
            65.24223904212316,
        ]
    )

    rmses = []
    r2 = []
    for model in models:
        fp = os.path.join(utils.RESULTS_URL, f"{model.id}.npy")
        # bp = os.path.join(utils.RESULTS_URL, f"{model.id}_best.npy")
        final = np.load(fp)

        rmse, _r2 = eval_rmse_r2(dset.z, final)
        rmses.append(rmse)
        r2.append(_r2)

    ds = [params[:6], r2[:6], training_times[:6]]
    ws = [params[6:], r2[6:], training_times[6:]]

    fig, axs = plt.subplots(
        nrows=2,
        figsize=(utils.APS_COL_W, 1.0 * utils.APS_COL_W),
        sharex=True,
    )
    ax = axs[0]
    ax.scatter(ds[0], ds[1], s=12, c="k", marker="x", label="Deep")
    ax.scatter(ws[0], ws[1], s=6, c="k", marker="D", label="Wide")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation R$^2$")
    ax.set_title("Model Success vs. Complexity")
    ax.legend()

    ax = axs[1]
    ax.scatter(ds[0], ds[2] * 60, s=12, c="k", marker="x", label="Deep")
    ax.scatter(ws[0], ws[2] * 60, s=6, c="k", marker="D", label="Wide")
    ax.set_yscale("log")
    ax.set_ylim(1e1, 1e4)

    ax.set_xscale("log")
    ax.set_xlim(1e3, 3e6)
    ax.set_ylabel("Training time\n(sec. per 10 epochs)")
    ax.set_xlabel(r"\# of trainable parameters")
    ax.legend()

    fpth = os.path.join(utils.FIGURES_URL, "model_mse")
    fig.savefig(fpth)
    plt.close(fig)


def plot_traces(path):
    f = np.load(path)
    traces = f["traces"]
    rates = f["rates"]

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.8 * utils.APS_COL_W))

    x = np.arange(1, traces.shape[1] + 1)

    n = 50
    window = np.ones(n) / n
    s = int(n / 2)
    f = n - s - 1

    batch_size = 256
    ds = GalaxyDataset(mode="validate")
    size = len(ds.z)
    batches_per_epcoh = size / batch_size

    x = x / batches_per_epcoh

    for i, (lr, trace) in enumerate(zip(rates, traces)):
        rep = np.format_float_scientific(lr)  # this is a little hacky
        mantissa, exp = rep.split("e")
        label = r"$\eta=" + f"{float(mantissa):.1f}" + r"\times" + f"10^{{{int(exp)}}}$"

        avg = np.convolve(trace, window, mode="valid")

        c = f"C{i}"
        ax.plot(x[s:-f], avg, label=label, c=c)
        # ax.plot(x[:s], trace[:s], c=c)
        # ax.plot(x, trace, c=c, alpha=0.3, lw=0.1)

    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim(2e-1, top=4e0)
    ax.set_xlim(x[s], x[-1])

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Batch RMSE")
    ax.set_title(r"Error During Descent for Various $\eta$")

    fig.savefig(os.path.join(utils.FIGURES_URL, "traces"))
    plt.close(fig)


def zz(
    pred,
    target,
    filename,
    title="",
    bins=150,
):

    # Reporting & Plotting
    fig, ax = plt.subplots(
        figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W),
    )

    # reference line
    ax.plot(np.linspace(0, 4, 30), np.linspace(0, 4, 30), c="k", lw=0.7, ls="--")

    norm = mpl.colors.LogNorm(vmin=1, vmax=1e3)

    _, _, _, img = ax.hist2d(
        target,
        pred,
        bins=bins,
        range=[[0, 4], [0, 4]],
        norm=norm,
    )

    ax.set_ylim(0, 4)
    ax.set_xlim(0, 4)
    ax.set_title(title)

    ax.set_ylabel(r"$z$ predicted")
    ax.set_xlabel(r"$z$ target")

    fig.colorbar(img, label=r"Frequency (\#)")

    fig.savefig(os.path.join(utils.FIGURES_URL, filename))

    plt.close(fig)
    torch.cuda.empty_cache()


def main():
    trace_path = os.path.join(utils.RESULTS_URL, "traces.npz")

    models = ["w1_best", "w6_best"]
    titles = [
        "Prediction Distribution, Smallest",
        "Prediction Distribution, Best CNN",
    ]
    filenames = [
        "zz_small",
        "zz_big",
    ]

    # plot_traces(trace_path)
    plot_evaluation()
    # small_demo()
    for model, title, name in zip(models, titles, filenames):
        preds = np.load(os.path.join(utils.RESULTS_URL, f"{model}.npy"))
        ds = GalaxyDataset(mode="validate")
        zz(preds, ds.z, name, title)
