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
    mean = np.mean(dset.z)
    for model in models:
        fp = os.path.join(utils.RESULTS_URL, f"{model.id}.npy")
        # bp = os.path.join(utils.RESULTS_URL, f"{model.id}_best.npy")
        final = np.load(fp)

        ss_res = np.sum((dset.z - final) ** 2)

        rmse = np.sqrt(ss_res / len(final))
        rmses.append(rmse)

        ss_tot = np.sum((dset.z - mean) ** 2)

        r2.append(1 - (ss_res / ss_tot))

        # best = np.load(bp)

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

    for lr, trace in zip(rates, traces):
        rep = np.format_float_scientific(lr)  # this is a little hacky
        mantissa, exp = rep.split("e")
        label = r"$\eta=" + f"{float(mantissa):.1f}" + r"\times" + f"10^{{{int(exp)}}}$"
        ax.plot(x, trace, label=label, alpha=0.5)

    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Batch")
    ax.set_ylabel("Batch error")
    ax.set_title(r"Error During Descent for Various $\eta$")

    fig.savefig(os.path.join(utils.FIGURES_URL, "traces"))
    plt.close(fig)


def zz(eval_path, name, bins=150):
    preds = np.load(eval_path)

    # model = init_model("d1")
    # model = init_model(id)

    ds = GalaxyDataset(mode="validate")

    # Reporting & Plotting
    fig, ax = plt.subplots(
        figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W),
    )

    # reference line
    ax.plot(np.linspace(0, 4, 30), np.linspace(0, 4, 30), c="k", lw=0.7, ls="--")

    norm = mpl.colors.LogNorm(vmin=1, vmax=1e3)

    _, _, _, img = ax.hist2d(
        ds.z,
        preds,
        bins=bins,
        range=[[0, 4], [0, 4]],
        norm=norm,
    )

    ax.set_ylim(0, 4)
    ax.set_xlim(0, 4)
    # ax.set_title(model.id)

    ax.set_ylabel(r"$z$ predicted")
    ax.set_xlabel(r"$z$ target")

    fig.colorbar(img)

    fig.savefig(os.path.join(utils.FIGURES_URL, f"zz_{name}"))

    plt.close(fig)
    torch.cuda.empty_cache()


def main():
    trace_path = os.path.join(utils.RESULTS_URL, "traces.npz")

    # model = "d2"

    # plot_traces(trace_path)
    # plot_evaluation()
    # small_demo()
    for model in _model_params.keys():
        zz(os.path.join(utils.RESULTS_URL, f"{model}.npy"), model)
        zz(os.path.join(utils.RESULTS_URL, f"{model}.npy"), f"{model}_best")
