import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from src import utils
from .cnn_training import init_model
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


def plot_evaluation(path):
    f = np.load(path)
    scores = np.array(f["scores"])
    print("scores", scores)
    params = np.array(f["params"])
    print("params", params)
    ids = f["ids"]
    print("ids", ids)

    ds = [params[:6], scores[:6]]
    ws = [params[6:], scores[6:]]

    fig, ax = plt.subplots(figsize=(utils.APS_COL_W, 0.6 * utils.APS_COL_W))
    ax.scatter(ds[0], ds[1], s=12, c="k", marker="x", label="Deep")
    ax.scatter(ws[0], ws[1], s=6, c="k", marker="D", label="Wide")

    ax.set_xscale("log")
    ax.set_ylim(0, 0.17)
    ax.set_xlim(1e3, 1e6)
    ax.set_xlabel(r"\# of trainable parameters")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Model Success vs. Complexity")

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
        ax.plot(x, trace, label=label)

    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Batch")
    ax.set_ylabel("Batch error")
    ax.set_title(r"Error During Descent for Various $\eta$")

    fig.savefig(os.path.join(utils.FIGURES_URL, "traces"))
    plt.close(fig)


def zz(eval_path, name, bins=100):
    preds = np.load(eval_path)

    # model = init_model("d1")
    # model = init_model(id)

    ds = GalaxyDataset(mode="validate")
    # loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Reporting & Plotting
    fig, ax = plt.subplots(
        figsize=(utils.APS_COL_W, 0.7 * utils.APS_COL_W),
    )

    # preds = []
    # with torch.no_grad():
    #     model.to(utils.device)
    #     for imgs, _ in loader:
    #         imgs = imgs.to(utils.device)
    #         output = model(imgs).squeeze().cpu()
    #         preds.append(output)
    # preds = torch.cat(preds)

    # reference line
    ax.plot(np.linspace(0, 4, 30), np.linspace(0, 4, 30), c="k", lw=1)

    ax.hist2d(
        ds.z,
        preds,
        bins=bins,
        range=[[0, 4], [0, 4]],
        norm="log",
    )

    # ax.set_ylim(0, 4)
    # ax.set_xlim(0, 4)
    # ax.set_title(model.id)

    fig.savefig(os.path.join(utils.FIGURES_URL, f"zz_{name}"))

    plt.close(fig)
    torch.cuda.empty_cache()


def main():
    eval_path = os.path.join(utils.RESULTS_URL, "evaluation.npz")
    trace_path = os.path.join(utils.RESULTS_URL, "traces.npz")

    model = "d2"

    # plot_traces(trace_path)
    # plot_evaluation(eval_path)
    # small_demo()
    zz(os.path.join(utils.RESULTS_URL, f"{model}.npy"), model)
    zz(os.path.join(utils.RESULTS_URL, f"{model}.npy"), f"{model}_best")
