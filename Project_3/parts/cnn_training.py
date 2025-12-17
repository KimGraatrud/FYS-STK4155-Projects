import gc
import os

# Try to prevent vram fragmentation
# has to be called before torch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
from src import utils
from src.Dataset import GalaxyDataset
import torch
import torch.nn as nn
from src.CNN import CNN, train
from torch.utils.data import DataLoader


_model_params = {
    # 'deep' models
    "d1": ([8, 16], {}),
    "d2": ([8, 16, 16], {}),
    "d3": ([16, 16, 32, 32], {}),
    "d4": ([16, 16, 32, 32, 32], {}),
    "d5": ([16, 16, 32, 32, 64, 64], {}),
    "d6": ([16, 16, 16, 32, 32, 32, 64, 64, 64], {}),
    # 'wide' model
    "w1": ([8], {"kernal_size": 5}),
    "w2": ([16], {"kernal_size": 5}),
    "w3": ([16, 32], {"kernal_size": 5}),
    "w4": ([16, 32, 64], {"kernal_size": 5}),
    "w5": ([16, 32, 64, 128], {"kernal_size": 5}),
    "w6": ([16, 32, 64, 128, 256], {"kernal_size": 5}),
}


# get a single model
def init_model(id):
    args, kwargs = _model_params[id]
    return CNN(*args, **kwargs, id=id)


# iterator to create models so they aren't all in memory at once
def init_models_iter():
    return map(init_model, _model_params.keys())


def train_models(**kwargs):
    """
    Train models of various architectures
    """
    for model in init_models_iter():
        print(model.id, f"{utils.trainable_params(model)} trainable params")
        train(model, device=utils.DEVICE, **kwargs)
        torch.save(model.state_dict(), model.filepath())
        del model
        gc.collect()
        torch.cuda.empty_cache()


def vary_lr(id="d1", savepath=None, **train_kwargs):
    """
    Train one model on various learning rates, recording the progress
    """
    lrs = np.logspace(-2, -6, 4)
    traces = []
    for lr in lrs:
        print(f"learning rate {lr:.2e}")
        model = init_model(id)
        trace = train(
            model,
            device=utils.DEVICE,
            lr=lr,
            trace=True,
            **train_kwargs,
        )

        traces.append(trace)

        del model
        gc.collect()

    if savepath is not None:
        np.savez(savepath, traces=traces, rates=lrs)


def _evaluate(model, dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    state = torch.load(model.filepath(), weights_only=True, map_location="cpu")
    model.load_state_dict(state)

    score = nn.MSELoss()

    total_score = 0.0
    i = 0
    with torch.no_grad():
        model.to(utils.DEVICE)
        for imgs, labs in loader:
            imgs = imgs.to(utils.DEVICE)
            pred = model(imgs).squeeze().cpu()

            total_score += score(pred, labs).item()
            i += 1

    return np.sqrt(total_score / i)


def evaluate_models(mode="validate", savepath=None, batch_size=32):
    dataset = GalaxyDataset(mode=mode)

    scores = []
    params = []
    ids = []

    for model in init_models_iter():
        print("evaluating:", model.id)
        state = torch.load(model.filepath(), weights_only=True, map_location="cpu")
        model.load_state_dict(state)

        # get the score
        score = _evaluate(model, dataset, batch_size=batch_size)
        scores.append(score)
        params.append(utils.trainable_params(model))
        ids.append(model.id)

        # _zz_plot(model, dataset, **kwargs)

        # Free memory the model used and force garbage collection.
        model.to("cpu")

        del model
        del state

        torch.cuda.empty_cache()
        gc.collect()

    if savepath is not None:
        np.savez(
            savepath,
            scores=scores,
            params=params,
            ids=ids,
        )


def main():
    eval_path = os.path.join(utils.RESULTS_URL, "evaluation.npz")
    trace_path = os.path.join(utils.RESULTS_URL, "traces.npz")

    epoc = 10
    batch = 256

    print("Training different architectures")
    train_models(
        epochs=epoc,
        batch_size=batch,
    )

    print("Training different learning rates")
    vary_lr(
        id="d1",
        savepath=trace_path,
        epochs=epoc,
        batch_size=batch,
    )

    print("Evaluating models")
    evaluate_models(
        mode="validate",
        batch_size=batch,
        savepath=eval_path,
    )


if __name__ == "__main__":
    main()
