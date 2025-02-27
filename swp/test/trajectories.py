from typing import Any, Callable, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from swp.models.autoencoder import Bimodel, Unimodel


class BufferDict(TypedDict):
    activiations: list[np.ndarray]
    is_batched: bool


def create_LSTM_hook(
    buffer: BufferDict,
    include_cell: bool,
) -> Callable[
    [
        nn.Module,
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ],
    None,
]:
    def LSTM_hook(
        module: nn.Module,
        input: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        output: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        out, (h, c) = (
            output  # size (batch, time, out), ((num_layers, batch, out), (num_layers, batch, out)) when batch_first
        )
        buffer["is_batched"] = len(out.shape) == 3
        if include_cell:
            h_free = h.detach().cpu()
            c_free = c.detach().cpu()
            if buffer["is_batched"]:
                h_free = h_free.permute(1, 0, 2)
                c_free = c_free.permute(1, 0, 2)
            h_np = h_free.numpy()
            c_np = c_free.numpy()
            cat_act = np.concatenate((h_np, c_np), axis=2)
            if buffer["is_batched"]:
                processed_output = cat_act.reshape((cat_act.shape[0], -1))
            else:
                processed_output = cat_act.flatten()
        else:
            processed_output = out.detach().cpu().numpy()
        buffer["activiations"].append(processed_output)

    return LSTM_hook


def hook_model(
    model: Unimodel | Bimodel,
    hook: Callable[[nn.Module, Any, Any], Any],
    layers: str,
) -> list[RemovableHandle]:
    layers = layers.lower()
    if layers not in {"all", "encoder", "decoder"}:
        raise ValueError(f"{layers} not recognized as a layer of the model")
    handles = []
    if layers in {"all", "encoder"}:
        enc_handle = model.encoder.recurrent.register_forward_hook(hook)
        handles.append(enc_handle)
    if layers in {"all", "decoder"}:
        dec_handle = model.decoder.recurrent.register_forward_hook(hook)
        handles.append(dec_handle)
    return handles


def trajectories(
    model: Unimodel,
    device: str | torch.device,
    test_df: pd.DataFrame,
    test_loader: DataLoader,
    mode: str,
    include_cell: bool,
    include_start: bool = True,
    layers: str = "all",
):
    buffer = BufferDict({"activiations": [], "is_batched": False})
    if include_cell:
        model.to_unroll()
    model.to(device)
    model.eval()
    hook = create_LSTM_hook(buffer, include_cell=include_cell)
    handles = hook_model(model, hook, layers=layers)
    concat_act: np.ndarray | None = None
    length_to_split = []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            _ = model(inputs, target)
            if include_cell:
                current_acts = np.stack(buffer["activiations"], axis=-2)
            else:
                current_acts = np.concatenate(buffer["activiations"], axis=-2)
            if include_start:
                start_shape = list(current_acts.shape)
                start_shape[-2] = 1
                start = np.zeros(start_shape)
                current_acts = np.concatenate([start, current_acts], axis=-2)
            if buffer["is_batched"]:
                to_cat = []
                to_split = []
                for i in range(current_acts.shape[0]):
                    to_cat.append(current_acts[i])
                    to_split.append(current_acts.shape[1])
                to_stack = np.concatenate(to_cat, axis=0)
            else:
                to_stack = current_acts
                to_split = [len(current_acts)]
            if concat_act is None:
                concat_act = to_stack
            else:
                concat_act = np.concatenate((concat_act, to_stack), axis=0)
            length_to_split.extend(to_split)
            buffer["activiations"].clear()
    for handle in handles:
        handle.remove()
    if concat_act is None:
        raise ValueError("Trying to compute trajectories on empty dataset")
    if mode == "MDS":
        embedder = MDS()
        concat_act = concat_act.astype(np.float64)
        # use precomputed matrix in float32, might be faster
        # from scipy.spatial.distance import squareform,pdist
        # similarities = squareform(pdist(data,'speuclidean'))
    elif mode == "PCA":
        embedder = PCA(n_components=2)
    else:
        raise ValueError(f"Mode {mode} is not recognized to embed trajectories in 2D")
    embedded_acts = embedder.fit_transform(concat_act)
    cumsum = 0
    cumsum_to_split = []
    for length in length_to_split:
        cumsum += length
        cumsum_to_split.append(cumsum)
    splitted = np.split(embedded_acts, cumsum_to_split[:-1], axis=0)
    trajs = [traj.transpose().tolist() for traj in splitted]
    test_df["Trajectory"] = trajs
    return test_df
