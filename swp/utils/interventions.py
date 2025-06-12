import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from swp.utils.datasets import get_phoneme_to_id
from swp.utils.paths import get_dataframe_dir, get_intervention_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p2i = get_phoneme_to_id()
i2p = {i: p for p, i in p2i.items()}
stop_token = p2i["<EOS>"]

# fmt: off
vowels = [
    "AH", "OY", "AA", "AY", "ER", "AO", "UW", "IH", 
    "EH", "UH", "IY", "EY", "OW", "AE", "AW"
]

consonants = [
    "TH", "D", "G", "W", "F", "B", "P", "K", "ZH",
    "L", "T", "R", "M", "N", "NG", "Z", "S", "Y", 
    "JH", "SH", "HH", "CH", "V", "DH"
]
# fmt: on

c_tokens = [p2i[p] for p in consonants]
v_tokens = [p2i[p] for p in vowels]
all_tokens = c_tokens + v_tokens

t_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device)
c_tensor = torch.tensor(c_tokens, dtype=torch.long, device=device)
v_tensor = torch.tensor(v_tokens, dtype=torch.long, device=device)


# ─── caching utilities ────────────────────────────────────────────────────


def _cache_path(split: str, length: int) -> Path:
    """
    Compute the on‑disk path for a cached intervention dataset.
    Example:  .../dataframes/intv_train_3.pt
    """
    df_dir = Path(get_dataframe_dir())
    df_dir.mkdir(parents=True, exist_ok=True)
    return df_dir / f"intv_{split}_{length}.pt"


def _save_dataset(tensors: list[torch.Tensor], split: str, length: int) -> None:
    """Serialize the tensors list to a single file."""
    torch.save(tensors, _cache_path(split, length))


def _load_dataset(split: str, length: int) -> list[torch.Tensor] | None:
    """Return cached tensors if the file exists, else None."""
    p = _cache_path(split, length)
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=True)
    else:
        return None


# ─── helper functions ────────────────────────────────────────────────


def get_cv_structures(length: int, max_c: int = 3, max_v: int = 2) -> list[str]:
    """
    Enumerate CV strings of given `length` that satisfy cluster-size
    constraints, with `target` forced at position `index`.

    Returns a list of strings like ["CVCV", "CVCC", …].
    """
    results = []

    def backtrack(pos: int, curr: list[str], run_ch: str, run_len: int) -> None:
        """
        Build the pattern left→right.
        `pos`      – current position we’re about to fill
        `curr`     – list of chars chosen so far
        `run_ch`   – current run character ('C' or 'V')
        `run_len`  – consecutive length of `run_ch`
        """
        # Base case
        if pos == length:
            results.append("".join(curr))
            return

        choices = ("C", "V")

        for ch in choices:
            # Update run length / character
            if pos == 0 or ch != run_ch:  # new run starts
                new_run_ch, new_run_len = ch, 1
            else:
                new_run_ch, new_run_len = run_ch, run_len + 1

            # Enforce cluster limits
            if (new_run_ch == "C" and new_run_len > max_c) or (
                new_run_ch == "V" and new_run_len > max_v
            ):
                continue  # prune this branch

            curr.append(ch)
            backtrack(pos + 1, curr, new_run_ch, new_run_len)
            curr.pop()

    backtrack(0, [], "", 0)
    return results


def choose_cv_structure(cv_structures) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick one pattern and return bool masks for C and V columns."""
    pattern = random.choice(cv_structures)
    c_mask = torch.tensor([ch == "C" for ch in pattern], device=device)
    v_mask = ~c_mask
    return c_mask, v_mask


def make_random_sequences(c_mask, v_mask, sample_size, length) -> torch.Tensor:
    """Fill C slots with random consonants, V slots with random vowels."""
    x = torch.empty(sample_size, length + 1, dtype=torch.long, device=device)
    x[:, -1] = stop_token  # EOS
    if c_mask.any():
        x[:, :-1][:, c_mask] = c_tensor[
            torch.randint(0, len(c_tensor), (sample_size, c_mask.sum()), device=device)
        ]
    if v_mask.any():
        x[:, :-1][:, v_mask] = v_tensor[
            torch.randint(0, len(v_tensor), (sample_size, v_mask.sum()), device=device)
        ]
    return x


def dedupe_against_valid(x_cand, valid_set) -> torch.Tensor:
    """Drop rows already present in the validation split."""
    if valid_set is None:
        return x_cand
    dtype = valid_set.dtype.fields["f0"][0]
    view = x_cand.cpu().numpy().astype(dtype, copy=False).view(valid_set.dtype)
    keep = torch.from_numpy(~np.isin(view.ravel(), valid_set)).to(device)
    return x_cand[keep]


@torch.no_grad()
def filter_perfect_repeats(
    batch: torch.Tensor,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    start_token: torch.Tensor,
    return_mask: bool = False,
) -> tuple:
    """Return only examples the base model reproduces exactly."""
    h, c = encoder(batch)
    start = start_token.long().repeat(batch.size(0), 1).to(device)
    preds = decoder(start, (h, c), batch).argmax(-1)
    mask = (preds == batch).all(1)
    return batch[mask], h.squeeze(0)[mask], c.squeeze(0)[mask], mask


def choose_edit_indices(c_mask, v_mask, n_rows, swap_bool) -> torch.Tensor:
    """One random valid column per row, consonant→vowel or vice-versa."""
    valid = torch.nonzero(c_mask if swap_bool else v_mask).squeeze(1)
    if valid.numel() == 0:  # flip once if empty
        valid = torch.nonzero(v_mask if swap_bool else c_mask).squeeze(1)
    idx = torch.randint(0, valid.numel(), (n_rows,), device=device)
    return valid[idx]


def replace_with_random(row_tensor, indices) -> None:
    """In-place swap at given columns with a random phoneme from t_tensor."""
    row_tensor[torch.arange(len(row_tensor)), indices] = t_tensor[
        torch.randint(0, len(t_tensor), (len(row_tensor),), device=device)
    ]


def tensors_to_dataset(batch_size: int, *tensors) -> DataLoader:
    """Package tensors into a DataLoader; keeps __getstate__ picklable."""
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)


# ─── main function ─────────────────────────────────────────────────────────


def get_intervention_dataset(
    split: str,
    length: int,
    total_size: int,
    batch_size: int,
    sample_size: int,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    start_token: torch.Tensor,
    valid_set: np.ndarray | None = None,
    cv_swap: bool = True,
) -> tuple[DataLoader, np.ndarray | None]:

    cv_structures = get_cv_structures(length)

    # Check if we can load the dataset from cache
    cached = _load_dataset(split, length)
    if cached is not None:
        print("Loading cached dataset...")
        loader = tensors_to_dataset(batch_size, *cached)
        # When loaded from cache we no longer need valid_set
        return loader, None

    swap_bool, counter = True, 0
    buckets = {b: [] for b in ("X Xh Xc Xt Y Yh Yc Yt I".split())}

    while counter < total_size:
        print(f"Dataset size: {counter} / {total_size}   ", end="\r")

        c_mask, v_mask = choose_cv_structure(cv_structures)
        x_cand = make_random_sequences(c_mask, v_mask, sample_size, length)
        x_cand = dedupe_against_valid(x_cand, valid_set)

        x_keep, xh_keep, xc_keep, _ = filter_perfect_repeats(
            x_cand, encoder, decoder, start_token
        )
        if x_keep.numel() == 0:
            continue

        # generate target batch
        y_cand = x_keep.clone()
        indices = (
            choose_edit_indices(c_mask, v_mask, y_cand.size(0), swap_bool)
            if cv_swap
            else torch.randint(0, length, (y_cand.size(0),), device=device)
        )
        replace_with_random(y_cand, indices)

        # keep only rows the model reproduces perfectly **and** align indices
        y_keep, yh_keep, yc_keep, mask = filter_perfect_repeats(
            y_cand, encoder, decoder, start_token
        )
        if y_keep.numel() == 0:
            continue

        # apply the identical mask to the source tensors and indices
        x_keep, xh_keep, xc_keep = (x_keep[mask], xh_keep[mask], xc_keep[mask])
        indices = indices[mask]

        # grab the source and target tokens at the intervention index
        x_tok = x_keep[torch.arange(len(x_keep)), indices]
        y_tok = y_keep[torch.arange(len(y_keep)), indices]

        for key, tensor in zip(
            buckets,
            (x_keep, xh_keep, xc_keep, x_tok, y_keep, yh_keep, yc_keep, y_tok, indices),
        ):
            buckets[key].append(tensor.cpu())

        counter += len(y_keep)
        if cv_swap:
            swap_bool = not swap_bool

    tensors = [torch.cat(buckets[k]) for k in buckets]
    loader = tensors_to_dataset(batch_size, *tensors)
    _save_dataset(tensors, split, length)

    if valid_set is None:
        arr = tensors[0].numpy().astype(np.int16)
        arr = arr.view([("", arr.dtype)] * arr.shape[1])
        valid_set = np.unique(arr).ravel()
        return loader, valid_set
    else:
        return loader, None
