# TODO: cite Csordas et al. 2025

import argparse
import os
import random
import sys
import warnings
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from torch import nn

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from swp.utils.datasets import get_phoneme_to_id
from swp.utils.models import get_model, load_weights
from swp.utils.paths import get_intervention_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

p2i = get_phoneme_to_id()
i2p = {i: p for p, i in p2i.items()}
stop_token = p2i["<EOS>"]

c_tokens = [p2i[p] for p in consonants]
v_tokens = [p2i[p] for p in vowels]
all_tokens = c_tokens + v_tokens

t_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device)
c_tensor = torch.tensor(c_tokens, dtype=torch.long, device=device)
v_tensor = torch.tensor(v_tokens, dtype=torch.long, device=device)

converters = {
    "Word": str,
    "Phonemes": literal_eval,
    "No_Stress": literal_eval,
    "Prediction": literal_eval,
    "Real": bool,
}


def get_cv_structures(
    length: int,
    max_c: int = 3,
    max_v: int = 2,
) -> list[str]:
    """
    Enumerate CV strings of given `length` that satisfy cluster-size
    constraints, with `target` forced at position `index`.

    Returns a list of strings like ["CVCV", "CVCC", …].
    """
    results = []

    def backtrack(
        pos: int,
        curr: list[str],
        run_ch: str,
        run_len: int,
    ) -> None:
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


def get_intervention_dataset(
    length: int,
    total_size: int,
    batch_size: int,
    sample_size: int,
    valid_set: np.ndarray | None = None,
    cv_swap: bool = True,
) -> tuple[DataLoader, np.ndarray | None]:

    X, Xh, Xc, Xt = [], [], [], []
    Y, Yh, Yc, Yt = [], [], [], []
    I = []

    counter = 0
    swap_bool = True  # bool to swap CV direction
    cv_structures = get_cv_structures(length=length)

    while counter < total_size:
        # if args.verbose:
        print(f"Dataset size: {counter} / {total_size}   ", end="\r")

        # choose a random CV structure
        cv_structure = random.choice(cv_structures)
        c_columns = torch.tensor(
            [1 if ch == "C" else 0 for ch in cv_structure],
            dtype=torch.bool,
            device=device,
        )
        v_columns = ~c_columns

        # generate candidates for inputs
        x_cand = torch.empty(sample_size, length + 1, dtype=torch.long, device=device)
        x_cand[:, -1] = stop_token

        # fill in the input with random consonants and vowels
        Nc = int(c_columns.sum())
        if Nc > 0:
            idx = torch.randint(0, c_tensor.size(0), (sample_size, Nc), device=device)
            x_cand[:, :-1][:, c_columns] = c_tensor[idx]

        Nv = int(v_columns.sum())
        if Nv > 0:
            idx = torch.randint(0, v_tensor.size(0), (sample_size, Nv), device=device)
            x_cand[:, :-1][:, v_columns] = v_tensor[idx]

        # Exclude candidates already in the validation set
        if valid_set is not None:
            dtype = valid_set.dtype.fields["f0"][0]  # e.g. np.int16
            cand_np = x_cand.detach().cpu().numpy().astype(dtype, copy=False)
            cand_view = cand_np.view(valid_set.dtype).ravel()  # (B,)
            keep_np = np.isin(cand_view, valid_set, invert=True)  # True = novel

            if not keep_np.any():
                continue
            keep_mask = torch.from_numpy(keep_np).to(device, dtype=torch.bool)
            x_cand = x_cand[keep_mask]

        # keep only candidates that the model repeats perfectly
        with torch.no_grad():
            xh, xc = encoder(x_cand)
            start = start_token.long().repeat(x_cand.size(0), 1).to(device)
            x_out = decoder(start, (xh, xc), x_cand)
            x_preds = x_out.argmax(dim=-1)
        x_mask = (x_preds == x_cand).all(dim=1)

        if not x_mask.any():
            continue

        x_keep = x_cand[x_mask]
        xh_keep = xh.squeeze(0)[x_mask]
        xc_keep = xc.squeeze(0)[x_mask]
        N = x_keep.size(0)

        # TODO: leverage this to repeat the following steps
        # thus giving us more candidate targets for valid inputs
        # x_keep = x_keep.repeat(len(target_tokens), 1)
        # xh_keep = xh_keep.repeat(len(target_tokens), 1)
        # xc_keep = xc_keep.repeat(len(target_tokens), 1)

        # generate candidates for targets
        y_cand = x_keep.clone()

        # for internvations on token types (c or v)
        if cv_swap:
            # valid indices depend on the current swap direction
            val_idx = torch.nonzero(c_columns if swap_bool else v_columns)

            # edge case: chosen direction has no valid slots
            if val_idx.numel() == 0:
                val_idx = torch.nonzero(v_columns if swap_bool else v_columns)

            # edge case: if still empty, try a new CV pattern
            if val_idx.numel() == 0:
                continue

            # one random valid position **per sample**
            rand = torch.randint(0, val_idx.numel(), (y_cand.size(0),), device=device)
            indices = val_idx.squeeze(1)[rand]

        # for intenventions on unique token identities
        else:
            indices = torch.randint(0, length, (y_cand.size(0),), device=device)

        # replace target indicies with random target tokens from t_tensor
        # TODO: could repeat this step since dedupe and 1st pass are successful
        y_cand[:, indices] = t_tensor[
            torch.randint(0, t_tensor.size(0), (N,), device=device)
        ]

        # keep only the targets that the model repeats perfectly
        with torch.no_grad():
            yh, yc = encoder(y_cand)
            start = start_token.long().repeat(y_cand.size(0), 1).to(device)
            y_out = decoder(start, (yh, yc), y_cand)
            y_preds = y_out.argmax(dim=-1)
        y_mask = (y_preds == y_cand).all(dim=1)

        #     for i in range(0, y_cand.size(0), batch_size):
        #         y_chunk = y_cand[i : i + batch_size]
        #         yh_chunk, yc_chunk = encoder(y_chunk)
        #         start = start_token.long().repeat(y_chunk.size(0), 1).to(device)
        #         y_out_chunk = decoder(start, (yh_chunk, yc_chunk), y_chunk)

        #         y_chunks.append(y_out_chunk.argmax(dim=-1))
        #         yh_chunks.append(yh_chunk.squeeze(0))  # (m, H)
        #         yc_chunks.append(yc_chunk.squeeze(0))  # (m, H)

        # y_preds = torch.cat(y_chunks, dim=0)  # (M, L+1)
        # yh = torch.cat(yh_chunks, dim=0).unsqueeze(0)  # (1, M, H)
        # yc = torch.cat(yc_chunks, dim=0).unsqueeze(0)  # (1, M, H)

        if not y_mask.any():
            continue

        # apply the mask to the candidates
        x_keep = x_keep[y_mask]
        xh_keep = xh_keep[y_mask]
        xc_keep = xc_keep[y_mask]

        y_keep = y_cand[y_mask]
        yh_keep = yh.squeeze(0)[y_mask]
        yc_keep = yc.squeeze(0)[y_mask]

        indices = indices[y_mask]

        # grab the source and target phoneme tokens
        x_tokens = x_keep[torch.arange(x_keep.size(0)), indices]
        y_tokens = y_keep[torch.arange(y_keep.size(0)), indices]

        # append to the dataset
        X.append(x_keep.detach().cpu())
        Xh.append(xh_keep.detach().cpu())
        Xc.append(xc_keep.detach().cpu())
        Xt.append(x_tokens.detach().cpu())
        Y.append(y_keep.detach().cpu())
        Yh.append(yh_keep.detach().cpu())
        Yc.append(yc_keep.detach().cpu())
        Yt.append(y_tokens.detach().cpu())
        I.append(indices.detach().cpu())

        if cv_swap:
            swap_bool = not swap_bool
        counter += x_keep.size(0)

    # concatenate the dataset
    X = torch.cat(X, dim=0)
    Xh = torch.cat(Xh, dim=0)
    Xc = torch.cat(Xc, dim=0)
    Xt = torch.cat(Xt, dim=0)
    Y = torch.cat(Y, dim=0)
    Yh = torch.cat(Yh, dim=0)
    Yc = torch.cat(Yc, dim=0)
    Yt = torch.cat(Yt, dim=0)
    I = torch.cat(I, dim=0)

    dataloader = DataLoader(
        TensorDataset(X, Xh, Xc, Xt, Y, Yh, Yc, Yt, I),
        batch_size=batch_size,
        shuffle=valid_set is not None,
    )

    if valid_set is None:
        # Build a deduplicated set, then convert to 1‑D view
        X_arr = X.detach().cpu().numpy().astype(np.int16)
        X_arr = X_arr.view([("", X_arr.dtype)] * X_arr.shape[1])
        valid_set = np.unique(X_arr).ravel()

        return dataloader, valid_set
    else:
        # if the validation set is provided, return just the dataloader
        return dataloader, None


# TODO: set seed for reproducibility
def get_random_mix(*iterables, generator: np.random.Generator):

    iterators = [iter(it) for it in iterables]
    active = list(range(len(iterators)))

    while len(active) != 0:
        i = generator.choice(active)
        try:
            yield next(iterators[i])
        except StopIteration:
            active.remove(i)


class OnionEdit(nn.Module):
    """
    An intervention model that edits the hidden state of a phoneme encoder,
    it uses the position of the phoneme to be edited in the input sequence
    to scale the change in the hidden state.

    Init parameters:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Size of the hidden state.
        gamma (bool): Scales exponentially with position.
        alpha (bool): Linear scale for the gamma term.
        beta (bool): Scales linearly with position.
        bias (bool): Additive bias term.

    Forward parameters:
        inputs (torch.Tensor): Input phoneme sequence.
        targets (torch.Tensor): Target phoneme sequence.
        h_inputs (torch.Tensor): Hidden state after processing inputs.
        indices (torch.Tensor): Position indices of the phonemes to be edited.
        activation (bool): Whether to apply activation function (default: False).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        gamma: bool = True,
        alpha: bool = True,
        beta: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Compact parameter/buffer initialization
        specs = [
            ("gamma", torch.ones(hidden_size), gamma),
            ("alpha", torch.ones(hidden_size), alpha),
            ("beta", torch.zeros(hidden_size), beta),
            ("bias", torch.zeros(hidden_size), bias),
        ]

        for name, tensor, trainable in specs:
            if trainable:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        h_inputs: torch.Tensor,
        indices: torch.Tensor,
        activation: bool = True,
    ):
        x = self.embedding(inputs)
        y = self.embedding(targets)

        if activation == True:
            x = F.tanh(x)
            y = F.tanh(y)

        scale = (
            self.alpha * (self.gamma ** indices[:, None])
            + self.beta * indices[:, None]
            + self.bias
        )
        h_prime = h_inputs + (y - x) * scale

        return h_prime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name string",
    )
    parser.add_argument(
        "--train_name",
        type=str,
        required=True,
        help="Training name string",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--edit_type",
        type=str,
        help="Type of intervention: substitution, deletion, or insertion",
        default="substitution",
    )
    parser.add_argument(
        "--onion_type",
        type=str,
        default="matrix",
        help="The type of intervention model (vector, matrix, ?)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Test dataloader batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for intervention model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    args = parser.parse_args()

    # base model parameters
    train_name = args.train_name
    model_name = args.model_name
    checkpoint = args.checkpoint

    # intervention model parameters
    edit_type = args.edit_type
    onion_type = args.onion_type
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    model = get_model(model_name)
    load_weights(
        model=model,
        model_name=model_name,
        train_name=train_name,
        checkpoint=checkpoint,
        device=device,
    )
    start_token = torch.Tensor([model.start_token_id])
    encoder_hidden = model.encoder.hidden_size
    decoder_hidden = model.decoder.hidden_size
    encoder = model.encoder.to(device)
    decoder = model.decoder.to(device)
    encoder.eval()
    decoder.eval()

    results = {
        "Epoch": [],
        "Distance": [],
        "CV_Accuracy": [],
        "ID_Accuracy": [],
        "TT_Accuracy": [],
        "Stability": [],
    }

    # Include parameters in filename for better organization
    results_path = (
        get_intervention_dir()
        / f"{model_name}_{train_name}_{checkpoint}"
        / f"{edit_type[:3]}_{onion_type}_b{batch_size}_lr{learning_rate}.csv"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Generate the dataloaders
    valid_loaders = []
    train_loaders = []
    sample_size = 4096
    lengths = range(3, 6)

    split = 0.1
    sizes = np.linspace(0, 200000, len(lengths) + 1).astype(int)
    total_valid_size = 0
    total_train_size = 0

    for i, length in enumerate(lengths):

        total_size = sizes[i + 1]
        valid_size = int(total_size * split)
        train_size = total_size - valid_size

        print("Generating dataset for length ", length)

        valid_loader, valid_set = get_intervention_dataset(
            length=length,
            total_size=valid_size,
            batch_size=batch_size,
            sample_size=sample_size,
        )
        valid_loaders.append(valid_loader)
        total_valid_size += len(valid_loader.dataset)  # type: ignore
        train_loader, _ = get_intervention_dataset(
            length=length,
            total_size=train_size,
            batch_size=batch_size,
            sample_size=sample_size,
            valid_set=valid_set,
        )
        train_loaders.append(train_loader)
        total_train_size += len(train_loader.dataset)  # type: ignore

    # if args.verbose:
    #     if target_token == 0:
    #         target_id = "V"
    #     elif target_token == 1:
    #         target_id = "C"
    #     else:
    #         target_id = i2p[target_token]
    #     print(f"\nIntervention: {target_id} at index {index}")

    # if target_type == "type":
    #     p_type = v_tokens if target_id == "V" else c_tokens

    # early stopping parameters
    best_accuracy = 0
    patience = 5
    wait = 0

    model = OnionEdit(
        vocab_size=len(p2i),
        hidden_size=encoder_hidden,
        gamma=True,
        alpha=True,
        beta=True,
        bias=True,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    generator = np.random.default_rng(42)

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch} / {num_epochs}   ", end="\r")

        # get random mix of dataloaders
        train_loader = get_random_mix(*train_loaders, generator=generator)

        # generate training dataset
        # train_loader, _ = get_intervention_dataset(
        #     size=train_size,
        #     index=index,
        #     length=length,
        #     target=target_token,
        #     valid_set=valid_set,
        #     batch_size=batch_size,
        # )
        # print(f"Training set size: {len(train_loader.dataset)} ({train_size})")  # type: ignore

        ### Training
        model.train()
        running_loss = 0.0

        for _, xh, _, xt, _, yh, _, yt, index in train_loader:
            xh = xh.to(device)
            xt = xt.to(device)
            yh = yh.to(device)
            yt = yt.to(device)
            index = index.to(device)

            optimizer.zero_grad()
            h_prime = model(xt, yt, xh, index)
            loss = criterion(h_prime, yh)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xh.size(0)

        epoch_loss = running_loss / total_train_size

        ### Validation
        model.eval()
        decoder.eval()
        cv_correct = 0  # consonant↔vowel accuracy
        id_correct = 0  # phoneme↔phoneme accuracy
        sb_correct = 0  # non-target phoneme accuracy
        total_tokens = 0

        for inputs, xh, _, xt, target, yh, yc, yt, indices in valid_loader:
            inputs = inputs.to(device)
            xh = xh.to(device)
            xt = xt.to(device)
            target = target.to(device)
            yh = yh.to(device)
            yc = yc.to(device)
            yt = yt.to(device)
            indices = indices.to(device)

            h_prime = model(xt, yt, xh, indices).unsqueeze(0)
            cell = yc.unsqueeze(0)

            start = start_token.long().repeat(xh.size(0), 1).to(device)
            preds = decoder(start, (h_prime, cell), target)
            preds = preds.argmax(dim=-1)

            # C/V‑type accuracy
            trgt_tokens = target[torch.arange(target.size(0)), indices]  # (B,)
            pred_tokens = preds[torch.arange(preds.size(0)), indices]  # (B,)

            trgt_is_c = torch.isin(trgt_tokens, c_tensor)
            pred_is_c = torch.isin(pred_tokens, c_tensor)

            cv_hits = trgt_is_c == pred_is_c  # same C/V category
            cv_correct += cv_hits.sum().item()

            # Identity (exact-token) accuracy
            id_hits = pred_tokens == trgt_tokens
            id_correct += id_hits.sum().item()

            # Stability = percentage of correct predictions w/o target indices
            non_target = torch.arange(target.size(1)) != indices[:, None]
            sb_correct += (preds[non_target] == target[non_target]).sum().item()

            t_correct += (preds == target).sum().item()
            total_tokens += target.shape[0] * target.shape[1]

        cv_accuracy = cv_correct / total_valid_size
        id_accuracy = id_correct / total_valid_size
        tt_accuracy = t_correct / total_tokens
        stability = sb_correct / total_tokens

        results["Epoch"].append(epoch)
        results["Distance"].append(epoch_loss)
        results["CV_Accuracy"].append(cv_accuracy)
        results["ID_Accuracy"].append(id_accuracy)
        results["TT_Accuracy"].append(tt_accuracy)
        results["Stability"].append(stability)

        if args.verbose:
            # if epoch % 20 == 0:
            print(
                f"E: {epoch} "
                + f"L: {epoch_loss:.3f} "
                + f"T: {cv_accuracy:.3f} "
                + f"I: {id_accuracy:.3f} "
                + f"A: {tt_accuracy:.3f} "
                + f"S: {stability:.3f}"
            )

        # early stopping
        if round(cv_accuracy, 3) > best_accuracy:
            best_accuracy = round(cv_accuracy, 3)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Stopping early...")
                # print(
                #     f"E: {epoch} L: {epoch_loss:.3f} A: {accuracy:.3f} S: {stability:.3f}"
                # )
                break

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
