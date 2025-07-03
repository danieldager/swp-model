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
stop_token = p2i["<EOS>"]
v_tokens = [p2i[p] for p in vowels]
c_tokens = [p2i[p] for p in consonants]
all_tokens = v_tokens + c_tokens
i2p = {i: p for p, i in p2i.items()}

converters = {
    "Word": str,
    "Phonemes": literal_eval,
    "No_Stress": literal_eval,
    "Prediction": literal_eval,
    "Real": bool,
}


def get_cv_structures(
    index: int,
    length: int,
    target: int,  # 0 → consonant, 1 → vowel
    max_c: int = 3,
    max_v: int = 2,
) -> list[str]:
    """
    Enumerate CV strings of given `length` that satisfy cluster-size
    constraints, with `target` forced at position `index`.

    Returns a list of strings like ["CVCV", "CVCC", …].
    """

    # if not (0 <= index < length):
    #     raise ValueError("index must be in 0 … length-1")
    # if target not in (0, 1):
    #     raise ValueError("target must be 0 (C) or 1 (V)")

    tgt_char = "C" if target == 0 else "V"
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

        # # Might not be necessary
        # if pos == index:                         # forced target
        #     choices = [tgt_char]
        # else:
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
    size: int,
    index: int,
    length: int,
    target: int,
    batch_size: int,
    valid_set: np.ndarray | None = None,
) -> tuple[DataLoader, np.ndarray | None]:

    X, Xh, Xc = [], [], []
    Y, Yh, Yc = [], [], []

    if target == 0:
        source_tokens = c_tokens
        target_tokens = v_tokens
    elif target == 1:
        source_tokens = v_tokens
        target_tokens = c_tokens
    else:
        source_tokens = [t for t in all_tokens if t != target]
        target_tokens = [target]

    source_tensor = torch.tensor(source_tokens, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_tokens, dtype=torch.long, device=device)
    c_tensor = torch.tensor(c_tokens, dtype=torch.long, device=device)
    v_tensor = torch.tensor(v_tokens, dtype=torch.long, device=device)
    cv_structures = get_cv_structures(index=index, length=length, target=target)

    counter = 0
    while counter < size:
        # if args.verbose:
        print(f"Dataset size: {counter} / {size}", end="\r")

        # generate a random CV structure
        cv_structure = random.choice(cv_structures)
        c_columns = torch.tensor(
            [1 if ch == "C" else 0 for ch in cv_structure],
            dtype=torch.bool,
            device=device,
        )
        v_columns = ~c_columns

        # generate candidates for inputs
        x_cand = torch.empty(batch_size, length + 1, dtype=torch.long, device=device)
        x_cand[:, -1] = stop_token

        # fill in the input with random consonants and vowels
        Nc = int(c_columns.sum())
        if Nc > 0:
            idx = torch.randint(0, c_tensor.size(0), (batch_size, Nc), device=device)
            x_cand[:, :-1][:, c_columns] = c_tensor[idx]

        Nv = int(v_columns.sum())
        if Nv > 0:
            idx = torch.randint(0, v_tensor.size(0), (batch_size, Nv), device=device)
            x_cand[:, :-1][:, v_columns] = v_tensor[idx]

        if target_type == "type":
            choices = random.choices(source_tokens, k=batch_size)
            x_cand[:, index] = torch.tensor(choices, device=device)

        elif target_type == "identity":
            x_cand[:, index] = target

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

        # repeat for future alignment with targets
        x_keep = x_keep.repeat(len(target_tokens), 1)
        xh_keep = xh_keep.repeat(len(target_tokens), 1)
        xc_keep = xc_keep.repeat(len(target_tokens), 1)

        # generate candidates for targets
        y_cand = x_keep.clone()
        y_cand[:, index] = target_tensor.repeat_interleave(N)

        # keep only the targets that the model repeats perfectly
        y_chunks, yh_chunks, yc_chunks = [], [], []
        with torch.no_grad():
            # process y_cand in minibatches
            for i in range(0, y_cand.size(0), batch_size):
                y_chunk = y_cand[i : i + batch_size]
                yh_chunk, yc_chunk = encoder(y_chunk)
                start = start_token.long().repeat(y_chunk.size(0), 1).to(device)
                y_out_chunk = decoder(start, (yh_chunk, yc_chunk), y_chunk)

                y_chunks.append(y_out_chunk.argmax(dim=-1))
                yh_chunks.append(yh_chunk.squeeze(0))  # (m, H)
                yc_chunks.append(yc_chunk.squeeze(0))  # (m, H)

        y_preds = torch.cat(y_chunks, dim=0)  # (M, L+1)
        yh = torch.cat(yh_chunks, dim=0).unsqueeze(0)  # (1, M, H)
        yc = torch.cat(yc_chunks, dim=0).unsqueeze(0)  # (1, M, H)
        y_mask = (y_preds == y_cand).all(dim=1)

        if not y_mask.any():
            continue

        # apply the mask to the candidates
        x_keep = x_keep[y_mask]
        xh_keep = xh_keep[y_mask]
        xc_keep = xc_keep[y_mask]

        y_keep = y_cand[y_mask]
        yh_keep = yh.squeeze(0)[y_mask]
        yc_keep = yc.squeeze(0)[y_mask]

        # append to the dataset
        X.append(x_keep.detach().cpu())
        Xh.append(xh_keep.detach().cpu())
        Xc.append(xc_keep.detach().cpu())
        Y.append(y_keep.detach().cpu())
        Yh.append(yh_keep.detach().cpu())
        Yc.append(yc_keep.detach().cpu())
        counter += x_keep.size(0)

    # concatenate the dataset
    X = torch.cat(X, dim=0)
    Xh = torch.cat(Xh, dim=0)
    Xc = torch.cat(Xc, dim=0)
    Y = torch.cat(Y, dim=0)
    Yh = torch.cat(Yh, dim=0)
    Yc = torch.cat(Yc, dim=0)

    # batch_size = 512

    dataloader = DataLoader(
        TensorDataset(X, Xh, Xc, Y, Yh, Yc),
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


class Net(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.
    """

    def __init__(self, encoder_hidden, decoder_hidden, bias_only: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(encoder_hidden, decoder_hidden)

        with torch.no_grad():
            # TODO: this should work with asymmetrical dimensions
            self.fc1.weight.copy_(torch.eye(encoder_hidden))  # W = I
            self.fc1.bias.zero_()  # b = 0

        if bias_only:  # freeze the weight
            self.fc1.weight.requires_grad_(False)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        return x


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
        "--batch_size",
        type=int,
        default=128,
        help="Test dataloader batch size",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--length",
        type=int,
        help="Length of sequences in dataset",
        required=True,
    )
    parser.add_argument(
        "--edit_type",
        type=str,
        help="Type of intervention: substitution, deletion, or insertion",
        default="substitution",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        help="Target of intervation: type or identity",
        default="type",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vector",
        help="The type of intervention model (vector, matrix, ?)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for intervention model",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    # base model parameters
    train_name = args.train_name
    model_name = args.model_name
    batch_size = args.batch_size
    checkpoint = args.checkpoint

    # intervention task parameters
    length = args.length
    edit_type = args.edit_type
    target_type = args.target_type

    # intervention model parameters
    num_epochs = args.num_epochs
    model_type = args.model_type
    learning_rate = args.learning_rate

    if edit_type == "substitution":
        indices = [i for i in range(length)]

    if target_type == "type":
        targets = [0, 1]  # 0 for vowels, 1 for consonants

    elif target_type == "identity":
        targets = v_tokens + c_tokens

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
        "Index": [],
        "Target": [],
        "Distance": [],
        "Accuracy": [],
        "Stability": [],
        "Batch_Size": [],
        "Hidden_Size": [],
        "Learning_Rate": [],
        "Model_Type": [],
    }

    # Include parameters in filename for better organization
    results_path = (
        get_intervention_dir()
        / f"{model_name}_{train_name}_{checkpoint}"
        / f"{edit_type[:3]}_{target_type[:4]}_{length}_{model_type}_b{batch_size}_lr{learning_rate}.csv"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Generate validation dataset
    total_size, tv_split = 100000, 0.9
    train_size = int(total_size * tv_split)
    valid_size = total_size - train_size

    for index in indices:
        for target_token in targets:

            valid_loader, valid_set = get_intervention_dataset(
                size=valid_size,
                index=index,
                length=length,
                target=target_token,
                batch_size=batch_size,
            )

            train_loader, _ = get_intervention_dataset(
                size=train_size,
                index=index,
                length=length,
                target=target_token,
                valid_set=valid_set,
                batch_size=batch_size,
            )

            if args.verbose:
                if target_token == 0:
                    target_id = "V"
                elif target_token == 1:
                    target_id = "C"
                else:
                    target_id = i2p[target_token]
                print(f"\nIntervention: {target_id} at index {index}")

            if target_type == "type":
                p_type = v_tokens if target_id == "V" else c_tokens

            # early stopping parameters
            best_accuracy = 0
            patience = 3
            wait = 0

            bias_only = True if model_type == "vector" else False
            model = Net(encoder_hidden, decoder_hidden, bias_only=bias_only).to(device)
            criterion = nn.MSELoss()

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=1e-4
            )

            for epoch in range(1, num_epochs + 1):
                print(f"Epoch {epoch} / {num_epochs}   ", end="\r")

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

                for _, xh, _, _, yh, _ in train_loader:
                    xh = xh.to(device)
                    yh = yh.to(device)

                    optimizer.zero_grad()
                    hidden = model(xh)
                    loss = criterion(hidden, yh)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * xh.size(0)

                epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore

                ### Validation
                model.eval()
                decoder.eval()
                total = 0
                correct = 0
                s_correct = 0

                for inputs, xh, _, target, yh, yc in valid_loader:
                    inputs = inputs.to(device)
                    xh = xh.to(device)
                    yh = yh.to(device)
                    yc = yc.to(device)
                    target = target.to(device)

                    start = start_token.long().repeat(xh.size(0), 1).to(device)
                    hidden = model(xh).unsqueeze(0)
                    cell = yc.unsqueeze(0)

                    preds = decoder(start, (hidden, cell), target)
                    preds = preds.argmax(dim=-1)

                    # check if the target phoneme is correct
                    if target_type == "type":
                        correct += np.isin(preds[:, index].cpu().numpy(), p_type).sum()
                    elif target_type == "identity":
                        correct += (preds[:, index] == target[:, index]).sum().item()

                    # calculate percentage of correct predictions (w/o target index)
                    idxs = np.delete(np.arange(preds.shape[1]), index)
                    s_correct += (preds[:, idxs] == target[:, idxs]).sum().item()
                    total += target.shape[0]

                accuracy = correct / total
                stability = s_correct / (total * length)

                results["Epoch"].append(epoch)
                results["Index"].append(index)
                results["Target"].append(target_id)
                results["Distance"].append(epoch_loss)
                results["Accuracy"].append(accuracy)
                results["Stability"].append(stability)
                results["Batch_Size"].append(batch_size)
                results["Hidden_Size"].append(encoder_hidden)
                results["Learning_Rate"].append(learning_rate)
                results["Model_Type"].append(model_type)

                if args.verbose:
                    # if epoch % 20 == 0:
                    print(
                        f"E: {epoch} L: {epoch_loss:.3f} A: {accuracy:.3f} S: {stability:.3f}"
                    )

                # early stopping
                if round(accuracy, 3) > best_accuracy:
                    best_accuracy = round(accuracy, 3)
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
