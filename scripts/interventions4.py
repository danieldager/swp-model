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
from swp.utils.interventions import get_intervention_dataset
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
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        gamma: bool = True,
        alpha: bool = True,
        beta: bool = True,
        bias: bool = True,
        mode: str = "matrix",
    ):
        """
        Init parameters:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Size of the hidden state.
        gamma (bool): Scales exponentially with position.
        alpha (bool): Linear scale for the gamma term.
        beta (bool): Scales linearly with position.
        bias (bool): Additive bias term.
        """
        super().__init__()
        assert mode in {"vector", "matrix"}, "mode must be 'vector' or 'matrix'"
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # self.embedding = nn.Embedding(2, hidden_size)

        if self.mode == "vector":
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
        elif self.mode == "matrix":
            # single square matrix to be exponentiated: M^(index)
            self.M = nn.Parameter(torch.eye(hidden_size))

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        h_inputs: torch.Tensor,
        indices: torch.Tensor,
        activation: bool = False,
    ):
        """
        Forward parameters:
        inputs (torch.Tensor): Input phoneme sequence.
        targets (torch.Tensor): Target phoneme sequence.
        h_inputs (torch.Tensor): Hidden state after processing inputs.
        indices (torch.Tensor): Position indices of the phonemes to be edited.
        activation (bool): Whether to apply activation function (default: False).
        """

        inputs = torch.isin(inputs, c_tensor).int()
        targets = torch.isin(targets, c_tensor).int()

        x = self.embedding(inputs)
        y = self.embedding(targets)

        if activation == True:
            x = F.tanh(x)
            y = F.tanh(y)

        if self.mode == "vector":
            scale = (
                self.alpha * (self.gamma ** indices[:, None])
                + self.beta * indices[:, None]
                + self.bias
            )
            h_prime = h_inputs + (y - x) * scale

        elif self.mode == "matrix":
            delta = y - x

            # Build a batch of M^k for each sample k = index[i]
            unique, idx_map = torch.unique(indices, return_inverse=True)
            M_k = torch.stack(
                [torch.linalg.matrix_power(self.M, int(k)) for k in unique]
            )
            M_batch = M_k[idx_map]

            # Apply: (delta row‑vector) * M^k  → row vector
            h_prime = h_inputs + torch.bmm(delta.unsqueeze(1), M_batch).squeeze(1)

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
    lengths = range(3, 7)

    split = 0.1
    # sizes = np.linspace(0, 200000, len(lengths) + 1).astype(int)
    sizes = np.array([0, 30000, 500000, 500000, 500000])
    total_valid_size = 0
    total_train_size = 0

    for i, length in enumerate(lengths):

        total_size = sizes[i + 1]
        valid_size = int(total_size * split)
        train_size = total_size - valid_size

        print("Generating dataset for length ", length)

        valid_loader, valid_set = get_intervention_dataset(
            split="valid",
            length=length,
            encoder=encoder,
            decoder=decoder,
            total_size=valid_size,
            batch_size=batch_size,
            sample_size=sample_size,
            start_token=start_token,
        )
        valid_loaders.append(valid_loader)
        total_valid_size += len(valid_loader.dataset)  # type: ignore
        train_loader, _ = get_intervention_dataset(
            split="train",
            length=length,
            encoder=encoder,
            decoder=decoder,
            total_size=train_size,
            batch_size=batch_size,
            sample_size=sample_size,
            start_token=start_token,
            valid_set=valid_set,
        )
        train_loaders.append(train_loader)
        total_train_size += len(train_loader.dataset)  # type: ignore

    print("Total valid size:", total_valid_size)

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
        mode=onion_type,
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
        tt_correct = 0  # total token accuracy
        total_tokens = 0
        total_tokens_wo = 0

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

            tt_correct += (preds == target).sum().item()
            total_tokens += target.shape[0] * target.shape[1]
            total_tokens_wo += non_target.sum().item()

        cv_accuracy = cv_correct / total_valid_size
        id_accuracy = id_correct / total_valid_size
        tt_accuracy = tt_correct / total_tokens
        stability = sb_correct / total_tokens_wo

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
