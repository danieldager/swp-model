import argparse
import os
import sys
import warnings
from ast import literal_eval
from typing import Callable

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
from torch.utils.data import DataLoader, TensorDataset, random_split

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
i2p = {i: p for p, i in p2i.items()}

converters = {
    "Word": str,
    "Phonemes": literal_eval,
    "No_Stress": literal_eval,
    "Prediction": literal_eval,
    "Real": bool,
}


def get_intervention_dataset(
    data: pd.DataFrame,
    emb_h: np.ndarray,
    emb_c: np.ndarray,
    index: int,
    length: int,
    batch_size: int,
    target_id: str,
    target_type: str,
) -> tuple[DataLoader, DataLoader]:
    """
    Create a dataset for the intervention task.
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        emb_h (np.ndarray): Hidden state embeddings.
        emb_c (np.ndarray): Cell state embeddings.
        index (int): Index of the phoneme to intervene.
        length (int): Length of the phoneme sequence.
        batch_size (int): Batch size for the DataLoader.
        target_id (str): Target phoneme ID.
        target_type (str): Type of intervention ("type" or "identity").
    Returns:
        tuple[DataLoader, DataLoader]: Train and validation DataLoaders.
    """
    I, Xh, Xc, Yh, Yc, T = [], [], [], [], [], []

    data["Tokens"] = data["No_Stress"].apply(
        lambda x: [p2i[p] for p in x] + [stop_token]
    )

    index_cols = [f"P{i}" for i in range(length)]
    match_cols = [c for c in index_cols if c != f"P{index}"]
    groups = data.groupby(match_cols, sort=False)

    if target_type == "type":
        target_set = vowels if target_id == "V" else consonants

    # iterate over the groups
    for keys, group in groups:

        if target_type == "type":
            mask = group[f"P{index}"].isin(target_set)
            dfx = group[~mask]  # not in the target set
            dfy = group[mask]  # in the target set

            if dfy.empty or dfx.empty:
                continue  # nothing to pair here

            i = np.array(dfx["Tokens"].to_list())
            xh = emb_h[dfx.index]
            xc = emb_c[dfx.index]
            n = len(dfx)

            t = i  # we just need the length
            yh = emb_h[dfy.index]
            yc = emb_c[dfy.index]
            yh = yh.mean(axis=0, keepdims=True)
            yc = yc.mean(axis=0, keepdims=True)
            yh = np.repeat(yh, n, axis=0)
            yc = np.repeat(yc, n, axis=0)

        elif target_type == "identity":
            raise NotImplementedError("Identity intervention still TODO")

        else:
            raise ValueError(f"Unknown target type: {target_type}")

        I.append(i)
        Xh.append(xh)
        Xc.append(xc)
        T.append(t)
        Yh.append(yh)
        Yc.append(yc)

    I = np.concatenate(I, axis=0)
    Xh = np.concatenate(Xh, axis=0)
    Xc = np.concatenate(Xc, axis=0)
    T = np.concatenate(T, axis=0)
    Yh = np.concatenate(Yh, axis=0)
    Yc = np.concatenate(Yc, axis=0)

    I = torch.tensor(I, dtype=torch.long)
    Xh = torch.tensor(Xh, dtype=torch.float32)
    Xc = torch.tensor(Xc, dtype=torch.float32)
    T = torch.tensor(T, dtype=torch.long)
    Yh = torch.tensor(Yh, dtype=torch.float32)
    Yc = torch.tensor(Yc, dtype=torch.float32)

    dataset = TensorDataset(I, Xh, Xc, T, Yh, Yc)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


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


class CombinedLoss(nn.Module):
    """
    mixed_loss = mse + alpha * (1 - cosine)
    """

    def __init__(self, alpha: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        e_dist = self.mse(pred, target)
        c_dist = 1.0 - F.cosine_similarity(pred, target, dim=1)
        if c_dist.ndim:  # keep 'mean' behaviour consistent
            c_dist = c_dist.mean()

        # with torch.no_grad():  # use to tune ratio
        #     print("mse", e_dist.item(), "cos", c_dist.item())

        return e_dist + self.alpha * c_dist


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
        "--retest",
        action="store_true",
        help="Regenerate test results",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="matrix",
        help="The type of intervention model (bias, matrix, ?)",
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

    path = f"results/evaluation/{model_name}/{train_name}/{checkpoint}"

    csv_path = f"{path}/control/{length}grams.csv"
    data = pd.read_csv(csv_path, index_col=0, converters=converters)
    if not args.include_stress:
        data["Phonemes"] = data["No_Stress"]

    h_path = f"{path}/control/{length}grams_h.npy"
    emb_h = np.load(h_path)  # [np.arange(len(data)), data.Length, :]

    c_path = f"{path}/control/{length}grams_c.npy"
    emb_c = np.load(c_path)

    if edit_type == "substitution":
        indices = [i for i in range(length)]

    if target_type == "type":
        targets = ["V", "C"]

    elif target_type == "identity":
        targets = vowels + consonants

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
    decoder = model.decoder.to(device)
    decoder.eval()

    results = {
        "Epoch": [],
        "Index": [],
        "Target": [],
        "Distance": [],
        "Accuracy": [],
        "Stability": [],
        "All_Accuracy": [],
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

    for index in indices:
        for target_id in targets:
            if args.verbose:
                print(f"\nIntervention: {target_id} at index {index}")

            if target_type == "type":
                p_type = v_tokens if target_id == "V" else c_tokens
                # print([i2p[p] for p in p_type])

            train_loader, valid_loader = get_intervention_dataset(
                data,
                emb_h,
                emb_c,
                index=index,
                length=length,
                batch_size=batch_size,
                target_id=target_id,
                target_type=target_type,
            )

            bias_only = True if model_type == "bias" else False
            model = Net(encoder_hidden, decoder_hidden, bias_only=bias_only).to(device)
            criterion = nn.MSELoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # optimizer = torch.optim.AdamW(
            #     model.parameters(), lr=learning_rate, weight_decay=1e-4
            # )

            for epoch in range(1, num_epochs + 1):

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
                # all_correct = 0

                # counts = {
                #     "CC": 0,
                #     "CV": 0,
                #     "VC": 0,
                #     "VV": 0,
                # }

                for inputs, xh, _, target, yh, yc in valid_loader:
                    inputs = inputs.to(device)
                    xh = xh.to(device)
                    yh = yh.to(device)
                    yc = yc.to(device)
                    target = target.to(device)

                    start = start_token.long().repeat(xh.size(0), 1).to(device)
                    # start = start_token.repeat(xh.size(0), 1).to(device)
                    hidden = model(xh).unsqueeze(0)
                    cell = yc.unsqueeze(0)

                    preds = decoder(start, (hidden, cell), target)
                    preds = preds.argmax(dim=-1)

                    # print first input, pred, and target
                    # print("Input: ", [i2p[p] for p in inputs[0].tolist()])
                    # print("Pred: ", [i2p[p] for p in preds[0].tolist()], "\n")
                    # print("Target: ", [i2p[p] for p in target[0].tolist()])

                    # check if the target phoneme is correct
                    if target_type == "type":
                        correct += np.isin(preds[:, index].cpu().numpy(), p_type).sum()
                    elif target_type == "identity":
                        correct += (preds[:, index] == target[:, index]).sum().item()

                    # calculate percentage of correct predictions (w/o target index)
                    idxs = np.delete(np.arange(preds.shape[1].cpu().numpy()), index)
                    s_correct += (preds[:, idxs] == target[:, idxs]).sum().item()

                    # calculate overall accuracy (all tokens)
                    # all_correct += (preds == target).sum().item()

                    total += target.shape[0]
                    # total_tokens = target.numel()

                    # # get only incorrect predictions for printing
                    # if epoch == num_epochs:
                    #     mask = ~np.isin(preds[:, 1].cpu().numpy(), Vs)
                    #     inputs = inputs[mask]
                    #     preds = preds[mask]

                    # for i, p in zip(wrong_i, wrong_p):
                    #     print(" ".join(i), "   ->   ", " ".join(p))
                    #     if i[0] in consonants and i[1] in consonants:
                    #         counts["CC"] += 1
                    #     elif i[0] in consonants and i[1] in vowels:
                    #         counts["CV"] += 1
                    #     elif i[0] in vowels and i[1] in consonants:
                    #         counts["VC"] += 1
                    #     elif i[0] in vowels and i[1] in vowels:
                    #         counts["VV"] += 1

                accuracy = correct / total
                stability = s_correct / (total * (length - 1))
                # all_accuracy = all_correct / total_tokens

                results["Epoch"].append(epoch)
                results["Index"].append(index)
                results["Target"].append(target_id)
                results["Distance"].append(epoch_loss)
                results["Accuracy"].append(accuracy)
                results["Stability"].append(stability)
                # results["All_Accuracy"].append(all_accuracy)
                results["Batch_Size"].append(batch_size)
                results["Hidden_Size"].append(encoder_hidden)
                results["Learning_Rate"].append(learning_rate)
                results["Model_Type"].append(model_type)

                if args.verbose:
                    if epoch % 20 == 0:
                        print(
                            f"E: {epoch} L: {epoch_loss:.4f} A: {accuracy:.4f} S: {stability:.4f}"  # AA: {all_accuracy:.4f}"
                        )

                    # if epoch == num_epochs:
                    #     print(counts, total)

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
