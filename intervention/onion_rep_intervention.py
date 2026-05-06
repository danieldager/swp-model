from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from ast import literal_eval
from torch.utils.data import DataLoader, Dataset

# Ensure the repository root is importable when running from the intervention folder.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swp.utils.datasets import get_train_dataset
from swp.utils.models import get_model
from swp.utils.setup import seed_everything, set_device
from swp.datasets.phonemes import get_phoneme_to_id


def get_device() -> torch.device:
    return set_device()


class OnionIntervention(nn.Module):
    """
    Onion representation intervention.

    If single_param=True:
        state_new = state + (embed(new) - embed(old)) * scale[position]
    else:
        state_new = state + (embed(new) - embed(old)) * (g * gamma^pos + pos * lin + b)
    linear:
        scale =  pos * lin + b
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        max_position: int = 9,
        pretrained_embedding: nn.Embedding | None = None,
        single_param: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.single_param = single_param
        self.max_position = max_position
        self.device = get_device()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if pretrained_embedding is not None:
            self.embedding.weight.data.copy_(pretrained_embedding.weight.data)
            self.embedding.weight.requires_grad = False

        if self.single_param:
            self.scale = nn.Parameter(torch.ones(max_position, hidden_size))
        else:
            self.gamma = nn.Parameter(torch.ones(hidden_size) * 0.9)
            self.lin = nn.Parameter(torch.zeros(hidden_size))
            self.g = nn.Parameter(torch.ones(hidden_size))
            self.b = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, state: torch.Tensor, old_token: torch.Tensor, new_token: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        x = self.embedding(old_token).tanh()
        y = self.embedding(new_token).tanh()

        if self.single_param:
            scale = self.scale[position]
        else:
            pos = position[:, None].float()
            scale = self.g * (self.gamma.pow(pos)) + pos * self.lin + self.b
            # scale =  pos * self.lin + self.b
 
        delta = (y - x) * scale
        return state + delta

    def get_scale(self, pos: int) -> torch.Tensor:
        if self.single_param:
            return self.scale[pos]
        p = torch.tensor([[pos]], device=self.device, dtype=torch.float)
        return self.g * (self.gamma.pow(p)) + p * self.lin + self.b
        # return p * self.lin + self.b

    def scale_parameters(self) -> dict[str, np.ndarray]:
        if self.single_param:
            return {
                "scales": self.scale.detach().cpu().numpy(),
                "embedding": self.embedding.weight.detach().cpu().numpy(),
            }

        return {
            "gamma": self.gamma.detach().cpu().numpy(),
            "lin": self.lin.detach().cpu().numpy(),
            "g": self.g.detach().cpu().numpy(),
            "b": self.b.detach().cpu().numpy(),
            "scales": [
                self.get_scale(pos).detach().cpu().numpy()
                for pos in range(self.max_position)
            ],
            "embedding": self.embedding.weight.detach().cpu().numpy(),
        }


class InterventionDataset(Dataset):
    def __init__(
        self,
        phoneme_sequences: list[list[int]],
        vocab_size: int,
        pad_id: int,
        eos_id: int,
        start_id: int,
        max_len: int = 20,
        max_pos: int = 9,
        random_replace_pos: bool = True,
        test_run: bool = False,
    ):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.start_id = start_id
        self.max_len = max_len
        self.max_pos = max_pos
        self.special_tokens = {pad_id, eos_id, start_id}
        self.random_replace_pos = random_replace_pos
        self.test_run = test_run

        self.examples = []
        for seq in phoneme_sequences:
            seq = seq.copy()
            if seq[-1] != self.eos_id:
                seq.append(self.eos_id)

            if self.random_replace_pos:
                replace_pos = np.random.randint(0, min(self.max_pos, len(seq) - 1))
                pos_list = [replace_pos]
            else:
                pos_list = list(range(min(self.max_pos, len(seq) - 1)))

            for replace_pos in pos_list:
                old_token = seq[replace_pos]
                modified_seq = seq.copy()
                if self.test_run:
                    new_token = old_token
                else:
                    valid_tokens = [
                        t
                        for t in range(self.vocab_size)
                        if t != old_token and t not in self.special_tokens
                    ]
                    new_token = np.random.choice(valid_tokens)
                    modified_seq[replace_pos] = new_token

                self.examples.append(
                    {
                        "seq": seq,
                        "modified_seq": modified_seq,
                        "replace_pos": replace_pos,
                        "old_token": old_token,
                        "new_token": new_token,
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        seq = example["seq"]
        modified_seq = example["modified_seq"]

        input_padded = seq + [self.pad_id] * max(0, self.max_len - len(seq))
        target_padded = modified_seq + [self.pad_id] * max(0, self.max_len - len(modified_seq))

        return {
            "input": torch.tensor(input_padded[: self.max_len], dtype=torch.long),
            "target": torch.tensor(target_padded[: self.max_len], dtype=torch.long),
            "old_token": torch.tensor(example["old_token"], dtype=torch.long),
            "new_token": torch.tensor(example["new_token"], dtype=torch.long),
            "position": torch.tensor(example["replace_pos"], dtype=torch.long),
            "seq_len": torch.tensor(len(seq), dtype=torch.long),
        }


def create_dataloader(
    df: pd.DataFrame,
    phoneme_col: str,
    phoneme_to_id: dict[str, int],
    batch_size: int = 32,
    shuffle: bool = True,
    max_len: int = 20,
    max_pos: int = 9,
    random_replace_pos: bool = True,
    test_run: bool = False,
) -> DataLoader:
    sequences: list[list[int]] = []
    for phonemes in df[phoneme_col]:
        if isinstance(phonemes, str):
            phonemes = literal_eval(phonemes)
        sequences.append([phoneme_to_id[p] for p in phonemes])

    dataset = InterventionDataset(
        sequences,
        len(phoneme_to_id),
        phoneme_to_id["<PAD>"],
        phoneme_to_id["<EOS>"],
        phoneme_to_id["<SOS>"],
        max_len=max_len,
        max_pos=max_pos,
        random_replace_pos=random_replace_pos,
        test_run=test_run,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_encoder_hidden(model: nn.Module, input_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        h, c = model.encoder(input_ids.to(device))
        return h[-1], c[-1]


def decode_with_hidden(
    model: nn.Module,
    h: torch.Tensor,
    c: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    teacher_forcing: bool = False,
) -> torch.Tensor:
    h, c = h.unsqueeze(0), c.unsqueeze(0)
    batch_size, max_len = target.shape

    outputs: list[torch.Tensor] = []
    hidden = (h, c)
    inp = torch.full((batch_size, 1), model.start_token_id, device=device, dtype=torch.long)

    for t in range(max_len):
        embedded = model.decoder.dropout(model.decoder.embedding(inp))
        out, hidden = model.decoder.recurrent(embedded, hidden)
        logits = out @ model.decoder.embedding.weight.T
        outputs.append(logits)
        if teacher_forcing:
            inp = target[:, t : t + 1]
        else:
            inp = logits.argmax(dim=-1)

    return torch.cat(outputs, dim=1)


class InterventionTrainer:
    def __init__(
        self,
        model: nn.Module,
        intervention: OnionIntervention,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        pad_id: int,
    ):
        self.model = model
        self.intervention = intervention
        self.optimizer = optimizer
        self.device = device
        self.pad_id = pad_id
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

    def _compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=self.pad_id,
        )

    def _compute_accuracy(self, preds: torch.Tensor, targets: torch.Tensor, seq_lens: torch.Tensor) -> float:
        batch_size = preds.shape[0]
        seq_correct = 0
        for i in range(batch_size):
            sl = seq_lens[i].item()
            if torch.equal(preds[i, :sl], targets[i, :sl]):
                seq_correct += 1
        return seq_correct / batch_size

    def _run_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch["input"].to(self.device)
        target_ids = batch["target"].to(self.device)
        old_token = batch["old_token"].to(self.device)
        new_token = batch["new_token"].to(self.device)
        position = batch["position"].to(self.device)
        seq_len = batch["seq_len"]

        h, c = get_encoder_hidden(self.model, input_ids, self.device)
        c_mod = self.intervention(c, old_token, new_token, position)
        logits = decode_with_hidden(self.model, h, c_mod, target_ids, self.device)

        loss = self._compute_loss(logits, target_ids)
        preds = logits.argmax(dim=-1)
        return loss, preds, target_ids, seq_len

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.intervention.train()
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in loader:
            self.optimizer.zero_grad()
            loss, preds, targets, seq_lens = self._run_batch(batch)
            loss.backward()
            self.optimizer.step()

            acc = self._compute_accuracy(preds, targets, seq_lens)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        return total_loss / n_batches, total_acc / n_batches

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.intervention.eval()
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in loader:
            loss, preds, targets, seq_lens = self._run_batch(batch)
            acc = self._compute_accuracy(preds, targets, seq_lens)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        return total_loss / n_batches, total_acc / n_batches

    @torch.no_grad()
    def evaluate_with_predictions(self, loader: DataLoader, id_to_phoneme: dict[int, str]) -> pd.DataFrame:
        self.intervention.eval()
        self.model.eval()

        records: list[dict[str, object]] = []
        for batch in loader:
            _, preds, targets, seq_lens = self._run_batch(batch)
            batch_size = preds.shape[0]
            for i in range(batch_size):
                sl = seq_lens[i].item()
                pred_ids = preds[i, :sl].cpu().tolist()
                target_ids = targets[i, :sl].cpu().tolist()
                input_ids = batch["input"][i, :sl].tolist()

                records.append(
                    {
                        "input": " ".join([id_to_phoneme[t] for t in input_ids]),
                        "target": " ".join([id_to_phoneme[t] for t in target_ids]),
                        "prediction": " ".join([id_to_phoneme[p] for p in pred_ids]),
                        "position": batch["position"][i].item(),
                        "old_phoneme": id_to_phoneme[batch["old_token"][i].item()],
                        "new_phoneme": id_to_phoneme[batch["new_token"][i].item()],
                        "seq_len": sl,
                        "match": pred_ids == target_ids,
                        "token_acc": sum(p == t for p, t in zip(pred_ids, target_ids)) / sl,
                    }
                )

        return pd.DataFrame(records)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        patience: int = 5,
        min_delta: float = 1e-4,
        test_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.intervention.state_dict())
        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            test_loss, test_acc = (None, None)
            if test_loader is not None:
                test_loss, test_acc = self.evaluate(test_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            if test_loader is not None:
                self.history["test_loss"].append(test_loss)
                self.history["test_acc"].append(test_acc)

            message = (
                f"Epoch {epoch:3d} | "
                f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
            )
            if test_loader is not None:
                message += f" | Test loss={test_loss:.4f} acc={test_acc:.4f}"
            print(message)

            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.intervention.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping after {epoch} epochs: "
                    f"no improvement on val set for {patience} epochs."
                )
                break

        self.intervention.load_state_dict(best_state)
        return self.history

    def save_scale_params(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        params = self.intervention.scale_parameters()
        np.savez(save_dir / "scale_params.npz", **params)


def plot_training_history(history: dict[str, list[float]], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    if history.get("test_loss"):
        axes[0].plot(history["test_loss"], label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    if history.get("test_acc"):
        axes[1].plot(history["test_acc"], label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    final_train_acc = history["train_acc"][-1]
    final_val_acc = history["val_acc"][-1]
    final_test_acc = history["test_acc"][-1] if history.get("test_acc") else None
    box_text = f"Train: {final_train_acc:.4f}\nVal: {final_val_acc:.4f}"
    if final_test_acc is not None:
        box_text += f"\nTest: {final_test_acc:.4f}"

    axes[1].text(
        0.95,
        0.15,
        box_text,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def main(dir, single_param, pretrained) -> None:
    seed_everything(42)
    device = get_device()

    model_name = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
    weights_path = Path("../reproduce/weights/1024_75.pth")
    hidden_size = 128

    model = get_model(model_name)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    phoneme_to_id = get_phoneme_to_id()
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}

    pad_id = phoneme_to_id["<PAD>"]
    eos_id = phoneme_to_id["<EOS>"]
    sos_id = phoneme_to_id["<SOS>"]

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 200
    MAX_SEQ_LEN = 20
    MAX_POS_ITRV = 9
    PATIENCE = 7
    # TRAIN_SAMPLE_SIZE = 10000
    # VAL_SAMPLE_SIZE = 2000
    SAVE_DIR = Path(dir)

    train_df_all = get_train_dataset()
    wfe_df = pd.read_csv(
        "datasets/wfe_with_repetition.csv",
        converters={"Phonemes": literal_eval, "No_Stress": literal_eval},
    )

    train_df_all = train_df_all[~train_df_all["Word"].isin(wfe_df["Word"])].copy()
    train_df_all["Length"] = train_df_all["No_Stress"].apply(len)
    train_val_df = train_df_all[
        (train_df_all["Length"] > 2) & (train_df_all["Length"] < MAX_POS_ITRV + 2)
    ]

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.05,
        random_state=42,
        shuffle=True,
        stratify=train_val_df["Length"],
    )

    test_df = wfe_df[wfe_df["can_repeat"] == True]

    train_loader = create_dataloader(
        train_df,
        "No_Stress",
        phoneme_to_id,
        batch_size=BATCH_SIZE,
        shuffle=True,
        max_len=MAX_SEQ_LEN,
        max_pos=MAX_POS_ITRV,
        random_replace_pos=True,
    )
    val_loader = create_dataloader(
        val_df,
        "No_Stress",
        phoneme_to_id,
        batch_size=BATCH_SIZE,
        shuffle=False,
        max_len=MAX_SEQ_LEN,
        max_pos=MAX_POS_ITRV,
        random_replace_pos=False,
        test_run=False,
    )
    test_loader = create_dataloader(
        test_df,
        "No_Stress",
        phoneme_to_id,
        batch_size=BATCH_SIZE,
        shuffle=False,
        max_len=MAX_SEQ_LEN,
        max_pos=MAX_POS_ITRV,
        random_replace_pos=False,
        test_run=False,
    )

    print(f"Batches: Train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    intervention = OnionIntervention(
        hidden_size=hidden_size,
        vocab_size=len(phoneme_to_id),
        max_position=MAX_POS_ITRV,
        pretrained_embedding=model.encoder.embedding if pretrained else None,
        single_param=single_param,
    ).to(device)

    optimizer = torch.optim.Adam(intervention.parameters(), lr=LEARNING_RATE)
    trainer = InterventionTrainer(model, intervention, optimizer, device, pad_id)

    print(f"Intervention params: {sum(p.numel() for p in intervention.parameters()):,}")
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        test_loader=test_loader,
    )

    plot_training_history(history, SAVE_DIR)

    test_loss, test_acc = trainer.evaluate(test_loader)
    print(dir)
    print(f"\nFinal test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    predictions = trainer.evaluate_with_predictions(test_loader, id_to_phoneme)
    predictions.to_csv(SAVE_DIR / "predictions.csv", index=False)
    trainer.save_scale_params(SAVE_DIR)


if __name__ == "__main__":

    # main("results/onion/C_linear_param_freeze_embedding/", single_param=False, pretrained=True)
    #C_linear_param_freeze_embedding, Final test: loss=1.8263, acc=0.4024
    #C_single_param_freeze_embedding, Final test: loss=1.6009, acc=0.4389
    main("results/onion/C_single_param_train_embedding_more_epochs/", single_param=True, pretrained=False)
    # main("results/onion/C_single_param_train_embedding/", single_param=True, pretrained=False)
    # main("results/onion/C_onion_param_train_embedding/", single_param=False, pretrained=False)
    # main("results/onion/C_onion_param_freeze_embedding/", single_param=False, pretrained=True)

# to do: 
# log epoch 0 before training, 
# more formulas, s + w(p), A (rotation and scaling with svd), for each phone or c,v.
# grid search
# different models? 
