from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


import copy
import torch.nn.functional as F
from intervention.scale_intervention_model import ScaleIntervention


def get_id_to_token_type(
    id_to_phoneme: dict[int, str],
    vowels: set[str],
    consonants: set[str],
) -> dict[int, str]:
    return {
        token_id: ("V" if token in vowels else "C")
        for token_id, token in id_to_phoneme.items()
        if token not in {"<PAD>", "<EOS>", "<SOS>"}
    }


def build_train_reference_sets(
    sequences: list[list[int]],
    id_to_phoneme: dict[int, str],
    vowels: set[str],
    consonants: set[str],
    special_tokens: set[int] | None = None,
) -> tuple[set[str], set[tuple[int, int]], set[tuple[int, int, int]], dict[int, str]]:
    special_tokens = special_tokens or set()
    id_to_type = get_id_to_token_type(id_to_phoneme, vowels, consonants)

    train_cv_patterns: set[str] = set()
    train_bigrams: set[tuple[int, int]] = set()
    train_trigrams: set[tuple[int, int, int]] = set()

    for seq in sequences:
        clean = [token for token in seq if token not in special_tokens]
        if not clean:
            continue
        train_cv_patterns.add("".join(id_to_type.get(token) for token in clean))
        for i in range(len(clean) - 1):
            train_bigrams.add((clean[i], clean[i + 1]))
        for i in range(len(clean) - 2):
            train_trigrams.add((clean[i], clean[i + 1], clean[i + 2]))

    return train_cv_patterns, train_bigrams, train_trigrams, id_to_type


def build_real_real_pairs(
    df: pd.DataFrame,
    phoneme_col: str,
    phoneme_to_id: dict[str, int],
) -> list[tuple[list[int], list[int], int, int, int]]:
    groups: dict[tuple[int, tuple[int, ...]], list[list[int]]] = defaultdict(list)
    for _, row in df.iterrows():
        seq = row[phoneme_col]
        if isinstance(seq, str):
            seq = literal_eval(seq)
        seq = [phoneme_to_id[p] for p in seq]
        for pos in range(len(seq)):
            key = (pos, tuple(seq[:pos] + seq[pos + 1 :]))
            groups[key].append(seq)

    pairs: list[tuple[list[int], list[int], int, int, int]] = []
    for (pos, _), seqs in groups.items():
        if len(seqs) < 2:
            continue
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                a = seqs[i]
                b = seqs[j]
                pairs.append((a, b, pos, a[pos], b[pos]))
                pairs.append((b, a, pos, b[pos], a[pos]))
    return pairs


class RealRealInterventionDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[list[int], list[int], int, int, int]],
        pad_id: int,
        eos_id: int,
        max_len: int = 20,
    ):
        self.pairs = pairs
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        source_seq, target_seq, replace_pos, old_token, new_token = self.pairs[idx]
        source = source_seq + [self.eos_id]
        target = target_seq + [self.eos_id]
        source_padded = source + [self.pad_id] * max(0, self.max_len - len(source))
        target_padded = target + [self.pad_id] * max(0, self.max_len - len(target))
        return {
            "input": torch.tensor(source_padded[: self.max_len], dtype=torch.long),
            "target": torch.tensor(target_padded[: self.max_len], dtype=torch.long),
            "old_token": torch.tensor(old_token, dtype=torch.long),
            "new_token": torch.tensor(new_token, dtype=torch.long),
            "position": torch.tensor(replace_pos, dtype=torch.long),
            "seq_len": torch.tensor(len(source), dtype=torch.long),
        }


def create_real_real_dataloader(
    df: pd.DataFrame,
    phoneme_col: str,
    phoneme_to_id: dict[str, int],
    batch_size: int = 32,
    shuffle: bool = True,
    max_len: int = 20,
) -> DataLoader:
    pairs = build_real_real_pairs(df, phoneme_col, phoneme_to_id)
    dataset = RealRealInterventionDataset(
        pairs,
        phoneme_to_id["<PAD>"],
        phoneme_to_id["<EOS>"],
        max_len=max_len,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    repeat_model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    rng: np.random.Generator | np.random.RandomState | None = None,
    max_attempts: int = 3,
    cache_path: Path | None = None,
    check_repeat: bool = True,
    check_cv: bool = False,
    check_n_gram: int = 0,
    train_cv_patterns: set[str] | None = None,
    train_bigrams: set[tuple[int, int]] | None = None,
    train_trigrams: set[tuple[int, int, int]] | None = None,
    id_to_type: dict[int, str] | None = None,
    dataset_condition: list[str] | None = None,
) -> DataLoader:


    sequences: list[list[int]] = []
    for phonemes in df[phoneme_col]:
        if isinstance(phonemes, str):
            phonemes = literal_eval(phonemes)
        sequences.append([phoneme_to_id[p] for p in phonemes])

    if rng is None:
        rng = np.random
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
        repeat_model=repeat_model,
        device=device,
        rng=rng,
        max_attempts=max_attempts,
        cache_path=cache_path,
        check_repeat=check_repeat,
        check_cv=check_cv,
        check_n_gram=check_n_gram,
        train_cv_patterns=train_cv_patterns,
        train_bigrams=train_bigrams,
        train_trigrams=train_trigrams,
        id_to_type=id_to_type,
        dataset_condition=dataset_condition,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
        repeat_model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        rng: np.random.Generator | np.random.RandomState | None = None,
        max_attempts: int = 1,
        cache_path: Path | None = None,
        check_repeat: bool = True,
        check_cv: bool = False,
        check_n_gram: int = 0,
        train_cv_patterns: set[str] | None = None,
        train_bigrams: set[tuple[int, int]] | None = None,
        train_trigrams: set[tuple[int, int, int]] | None = None,
        id_to_type: dict[int, str] | None = None,
        dataset_condition: list[str] | None = None,
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
        self.repeat_model = repeat_model
        self.device = device
        self.rng = np.random if rng is None else rng
        self._randint = self.rng.integers if hasattr(self.rng, "integers") else self.rng.randint
        self.max_attempts = max_attempts
        self.cache_path = cache_path
        self.check_repeat = check_repeat
        self.check_cv = check_cv
        self.check_n_gram = check_n_gram
        self.train_cv_patterns = train_cv_patterns
        self.train_bigrams = train_bigrams
        self.train_trigrams = train_trigrams
        self.id_to_type = id_to_type
        self.dataset_condition = dataset_condition
        if self.check_cv and self.id_to_type is None:
            raise ValueError("id_to_type is required for CV pattern checking")
        if self.check_cv and self.train_cv_patterns is None:
            raise ValueError("train_cv_patterns is required for CV pattern checking")
        if self.check_n_gram == 2 and self.train_bigrams is None:
            raise ValueError("train_bigrams is required for bigram checking")
        if self.check_n_gram == 3 and self.train_trigrams is None:
            raise ValueError("train_trigrams is required for trigram checking")

        self.examples: list[dict[str, int | list[int]]] = []
        if self.cache_path is not None and self.cache_path.exists():
            print(f"Loading cached dataset from {self.cache_path}")
            self.examples = torch.load(self.cache_path, weights_only=False)
            return
        print("Creating intervention dataset...")
        for i, seq in enumerate(phoneme_sequences):
            seq = seq.copy()

            if self.random_replace_pos:
                replace_pos = self._randint(0, min(self.max_pos, len(seq)))
                result = self.create_modified_seq(
                    seq,
                    replace_pos,
                    self.check_cv,
                    self.check_n_gram,
                    check_repeat=self.check_repeat,
                )
                if result is None:
                    continue

                modified_seq, old_token, new_token = result
                self.examples.append(
                    {
                        "seq": seq,
                        "modified_seq": modified_seq,
                        "replace_pos": replace_pos,
                        "old_token": old_token,
                        "new_token": new_token,
                    }
                )

            else:
                pos_list = list(range(min(self.max_pos, len(seq))))
                for replace_pos in pos_list:
                    result = self.create_modified_seq(
                        seq,
                        replace_pos,
                        self.check_cv,
                        self.check_n_gram,
                        check_repeat=self.check_repeat,
                    )
                    if result is None:
                        continue

                    modified_seq, old_token, new_token = result
                    self.examples.append(
                        {
                            "seq": seq,
                            "modified_seq": modified_seq,
                            "replace_pos": replace_pos,
                            "old_token": old_token,
                            "new_token": new_token,
                        }
                    )

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.examples, self.cache_path)

    def create_modified_seq(
        self,
        source_seq: list[int],
        replace_pos: int,
        check_cv: bool,
        check_n_gram: int,
        check_repeat: bool = True,
    ) -> tuple[list[int], int, int] | None:
        old_token = source_seq[replace_pos]
        if self.test_run:
            return source_seq.copy(), old_token, old_token

        valid_tokens = [
            t
            for t in range(self.vocab_size)
            if t != old_token and t not in self.special_tokens
        ]

        for _ in range(self.max_attempts):
            new_token = self.rng.choice(valid_tokens)
            modified_seq = source_seq.copy()
            modified_seq[replace_pos] = new_token

            if check_cv and self.train_cv_patterns is not None:
                cleaned = [token for token in modified_seq if token not in self.special_tokens]
                pattern = "".join(self.id_to_type.get(token, "C") for token in cleaned)
                if pattern not in self.train_cv_patterns:
                    continue

            if check_n_gram == 2:
                clean = [token for token in modified_seq if token not in self.special_tokens]
                if len(clean) >= 2 and any(
                    tuple(clean[i : i + 2]) not in self.train_bigrams
                    for i in range(len(clean) - 1)
                ):
                    continue
            elif check_n_gram == 3:
                clean = [token for token in modified_seq if token not in self.special_tokens]
                if len(clean) >= 3 and any(
                    tuple(clean[i : i + 3]) not in self.train_trigrams
                    for i in range(len(clean) - 2)
                ):
                    continue

            if check_repeat and self.repeat_model is not None:
                if not can_repeat(self.repeat_model, modified_seq, self.device):
                    continue

            return modified_seq, old_token, new_token

        return None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        seq = example["seq"] + [self.eos_id]
        modified_seq = example["modified_seq"] + [self.eos_id]

        source_padded = seq + [self.pad_id] * max(0, self.max_len - len(seq))
        modified_padded = modified_seq + [self.pad_id] * max(0, self.max_len - len(modified_seq))

        if self.dataset_condition == "source-modified":
            return {
                # predict the modified sequence from the original sequence:
                "input": torch.tensor(source_padded[: self.max_len], dtype=torch.long),
                "target": torch.tensor(modified_padded[: self.max_len], dtype=torch.long),
                "old_token": torch.tensor(example["old_token"], dtype=torch.long),
                "new_token": torch.tensor(example["new_token"], dtype=torch.long),
                "position": torch.tensor(example["replace_pos"], dtype=torch.long),
                "seq_len": torch.tensor(len(seq), dtype=torch.long),
            }

        elif self.dataset_condition == "modified-source":
            return {
                # Store the modified sequence as input, and the original sequence as target.
                "input": torch.tensor(modified_padded[: self.max_len], dtype=torch.long),
                "target": torch.tensor(source_padded[: self.max_len], dtype=torch.long),
                "old_token": torch.tensor(example["new_token"], dtype=torch.long),
                "new_token": torch.tensor(example["old_token"], dtype=torch.long),
                "position": torch.tensor(example["replace_pos"], dtype=torch.long),
                "seq_len": torch.tensor(len(seq), dtype=torch.long),
            }
        # todo, modified-modified and source-source




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
def can_repeat(model: nn.Module, input_ids: torch.Tensor, device: torch.device) -> bool:
    model.eval()
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        h, c = get_encoder_hidden(model, input_ids, device)
        logits = decode_with_hidden(model, h, c, target=input_ids, device=device, teacher_forcing=False)
        preds = logits.argmax(dim=-1)
        return torch.equal(preds, input_ids)

def _accuracy(preds: torch.Tensor, targets: torch.Tensor, seq_lens: torch.Tensor) -> float:
    batch_size = preds.shape[0]
    seq_correct = 0
    for i in range(batch_size):
        sl = seq_lens[i].item()
        if torch.equal(preds[i, :sl], targets[i, :sl]):
            seq_correct += 1
    return seq_correct / batch_size


class InterventionTrainer:
    def __init__(
        self,
        repeat_model: torch.nn.Module,
        intervention: ScaleIntervention,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        pad_id: int,
        teacher_forcing: bool = False,
    ):
        self.repeat_model = repeat_model
        self.intervention = intervention
        self.teacher_forcing = teacher_forcing
        self.optimizer = optimizer
        self.device = device
        self.pad_id = pad_id
        self.history: dict[str, list[float]] = {}

    def _compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=self.pad_id,
        )

    def _run_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch["input"].to(self.device)
        target_ids = batch["target"].to(self.device)
        old_token = batch["old_token"].to(self.device)
        new_token = batch["new_token"].to(self.device)
        position = batch["position"].to(self.device)
        seq_len = batch["seq_len"]

        h, c = get_encoder_hidden(self.repeat_model, input_ids, self.device)
        h_mod, c_mod = self.intervention.intervene(h, c, old_token, new_token, position)
        logits = decode_with_hidden(self.repeat_model, h_mod, c_mod, target_ids, self.device, self.teacher_forcing)

        loss = self._compute_loss(logits, target_ids)
        preds = logits.argmax(dim=-1)
        return loss, preds, target_ids, seq_len

    def _run_loader(self, loader: DataLoader, training: bool) -> tuple[float, float]:
        if training:
            self.intervention.train()
            self.repeat_model.train()
        else:
            self.intervention.eval()
            self.repeat_model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in loader:
            if training:
                self.optimizer.zero_grad()

            loss, preds, targets, seq_lens = self._run_batch(batch)

            if training:
                loss.backward()
                self.optimizer.step()

            acc = _accuracy(preds, targets, seq_lens)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        return total_loss / n_batches, total_acc / n_batches

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        return self._run_loader(loader, training=True)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        return self._run_loader(loader, training=False)

    @torch.no_grad()
    def evaluate_with_predictions(self, loader: DataLoader, id_to_phoneme: dict[int, str]) -> pd.DataFrame:
        self.intervention.eval()
        self.repeat_model.eval()

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
        verbose: bool = True,   
    ) -> dict[str, list[float]]:
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.intervention.state_dict())
        epochs_without_improvement = 0

        train_loss, train_acc = self.evaluate(train_loader)
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
        if verbose:
            print(
                f"Epoch {0:3d} | "
                f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
                + (f" | Test loss={test_loss:.4f} acc={test_acc:.4f}" if test_loader is not None else "")
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(self.intervention.state_dict())
            best_epoch = 0

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
            if verbose:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
                    + (f" | Test loss={test_loss:.4f} acc={test_acc:.4f}" if test_loader is not None else "")
                )

            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.intervention.state_dict())
                best_epoch = epoch
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
        if verbose:
            print(f"Loaded best intervention state from epoch {best_epoch} with val_loss={best_val_loss:.4f}")
        return self.history

    def save_scale_params(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        params = self.intervention.get_parameters()
        np.savez(save_dir / "scale_params.npz", **params)

