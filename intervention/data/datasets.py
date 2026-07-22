"""Intervention datasets and the (cache-worthy) example builders.

Two families share one batch schema (``input, target, old_token, new_token, position,
seq_len``):
  * real-real       — minimal pairs of real words differing at one position
  * source/modified — a real word paired with a synthetically edited version

The synthetic edit replaces one phoneme (``edit_ngram=1``) or splices one attested
bi-/tri-gram over an n-position window (``edit_ngram`` 2 or 3); in the n-gram case
``old_token``/``new_token`` index the n-gram vocabulary from :func:`build_ngram_vocab`.
Datasets themselves are cheap — they only pad/tensorise pre-built examples.
"""
from __future__ import annotations

from ast import literal_eval
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from intervention.models.repeat_model import can_repeat

SPECIAL_PHONEMES = ("<PAD>", "<EOS>", "<SOS>")


# --------------------------------------------------------------------------- #
# Reference sets + edit filters (keep synthetic edits "in distribution")
# --------------------------------------------------------------------------- #
def _token_type(id_to_phoneme: dict[int, str], vowels: set[str]) -> dict[int, str]:
    return {
        token_id: ("V" if token in vowels else "C")
        for token_id, token in id_to_phoneme.items()
        if token not in set(SPECIAL_PHONEMES)
    }


def build_reference_sets(
    sequences: list[list[int]],
    id_to_phoneme: dict[int, str],
    vowels: set[str],
    special_tokens: set[int],
) -> dict[str, object]:
    """Attested consonant/vowel patterns and bi-/tri-grams from the training words."""
    id_to_type = _token_type(id_to_phoneme, vowels)
    cv_patterns: set[str] = set()
    bigrams: set[tuple[int, int]] = set()
    trigrams: set[tuple[int, int, int]] = set()

    for seq in sequences:
        clean = [t for t in seq if t not in special_tokens]
        if not clean:
            continue
        cv_patterns.add("".join(id_to_type.get(t, "C") for t in clean))
        bigrams.update(zip(clean, clean[1:]))
        trigrams.update(zip(clean, clean[1:], clean[2:]))

    return {"id_to_type": id_to_type, "cv_patterns": cv_patterns,
            "bigrams": bigrams, "trigrams": trigrams}


def build_ngram_vocab(sequences: list[list[int]], n: int) -> dict[tuple[int, ...], int]:
    """Attested n-grams across ``sequences`` -> stable ids (sorted, so deterministic
    across runs/seeds given the same words). This is the edit inventory *and* the
    embedding vocabulary for ``edit_ngram > 1`` runs."""
    grams = sorted({tuple(seq[i : i + n]) for seq in sequences for i in range(len(seq) - n + 1)})
    return {gram: idx for idx, gram in enumerate(grams)}


def _passes_filters(
    seq: list[int],
    special_tokens: set[int],
    check_cv_pattern: bool,
    check_n_gram: int,
    refs: dict[str, object],
) -> bool:
    clean = [t for t in seq if t not in special_tokens]
    if check_cv_pattern:
        pattern = "".join(refs["id_to_type"].get(t, "C") for t in clean)
        if pattern not in refs["cv_patterns"]:
            return False
    if check_n_gram == 2 and len(clean) >= 2:
        if any(g not in refs["bigrams"] for g in zip(clean, clean[1:])):
            return False
    if check_n_gram == 3 and len(clean) >= 3:
        if any(g not in refs["trigrams"] for g in zip(clean, clean[1:], clean[2:])):
            return False
    return True


# --------------------------------------------------------------------------- #
# Example builders (the expensive, cache-worthy step)
# --------------------------------------------------------------------------- #
def build_real_real_pairs(
    df, phoneme_col: str, phoneme_to_id: dict[str, int]
) -> list[tuple[list[int], list[int], int, int, int]]:
    """All ordered minimal pairs (a, b) that differ at exactly one position."""
    groups: dict[tuple[int, tuple[int, ...]], list[list[int]]] = defaultdict(list)
    for seq in df[phoneme_col]:
        seq = literal_eval(seq) if isinstance(seq, str) else seq
        ids = [phoneme_to_id[p] for p in seq]
        for pos in range(len(ids)):
            groups[(pos, tuple(ids[:pos] + ids[pos + 1 :]))].append(ids)

    pairs: list[tuple[list[int], list[int], int, int, int]] = []
    for (pos, _), seqs in groups.items():
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                a, b = seqs[i], seqs[j]
                pairs.append((a, b, pos, a[pos], b[pos]))
                pairs.append((b, a, pos, b[pos], a[pos]))
    return pairs


def build_real_real_ngram_pairs(
    df, phoneme_col: str, phoneme_to_id: dict[str, int], n: int,
    ngram_vocab: dict[tuple[int, ...], int],
) -> list[tuple[list[int], list[int], int, int, int]]:
    """All ordered pairs of real words that share everything outside one ``n``-position
    window (any difference inside it counts, matching the synthetic replacement rule).
    ``old/new_token`` are ids into ``ngram_vocab``; unattested windows are skipped."""
    groups: dict[tuple[int, tuple[int, ...]], list[list[int]]] = defaultdict(list)
    for seq in df[phoneme_col]:
        seq = literal_eval(seq) if isinstance(seq, str) else seq
        ids = [phoneme_to_id[p] for p in seq]
        for pos in range(len(ids) - n + 1):
            groups[(pos, tuple(ids[:pos] + ids[pos + n :]))].append(ids)

    pairs: list[tuple[list[int], list[int], int, int, int]] = []
    for (pos, _), seqs in groups.items():
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                a, b = seqs[i], seqs[j]
                ga, gb = tuple(a[pos : pos + n]), tuple(b[pos : pos + n])
                if ga == gb or ga not in ngram_vocab or gb not in ngram_vocab:
                    continue
                pairs.append((a, b, pos, ngram_vocab[ga], ngram_vocab[gb]))
                pairs.append((b, a, pos, ngram_vocab[gb], ngram_vocab[ga]))
    return pairs


def build_synthetic_examples(
    sequences: list[list[int]],
    *,
    vocab_size: int,
    special_tokens: set[int],
    max_pos: int,
    random_replace_pos: bool,
    rng: np.random.Generator,
    repeat_model: torch.nn.Module | None,
    device: torch.device | None,
    check_repeat: bool,
    check_cv_pattern: bool,
    check_n_gram: int,
    refs: dict[str, object],
    max_attempts: int = 5,
    edit_ngram: int = 1,
    ngram_vocab: dict[tuple[int, ...], int] | None = None,
) -> list[dict[str, object]]:
    """For each word, pick edited start position(s) and sample a replacement that survives
    the enabled filters. ``random_replace_pos`` picks one random position per word;
    otherwise every valid position (up to ``max_pos``) is used.

    ``edit_ngram = 1`` replaces one phoneme (``old/new_token`` are phoneme ids).
    ``edit_ngram > 1`` splices in an attested n-gram drawn from ``ngram_vocab``
    (``old/new_token`` are then ids into that vocabulary); words shorter than the
    n-gram, or whose edited window is unattested, are skipped."""
    if edit_ngram > 1 and ngram_vocab is None:
        raise ValueError("edit_ngram > 1 requires an ngram_vocab")
    ngram_list = list(ngram_vocab) if ngram_vocab else None

    examples: list[dict[str, object]] = []
    for seq in sequences:
        n_starts = min(max_pos, len(seq) - edit_ngram + 1)
        if n_starts <= 0:
            continue
        positions = (
            [int(rng.integers(0, n_starts))] if random_replace_pos else list(range(n_starts))
        )
        for pos in positions:
            if edit_ngram == 1:
                edit = _sample_edit(
                    seq, pos, vocab_size, special_tokens, rng, repeat_model, device,
                    check_repeat, check_cv_pattern, check_n_gram, refs, max_attempts,
                )
            else:
                edit = _sample_ngram_edit(
                    seq, pos, edit_ngram, ngram_vocab, ngram_list, special_tokens, rng,
                    repeat_model, device, check_repeat, check_cv_pattern, check_n_gram,
                    refs, max_attempts,
                )
            if edit is None:
                continue
            modified_seq, old_token, new_token = edit
            examples.append({"seq": seq, "modified_seq": modified_seq,
                             "replace_pos": pos, "old_token": old_token, "new_token": new_token})
    return examples


def _sample_edit(
    seq, pos, vocab_size, special_tokens, rng, repeat_model, device,
    check_repeat, check_cv_pattern, check_n_gram, refs, max_attempts,
) -> tuple[list[int], int, int] | None:
    """Draw a single-position replacement that passes every enabled filter."""
    old_token = seq[pos]
    valid = [t for t in range(vocab_size) if t != old_token and t not in special_tokens]
    for _ in range(max_attempts):
        new_token = int(rng.choice(valid))
        modified = seq.copy()
        modified[pos] = new_token
        if (check_cv_pattern or check_n_gram) and not _passes_filters(
            modified, special_tokens, check_cv_pattern, check_n_gram, refs
        ):
            continue
        if check_repeat and repeat_model is not None and not can_repeat(repeat_model, modified, device):
            continue
        return modified, old_token, new_token
    return None


def _sample_ngram_edit(
    seq, pos, n, ngram_vocab, ngram_list, special_tokens, rng, repeat_model, device,
    check_repeat, check_cv_pattern, check_n_gram, refs, max_attempts,
) -> tuple[list[int], int, int] | None:
    """Splice an attested n-gram over ``seq[pos:pos+n]``; returns ids into ``ngram_vocab``.

    The replacement may share phonemes with the old n-gram (any attested n-gram != old
    counts as a counterfactual); the usual content filters still apply to the result."""
    old_gram = tuple(seq[pos : pos + n])
    old_id = ngram_vocab.get(old_gram)
    if old_id is None:  # window unattested in the training inventory (e.g. some test words)
        return None
    for _ in range(max_attempts):
        new_gram = ngram_list[int(rng.integers(0, len(ngram_list)))]
        if new_gram == old_gram:
            continue
        modified = seq[:pos] + list(new_gram) + seq[pos + n :]
        if (check_cv_pattern or check_n_gram) and not _passes_filters(
            modified, special_tokens, check_cv_pattern, check_n_gram, refs
        ):
            continue
        if check_repeat and repeat_model is not None and not can_repeat(repeat_model, modified, device):
            continue
        return modified, old_id, ngram_vocab[new_gram]
    return None


# --------------------------------------------------------------------------- #
# Datasets (cheap: pad/tensorise pre-built examples)
# --------------------------------------------------------------------------- #
def _pad(seq: list[int], pad_id: int, max_len: int) -> torch.Tensor:
    seq = seq + [pad_id] * max(0, max_len - len(seq))
    return torch.tensor(seq[:max_len], dtype=torch.long)


class RealRealDataset(Dataset):
    def __init__(self, pairs, pad_id: int, eos_id: int, max_len: int = 20):
        self.pairs, self.pad_id, self.eos_id, self.max_len = pairs, pad_id, eos_id, max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src, tgt, pos, old_token, new_token = self.pairs[idx]
        return {
            "input": _pad(src + [self.eos_id], self.pad_id, self.max_len),
            "target": _pad(tgt + [self.eos_id], self.pad_id, self.max_len),
            "old_token": torch.tensor(old_token, dtype=torch.long),
            "new_token": torch.tensor(new_token, dtype=torch.long),
            "position": torch.tensor(pos, dtype=torch.long),
            "seq_len": torch.tensor(len(src) + 1, dtype=torch.long),
        }


class SyntheticDataset(Dataset):
    """``role`` (the dataset name) swaps which of the original/edited word is input vs target."""

    def __init__(self, examples, role: str, pad_id: int, eos_id: int, max_len: int = 20):
        self.examples, self.role = examples, role
        self.pad_id, self.eos_id, self.max_len = pad_id, eos_id, max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        original = ex["seq"] + [self.eos_id]
        modified = ex["modified_seq"] + [self.eos_id]
        if self.role == "source-modified":         # original in -> predict edited
            inp, tgt, old_token, new_token = original, modified, ex["old_token"], ex["new_token"]
        else:                                       # modified-source: edited in -> predict original
            inp, tgt, old_token, new_token = modified, original, ex["new_token"], ex["old_token"]
        return {
            "input": _pad(inp, self.pad_id, self.max_len),
            "target": _pad(tgt, self.pad_id, self.max_len),
            "old_token": torch.tensor(old_token, dtype=torch.long),
            "new_token": torch.tensor(new_token, dtype=torch.long),
            "position": torch.tensor(ex["replace_pos"], dtype=torch.long),
            "seq_len": torch.tensor(len(original), dtype=torch.long),
        }
