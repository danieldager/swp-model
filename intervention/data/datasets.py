"""Intervention examples and the (cache-worthy) builders.

Every example, whatever the dataset family, is the same four things:

    input   the word the frozen model encodes  (the DAS "base")
    target  the word it must produce           (the counterfactual)
    source  the word the intervention reads from
    position, old_token, new_token             where the edit is and what it swapped

``source`` is always *random-context*: it carries only the phonemes the target needs at
the edit window (see :func:`keep_window`) and resamples everything else from the attested-
bigram chain, so the intervention cannot read surrounding context off it.

Two families produce those examples:
  * real-real       — ordered minimal pairs of real words
  * source/modified — a real word paired with a synthetically edited version

``edit_ngram`` (1|2|3) is the width of the edit; for n > 1 ``old_token``/``new_token``
index the n-gram vocabulary from :func:`build_ngram_vocab` instead of the phoneme table.
"""
from __future__ import annotations

from ast import literal_eval
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from intervention.data.markov import sample_sequence
from intervention.models.repeat_model_utils import can_repeat

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


class Gate:
    """The content filters + the frozen-model repeat check, bundled once.

    Applied to every *generated* sequence — synthetic edits and the sampled source.
    Real lexicon words are never gated, so all three dataset families treat attested
    words the same way.
    """

    def __init__(self, *, special_tokens, check_repeat, check_cv_pattern, check_n_gram,
                 refs, repeat_model, device, eos_id=None, pad_id=0, max_len=20):
        self.special_tokens = special_tokens
        self.check_repeat = check_repeat
        self.check_cv_pattern = check_cv_pattern
        self.check_n_gram = check_n_gram
        self.refs = refs
        self.repeat_model = repeat_model
        self.device = device
        # The repeat check runs in the same format the trainer feeds (seq + EOS, padded),
        # so "repeatable" and "reachable by the intervention" mean the same thing.
        self.eos_id, self.pad_id, self.max_len = eos_id, pad_id, max_len

    def accepts(self, seq: list[int]) -> bool:
        if (self.check_cv_pattern or self.check_n_gram) and not _passes_filters(
            seq, self.special_tokens, self.check_cv_pattern, self.check_n_gram, self.refs
        ):
            return False
        if self.check_repeat and self.repeat_model is not None:
            return can_repeat(self.repeat_model, seq, self.device,
                              self.eos_id, self.pad_id, self.max_len)
        return True


# --------------------------------------------------------------------------- #
# Sampling: one edit sampler, one source sampler
# --------------------------------------------------------------------------- #
def sample_edit(
    seq: list[int],
    pos: int,
    n: int,
    *,
    gate: Gate,
    rng: np.random.Generator,
    chain: dict | None,
    ngram_vocab: dict[tuple[int, ...], int] | None,
    ngram_list: list[tuple[int, ...]] | None,
    vocab_size: int,
    edit_sampler: str,
    max_attempts: int = 5,
) -> tuple[list[int], int, int] | None:
    """Replace ``seq[pos:pos+n]``; returns ``(modified, old_token, new_token)`` or None.

    ``edit_sampler='uniform'`` draws the replacement without regard to its neighbours
    (a single phoneme for n=1, an attested n-gram otherwise); ``'markov'`` resamples the
    window from the attested-bigram chain, so the edit is attested in context.
    """
    old_gram = tuple(seq[pos : pos + n])
    if ngram_vocab is not None and old_gram not in ngram_vocab:
        return None  # window unattested in the training inventory
    tok = (lambda g: ngram_vocab[g]) if ngram_vocab is not None else (lambda g: g[0])

    outside = {i: t for i, t in enumerate(seq) if not pos <= i < pos + n}
    uniform_pool = [t for t in range(vocab_size) if t not in gate.special_tokens]

    for _ in range(max_attempts):
        if edit_sampler == "markov":
            modified = sample_sequence(len(seq), outside, chain, rng)
            if modified is None:
                continue
        elif n == 1:
            modified = seq.copy()
            modified[pos] = int(rng.choice(uniform_pool))
        else:
            gram = ngram_list[int(rng.integers(0, len(ngram_list)))]
            modified = seq[:pos] + list(gram) + seq[pos + n :]

        new_gram = tuple(modified[pos : pos + n])
        if new_gram == old_gram:
            continue
        if ngram_vocab is not None and new_gram not in ngram_vocab:
            continue
        if not gate.accepts(modified):
            continue
        return modified, tok(old_gram), tok(new_gram)
    return None


def keep_window(pos: int, length: int, edit_ngram: int, var_ngram: int) -> tuple[int, int]:
    """Inclusive span of the target the source must carry.

    A DAS variable holds ``var_ngram`` phonemes, so every variable overlapping the edit
    ``[pos, pos+edit_ngram-1]`` spans somewhere in ``[pos-var_ngram+1, pos+edit_ngram+var_ngram-2]``.
    That is the whole window the source has to get right; the rest is free to be noise.
    With ``var_ngram=1, edit_ngram=1`` it collapses to the single edited phoneme.
    """
    return max(0, pos - var_ngram + 1), min(length - 1, pos + edit_ngram + var_ngram - 2)


def sample_source(
    target: list[int],
    pos: int,
    *,
    gate: Gate,
    rng: np.random.Generator,
    chain: dict,
    edit_ngram: int,
    var_ngram: int,
    max_attempts: int = 5,
) -> list[int] | None:
    """A same-length sequence agreeing with ``target`` on :func:`keep_window` only.

    Returns ``None`` if the window already covers the whole word (nothing left to
    randomise, so the source would just equal the target) or if no sampled filler
    survives the gate.
    """
    lo, hi = keep_window(pos, len(target), edit_ngram, var_ngram)
    if lo == 0 and hi == len(target) - 1:
        return None
    clamped = {i: target[i] for i in range(lo, hi + 1)}
    for _ in range(max_attempts):
        cand = sample_sequence(len(target), clamped, chain, rng)
        if cand is None or cand == target:
            continue
        if gate.accepts(cand):
            return cand
    return None


# --------------------------------------------------------------------------- #
# Example builders (the expensive, cache-worthy step)
# --------------------------------------------------------------------------- #
class SourceMaker:
    """Attaches a random-context ``source`` to an example, memoised on (target, position).

    real-real produces far more pairs than distinct targets, and every source costs a
    frozen-model forward pass, so the memo is what makes source sampling affordable.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._memo: dict[tuple, list[int] | None] = {}

    def __call__(self, target: list[int], pos: int) -> list[int] | None:
        key = (tuple(target), pos)
        if key not in self._memo:
            self._memo[key] = sample_source(target, pos, **self.kwargs)
        return self._memo[key]


def _example(inp, tgt, src, pos, old_token, new_token) -> dict[str, object]:
    return {"input": inp, "target": tgt, "source": src,
            "position": pos, "old_token": old_token, "new_token": new_token}


def usable_words(sequences: list[list[int]], min_word_len: int) -> list[list[int]]:
    """Drop words too short to carry a variable or an edit.

    ``min_word_len`` counts *phonemes*, not tokens: EOS and padding are added later by
    :class:`PairDataset` and are not positions the intervention can address.

    Real words are *not* gated on ``check_repeat``: the filters apply to generated
    sequences (synthetic edits, sampled sources), never to attested lexicon entries, so
    every family treats real words identically.
    """
    return [s for s in sequences if len(s) >= min_word_len]


def build_real_real(
    df, phoneme_col: str, phoneme_to_id: dict[str, int], n: int,
    ngram_vocab: dict[tuple[int, ...], int] | None,
    source_maker: SourceMaker, min_word_len: int,
) -> list[dict[str, object]]:
    """Ordered pairs of real words that share everything outside one ``n``-position
    window. For n=1 that is the classic minimal pair; for n>1 any difference inside the
    window counts and ``old/new_token`` index ``ngram_vocab``."""
    words = [literal_eval(s) if isinstance(s, str) else s for s in df[phoneme_col]]
    words = usable_words([[phoneme_to_id[p] for p in s] for s in words], min_word_len)

    groups: dict[tuple[int, tuple[int, ...]], list[list[int]]] = defaultdict(list)
    for ids in words:
        for pos in range(len(ids) - n + 1):
            groups[(pos, tuple(ids[:pos] + ids[pos + n :]))].append(ids)

    tok = (lambda g: ngram_vocab[g]) if ngram_vocab is not None else (lambda g: g[0])
    examples: list[dict[str, object]] = []
    for (pos, _), seqs in groups.items():
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                a, b = seqs[i], seqs[j]
                ga, gb = tuple(a[pos : pos + n]), tuple(b[pos : pos + n])
                if ga == gb:
                    continue
                if ngram_vocab is not None and (ga not in ngram_vocab or gb not in ngram_vocab):
                    continue
                for base, other, g_base, g_other in ((a, b, ga, gb), (b, a, gb, ga)):
                    src = source_maker(other, pos)
                    if src is not None:
                        examples.append(_example(base, other, src, pos, tok(g_base), tok(g_other)))
    return examples


def build_synthetic(
    sequences: list[list[int]],
    role: str,
    *,
    max_pos: int,
    random_replace_pos: bool,
    rng: np.random.Generator,
    edit_ngram: int,
    source_maker: SourceMaker,
    min_word_len: int,
    **edit_kwargs,
) -> list[dict[str, object]]:
    """Edit each word at one random position (or every position) and keep what survives.

    ``role='source-modified'`` feeds the original and asks for the edited word;
    ``'modified-source'`` is the reverse. The source is always built against the target.
    """
    examples: list[dict[str, object]] = []
    for seq in usable_words(sequences, min_word_len):
        n_starts = min(max_pos, len(seq) - edit_ngram + 1)
        if n_starts <= 0:
            continue
        positions = [int(rng.integers(0, n_starts))] if random_replace_pos else range(n_starts)
        for pos in positions:
            edit = sample_edit(seq, pos, edit_ngram, rng=rng, **edit_kwargs)
            if edit is None:
                continue
            modified, old_token, new_token = edit
            if role == "source-modified":
                inp, tgt, old, new = seq, modified, old_token, new_token
            else:
                inp, tgt, old, new = modified, seq, new_token, old_token
            src = source_maker(tgt, pos)
            if src is not None:
                examples.append(_example(inp, tgt, src, pos, old, new))
    return examples


# --------------------------------------------------------------------------- #
# Dataset (cheap: pad/tensorise pre-built examples)
# --------------------------------------------------------------------------- #
def _pad(seq: list[int], pad_id: int, max_len: int) -> torch.Tensor:
    seq = seq + [pad_id] * max(0, max_len - len(seq))
    return torch.tensor(seq[:max_len], dtype=torch.long)


class PairDataset(Dataset):
    """Pads pre-built examples. ``input``/``target``/``source`` are always equal length,
    so ``seq_len`` describes all three and ``seq_len - 1`` is the phoneme count."""

    def __init__(self, examples: list[dict], pad_id: int, eos_id: int, max_len: int = 20):
        self.examples, self.pad_id, self.eos_id, self.max_len = examples, pad_id, eos_id, max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "input": _pad(ex["input"] + [self.eos_id], self.pad_id, self.max_len),
            "target": _pad(ex["target"] + [self.eos_id], self.pad_id, self.max_len),
            "source": _pad(ex["source"] + [self.eos_id], self.pad_id, self.max_len),
            "old_token": torch.tensor(ex["old_token"], dtype=torch.long),
            "new_token": torch.tensor(ex["new_token"], dtype=torch.long),
            "position": torch.tensor(ex["position"], dtype=torch.long),
            "seq_len": torch.tensor(len(ex["input"]) + 1, dtype=torch.long),
        }
