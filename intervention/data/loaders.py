"""The single entry point for building train/val/test loaders: ``build_loaders``.

Used by every method (scale and DAS). It owns the train/val/test split and a
**content-addressed cache**: each split's cache filename hashes every field that changes
its content (dataset, seed, filters, ...), so repeated runs — e.g. across CV seeds — are
fast and can never serve a stale dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from intervention.config import DataConfig
from intervention.data.datasets import (
    SPECIAL_PHONEMES,
    Gate,
    PairDataset,
    SourceMaker,
    build_ngram_vocab,
    build_real_real,
    build_reference_sets,
    build_synthetic,
)
from intervention.data.markov import build_chain
from intervention.paths import DATASETS_DIR, get_train_dataset

_SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}  # deterministic RNG stream per split


# --------------------------------------------------------------------------- #
# Content-addressed cache
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: Path | None, dataset: str, split: str, key_fields: dict,
                tag: str = "") -> Path | None:
    """Filename encodes *every* field that changes dataset content (via ``digest``), so
    different seeds/filters can never collide on the same file (the old bug). ``tag`` adds
    the human-readable filter suffix so caches are self-describing at a glance."""
    if cache_dir is None:
        return None
    payload = json.dumps(key_fields, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:10]
    name = "_".join(p for p in (dataset, tag, split, digest) if p)
    return Path(cache_dir) / f"{name}.pt"


def _load_or_build(cache_path: Path | None, build_fn, verbose: bool):
    if cache_path is not None and cache_path.exists():
        if verbose:
            print(f"  [cache] load {cache_path.name}")
        return torch.load(cache_path, weights_only=False)
    examples = build_fn()
    if cache_path is not None:
        # Write-then-rename so parallel workers can't read a half-written cache file.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(f"{cache_path.suffix}.tmp.{os.getpid()}")
        torch.save(examples, tmp)
        os.replace(tmp, cache_path)
    return examples


# --------------------------------------------------------------------------- #
# Source frames + splitting
# --------------------------------------------------------------------------- #
def _load_source_frames() -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Training words (minus the held-out WFE test words), the WFE frame, and vowels."""
    train_df = get_train_dataset()
    wfe_df = pd.read_csv(
        DATASETS_DIR / "wfe_with_repetition.csv",
        converters={"Phonemes": literal_eval, "No_Stress": literal_eval},
    )
    train_df = train_df[~train_df["Word"].isin(wfe_df["Word"])].copy()
    train_df["Length"] = train_df["No_Stress"].apply(len)

    # Optional cap for fast smoke tests (inherited by parallel workers via the env).
    max_words = os.environ.get("INTERVENTION_MAX_WORDS")
    if max_words:
        train_df = train_df.sample(n=min(int(max_words), len(train_df)), random_state=0).copy()

    phonemes = pd.read_csv(DATASETS_DIR / "phonemes.csv")
    vowels = set(phonemes["Phoneme"][phonemes["Type"] == "V"])
    return train_df, wfe_df, vowels


def _safe_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify by word length when every length has >=2 members, else plain split."""
    stratify = df["Length"] if df["Length"].value_counts().min() >= 2 else None
    return train_test_split(df, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify)


def _split_frames(dataset: str, seed: int, val_ratio: float,
                  train_df: pd.DataFrame, wfe_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if dataset == "real-real":
        train, holdout = _safe_split(train_df, test_size=0.4, seed=seed)
        val, test = _safe_split(holdout, test_size=0.7, seed=seed)
        return {"train": train, "val": val, "test": test}
    train, val = _safe_split(train_df, test_size=val_ratio, seed=seed)
    return {"train": train, "val": val, "test": wfe_df[wfe_df["can_repeat"] == True]}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_loaders(
    data_cfg: DataConfig,
    seed: int,
    phoneme_to_id: dict[str, int],
    repeat_model: torch.nn.Module | None,
    device: torch.device | None,
    batch_size: int = 32,
    cache_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[dict[str, DataLoader], int, dict[tuple[int, ...], int] | None]:
    """Return ``{"train"/"val"/"test": DataLoader}``, the max word length (= scale
    ``max_position`` = DAS ``n_variables``), and — for ``edit_ngram > 1`` — the n-gram
    vocabulary that ``old_token``/``new_token`` index (``None`` for phoneme edits)."""
    data_cfg.validate()
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    pad_id, eos_id = phoneme_to_id["<PAD>"], phoneme_to_id["<EOS>"]
    special_tokens = {phoneme_to_id[p] for p in SPECIAL_PHONEMES}

    train_df, wfe_df, vowels = _load_source_frames()
    max_position = int(train_df["Length"].max())
    frames = _split_frames(data_cfg.dataset, seed, data_cfg.val_ratio, train_df, wfe_df)

    # Reference sets, the n-gram inventory and the Markov chain all come from *all* train
    # words, so they are identical across seeds/splits and cached examples hold stable ids.
    all_sequences = [[phoneme_to_id[p] for p in s] for s in train_df["No_Stress"]]
    refs: dict[str, object] = {}
    ngram_vocab: dict[tuple[int, ...], int] | None = None
    # The sampled source is a generated sequence too, so it needs the filters even when
    # the (real-real) edits themselves do not.
    if data_cfg.check_cv_pattern or data_cfg.check_n_gram:
        refs = build_reference_sets(all_sequences, id_to_phoneme, vowels, special_tokens)
    if data_cfg.edit_ngram > 1:
        ngram_vocab = build_ngram_vocab(all_sequences, data_cfg.edit_ngram)
        if verbose:
            print(f"  ngram vocab: {len(ngram_vocab)} attested {data_cfg.edit_ngram}-grams")
    # The random-context source always samples from the chain (markov edits do too).
    chain = build_chain(all_sequences, special_tokens)
    if verbose:
        print(f"  markov chain: {sum(len(v) for v in chain['succ'].values())} attested bigrams")

    gate = Gate(special_tokens=special_tokens, check_repeat=data_cfg.check_repeat,
                check_cv_pattern=data_cfg.check_cv_pattern, check_n_gram=data_cfg.check_n_gram,
                refs=refs, repeat_model=repeat_model, device=device,
                eos_id=eos_id, pad_id=pad_id, max_len=data_cfg.max_seq_len)

    loaders: dict[str, DataLoader] = {}
    for split, frame in frames.items():
        # The cache key is the full data identity for this split (never batch_size).
        key = {**data_cfg.cache_fields(), "seed": seed, "split": split}
        cache_path = _cache_path(cache_dir, data_cfg.dataset, split, key,
                                 tag="_".join(data_cfg.filter_tokens()))

        # Independent per-(seed, split) RNG -> each split reproducible and cacheable alone.
        split_rng = np.random.default_rng([seed, _SPLIT_INDEX[split]])
        source_maker = SourceMaker(
            gate=gate, rng=split_rng, chain=chain,
            edit_ngram=data_cfg.edit_ngram, var_ngram=data_cfg.var_ngram,
        )

        if data_cfg.dataset == "real-real":
            build_fn = lambda f=frame, sm=source_maker: build_real_real(
                f, "No_Stress", phoneme_to_id, data_cfg.edit_ngram, ngram_vocab, sm,
                data_cfg.min_word_len)
        else:
            sequences = [[phoneme_to_id[p] for p in s] for s in frame["No_Stress"]]
            random_replace_pos = (split == "train") and not data_cfg.train_all_pos
            build_fn = lambda s=sequences, rp=random_replace_pos, r=split_rng, sm=source_maker: build_synthetic(
                s, data_cfg.dataset, max_pos=max_position, random_replace_pos=rp, rng=r,
                edit_ngram=data_cfg.edit_ngram, source_maker=sm,
                min_word_len=data_cfg.min_word_len,
                gate=gate, chain=chain, ngram_vocab=ngram_vocab,
                ngram_list=list(ngram_vocab) if ngram_vocab else None,
                vocab_size=len(phoneme_to_id), edit_sampler=data_cfg.edit_sampler,
            )

        examples = _load_or_build(cache_path, build_fn, verbose)
        dataset: Dataset = PairDataset(examples, pad_id, eos_id, data_cfg.max_seq_len)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
        if verbose:
            print(f"  {split}_loader: {len(dataset)} examples, {len(loaders[split])} batches")

    return loaders, max_position, ngram_vocab
