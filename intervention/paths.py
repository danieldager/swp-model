"""Where the bundled resources live, and how to read them.

Everything the pipeline needs at runtime ships under ``resources/`` so the package is
self-contained — no repo-root lookups, no ``sys.path`` juggling, and nothing resolved
against the current working directory:

    resources/weights/      the frozen repeat model checkpoint
    resources/phonemes/     phoneme -> id vocabularies (stress / no stress)
    resources/datasets/     word lists and phoneme feature tables
    resources/embeddings/   token-state statistics for the ``delta_*`` initialisations

The readers below replace the equivalents in the parent SWP project. Those regenerated
their inputs from scratch on a cache miss, which pulled in the whole grapheme-to-phoneme
stack (spacy, g2p_en, nltk, morphemes, wordfreq). Here the shipped artifacts *are* the
input, so a missing file is an error rather than a silent multi-minute rebuild.
"""
from __future__ import annotations

import json
from ast import literal_eval
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESOURCES_DIR = ROOT / "resources"

WEIGHTS_DIR = RESOURCES_DIR / "weights"
PHONEMES_DIR = RESOURCES_DIR / "phonemes"
DATASETS_DIR = RESOURCES_DIR / "datasets"
EMBEDDINGS_DIR = RESOURCES_DIR / "embeddings"
PLOTS_DIR = ROOT / "plots"

# Not a resource: bulk hidden-state dumps written by state_analysis. Anchored to the
# package so the cell-style scripts there no longer depend on the working directory.
STATES_DIR = ROOT / "states_ds"

# Phonemes stored as a stringified list ("['K', 'AE', 'T']") in every shipped csv.
_SEQ_CONVERTERS = {"Phonemes": literal_eval, "No_Stress": literal_eval}


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")
    return path


def resolve_weights(path_str: str) -> Path:
    """Locate a checkpoint given as absolute, package-relative, or a bare filename."""
    candidates = [Path(path_str), ROOT / path_str, WEIGHTS_DIR / Path(path_str).name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find weights at any of: {[str(c) for c in candidates]}"
    )


@lru_cache(maxsize=2)
def get_phoneme_to_id(include_stress: bool = False) -> dict[str, int]:
    """The repeat model's vocabulary: phoneme (or ``<PAD>``/``<SOS>``/``<EOS>``) -> id."""
    name = f'phonemes_to_id{"_sw" if include_stress else "_sn"}.json'
    with _require(PHONEMES_DIR / name, "phoneme vocabulary").open() as f:
        return json.load(f)


def get_train_dataset() -> pd.DataFrame:
    """The real-word training set the source and target sequences are drawn from."""
    return pd.read_csv(
        _require(DATASETS_DIR / "training.csv", "training word list"),
        index_col=0,
        converters={"Word": str, **_SEQ_CONVERTERS},
    )


def get_wfe_dataset(with_repetition: bool = True) -> pd.DataFrame:
    """Held-out word/non-word evaluation frame (excluded from the training words)."""
    name = "wfe_with_repetition.csv" if with_repetition else "wfe.csv"
    return pd.read_csv(_require(DATASETS_DIR / name, "WFE frame"), converters=_SEQ_CONVERTERS)


def get_phoneme_features() -> dict[str, dict[str, str]]:
    """Per-phoneme articulatory features, keyed by phoneme symbol."""
    info = pd.read_csv(_require(DATASETS_DIR / "phonemes.csv", "phoneme feature table"))
    return info.set_index("Phoneme").to_dict("index")


def state_stats_path(n: int = 1) -> Path:
    """Token-state statistics for ``embedding_init='delta_*'`` over the n-gram vocabulary."""
    name = "phoneme_state_embeddings.npz" if n == 1 else f"ngram{n}_state_embeddings.npz"
    return EMBEDDINGS_DIR / name
