from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "item_id", "phoneme", "phoneme_index", "start_s", "end_s", "duration_s",
]

OPTIONAL_PASSTHROUGH = ["word", "word_index", "tier", "textgrid_path"]

# Phone labels treated as silence / non-speech
SILENCE_LABELS: frozenset[str] = frozenset(
    {"sp", "sil", "SIL", "<eps>", "spn", "SPN", "ssp", "<sil>"}
)


def load_boundaries(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a phoneme boundary CSV.

    Required columns: item_id, phoneme, phoneme_index, start_s, end_s, duration_s
    Optional passthrough columns: word, word_index, tier, textgrid_path

    Returns a DataFrame sorted by (item_id, phoneme_index).
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Boundary CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    df = df.sort_values(["item_id", "phoneme_index"]).reset_index(drop=True)
    return df
