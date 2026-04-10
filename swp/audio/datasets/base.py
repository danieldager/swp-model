from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from swp.audio.datasets.utils import load_audio

# Columns that must be present in a processed subset.csv for the pipeline to work.
# Note: frequency_bin is intentionally excluded — it is nullable (NA for nonwords).
REQUIRED_COLUMNS = [
    "item_id",
    "wav_path",
    "original_sample_rate",
    "duration_s",
    "speaker",
    "condition",
    "lexicality",
    "length_bin",
    "morphology",
]


class ParadigmDataset(Dataset):
    """Dataset adapter for the lab paradigm audio data.

    Reads a normalized ``processed/subset.csv`` and serves
    ``(waveform, metadata)`` pairs.

    Args:
        csv_path: path to processed subset.csv
        target_sr: target sample rate for audio loading;
            if None, audio is loaded at its original sample rate
        repo_root: root directory for resolving relative ``wav_path`` values;
            defaults to the current working directory (i.e. the repo root
            when scripts are run from ``swp-model/``)
    """

    def __init__(
        self,
        csv_path: str | Path,
        target_sr: int | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.target_sr = target_sr
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()

        self.df = pd.read_csv(self.csv_path)
        self._validate()

    def _validate(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"subset.csv is missing required columns: {missing}. "
                "Run scripts/audio/setup_paradigm.py to regenerate the processed CSV."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        row = self.df.iloc[idx]
        wav_path = self.repo_root / row["wav_path"]
        waveform, _ = load_audio(wav_path, target_sr=self.target_sr)
        metadata = row.to_dict()
        return waveform, metadata

    def item_ids(self) -> list[str]:
        return self.df["item_id"].tolist()