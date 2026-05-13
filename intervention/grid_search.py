from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from intervention.experiment import run_experiment

SUMMARY_FILENAME = "grid_search_summary.csv"


@dataclass
class InterventionConfig:
    model_name: str
    weights_path: str
    state_mode: str
    scale_param: str
    pretrained_embedding: bool
    learning_rate: float
    batch_size: int
    hidden_size: int
    num_epochs: int
    patience: int
    min_delta: float
    max_seq_len: int
    val_ratio: float
    seed: int
    freeze_embedding: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def should_skip_config(self) -> bool:
        return self.pretrained_embedding is False and self.freeze_embedding is True

    def save_experiment_config(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def save_history_csv(history: dict[str, list[float]], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)


def make_run_name(config: InterventionConfig) -> str:
    parts = [
        f"{config.scale_param}",
        f"state_{config.state_mode}",
        "load_embed" if config.pretrained_embedding else None,
        "train_embed" if not config.freeze_embedding else None,
    ]
    return "-".join(part for part in parts if part is not None)


def _run_config(config_dict: dict[str, object], base_save_dir: Path, skip_existing: bool) -> dict[str, object] | None:
    config = InterventionConfig(**config_dict)
    if config.should_skip_config():
        return None

    run_dir = base_save_dir / make_run_name(config)
    if skip_existing and run_dir.exists() and (run_dir / "history.csv").exists():
        return load_run_summary_row(config, run_dir)

    history = run_experiment(config, run_dir, verbose=False)
    return make_summary_row(config, run_dir, history)


def run_grid_search(
    grid: dict[str, list[object]] | Path,
    base_save_dir: Path,
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    if isinstance(grid, Path):
        grid = load_grid_from_json(grid)

    base_save_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for config_dict in grid_iter(grid):
        row = _run_config(config_dict, base_save_dir, skip_existing)
        if row is not None:
            summary_rows.append(row)

    save_grid_summary(summary_rows, base_save_dir)
    return summary_rows


def run_grid_search_parallel(
    grid: dict[str, list[object]] | Path,
    base_save_dir: Path,
    skip_existing: bool = True,
    n_jobs: int = -1,
) -> list[dict[str, object]]:
    if isinstance(grid, Path):
        grid = load_grid_from_json(grid)

    base_save_dir.mkdir(parents=True, exist_ok=True)
    jobs = list(grid_iter(grid))

    with tqdm_joblib(tqdm(desc="grid search", total=len(jobs))) as progress_bar:
        summary_rows = Parallel(n_jobs=n_jobs)(
            delayed(_run_config)(config_dict, base_save_dir, skip_existing)
            for config_dict in jobs
        )

    summary_rows = [row for row in summary_rows if row is not None]
    save_grid_summary(summary_rows, base_save_dir)
    return summary_rows


def _last(list_or_none: list[object] | object | None):
    if list_or_none is None:
        return None
    if isinstance(list_or_none, list):
        return list_or_none[-1] if list_or_none else None
    return list_or_none


def _history_stats(history: dict[str, list[float] | float]) -> dict[str, object]:
    last_epoch = len(history["val_loss"])
    best_epoch = int(min(range(last_epoch), key=lambda i: history["val_loss"][i]))

    return {
        "epochs": last_epoch,
        "best_epoch": best_epoch,
        "train_loss": _last(history.get("train_loss")),
        "train_acc": _last(history.get("train_acc")),
        "val_loss": _last(history.get("val_loss")),
        "val_acc": _last(history.get("val_acc")),
        "final_test_loss": _last(history.get("final_test_loss")) or _last(history.get("test_loss")),
        "final_test_acc": _last(history.get("final_test_acc")) or _last(history.get("test_acc")),
    }


def _history_stats_from_df(history_df: pd.DataFrame) -> dict[str, object]:
    last_row = history_df.iloc[-1]
    return {
        "epochs": len(history_df),
        "best_epoch": int(history_df["val_loss"].idxmin()),
        "train_loss": float(last_row["train_loss"]),
        "train_acc": float(last_row["train_acc"]),
        "val_loss": float(last_row["val_loss"]),
        "val_acc": float(last_row["val_acc"]),
        "final_test_loss": float(last_row["test_loss"]) if "test_loss" in history_df.columns else None,
        "final_test_acc": float(last_row["test_acc"]) if "test_acc" in history_df.columns else None,
    }


def _base_summary(config: InterventionConfig, run_dir: Path) -> dict[str, object]:
    return {
        "run_dir": str(run_dir),
        "state_mode": config.state_mode,
        "scale_param": config.scale_param,
        "pretrained_embedding": config.pretrained_embedding,
        "freeze_embedding": config.freeze_embedding,
        "hidden_size": config.hidden_size,
    }


def make_summary_row(config: InterventionConfig, run_dir: Path, history: dict[str, list[float]]) -> dict[str, object]:
    return {**_base_summary(config, run_dir), **_history_stats(history)}


def load_run_summary_row(config: InterventionConfig, run_dir: Path) -> dict[str, object]:
    history_df = pd.read_csv(run_dir / "history.csv")
    return {**_base_summary(config, run_dir), **_history_stats_from_df(history_df)}


def save_grid_summary(rows: list[dict[str, object]], save_dir: Path) -> None:
    if not rows:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_dir / SUMMARY_FILENAME
    new_df = pd.DataFrame(rows)

    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        old_df = old_df.reindex(columns=new_df.columns)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["run_dir"], keep="last")
    else:
        combined = new_df

    combined.to_csv(summary_path, index=False)


def update_grid_summary_row(row: dict[str, object], save_dir: Path) -> None:
    save_grid_summary([row], save_dir)


def grid_iter(grid: dict[str, Iterable[int | str | float | bool | None]]):
    keys = list(grid.keys())
    if not keys:
        return
    for instance in product(*grid.values()):
        yield dict(zip(keys, instance))


def load_grid_from_json(json_path: Path) -> dict[str, list[object]]:
    with open(json_path, "r", encoding="utf-8") as f:
        grid = json.load(f)
    if not isinstance(grid, dict):
        raise ValueError("Grid JSON must contain an object of parameter names to list values.")
    return grid
