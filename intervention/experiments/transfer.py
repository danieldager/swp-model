"""Evaluate trained interventions on test sets they were not trained on.

``run_transfer`` takes finished run directories and a list of evaluation ``DataConfig``s
and reports every (run, test set) cell. Because test sets differ in difficulty, each cell
carries its own floor and ceiling:

    floor    decode the base state with no intervention at all
    ceiling  encode the target directly and decode it — what the frozen model can do
    headroom (acc - floor) / (ceiling - floor), so cells are comparable across test sets

Raw accuracy across different test sets is not comparable; ``headroom`` is the column to
read. Requires ``intervention.pth``, so the source runs must be trained with
``save_model=True``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from intervention.config import ExperimentConfig
from intervention.data import build_loaders
from intervention.experiments.trainer import count_correct
from intervention.models.repeat_model_utils import decode_with_hidden, get_encoder_hidden
from intervention.experiments.runner import _build_method, _load_repeat_model
from intervention.paths import get_phoneme_to_id
from intervention.utils import set_device

# Fields that must agree between the trained model and the evaluation data: they change
# the shape or the meaning of what the intervention was fitted to.
LOCKED_FIELDS = ("edit_ngram", "var_ngram", "max_seq_len")


@torch.no_grad()
def _baselines(repeat_model, loader, device, pad_id, teacher_forcing) -> tuple[float, float]:
    """(floor, ceiling) sequence accuracy on ``loader``: no intervention, and oracle state."""
    hits = {"floor": 0, "ceiling": 0}
    total = 0
    for batch in loader:
        target = batch["target"].to(device)
        for name, ids in (("floor", batch["input"]), ("ceiling", batch["target"])):
            h, c = get_encoder_hidden(repeat_model, ids.to(device), device)
            logits = decode_with_hidden(repeat_model, h, c, target, device, teacher_forcing)
            hits[name] += count_correct(logits.argmax(-1), target, batch["seq_len"])
        total += target.shape[0]
    if total == 0:
        return float("nan"), float("nan")
    return hits["floor"] / total, hits["ceiling"] / total


def evaluate_run(
    run_dir: Path,
    eval_data: dict | ExperimentConfig,
    seed: int | None = None,
    cache_dir: Path | None = None,
    device: torch.device | None = None,
    save_predictions: bool = True,
    out_root: Path | None = None,
) -> dict[str, object]:
    """Load the intervention trained in ``run_dir`` and score it on ``eval_data``'s test split.

    ``eval_data`` is a flat dict of ``DataConfig`` overrides (e.g. ``{"dataset": "real-real"}``);
    everything not overridden is inherited from the trained run, so only the axes you name
    actually change. ``seed`` defaults to the run's own seed, which keeps the test split
    fixed; pass a different one to also resample the split.
    """
    run_dir = Path(run_dir)
    trained = ExperimentConfig.from_flat(json.loads((run_dir / "config.json").read_text()))
    seed = seed if seed is not None else int(trained.train.seeds[0])
    device = device or set_device()

    overrides = eval_data.data.__dict__ if isinstance(eval_data, ExperimentConfig) else eval_data
    cfg = ExperimentConfig.from_flat({**trained.to_flat(), **overrides, "seeds": [seed]}).validate()
    for name in LOCKED_FIELDS:
        if getattr(cfg.data, name) != getattr(trained.data, name):
            raise ValueError(
                f"{name} differs (trained {getattr(trained.data, name)}, eval "
                f"{getattr(cfg.data, name)}): the intervention was fitted to the other one"
            )

    phoneme_to_id = get_phoneme_to_id()
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    repeat_model = _load_repeat_model(trained.train, device)
    loaders, max_position, ngram_vocab = build_loaders(
        cfg.data, seed, phoneme_to_id, repeat_model, device,
        batch_size=trained.train.batch_size, cache_dir=cache_dir,
    )

    intervention, trainer = _build_method(trained, repeat_model, max_position, phoneme_to_id,
                                          device, phoneme_to_id["<PAD>"], ngram_vocab=ngram_vocab)
    intervention.load_state_dict(torch.load(run_dir / "intervention.pth", map_location=device))

    test = loaders["test"]
    loss, acc = trainer.evaluate(test)
    floor, ceiling = _baselines(repeat_model, test, device, phoneme_to_id["<PAD>"],
                                trained.train.teacher_forcing)

    span = ceiling - floor
    row = {
        "trained_on": trained.run_name(), "tested_on": cfg.run_name(), "seed": seed,
        "train_dataset": trained.data.dataset, "test_dataset": cfg.data.dataset,
        "model": trained.method.model, "state_mode": trained.method.state_mode,
        "edit_ngram": cfg.data.edit_ngram, "var_ngram": cfg.data.var_ngram,
        "in_domain": trained.run_name() == cfg.run_name(),
        "n_test": len(test.dataset), "test_acc": acc, "test_loss": loss,
        "floor": floor, "ceiling": ceiling,
        "headroom": (acc - floor) / span if span > 1e-9 else float("nan"),
    }

    # One directory per (trained run, test set, seed), holding the row plus BOTH full
    # configs, so a cell is self-describing without consulting the training run.
    if out_root is not None:
        cell = Path(out_root) / trained.run_name() / f"on__{cfg.run_name()}" / f"seed_{seed}"
        cell.mkdir(parents=True, exist_ok=True)
        with open(cell / "result.json", "w", encoding="utf-8") as f:
            json.dump({**row, "train_config": trained.to_flat(), "test_config": cfg.to_flat()},
                      f, indent=2, default=str)
        if save_predictions:
            trainer.evaluate_with_predictions(test, id_to_phoneme).to_csv(
                cell / "predictions.csv", index=False)
        row["cell_dir"] = str(cell)
    return row


def run_transfer(
    run_dirs: list[Path],
    eval_datasets: list[dict],
    out_root: Path,
    seeds: list[int] | None = None,
    cache_dir: Path | None = None,
    device: torch.device | None = None,
    save_predictions: bool = True,
) -> pd.DataFrame:
    """Every (run x eval dataset x seed) cell, written under ``out_root`` and rolled up.

    Layout::

        out_root/<trained_run>/on__<test_run>/seed_<n>/{result.json, predictions.csv}
        out_root/transfer_all.csv       one row per cell
        out_root/transfer_summary.csv   mean/std over seeds per (trained, tested)
    """
    out_root = Path(out_root)
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        for eval_data in eval_datasets:
            for seed in (seeds or [None]):
                try:
                    row = evaluate_run(run_dir, eval_data, seed, cache_dir, device,
                                       save_predictions, out_root)
                except (ValueError, FileNotFoundError) as err:
                    print(f"  skip {Path(run_dir).name} -> {eval_data}: {err}")
                    continue
                rows.append(row)
                tag = "in-domain" if row["in_domain"] else "transfer "
                print(f"  [{tag}] {row['train_dataset']:>16s} -> {row['test_dataset']:<16s} "
                      f"seed {row['seed']}: acc={row['test_acc']:.4f} "
                      f"floor={row['floor']:.4f} ceiling={row['ceiling']:.4f} "
                      f"headroom={row['headroom']:.4f}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "transfer_all.csv", index=False)

    keys = ["trained_on", "tested_on", "train_dataset", "test_dataset", "model", "in_domain"]
    summary = (df.groupby(keys, dropna=False)
                 .agg(n_seeds=("seed", "count"),
                      test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
                      headroom_mean=("headroom", "mean"), headroom_std=("headroom", "std"),
                      floor_mean=("floor", "mean"), ceiling_mean=("ceiling", "mean"),
                      n_test=("n_test", "max"))
                 .reset_index())
    summary.to_csv(out_root / "transfer_summary.csv", index=False)
    print(f"Wrote {len(df)} cells -> {out_root}/transfer_all.csv "
          f"({len(summary)} rows in transfer_summary.csv)")
    return df


def find_runs(results_dir: Path) -> list[Path]:
    """Every seed directory under ``results_dir`` that has saved weights."""
    return sorted(p.parent for p in Path(results_dir).glob("*/seed_*/intervention.pth"))
