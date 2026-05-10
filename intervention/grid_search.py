from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable
from ast import literal_eval

import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from swp.datasets.phonemes import get_phoneme_to_id
from swp.utils.datasets import get_train_dataset
from swp.utils.models import get_model
from swp.utils.setup import set_device as get_device

from analysis import plot_run_summary
from intervention_core import ScaleIntervention, create_dataloader, InterventionTrainer


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


def save_experiment_config(config: InterventionConfig, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)


def save_history_csv(history: dict[str, list[float]], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    num_epochs = len(history["train_loss"])

    for epoch in range(num_epochs):
        rows.append(
            {
                "epoch": epoch,
                "train_loss": history["train_loss"][epoch],
                "train_acc": history["train_acc"][epoch],
                "val_loss": history["val_loss"][epoch],
                "val_acc": history["val_acc"][epoch],
                "test_loss": history["test_loss"][epoch] if history.get("test_loss") else None,
                "test_acc": history["test_acc"][epoch] if history.get("test_acc") else None,
            }
        )

    pd.DataFrame(rows).to_csv(save_dir / "history.csv", index=False)


def make_run_name(config: InterventionConfig) -> str:
    parts = [
        f"scale_model={config.scale_param}",
        f"state={config.state_mode}",
        f"pretrained_embedding={int(config.pretrained_embedding)}",
        f"freeze_embedding={int(config.freeze_embedding)}",
    ]
    return "~".join(parts)


def should_skip_config(config: InterventionConfig) -> bool:
    return config.pretrained_embedding is False and config.freeze_embedding is True


def run_experiment(config: InterventionConfig, save_dir: Path, verbose: bool = False) -> dict[str, object]:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    device = get_device()
    model = get_model(config.model_name)
    state_dict = torch.load(config.weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    phoneme_to_id = get_phoneme_to_id()
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}

    pad_id = phoneme_to_id["<PAD>"]
    eos_id = phoneme_to_id["<EOS>"]
    sos_id = phoneme_to_id["<SOS>"]

    train_df_all = get_train_dataset()
    wfe_df = pd.read_csv(
        "datasets/wfe_with_repetition.csv",
        converters={"Phonemes": literal_eval, "No_Stress": literal_eval},
    )

    train_df_all = train_df_all[~train_df_all["Word"].isin(wfe_df["Word"])].copy()
    train_df_all["Length"] = train_df_all["No_Stress"].apply(len)
    max_position = int(train_df_all["Length"].max())

    train_df, val_df = train_test_split(
        train_df_all,
        test_size=config.val_ratio,
        random_state=config.seed,
        shuffle=True,
        stratify=train_df_all["Length"],
    )

    test_df = wfe_df[wfe_df["can_repeat"] == True]

    train_loader = create_dataloader(
        train_df,
        "No_Stress",
        phoneme_to_id,
        batch_size=config.batch_size,
        shuffle=True,
        max_len=config.max_seq_len,
        max_pos=max_position,
        random_replace_pos=True,
    )
    val_loader = create_dataloader(
        val_df,
        "No_Stress",
        phoneme_to_id,
        batch_size=config.batch_size,
        shuffle=False,
        max_len=config.max_seq_len,
        max_pos=max_position,
        random_replace_pos=False,
        test_run=False,
    )
    test_loader = create_dataloader(
        test_df,
        "No_Stress",
        phoneme_to_id,
        batch_size=config.batch_size,
        shuffle=False,
        max_len=config.max_seq_len,
        max_pos=max_position,
        random_replace_pos=False,
        test_run=False,
    )

    intervention = ScaleIntervention(
        hidden_size=config.hidden_size,
        vocab_size=len(phoneme_to_id),
        state_mode=config.state_mode,
        scale_param=config.scale_param,
        max_position=max_position,
        pretrained_embedding=model.encoder.embedding if config.pretrained_embedding else None,
        freeze_embedding=config.freeze_embedding,
    ).to(device)

    optimizer = torch.optim.Adam(intervention.parameters(), lr=config.learning_rate)
    trainer = InterventionTrainer(model, intervention, optimizer, device, pad_id)
    run_name = make_run_name(config)
    
    print(f"Starting training : {run_name}")
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=config.num_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        test_loader=test_loader,
        verbose=verbose,
    )
    # final evaluation on test set
    final_test_loss, final_test_acc = trainer.evaluate(test_loader)
    print(f"{run_name}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}, Final Test Loss: {final_test_loss:.4f}")

    save_experiment_config(config, save_dir)
    save_history_csv(history, save_dir)
    predictions = trainer.evaluate_with_predictions(test_loader, id_to_phoneme)
    predictions.to_csv(save_dir / "predictions.csv", index=False)
    trainer.save_scale_params(save_dir)
    torch.save(intervention.state_dict(), save_dir / "intervention_model.pth")
    plot_run_summary(
        save_dir,
        feature_cols=["position","Lexicality", "Size", "Morphology", "type-change", "Condition"],
    )

    return {
        "run_dir": str(save_dir),
        "state_mode": config.state_mode,
        "scale_param": config.scale_param,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "test_loss": history.get("test_loss"),
        "train_acc": history["train_acc"],
        "val_acc": history["val_acc"],
        "test_acc": history.get("test_acc"),
    }




def run_grid_search(grid: dict[str, list[object]] | Path, base_save_dir: Path) -> None:
    if isinstance(grid, Path):
        grid = load_grid_from_json(grid)

    base_save_dir.mkdir(parents=True, exist_ok=True)
    base_config = {
        "model_name": "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1",
        "weights_path": str(Path("../reproduce/weights/1024_75.pth")),
    }

    summary_rows: list[dict[str, object]] = []
    for config_dict in grid_iter(grid):
        config = InterventionConfig(**{**base_config, **config_dict})
        if should_skip_config(config):
            print(f"Skipping invalid config: {make_run_name(config)}")
            continue

        run_dir = base_save_dir / make_run_name(config)
        if run_dir.exists() and (run_dir / "history.csv").exists():
            summary_rows.append(load_run_summary_row(config, run_dir))
            continue

        history = run_experiment(config, run_dir, verbose=False)
        summary_rows.append(make_summary_row(config, run_dir, history))

    save_grid_summary(summary_rows, base_save_dir)


def _run_grid_item(config_dict: dict[str, object], base_save_dir: Path, base_config: dict[str, object]) -> dict[str, object] | None:
    config = InterventionConfig(**{**base_config, **config_dict})
    if should_skip_config(config):
        return None

    run_dir = base_save_dir / make_run_name(config)
    if run_dir.exists() and (run_dir / "history.csv").exists():
        return load_run_summary_row(config, run_dir)

    history = run_experiment(config, run_dir, verbose=False)
    return make_summary_row(config, run_dir, history)


def run_grid_search_parallel(
    grid: dict[str, list[object]] | Path,
    base_save_dir: Path,
    n_jobs: int = -1,
) -> list[dict[str, object]]:
    if isinstance(grid, Path):
        grid = load_grid_from_json(grid)

    base_save_dir.mkdir(parents=True, exist_ok=True)
    base_config = {
        "model_name": "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1",
        "weights_path": str(Path("../reproduce/weights/1024_75.pth")),
    }

    jobs = list(grid_iter(grid))
    summary_rows = Parallel(n_jobs=n_jobs)(
        delayed(_run_grid_item)(config_dict, base_save_dir, base_config)
        for config_dict in jobs
    )
    summary_rows = [row for row in summary_rows if row is not None]
    save_grid_summary(summary_rows, base_save_dir)
    return summary_rows


def make_summary_row(config: InterventionConfig, run_dir: Path, history: dict[str, list[float]]) -> dict[str, object]:
    return {
        "run_dir": str(run_dir),
        "state_mode": config.state_mode,
        "scale_param": config.scale_param,
        "pretrained_embedding": config.pretrained_embedding,
        "freeze_embedding": config.freeze_embedding,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "hidden_size": config.hidden_size,
        "num_epochs": config.num_epochs,
        "patience": config.patience,
        "min_delta": config.min_delta,
        "max_seq_len": config.max_seq_len,
        "val_ratio": config.val_ratio,
        "seed": config.seed,
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1],
        "test_loss": history["test_loss"][-1] if history.get("test_loss") else None,
        "train_acc": history["train_acc"][-1],
        "val_acc": history["val_acc"][-1],
        "test_acc": history["test_acc"][-1] if history.get("test_acc") else None,
    }


def load_run_summary_row(config: InterventionConfig, run_dir: Path) -> dict[str, object]:
    history_df = pd.read_csv(run_dir / "history.csv")
    return {
        "run_dir": str(run_dir),
        "state_mode": config.state_mode,
        "scale_param": config.scale_param,
        "pretrained_embedding": config.pretrained_embedding,
        "freeze_embedding": config.freeze_embedding,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "hidden_size": config.hidden_size,
        "num_epochs": config.num_epochs,
        "patience": config.patience,
        "min_delta": config.min_delta,
        "max_seq_len": config.max_seq_len,
        "val_ratio": config.val_ratio,
        "seed": config.seed,
        "train_loss": history_df["train_loss"].iloc[-1],
        "val_loss": history_df["val_loss"].iloc[-1],
        "test_loss": history_df["test_loss"].iloc[-1] if "test_loss" in history_df.columns else None,
        "train_acc": history_df["train_acc"].iloc[-1],
        "val_acc": history_df["val_acc"].iloc[-1],
        "test_acc": history_df["test_acc"].iloc[-1] if "test_acc" in history_df.columns else None,
    }


def save_grid_summary(rows: list[dict[str, object]], save_dir: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(save_dir / "grid_search_summary.csv", index=False)


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
