"""Runner: train one intervention for one seed and save its artifacts.

``run_experiment`` is method-agnostic — it dispatches on ``cfg.method.model`` to build
either the scale or the DAS intervention, then shares the same data, trainer, and I/O.
It writes ``config.json``, ``history.csv``, ``predictions.csv`` and ``params.npz``
(the interpretable parameters used for CI/bar plots); the trained weights are only saved
when ``save_model=True``. Plotting is a separate step (see ``analysis_plots``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from intervention.models.das import DASTrainer, build_das_intervention
from intervention.config import ExperimentConfig
from intervention.data import build_loaders
from intervention.models.additive_intervention import build_scale_intervention
from intervention.models.repeat_model import get_model
from intervention.experiments.trainer import InterventionTrainer
from intervention.paths import get_phoneme_to_id, resolve_weights
from intervention.utils import set_device


def _load_repeat_model(train_cfg, device) -> torch.nn.Module:
    model = get_model(train_cfg.model_name)
    model.load_state_dict(torch.load(resolve_weights(train_cfg.weights_path), map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _build_method(cfg: ExperimentConfig, repeat_model, max_position, phoneme_to_id,
                  device, pad_id, ngram_vocab=None):
    """Return (intervention, trainer) for the configured method."""
    if cfg.method.is_das:
        intervention = build_das_intervention(
            cfg.method, cfg.data, cfg.train.hidden_size, max_position
        ).to(device)
        trainer_cls = DASTrainer
    else:  # scale
        intervention = build_scale_intervention(
            cfg.method, repeat_model, cfg.train.hidden_size, max_position, phoneme_to_id,
            ngram_vocab=ngram_vocab,
        ).to(device)
        trainer_cls = InterventionTrainer

    optimizer = torch.optim.Adam(intervention.parameters(), lr=cfg.train.learning_rate)
    trainer = trainer_cls(repeat_model, intervention, optimizer, device, pad_id, cfg.train.teacher_forcing)
    return intervention, trainer


def run_experiment(
    cfg: ExperimentConfig,
    seed: int,
    save_dir: Path,
    cache_dir: Path | None = None,
    device: torch.device | None = None,
    save_model: bool = True,
    verbose: bool = False,
) -> dict[str, object]:
    cfg.validate()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or set_device()

    phoneme_to_id = get_phoneme_to_id()
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    pad_id = phoneme_to_id["<PAD>"]

    repeat_model = _load_repeat_model(cfg.train, device)
    loaders, max_position, ngram_vocab = build_loaders(
        cfg.data, seed, phoneme_to_id, repeat_model, device,
        batch_size=cfg.train.batch_size, cache_dir=cache_dir, verbose=verbose,
    )

    intervention, trainer = _build_method(cfg, repeat_model, max_position, phoneme_to_id,
                                          device, pad_id, ngram_vocab=ngram_vocab)
    n_params = sum(p.numel() for p in intervention.parameters() if p.requires_grad)
    print(f"[{cfg.run_name()} | seed {seed}] {cfg.method.model}, {n_params} trainable params")

    start = time.perf_counter()
    history = trainer.fit(
        loaders["train"], loaders["val"],
        num_epochs=cfg.train.num_epochs, patience=cfg.train.patience, min_delta=cfg.train.min_delta,
        test_loader=loaders.get("test"), verbose=verbose,
    )
    test_loss, test_acc = trainer.evaluate(loaders["test"])
    print(f"[{cfg.run_name()} | seed {seed}] test acc={test_acc:.4f} loss={test_loss:.4f} "
          f"({(time.perf_counter() - start) / 60:.2f} min)")

    # --- artifacts ---
    cfg.save(save_dir / "config.json")
    pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)
    trainer.save_params(save_dir)

    # For n-gram edits, old/new tokens index the n-gram vocab: persist the id -> "P1 P2"
    # mapping (for labelling embeddings later) and use it in the predictions CSV.
    edit_labels = None
    if ngram_vocab is not None:
        edit_labels = {i: " ".join(id_to_phoneme[t] for t in gram) for gram, i in ngram_vocab.items()}
        with open(save_dir / "ngram_vocab.json", "w", encoding="utf-8") as f:
            json.dump(edit_labels, f, indent=0)
    trainer.evaluate_with_predictions(
        loaders["test"], id_to_phoneme, edit_id_to_str=edit_labels
    ).to_csv(save_dir / "predictions.csv", index=False)
    if save_model:
        torch.save(intervention.state_dict(), save_dir / "intervention.pth")

    best_epoch = int(np.argmin(history["val_loss"]))
    return {
        "run_dir": str(save_dir),
        "run_name": cfg.run_name(),
        "seed": seed,
        "epochs": len(history["val_loss"]),
        "best_epoch": best_epoch,
        "val_acc": history["val_acc"][best_epoch],
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
    }
