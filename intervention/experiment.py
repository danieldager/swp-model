from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import TYPE_CHECKING
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from swp.datasets.phonemes import get_phoneme_to_id
from swp.utils.datasets import get_train_dataset
from swp.utils.models import get_model
from swp.utils.setup import set_device as get_device

from intervention.analysis_plots import plot_run_summary
from intervention.core import ScaleIntervention, create_dataloader, InterventionTrainer
from intervention.embedding_utils import load_token_embedding_from_stats

if TYPE_CHECKING:
    from intervention.grid_search import InterventionConfig


def run_experiment(config: InterventionConfig, save_dir: Path, verbose: bool = False) -> dict[str, object]:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    device = get_device()
    repeat_model = get_model(config.model_name)
    state_dict = torch.load(config.weights_path, map_location=device)
    repeat_model.load_state_dict(state_dict)
    repeat_model.to(device)
    repeat_model.eval()

    phoneme_to_id = get_phoneme_to_id()
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}

    pad_id = phoneme_to_id["<PAD>"]
    eos_id = phoneme_to_id["<EOS>"]

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

    condition_list = [
        "real-pseudo",
        "real-real",
        "pseudo-real",
        "pseudo-pseudo",
    ] if config.dataset_type == "all" else [config.dataset_type]
    train_loader = create_dataloader(
        train_df,
        "No_Stress",
        phoneme_to_id,
        batch_size=config.batch_size,
        shuffle=True,
        max_len=config.max_seq_len,
        max_pos=max_position,
        random_replace_pos=not config.train_all_pos,
        repeat_model=repeat_model,
        device=device,
        rng=rng,
        max_attempts=5,
        cache_path=Path(f"cache/train_{config.dataset_type}.pt"),
        lexicality_col=config.lexicality_col,
        conditions=condition_list,
    )
    if verbose:
            print(f"train_loader created with {len(train_loader)} batches, max_position: {max_position}")

    val_loader = create_dataloader(
        val_df,
        "No_Stress",
        phoneme_to_id,
        batch_size=config.batch_size,
        shuffle=False,
        max_len=config.max_seq_len,
        max_pos=max_position,
        random_replace_pos=False,
        repeat_model=repeat_model,
        device=device,
        rng=rng,
        max_attempts=5,
        cache_path=Path(f"cache/val_{config.dataset_type}.pt"),
        lexicality_col=config.lexicality_col,
        conditions=condition_list,
    )
    if verbose:
            print(f"val_loader created with {len(val_loader)} batches.")

    test_loader = create_dataloader(
        test_df,
        "No_Stress",
        phoneme_to_id,
        batch_size=config.batch_size,
        shuffle=False,
        max_len=config.max_seq_len,
        max_pos=max_position,
        random_replace_pos=False,
        repeat_model=repeat_model,
        device=device,
        rng=rng,
        max_attempts=5,
        cache_path=Path(f"cache/test_{config.dataset_type}.pt"),
        lexicality_col=config.lexicality_col,
        conditions=condition_list,
    )
    if verbose:
            print(f"test_loader created with {len(test_loader)} batches.")
    for repeat_param in repeat_model.parameters():
        repeat_param.requires_grad = False

    if config.embedding_init == "pretrained":
        embedding = repeat_model.encoder.embedding
    elif config.embedding_init == "none":
        embedding = None
    else:
        embedding = load_token_embedding_from_stats(
            config.embedding_init,
            Path("states_ds/phoneme_state_embeddings.npz"),
            phoneme_to_id,
            repeat_model.encoder.embedding,
        )

    intervention = ScaleIntervention(
        hidden_size=config.hidden_size,
        vocab_size=len(phoneme_to_id),
        state_mode=config.state_mode,
        scale_param=config.scale_param,
        max_position=max_position,
        pretrained_embedding=embedding,
        train_embedding=config.train_embedding,
    ).to(device)

    optimizer = torch.optim.Adam(intervention.parameters(), lr=config.learning_rate)
    trainer = InterventionTrainer(repeat_model, intervention, optimizer, device, pad_id, config.teacher_forcing)
    
    from intervention.grid_search import make_run_name
    run_name = make_run_name(config)
    print(f"Starting experiment: {run_name}")
    print(f"number of trainable parameters: {sum(p.numel() for p in intervention.parameters() if p.requires_grad)}")
    start_time = time.perf_counter()
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=config.num_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        test_loader=test_loader,
        verbose=verbose,
    )
    duration_min = (time.perf_counter() - start_time) / 60.0
    print(f"Training duration: {duration_min:.2f} minutes")

    final_test_loss, final_test_acc = trainer.evaluate(test_loader)
    print(f"{run_name} Test Accuracy: {final_test_acc:.4f}, Final Test Loss: {final_test_loss:.4f}")

    from intervention.grid_search import save_history_csv

    config.save_experiment_config(save_dir)
    save_history_csv(history, save_dir)
    predictions = trainer.evaluate_with_predictions(test_loader, id_to_phoneme)
    predictions.to_csv(save_dir / "predictions.csv", index=False)
    trainer.save_scale_params(save_dir)
    torch.save(intervention.state_dict(), save_dir / "intervention_model.pth")
    plot_run_summary(
        save_dir,
        feature_cols=["position", "Lexicality", "Size", "Morphology", "type-change", "Condition"],
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
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
    }
