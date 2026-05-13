from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from swp.datasets.phonemes import get_phoneme_to_id
from swp.utils.datasets import get_train_dataset
from swp.utils.models import get_model
from swp.utils.setup import set_device as get_device

from intervention.analysis_plots import plot_run_summary
from intervention.core import ScaleIntervention, create_dataloader, InterventionTrainer

if TYPE_CHECKING:
    from intervention.grid_search import InterventionConfig


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
        random_replace_pos= not config.train_all_pos,
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
    )

    intervention = ScaleIntervention(
        hidden_size=config.hidden_size,
        vocab_size=len(phoneme_to_id),
        state_mode=config.state_mode,
        scale_param=config.scale_param,
        max_position=max_position,
        pretrained_embedding=model.encoder.embedding if config.pretrained_embedding else None,
        train_embedding=config.train_embedding,
    ).to(device)

    optimizer = torch.optim.Adam(intervention.parameters(), lr=config.learning_rate)
    trainer = InterventionTrainer(model, intervention, optimizer, device, pad_id, config.teacher_forcing)

    from intervention.grid_search import make_run_name
    run_name = make_run_name(config)

    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=config.num_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        test_loader=test_loader,
        verbose=verbose,
    )

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
