import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..models.autoencoder import Bimodel, Unimodel
from ..utils.datasets import get_phoneme_to_id
from ..utils.paths import get_gridsearch_test_dir


def test(
    model: Unimodel | Bimodel,
    checkpoint: str,
    device: str | torch.device,
    test_df: pd.DataFrame,
    test_loader: DataLoader,
    model_name: str,
    train_name: str,
    include_stress: bool,
    override_extra_str: str | None = None,
    verbose: bool = False,
):
    r"""Takes any pd.df with Phonemes column, and return same df with corresponding phoneme preds"""

    if isinstance(model, Unimodel) and not model.is_auditory:
        raise ValueError("This model cannot be tested with auditory data")

    if isinstance(model, Bimodel):
        model.to_audio()

    test_error = 0
    last_index = 0
    predictions = []
    phoneme_key = "Phonemes" if include_stress else "No Stress"
    phoneme_to_id = get_phoneme_to_id(include_stress)
    id_to_phoneme = list(phoneme_to_id)

    model.eval()
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)

            # Forward pass
            output = model(inputs, target)
            preds = torch.argmax(output[0], dim=-1)

            # Error computation
            mask = target != phoneme_to_id["<PAD>"]
            test_error += torch.any((preds != target) * mask, dim=1).sum().item()

            # Save predictions
            batch_size = target.shape[0]
            for i in range(batch_size):
                ground_truth = test_df.iloc[last_index + i][phoneme_key]
                phonemes = [id_to_phoneme[id] for id in preds[i, : len(ground_truth)]]
                predictions.append(phonemes)
            last_index += batch_size
    test_df["Prediction"] = predictions

    if verbose:
        print(f"test error: {test_error}/{len(test_df)}")

    if override_extra_str is not None:
        checkpoint = f"{checkpoint}~{override_extra_str}"
    results_model_dir = get_gridsearch_test_dir() / f"{model_name}~{train_name}"
    results_model_dir.mkdir(exist_ok=True, parents=True)
    test_df.to_csv(results_model_dir / f"{checkpoint}.csv")

    # TODO move this to plots.py
    # Initialize the confusion matrix
    # confusions = {}
    # for t in phoneme_stats.keys():
    #     confusions[t] = {p: 0 for p in phoneme_stats.keys()}

    # Tabulate confusion between prediction and target
    # if len(target) == len(prediction):
    #     for t, p in zip(target, prediction):
    #         confusions[t][p] += 1
