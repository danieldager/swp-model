import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..datasets.phonemes import get_phoneme_testloader
from ..models.autoencoder import Bimodel, Unimodel
from ..models.decoders import DecoderLSTM, DecoderRNN
from ..models.encoders import EncoderLSTM, EncoderRNN
from ..utils.plots import confusion_matrix, create_error_plots
from ..utils.datasets import (
    get_phoneme_to_id,
    get_test_data,
    get_train_data,
    phoneme_statistics,
)
from ..utils.models import load_weigths
from ..utils.paths import get_weights_dir, get_results_dir
from .core import calculate_errors

results_test_dir = get_results_dir() / "test"
results_test_dir.mkdir(exist_ok=True, parents=True)


def beta_test(
    model: Unimodel | Bimodel,
    epoch: int,
    checkpoint: int | None,
    device: str | torch.device,
    input_df: pd.DataFrame,
    test_loader: DataLoader,
    model_name: str,
    training_name: str,
    include_stress: bool,
    override_extra_str: str | None = None,
):
    r"""Takes any pd.df with Phonemes column, and return same df with corresponding phoneme preds"""

    if include_stress:
        phoneme_key = "Phonemes"
    else:
        phoneme_key = "No Stress"
    id_to_phoneme = list(get_phoneme_to_id())
    last_index = 0
    predictions = []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs, target)
            auditory_out = output[0]
            preds = torch.argmax(auditory_out, dim=-1)

            batch_size = target.shape[0]
            for i in range(batch_size):
                ground_truth = input_df.iloc[last_index + i][phoneme_key]
                phoneme_list = [
                    id_to_phoneme[id] for id in preds[i, : len(ground_truth)]
                ]
                predictions.append(phoneme_list)
            last_index += batch_size
    input_df["Prediction"] = predictions

    model_results_dir = results_test_dir / f"{model_name}~{training_name}"
    model_results_dir.mkdir(exist_ok=True, parents=True)
    file_name = f"{epoch}"

    if checkpoint is not None:
        file_name = f"{file_name}_{checkpoint}"
    if override_extra_str is not None:
        file_name = f"{file_name}~{override_extra_str}"
    input_df.to_csv(model_results_dir / f"{file_name}.csv")


def test(
    test_loader: DataLoader,
    model: Unimodel | Bimodel,
    device: str | torch.device,
    model_name: str,
    verbose: bool,
):
    # TODO docstring
    if isinstance(model, Unimodel) and not model.is_auditory:
        raise ValueError(
            "The model provided is not made to be tested with auditory data"
        )
    if isinstance(model, Bimodel):
        model.to_audio()

    if verbose:
        print(f"\nTesting model: {model}")

    # Unpack parameters from model name
    m, h, l, v, d, t, s = [p[1:] for p in model_name.split("_")[1:]]
    if verbose:
        print(f"Parameters: e={e} h={h} l={l} d={d} t={t} l={r}")
    recurrent_type = "lstm" if m[0] == "S" else "rnn"
    h_size, n_layers, dropout, tf_ratio = int(h), int(l), int(d), float(t)

    # Unpack variables from Phonemes class
    test_data = get_test_data()
    vocab_size = len(get_phoneme_to_id())
    index_to_phone = {v: k for (k, v) in get_phoneme_to_id().items()}
    # phoneme_stats, _ = phoneme_statistics(list(get_train_data()["Phonemes"]))

    # TODO need to get num epochs
    # checkpoints = [f"1_{i}" for i in range(1, 11)]
    # epochs = checkpoints + list(range(2, n_epochs + 1))

    # For testing only the final epoch
    epochs = ["1_1", 20]

    dataframes = []
    for epoch in epochs:
        # print(f"Epoch {epoch+1}/{epochs}", end="\r")

        """LOAD MODEL"""
        model_weights_dir = get_weights_dir() / model

        if recurrent_type == "rnn":
            encoder = EncoderRNN(vocab_size, h_size, n_layers, dropout).to(device)
            decoder = DecoderRNN(vocab_size, h_size, n_layers, dropout, tf_ratio).to(
                device
            )

        elif recurrent_type == "lstm":
            encoder = EncoderLSTM(vocab_size, h_size, n_layers, dropout).to(device)
            decoder = DecoderLSTM(vocab_size, h_size, n_layers, dropout, tf_ratio).to(
                device
            )

        load_weigths(model_weights_dir, encoder, decoder, epoch, device)

        """ TESTING LOOP """
        predictions = []
        test_data = test_data.copy()

        deletions, insertions, substitutions = [], [], []
        error_indices, sequence_lengths = [], []
        edit_distances = []

        # Initialize the confusion matrix
        # confusions = {}
        # for t in phoneme_stats.keys():
        #     confusions[t] = {p: 0 for p in phoneme_stats.keys()}
        #     confusions[t]["OY0"] = 0
        # confusions["OY0"] = {p: 0 for p in phoneme_stats.keys()}
        # confusions["OY0"]["OY0"] = 0
        # TODO: Fix this hardcoding

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for inputs, target in test_loader:
                inputs = inputs.to(device)
                target = target.to(device)

                encoder_hidden = encoder(inputs)
                decoder_inputs = torch.zeros(1, inputs.shape[1], h_size, device=device)
                decoder_output = decoder(decoder_inputs, encoder_hidden)

                prediction = torch.argmax(decoder_output, dim=-1)
                prediction = prediction.squeeze().cpu().tolist()
                target = target.squeeze().cpu().tolist()

                # Calculate errors
                errors = calculate_errors(prediction, target)
                deletions.append(errors["dels"])
                insertions.append(errors["inss"])
                substitutions.append(errors["subs"])
                edit_distances.append(errors["total"])
                error_indices.append(errors["indices"])
                sequence_lengths.append(errors["length"])

                # Convert indices to phonemes
                prediction = [index_to_phone[i] for i in prediction]
                target = [index_to_phone[i] for i in target]

                # Tabulate confusion between prediction and target
                # if len(target) == len(prediction):
                #     for t, p in zip(target, prediction):
                #         confusions[t][p] += 1

                predictions.append(prediction[:-1])

        test_data["Prediction"] = predictions
        test_data["Deletions"] = deletions
        test_data["Insertions"] = insertions
        test_data["Substitutions"] = substitutions
        test_data["Edit Distance"] = edit_distances
        test_data["Error Indices"] = error_indices
        test_data["Sequence Length"] = sequence_lengths

        # error_plots(test_data, model_name, epoch)
        # confusion_matrix(confusions, model_name, epoch)

        dataframes.append(test_data)

    return dataframes


### LEGACY CODE ###
def test_repetition(model: str, device) -> list:
    print(f"\nTesting model: {model}")
    # Unpack parameters from model name
    e, h, l, d, t, r = [p[1:] for p in model.split("_")]
    print(f"Parameters: e={e} h={h} l={l} d={d} t={t} l={r}")
    n_epochs, h_size, n_layers, dropout = int(e), int(h), int(l), float(d)

    # Unpack variables from Phonemes class
    test_data = get_test_data()
    vocab_size = len(get_phoneme_to_id())
    phoneme_stats, _ = phoneme_statistics(list(get_train_data()["Phonemes"]))
    index_to_phone = {v: k for (k, v) in get_phoneme_to_id().items()}
    test_dataloader = get_phoneme_testloader(1, 0)

    # Add checkpoints for proper iteration
    checkpoints = [f"1_{i}" for i in range(1, 11)]
    epochs = checkpoints + list(range(2, n_epochs + 1))

    # For testing only the final epoch
    epochs = ["1_1", epochs[-1]]

    dataframes = []
    for epoch in epochs:
        # print(f"Epoch {epoch+1}/{epochs}", end="\r")

        """LOAD MODEL"""
        MODEL_WEIGHTS_DIR = get_weights_dir() / model
        encoder = EncoderRNN(vocab_size, h_size, n_layers, dropout).to(device)
        decoder = DecoderRNN(h_size, vocab_size, n_layers, dropout).to(device)

        encoder = EncoderLSTM(vocab_size, h_size, n_layers).to(device)
        decoder = DecoderLSTM(h_size, vocab_size, n_layers).to(device)
        load_weigths(MODEL_WEIGHTS_DIR, encoder, decoder, epoch, device)

        """ TESTING LOOP """
        predictions = []
        test_data = test_data.copy()

        deletions, insertions, substitutions = [], [], []
        error_indices, sequence_lengths = [], []
        edit_distances = []

        # Initialize the confusion matrix
        confusions = {}
        for t in phoneme_stats.keys():
            confusions[t] = {p: 0 for p in phoneme_stats.keys()}
            confusions[t]["OY0"] = 0
        confusions["OY0"] = {p: 0 for p in phoneme_stats.keys()}
        confusions["OY0"]["OY0"] = 0
        # TODO: Fix this hardcoding

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for inputs, target in test_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)

                encoder_hidden = encoder(inputs)
                decoder_inputs = torch.zeros(1, inputs.shape[1], h_size, device=device)
                decoder_output = decoder(decoder_inputs, encoder_hidden)

                prediction = torch.argmax(decoder_output, dim=-1)
                prediction = prediction.squeeze().cpu().tolist()
                target = target.squeeze().cpu().tolist()

                # Calculate errors
                errors = calculate_errors(prediction, target)
                deletions.append(errors["dels"])
                insertions.append(errors["inss"])
                substitutions.append(errors["subs"])
                edit_distances.append(errors["total"])
                error_indices.append(errors["indices"])
                sequence_lengths.append(errors["length"])

                # Convert indices to phonemes
                prediction = [index_to_phone[i] for i in prediction]
                target = [index_to_phone[i] for i in target]

                # Tabulate confusion between prediction and target
                if len(target) == len(prediction):
                    for t, p in zip(target, prediction):
                        confusions[t][p] += 1

                predictions.append(prediction[:-1])

        test_data["Prediction"] = predictions
        test_data["Deletions"] = deletions
        test_data["Insertions"] = insertions
        test_data["Substitutions"] = substitutions
        test_data["Edit Distance"] = edit_distances
        test_data["Error Indices"] = error_indices
        test_data["Sequence Length"] = sequence_lengths

        # error_plots(test_data, model, epoch)
        # confusion_matrix(confusions, model, epoch)

        dataframes.append(test_data)

    return dataframes
