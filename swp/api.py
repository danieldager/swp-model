from typing import overload

import torch.optim as optim
from torch.utils.data import DataLoader

from .datasets.graphemes import get_grapheme_trainloader, get_mixed_trainloader
from .datasets.phonemes import get_phoneme_trainloader
from .models.autoencoder import Bimodel, Unimodel
from .models.decoders import DecoderLSTM, DecoderRNN
from .models.encoders import CorNetEncoder, EncoderLSTM, EncoderRNN, VisualEncoder
from .models.losses import AuditoryXENT, TaskLosses
from .utils.datasets import get_phoneme_to_id
from .utils.models import (
    CNNArgs,
    ModelArgs,
    TrainArgs,
    get_model,
    get_model_name,
    get_train_args,
    get_train_name,
)


def get_traindata(
    batch_size: int,
    fold_id: int,
    include_stress: bool,
    mode: str,
) -> tuple[DataLoader, DataLoader]:
    # TODO code
    if mode == "Audio":
        train_loader = get_phoneme_trainloader(
            fold_id=fold_id,
            train=True,
            batch_size=batch_size,
            include_stress=include_stress,
        )
        valid_loader = get_phoneme_trainloader(
            fold_id=fold_id,
            train=False,
            batch_size=batch_size,
            include_stress=include_stress,
        )
    elif mode == "Visual":
        # TODO support for object/grapheme mixed dataset
        train_loader = get_grapheme_trainloader(
            fold_id=fold_id,
            train=True,
            batch_size=batch_size,
            include_stress=include_stress,
        )
        valid_loader = get_grapheme_trainloader(
            fold_id=fold_id,
            train=False,
            batch_size=batch_size,
            include_stress=include_stress,
        )
    # elif mode == "AugmentedVisual":
    elif mode == "Mixed":
        # TODO determine what it would look like
        raise NotImplementedError
    return train_loader, valid_loader


def get_train_method(
    model, training_name, batch_size, learn_rate, fold_id, include_stress, mode
):
    if training_name is not None:
        training_name = get_train_name(
            batch_size,
            learn_rate,
            fold_id,
            include_stress,
        )
    else:
        training_name = training_name
        training_args = get_train_args(training_name)
        batch_size = training_args["batch_size"]
        learn_rate = training_args["learning_rate"]
        fold_id = training_args["fold_id"]
    criterion = AuditoryXENT()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    return criterion, optimizer


@overload
def build_model(
    model_name: None,
    recurrent_type: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    tf_ratio: float,
    cnn_args: CNNArgs,
    mode: str,
) -> tuple[Unimodel | Bimodel, str]: ...


@overload
def build_model(
    model_name: str,
    recurrent_type: None,
    hidden_size: None,
    num_layers: None,
    dropout: None,
    tf_ratio: None,
    cnn_args: None,
    mode: None,
) -> tuple[Unimodel | Bimodel, str]: ...


def build_model(
    model_name: str | None = None,
    recurrent_type: str | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    tf_ratio: float | None = None,
    cnn_args: CNNArgs | None = None,
    mode: str | None = None,
) -> tuple[Unimodel | Bimodel, str]:
    # TODO docstring
    if model_name is None:
        if recurrent_type is None:
            raise  # TODO error
        if hidden_size is None:
            raise  # TODO error
        if num_layers is None:
            raise  # TODO error
        if dropout is None:
            raise  # TODO error
        if tf_ratio is None:
            raise  # TODO error
        rec_type = recurrent_type.upper()
        phoneme_to_id = get_phoneme_to_id()
        start_token_id = phoneme_to_id["<SOS>"]
        vocab_size = len(phoneme_to_id)
        if rec_type == "LSTM":
            audit_encoder = EncoderLSTM(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            decoder = DecoderLSTM(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                tf_ratio=tf_ratio,
            )
        elif rec_type == "RNN":
            audit_encoder = EncoderRNN(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            decoder = DecoderRNN(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                tf_ratio=tf_ratio,
            )
        else:
            raise ValueError(
                f"Recurrent subnetwork {rec_type} is not recognized. Try RNN or LSTM."
            )
        if mode == "Audio":
            model = Unimodel(
                encoder=audit_encoder,
                decoder=decoder,
                start_token_id=start_token_id,
            )
        else:
            if cnn_args is None:
                raise ValueError(
                    "No arguments corresponding to the visual encoder in a visual model"
                )
            visual_encoder = CorNetEncoder(
                cornet_model=cnn_args["cnn_model"],
                hidden_size=cnn_args["hidden_size"],
            )
            if mode == "Visual":
                model = Unimodel(
                    encoder=visual_encoder,
                    decoder=decoder,
                    start_token_id=start_token_id,
                )

            elif mode == "Mixed":
                model = Bimodel(
                    audit_encoder=audit_encoder,
                    visual_encoder=visual_encoder,
                    decoder=decoder,
                    start_token_id=start_token_id,
                )
            else:
                raise  # TODO error
        model_name = get_model_name(model)
    else:
        # TODO warnings if some args are not None
        if recurrent_type is not None:
            raise  # TODO Warning
        if hidden_size is not None:
            raise  # TODO Warning
        if num_layers is not None:
            raise  # TODO Warning
        if dropout is not None:
            raise  # TODO Warning
        if tf_ratio is not None:
            raise  # TODO Warning
        if cnn_args is not None:
            raise  # TODO Warning
        if mode is not None:
            raise  # TODO Warning
        model = get_model(model_name)
    return model, model_name
