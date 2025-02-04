from typing import TypedDict

import torch

from ..models.autoencoder import Bimodel, Unimodel
from ..models.decoders import DecoderLSTM, DecoderRNN
from ..models.encoders import CorNetEncoder, EncoderLSTM, EncoderRNN
from .paths import get_weights_dir


def save_weights(
    model_name: str,
    train_name: str,
    model: Unimodel | Bimodel,
    epoch: int,
    checkpoint: int | None = None,
) -> None:
    r"""Save weights of a model for a given training procedure."""
    save_dir = get_weights_dir() / model_name / train_name
    save_dir.mkdir(exist_ok=True, parents=True)
    epoch_str = f"{epoch}"
    if checkpoint is not None:
        epoch_str = f"{epoch_str}_{checkpoint}"
    model_path = save_dir / f"{epoch_str}.pth"
    torch.save(model.state_dict(), model_path)


def load_weights(
    model: Unimodel | Bimodel,
    model_name: str,
    train_name: str,
    checkpoint: str,
    device: torch.device,
) -> None:
    r"""Load the weights of a model for a given training procedure at a specific
    epoch and potential checkpoint.
    """
    save_dir = get_weights_dir() / model_name / train_name
    model_path = save_dir / f"{checkpoint}.pth"
    model.to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.bind()


class CNNArgs(TypedDict):
    r"""TypedDict containing values required to create a visual encoder :
    `hidden_size` : hidden size of the network
    `cnn_model` : expected to contain values `"R"`, `"RT"`, `"S"` or `"Z"`.
    """

    hidden_size: int
    cnn_model: str


class ModelArgs(TypedDict):
    r"""TypedDict containing values required to create a model :
    `model_class` : expected to contain values `"Ua"`, `"Uv"` or `"B"` for Unimodel auditory, Unimodel visual and Bimodel
    `recur_type` : expected to contain values `"LSTM"` or `"RNN"`
    `hidden_size` : hidden size of the network
    `num_layers` : number of recurrent layers
    `vocab_size` : size of the vocabulary
    `droprate` : dropout ratio
    `tf_ratio` : teacher forcing ratio
    `start_token_id` : id of the token to use as first input for decoding
    `cnn_args` : `CNNArgs` dict containing the information for the visual decoder, or None if not relevant
    """

    model_class: str
    recur_type: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    droprate: float
    tf_ratio: float
    start_token_id: int
    cnn_args: CNNArgs | None


class TrainArgs(TypedDict):
    batch_size: int
    learning_rate: float
    fold_id: int | None
    include_stress: bool


def get_model_args(model_name: str) -> ModelArgs:
    r"""Create a dictionnary containing the necessary arguments to build a model
    from a `model_name`. See `ModelArgs` class for more information."""
    # TODO make modular with other cnn encoders
    big_split = model_name.split("__")
    main_name = big_split[0]
    name_split = main_name.split("_")
    model_class = name_split[0]
    recur_type = name_split[1]
    str_args = {arg[0]: arg[1:] for arg in name_split[2:]}
    cnn_args = None
    if len(big_split) > 1:
        cnn_str = big_split[1][1:]
        str_cnn_args = {arg[0]: arg[1:] for arg in cnn_str.split("_")}
        cnn_args = CNNArgs(
            {
                "hidden_size": int(str_cnn_args["h"]),
                "cnn_model": str_cnn_args["m"],
            }
        )
    model_args = ModelArgs(
        {
            "model_class": model_class,
            "recur_type": recur_type,
            "hidden_size": int(str_args["h"]),
            "num_layers": int(str_args["l"]),
            "vocab_size": int(str_args["v"]),
            "droprate": float(str_args["d"]),
            "tf_ratio": float(str_args["t"]),
            "start_token_id": int(str_args["s"]),
            "cnn_args": cnn_args,
        }
    )
    return model_args


def get_model(model_name: str) -> Unimodel | Bimodel:
    r"""Create a model corresponding to the `model_name`"""
    # TODO make modular with other CNN encoders
    model_args = get_model_args(model_name)
    recur_type = model_args["recur_type"].upper()
    if recur_type == "LSTM":
        audit_encoder_class = EncoderLSTM
        decoder_class = DecoderLSTM
    elif recur_type == "RNN":
        audit_encoder_class = EncoderRNN
        decoder_class = DecoderRNN
    else:
        raise NotImplementedError(
            f"Recurrent type {recur_type} is not currently supported"
        )
    decoder = decoder_class(
        vocab_size=model_args["vocab_size"],
        hidden_size=model_args["hidden_size"],
        num_layers=model_args["num_layers"],
        dropout=model_args["droprate"],
        tf_ratio=model_args["tf_ratio"],
    )
    model_class = model_args["model_class"]
    if model_class.startswith("U"):
        if model_class[1] == "a":
            encoder = audit_encoder_class(
                vocab_size=model_args["vocab_size"],
                hidden_size=model_args["hidden_size"],
                num_layers=model_args["num_layers"],
                dropout=model_args["droprate"],
            )
        elif model_class[1] == "v":
            if model_args["cnn_args"] is None:
                raise ValueError(
                    "No arguments corresponding to the visual encoder in a visual model"
                )
            encoder = CorNetEncoder(
                hidden_size=model_args["cnn_args"]["hidden_size"],
                cornet_model=model_args["cnn_args"]["cnn_model"],
            )
        else:
            raise ValueError(
                f"Trying to name a Unimodel that is neither auditory nor visual, type : {model_class[1:]}"
            )
        model = Unimodel(
            encoder=encoder,
            decoder=decoder,
            start_token_id=model_args["start_token_id"],
        )
    elif model_class == "B":
        audit_encoder = audit_encoder_class(
            vocab_size=model_args["vocab_size"],
            hidden_size=model_args["hidden_size"],
            num_layers=model_args["num_layers"],
            dropout=model_args["droprate"],
        )
        if model_args["cnn_args"] is None:
            raise ValueError(
                "No arguments corresponding to the visual encoder in a visual model"
            )
        visual_encoder = CorNetEncoder(
            hidden_size=model_args["cnn_args"]["hidden_size"],
            cornet_model=model_args["cnn_args"]["cnn_model"],
        )
        model = Bimodel(
            audit_encoder=audit_encoder,
            visual_encoder=visual_encoder,
            decoder=decoder,
            start_token_id=model_args["start_token_id"],
        )
    else:
        raise ValueError(f"Model class not recognized : {model_class}")
    return model


def get_model_name(model: Unimodel | Bimodel) -> str:
    r"""Returns the codified `model_name` corresponding to the `model`"""
    # TODO make modular with other CNN encoders
    cnn_str = None
    if isinstance(model, Unimodel):
        if model.is_auditory:
            model_name = "Ua"
        else:
            model_name = "Uv"
            cnn_str = f"h{model.encoder.hidden_size}_m{model.encoder.cnn_model}"
    else:
        model_name = "B"
        cnn_str = (
            f"h{model.visual_encoder.hidden_size}_m{model.visual_encoder.cnn_model}"
        )
    if isinstance(model.decoder, DecoderLSTM):
        model_name = f"{model_name}_LSTM"
    elif isinstance(model.decoder, DecoderRNN):
        model_name = f"{model_name}_RNN"
    model_name = f"{model_name}_h{model.decoder.hidden_size}"
    model_name = f"{model_name}_l{model.decoder.num_layers}"
    model_name = f"{model_name}_v{model.decoder.vocab_size}"
    model_name = f"{model_name}_d{model.decoder.droprate}"
    model_name = f"{model_name}_t{model.decoder.tf_ratio}"
    model_name = f"{model_name}_s{model.start_token_id}"
    if cnn_str is not None:
        model_name = f"{model_name}__c{cnn_str}"
    return model_name


def get_model_name_from_args(
    model_class: str,
    recur_type: str,
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    droprate: float,
    tf_ratio: float,
    start_token_id: int,
    cnn_args: CNNArgs | None = None,
    **kwargs,
) -> str:
    r"""Generate the `model_name` from the arguments that would allow to generate the model"""
    # TODO make modular with other CNN encoders
    model_name = f"{model_class}_{recur_type.upper()}"
    model_name = f"{model_name}_h{hidden_size}"
    model_name = f"{model_name}_l{num_layers}"
    model_name = f"{model_name}_v{vocab_size}"
    model_name = f"{model_name}_d{droprate}"
    model_name = f"{model_name}_t{tf_ratio}"
    model_name = f"{model_name}_s{start_token_id}"
    if cnn_args is not None:
        cnn_str = f'h{cnn_args["hidden_size"]}_m{cnn_args["cnn_model"]}'
        model_name = f"{model_name}__c{cnn_str}"
    return model_name


def get_train_name(
    batch_size: int,
    learning_rate: float,
    fold_id: int | None,
    include_stress: bool,
    **kwargs,
) -> str:
    r"""Generate the `train_name` from the training arguments."""
    train_name = (
        f"b{batch_size}_l{learning_rate}_f{'all' if fold_id is None else fold_id}"
    )
    if include_stress:
        train_name = f"{train_name}_sw"
    else:
        train_name = f"{train_name}_sn"
    # TODO add support for visual dataset, mixed or not
    return train_name


def get_train_args(train_name: str) -> TrainArgs:
    r"""Returns a dictionnary containing the arguments corresponding to the `train_name`."""
    # TODO add support for visual dataset, mixed or not
    str_args = {arg[0]: arg[1:] for arg in train_name.split("_")}
    if str_args["s"] == "w":  # include_stress
        include_stress = True
    elif str_args["s"] == "n":
        include_stress = False
    else:
        raise ValueError(f'Stress value not recognized : {str_args["s"]}')
    train_args = TrainArgs(
        {
            "batch_size": int(str_args["b"]),
            "learning_rate": float(str_args["l"]),
            "fold_id": None if str_args["f"] == "all" else int(str_args["f"]),
            "include_stress": include_stress,
        }
    )
    return train_args
