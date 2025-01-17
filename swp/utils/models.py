from typing import Any

import torch

from swp.utils.paths import get_weights_dir

from ..models.autoencoder import Bimodel, Unimodel
from ..models.decoders import DecoderLSTM, DecoderRNN
from ..models.encoders import CorNetEncoder, EncoderLSTM, EncoderRNN


def save_weights(
    model_name: str,
    training_name: str,
    model: Unimodel | Bimodel,
    epoch: int,
    checkpoint: int | None = None,
) -> None:
    r"""Save weights of a model for a given training procedure."""
    save_dir = get_weights_dir() / model_name / training_name
    save_dir.mkdir(exist_ok=True, parents=True)
    epoch_str = f"{epoch}"
    if checkpoint is not None:
        epoch_str = f"{epoch_str}_{checkpoint}"
    model_path = save_dir / f"model_{epoch_str}.pth"
    torch.save(model.state_dict(), model_path)


def load_weigths(
    model_name: str,
    training_name: str,
    model: Unimodel | Bimodel,
    epoch: str,
    device: torch.device,
    checkpoint: int | None = None,
) -> None:
    r"""Load the weights of a model for a given training procedure at a specific
    epoch and potential checkpoint.
    """
    save_dir = get_weights_dir() / model_name / training_name
    epoch_str = f"{epoch}"
    if checkpoint is not None:
        epoch_str = f"{epoch_str}_{checkpoint}"
    model_path = save_dir / f"model_{epoch_str}.pth"
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.bind()


def get_args_from_model_name(model_name: str) -> tuple[str, str, dict[str, Any]]:
    r"""Create a dictionnary containing the necessary arguments to build a model
    from a `model_name`."""
    # TODO improve docstring
    # TODO make modular with other cnn encoders
    big_split = model_name.split("__")
    main_name = big_split[0]
    name_split = main_name.split("_")
    model_class = name_split[0]
    rec_type = name_split[1]
    str_args = {arg[0]: arg[1:] for arg in name_split[2:]}
    typed_args = {}
    typed_args["h"] = int(str_args["h"])
    typed_args["l"] = int(str_args["l"])
    typed_args["v"] = int(str_args["v"])
    typed_args["d"] = float(str_args["d"])
    typed_args["t"] = float(str_args["t"])
    typed_args["s"] = int(str_args["s"])
    if len(big_split) > 1:
        cnn_str = big_split[1][1:]
        str_cnn_args = {arg[0]: arg[1:] for arg in cnn_str.split("_")}
        typed_cnn_args = {}
        typed_cnn_args["h"] = int(str_cnn_args["h"])
        typed_cnn_args["m"] = str_cnn_args["m"]
        typed_args["c"] = typed_cnn_args
    return model_class, rec_type, typed_args


def get_model(model_name: str) -> Unimodel | Bimodel:
    r"""Create a model corresponding to the `model_name`"""
    # TODO make modular with other CNN encoders
    model_class, rec_type, typed_args = get_args_from_model_name(model_name)
    if rec_type.upper() == "LSTM":
        audit_encoder_class = EncoderLSTM
        decoder_class = DecoderLSTM
    elif rec_type.upper() == "RNN":
        audit_encoder_class = EncoderRNN
        decoder_class = DecoderRNN
    else:
        raise NotImplementedError(
            f"Recurrent type {rec_type} is not currently supported"
        )
    decoder = decoder_class(
        vocab_size=typed_args["v"],
        hidden_size=typed_args["h"],
        num_layers=typed_args["l"],
        dropout=typed_args["d"],
        tf_ratio=typed_args["t"],
    )
    if model_class.startswith("U"):
        if model_class[1] == "a":
            encoder = audit_encoder_class(
                vocab_size=typed_args["v"],
                hidden_size=typed_args["h"],
                num_layers=typed_args["l"],
                dropout=typed_args["d"],
            )
        elif model_class[1] == "v":
            encoder = CorNetEncoder(
                hidden_size=typed_args["c"]["h"], cornet_model=typed_args["c"]["m"]
            )
        else:
            raise ValueError(
                f"Trying to name a Unimodel that is neither auditory nor visual, type : {model_class[1:]}"
            )
        model = Unimodel(
            encoder=encoder, decoder=decoder, start_token_id=typed_args["s"]
        )
    elif model_class == "B":
        audit_encoder = audit_encoder_class(
            vocab_size=typed_args["v"],
            hidden_size=typed_args["h"],
            num_layers=typed_args["l"],
            dropout=typed_args["d"],
        )
        visual_encoder = CorNetEncoder(
            hidden_size=typed_args["c"]["h"], cornet_model=typed_args["c"]["m"]
        )
        model = Bimodel(
            audit_encoder=audit_encoder,
            visual_encoder=visual_encoder,
            decoder=decoder,
            start_token_id=typed_args["s"],
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
    rec_type: str,
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    droprate: float,
    tf_ratio: float,
    start_token_id: int,
    cnn_args: dict[str, Any] | None = None,
) -> str:
    r"""Generate the `model_name` from the arguments that would allow to generate the model"""
    # TODO make modular with other CNN encoders
    model_name = f"{model_class}_{rec_type.upper()}"
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


def get_training_name(batch_size: int, learning_rate: float, fold_id: int) -> str:
    r"""Generate the `training_name` from the training arguments."""
    training_name = f"b{batch_size}_l{learning_rate}_f{fold_id}"
    # TODO add support for visual dataset, mixed or not
    return training_name


def get_training_args(training_name: str) -> dict[str, Any]:
    r"""Returns a dictionnary containing the arguments corresponding to the `training_name`."""
    # TODO add support for visual dataset, mixed or not
    str_args = {arg[0]: arg[1:] for arg in training_name.split("_")}
    typed_args = {}
    typed_args["b"] = int(str_args["b"])
    typed_args["l"] = float(str_args["l"])
    typed_args["f"] = int(str_args["f"])
    return typed_args
