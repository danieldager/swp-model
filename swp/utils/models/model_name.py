from typing import TypedDict

from ...models.autoencoder import Bimodel, Unimodel
from ...models.decoders import DecoderLSTM, DecoderRNN
from ...models.encoders import CorNetEncoder, EncoderLSTM, EncoderRNN


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
    match recur_type:
        case "LSTM":
            audit_encoder_class = EncoderLSTM
            decoder_class = DecoderLSTM
        case "RNN":
            audit_encoder_class = EncoderRNN
            decoder_class = DecoderRNN
        case _:
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
    match model_class:
        case "Ua" | "Uv":
            if model_class == "Ua":
                encoder = audit_encoder_class(
                    vocab_size=model_args["vocab_size"],
                    hidden_size=model_args["hidden_size"],
                    num_layers=model_args["num_layers"],
                    dropout=model_args["droprate"],
                )
            else:
                if model_args["cnn_args"] is None:
                    raise ValueError(
                        "No arguments corresponding to the visual encoder in a visual model"
                    )
                encoder = CorNetEncoder(
                    hidden_size=model_args["cnn_args"]["hidden_size"],
                    cornet_model=model_args["cnn_args"]["cnn_model"],
                )
            model = Unimodel(
                encoder=encoder,
                decoder=decoder,
                start_token_id=model_args["start_token_id"],
            )
        case "B":
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
        case _ if model_class.startswith("U"):
            raise ValueError(
                f"Trying to name a Unimodel that is neither auditory nor visual, type : {model_class[1:]}"
            )
        case _:
            raise ValueError(f"Model class not recognized : {model_class}")
    return model


def get_model_name(model: Unimodel | Bimodel) -> str:
    r"""Returns the codified `model_name` corresponding to the `model`.
    Field keys in the name string are the following :
    First field : model class
    Second field : architecture of the recurrent part
    - h : Hidden size
    - l : number of Layers
    - v : Vocabulary size
    - d : Dropout rate
    - t : Teacher forcing rate
    - s : Start token id
    CNN fields are separated by `__c`
    - h : CNN Hidden size
    - m : CNN Model
    """
    # TODO make modular with other CNN encoders
    cnn_str = None
    match model:
        case Unimodel(is_auditory=True):
            model_name = "Ua"
        case Unimodel():
            model_name = "Uv"
            cnn_str = f"h{model.encoder.hidden_size}_m{model.encoder.cnn_model}"
        case Bimodel():
            model_name = "B"
            cnn_str = (
                f"h{model.visual_encoder.hidden_size}_m{model.visual_encoder.cnn_model}"
            )
        case _:
            raise TypeError("Model type not supported.")
    match model.decoder:
        case DecoderLSTM():
            model_name = f"{model_name}_LSTM"
        case DecoderRNN():
            model_name = f"{model_name}_RNN"
        case _:
            raise TypeError("Recurrent type of decoder not supported.")
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
