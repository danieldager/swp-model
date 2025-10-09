from typing import TypedDict

from ..datasets import check_query, unhash_query


class TrainArgs(TypedDict):
    r"""TypedDict containing values required to identify a training :
    `batch_size` : batch size
    `learning_rate` : learning rate
    `fold_id` : fold used for training
    `include_stress` : whether or not phoneme stress is used
    `loss` : the type of loss used for training
    `query` : query used for the data
    `seed` : seed of the dataloaders
    `train_part` : part of the model that is trained
    `mixed` : if the dataset is homogeneous or mixed
    """

    batch_size: int
    learning_rate: float
    fold_id: int | None
    include_stress: bool
    loss: str
    query: str | None
    seed: int
    train_part: str | None
    mixed: bool


def get_train_name(
    batch_size: int,
    learning_rate: float,
    fold_id: int | None,
    include_stress: bool,
    seed: int,
    loss: str = "classic",
    query: str | None = None,
    train_part: str | None = None,
    mixed: bool = False,
    **kwargs,
) -> str:
    r"""Generate the `train_name` from the training arguments.
    Field keys in the name string are the following :
    - b : Batch size
    - l : Learning rate
    - f : dataset Fold
    - g : Generator seed
    - s : phoneme Stress
    - e : Error function
    - q : Query hash
    - d : Dataset type
    - p : trained Part
    """
    fold_str = "all" if fold_id is None else fold_id
    train_name = f"b{batch_size}_l{learning_rate}_f{fold_str}_g{seed}"
    if include_stress:
        train_name = f"{train_name}_sw"
    else:
        train_name = f"{train_name}_sn"

    match loss:
        case "classic":
            train_name = f"{train_name}_ec"
        case "first":
            train_name = f"{train_name}_ef"
        case _:
            raise NotImplementedError(
                f"No support for loss {loss} is currently implemented"
            )
    if query is not None:
        hashed = check_query(query=query)
        train_name = f"{train_name}_q{hashed}"
    if train_part is not None:
        train_name = f"{train_name}_p{train_part[0]}"
    if mixed:
        train_name = f"{train_name}_dm"
    else:
        train_name = f"{train_name}_ds"
    return train_name


def get_train_args(train_name: str) -> TrainArgs:
    r"""Returns a dictionnary containing the arguments corresponding to the `train_name`."""
    str_args = {arg[0]: arg[1:] for arg in train_name.split("_")}
    match str_args["s"]:
        case "w":  # include_stress
            include_stress = True
        case "n":
            include_stress = False
        case _:
            raise ValueError(f'Stress value not recognized : {str_args["s"]}')
    match str_args["e"]:
        case "c":
            loss = "classic"
        case "f":
            loss = "first"
        case _:
            raise ValueError(f'Loss value not recognized : {str_args["e"]}')
    query = None
    if "q" in str_args:
        query = unhash_query(str_args["q"])
    train_part = None
    if "p" in str_args:
        match str_args["p"]:
            case "a":
                train_part = "all"
            case "h":
                train_part = "hidden"
            case "e":
                train_part = "encoder"
            case "d":
                train_part = "decoder"
            case _:
                raise ValueError(f'Part value not recognized : {str_args["p"]}')
    match str_args["d"]:
        case "m":
            mixed = True
        case "s":
            mixed = False
        case _:
            raise ValueError(f'Dataset type value not recognized : {str_args["d"]}')
    train_args = TrainArgs(
        {
            "batch_size": int(str_args["b"]),
            "learning_rate": float(str_args["l"]),
            "fold_id": None if str_args["f"] == "all" else int(str_args["f"]),
            "include_stress": include_stress,
            "loss": loss,
            "query": query,
            "seed": int(str_args["g"]),
            "train_part": train_part,
            "mixed": mixed,
        }
    )
    return train_args
