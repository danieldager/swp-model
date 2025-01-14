import os
import pathlib

_ON_JEAN_ZAY = os.getenv("SLURM_CLUSTER_NAME") == "jean-zay" or os.getenv(
    "HOSTNAME", ""
).startswith("jean-zay")

_ON_OBERON = False  # TODO commands to detect oberon

repo_root = pathlib.Path(os.path.realpath(__file__)).parent.parent.parent

if _ON_JEAN_ZAY:
    # script_dir = pathlib.Path(os.environ["HOME"]) / "single-word-processing-model" # TODO actualize this
    work_dir = pathlib.Path(os.environ["WORK"])
    dataset_dir = pathlib.Path(os.environ["WORK"]) / "stimuli"
    public_dataset_dir = pathlib.Path(os.environ["DSDIR"])
    # TODO add more paths for JZ
elif _ON_OBERON:
    pass  # TODO set paths for Oberon
else:  # personnal computer
    script_dir = repo_root / "scripts"
    dataset_dir = repo_root / "stimuli"
    weights_dir = repo_root / "weights"
    result_dir = repo_root / "results"
    public_dataset_dir = repo_root / "public_datasets"


def get_root() -> pathlib.Path:
    return repo_root


def get_dataset_dir() -> pathlib.Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def get_dataframe_dir() -> pathlib.Path:
    dataframe_dir = dataset_dir / "dataframe"
    dataframe_dir.mkdir(parents=True, exist_ok=True)
    return dataframe_dir


def get_folds_dir() -> pathlib.Path:
    folds_dir = dataset_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    return folds_dir


def get_weights_dir() -> pathlib.Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def get_result_dir() -> pathlib.Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def get_figures_dir() -> pathlib.Path:
    figures_dir = result_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def get_graphemes_dir() -> pathlib.Path:
    graphemes_dir = get_dataset_dir() / "graphemes"
    graphemes_dir.mkdir(parents=True, exist_ok=True)
    return graphemes_dir


def get_imagenet_dir() -> pathlib.Path:
    public_dataset_dir.mkdir(
        parents=True, exist_ok=True
    )  # TODO check it doesn't break JZ
    return public_dataset_dir
