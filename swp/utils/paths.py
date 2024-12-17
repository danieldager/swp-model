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
    dataset_dir = pathlib.Path(os.environ["DSDIR"])
elif _ON_OBERON:
    pass  # TODO set paths for Oberon
else:  # personnal computer
    script_dir = repo_root / "scripts"
    dataset_dir = repo_root / "stimuli"
    checkpoint_dir = repo_root / "weights"
    result_dir = repo_root / "results"


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


def get_checkpoint_dir() -> pathlib.Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_result_dir() -> pathlib.Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def get_figures_dir() -> pathlib.Path:
    figures_dir = result_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir
