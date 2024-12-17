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
    weights_dir = repo_root / "weights"
    result_dir = repo_root / "results"


def get_root() -> pathlib.Path:
    return repo_root


def get_dataset_dir() -> pathlib.Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


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
