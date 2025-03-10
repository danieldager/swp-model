import os
import pathlib

_ON_JEAN_ZAY = os.getenv("SLURM_CLUSTER_NAME") == "jean-zay" or os.getenv(
    "HOSTNAME", ""
).startswith("jean-zay")

_ON_OBERON = False  # TODO commands to detect oberon

repo_root = pathlib.Path(os.path.realpath(__file__)).parent.parent.parent

if _ON_JEAN_ZAY:
    work_dir = pathlib.Path(os.environ["WORK"])
    gen_script_dir = work_dir / "generated_scripts"
    stimuli_dir = work_dir / "stimuli"
    weights_dir = work_dir / "weights"
    results_dir = work_dir / "results"
    public_dataset_dir = pathlib.Path(os.environ["DSDIR"])
elif _ON_OBERON:
    pass  # TODO set paths for Oberon
else:  # personnal computer
    gen_script_dir = repo_root / "generated_scripts"
    stimuli_dir = repo_root / "stimuli"
    weights_dir = repo_root / "weights"
    results_dir = repo_root / "results"
    public_dataset_dir = repo_root / "public_datasets"


def get_root() -> pathlib.Path:
    return repo_root


def get_stimuli_dir() -> pathlib.Path:
    stimuli_dir.mkdir(parents=True, exist_ok=True)
    return stimuli_dir


def get_dataframe_dir() -> pathlib.Path:
    dataframe_dir = stimuli_dir / "dataframe"
    dataframe_dir.mkdir(parents=True, exist_ok=True)
    return dataframe_dir


def get_handmade_dir() -> pathlib.Path:
    handmade_dir = stimuli_dir / "handmade"
    handmade_dir.mkdir(parents=True, exist_ok=True)
    return handmade_dir


def get_folds_dir() -> pathlib.Path:
    folds_dir = stimuli_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    return folds_dir


def get_weights_dir() -> pathlib.Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def get_results_dir() -> pathlib.Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_figures_dir() -> pathlib.Path:
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def get_gridsearch_dir() -> pathlib.Path:
    gridsearch_dir = results_dir / "gridsearch"
    gridsearch_dir.mkdir(parents=True, exist_ok=True)
    return gridsearch_dir


def get_train_dir() -> pathlib.Path:
    train_dir = get_gridsearch_dir() / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    return train_dir


def get_evaluation_dir() -> pathlib.Path:
    eval_dir = get_results_dir() / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def get_graphemes_dir() -> pathlib.Path:
    graphemes_dir = get_stimuli_dir() / "graphemes"
    graphemes_dir.mkdir(parents=True, exist_ok=True)
    return graphemes_dir


def get_generated_scripts_dir() -> pathlib.Path:
    gen_script_dir.mkdir(parents=True, exist_ok=True)
    return gen_script_dir


def get_python_scripts_dir() -> pathlib.Path:
    python_scripts_dir = repo_root / "scripts"
    return python_scripts_dir


def get_imagenet_dir() -> pathlib.Path:
    public_dataset_dir.mkdir(
        parents=True, exist_ok=True
    )  # TODO check it doesn't break JZ
    return public_dataset_dir


def get_ablations_dir() -> pathlib.Path:
    ablations_dir = results_dir / "ablations"
    ablations_dir.mkdir(parents=True, exist_ok=True)
    return ablations_dir


def get_morphemes_dir() -> pathlib.Path:
    morphemes_dir = get_stimuli_dir() / "morphemes_data"
    morphemes_dir.mkdir(parents=True, exist_ok=True)
    return morphemes_dir
