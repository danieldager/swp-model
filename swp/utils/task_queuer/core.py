import logging
import pathlib

from swp.utils.models import get_model_name_from_args, get_train_name

from ..grid_search import Grid, grid_iter
from ..paths import get_generated_scripts_dir
from .script_gen import autoarg_slurmarray_file_generator, base_slurm_file_generator
from .script_queueing import queue_job

logger = logging.getLogger()


def create_and_queue_datagen(
    vocab_size: int = 50000, epoch_size: int = 1000000, num_folds: int = 5
) -> tuple[str, list[str]]:
    script_options = (
        f"--vocab_size {vocab_size} --epoch_size {epoch_size} --num_folds {num_folds}"
    )
    data_gen_path = base_slurm_file_generator(
        job_name="data_generation",
        partition="prepost",
        qos=None,
        timestr="5:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="generate_main_data.py",
        script_options=script_options,
    )
    datagen_id_var, datagen_commands = queue_job(
        job_path=data_gen_path, is_required_as_dependency=True
    )
    return datagen_id_var, datagen_commands


def create_and_queue_train_repetition(
    dependency_id_vars: list[str], array_arg_file: pathlib.Path, array_len: int
) -> tuple[str, list[str]]:
    training_path = autoarg_slurmarray_file_generator(
        job_name="train_rep_array",
        partition="gpu_p5",
        qos="qos_gpu-t3",
        timestr="20:00:00",
        n_gpus=1,
        n_cpus=4,
        arg_file=array_arg_file,
        array_stop=array_len,
        python_script="train_repetition.py",
    )
    train_array_id, train_commands = queue_job(
        job_path=training_path,
        is_required_as_dependency=True,
        dependency_id_vars=dependency_id_vars,
    )
    return train_array_id, train_commands


def create_and_queue_aggregate(dependency_id_vars: list[str]) -> tuple[str, list[str]]:
    aggregate_path = base_slurm_file_generator(
        job_name="aggregate",
        partition="prepost",
        qos=None,
        timestr="2:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="generate_main_data.py",
        script_options="",
    )
    aggregate_id, aggregate_commands = queue_job(
        aggregate_path,
        is_required_as_dependency=True,
        dependency_id_vars=dependency_id_vars,
    )
    return aggregate_id, aggregate_commands


def create_jean_zay_train_repetition_queuer(
    grid: Grid,
    bypass_datagen: bool = False,
):
    # TODO docstring
    gen_script_dir = get_generated_scripts_dir()
    logger.info("Beginning script generation")
    commands = ["#!/bin/bash"]
    train_dependency_id_vars = []
    if not bypass_datagen:
        datagen_id_var, datagen_commands = create_and_queue_datagen()
        commands.extend(datagen_commands)
        train_dependency_id_vars.append(datagen_id_var)
    job_array_len = 0
    all_args = []
    for arg_dict in grid_iter(grid):  # type: ignore
        model_name = get_model_name_from_args(**arg_dict)  # type: ignore
        training_name = get_train_name(**arg_dict)  # type: ignore
        args = f"--model_name {model_name} --train_name {training_name}"
        all_args.append(args)
        job_array_len += 1
    arg_file = gen_script_dir / "train_repetition_args.txt"
    with arg_file.open("w") as f:
        f.writelines(f"{args}\n" for args in all_args)
    train_array_id_var, train_commands = create_and_queue_train_repetition(
        dependency_id_vars=train_dependency_id_vars,
        array_arg_file=arg_file,
        array_len=job_array_len - 1,  # Slurm is not exclusive of upper bound
    )
    commands.extend(train_commands)
    aggregate_id_var, aggregate_commands = create_and_queue_aggregate(
        dependency_id_vars=[train_array_id_var]
    )
    commands.extend(aggregate_commands)
    # queue training plots ?
    script_name = f"train_repetition_queuer.sh"
    with (gen_script_dir / script_name).open("w") as f:
        f.writelines(f"{line}\n" for line in commands)
    logger.info(f"Script created : {script_name}")
