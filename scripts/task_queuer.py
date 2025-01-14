import pathlib
from typing import Any, Optional, Union

from loguru import logger
from torch import Value

# import grids
# import utils


def base_slurm_file_generator(
    job_name: str,
    partition: str,
    qos: Optional[str],
    timestr: str,
    n_gpus: int,
    n_cpus: int,
    python_script: str,
    script_options: str,
    array_arg: Optional[str] = None,
) -> pathlib.Path:
    logger.info(f"Generating slurm file for {job_name}")
    slurm_directory = utils.paths.get_slurm_folder()
    file_path = slurm_directory / f"{job_name}.slurm"
    file_as_string = f"""
    #!/bin/bash
    #SBATCH --job-name={job_name}
    #SBATCH --partition={partition}
    #SBATCH --output={str(slurm_directory.absolute())}/{job_name}_{'%j' if array_arg is None else '%A_%a'}.out
    #SBATCH --error={str(slurm_directory.absolute())}/{job_name}_{'%j' if array_arg is None else '%A_%a'}.err
    #SBATCH --time={timestr}
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --hint=nomultithread
    #SBATCH --gres=gpu:{n_gpus}
    #SBATCH --cpus-per-task={n_cpus}
    {f'#SBATCH --qos={qos}' if qos is not None else ""}
    {f'#SBATCH --array={array_arg}' if array_arg is not None else ""}

    module purge
    conda deactivate

    module load pytorch-gpu/py3/1.11.0
    set -x

    cd $HOME/brain-net-alignment/python_scripts
    srun python {python_script} {script_options}
    """
    with file_path.open("w") as f:
        f.write(utils.trim(file_as_string))
    logger.info(f"Generated slurm file for {job_name}")
    return file_path


def train_slurm_file_generator(
    task: str,
    net_type: str,
    net_hash: str,
) -> pathlib.Path:
    option_string = (
        f"--parallelize --task {task} --net_type {net_type} --config_hash {net_hash}"
    )
    file_path = base_slurm_file_generator(
        job_name=f"train_{net_hash}",
        partition="gpu_p2",
        qos="qos_gpu-t3",
        timestr="20:00:00",
        n_gpus=1,
        n_cpus=4,
        python_script="run_training.py",
        script_options=option_string,
    )
    return file_path


def record_slurm_file_generator(
    task: str,
    net_type: str,
    net_hash: str,
    num_samples_per_label: Optional[int] = None,
    recordloader_seed: Optional[int] = None,
) -> pathlib.Path:
    option_string = (
        f"--parallelize --task {task} --net_type {net_type} --config_hash {net_hash}"
    )
    if num_samples_per_label is not None:
        option_string = (
            f"{option_string} --num_samples_per_label {num_samples_per_label}"
        )
    if recordloader_seed is not None:
        option_string = f"{option_string} --seed {recordloader_seed}"
    file_path = base_slurm_file_generator(
        job_name=f"record_{net_hash}",
        partition="prepost",
        qos=None,
        timestr="20:00:00",
        n_cpus=4,
        n_gpus=0,
        python_script="run_recordings.py",
        script_options=option_string,
    )
    return file_path


def precomputRMD_slurm_file_generator(
    net_type: str,
    net_hash: str,
    mode: str,
    extraction_mode: Optional[str],
    optimize_layer: Optional[int],
    zscore: bool,
) -> pathlib.Path:
    option_string = (
        f"--net_type {net_type} --net_hash {net_hash} --mode {mode} --zscore {zscore}"
    )
    if optimize_layer is not None:
        job_name_complement = f"{mode}_{optimize_layer}"
    else:
        job_name_complement = mode
    if extraction_mode is not None:
        option_string = f"{option_string} --extraction_mode {extraction_mode}"
        job_name_complement = f"{job_name_complement}_{extraction_mode}"
    if optimize_layer is not None:
        option_string = f"{option_string} --optimize_layer {optimize_layer}"
    file_path = base_slurm_file_generator(
        job_name=f"precomputRDM_{job_name_complement}_{net_hash}_{zscore}",
        partition="prepost",
        qos=None,
        timestr="20:00:00",
        n_cpus=24,
        n_gpus=0,
        python_script="run_precomputRDM.py",
        script_options=option_string,
    )
    return file_path


def projinit_slurm_file_generator(
    task: str,
    net_type: str,
    project_name: str,
    from_grid: bool = True,
) -> pathlib.Path:
    option_string = f"--net_type {net_type} --task {task} --project_name {project_name}"
    if from_grid:
        option_string = f"{option_string} --from_grid"
    file_path = base_slurm_file_generator(
        job_name=f"projinit_{project_name}",
        partition="prepost",
        qos=None,
        timestr="00:15:00",
        n_cpus=4,
        n_gpus=0,
        python_script="run_projinit.py",
        script_options=option_string,
    )
    return file_path


def distcomput_slurm_file_generator(
    net_type: str,
    project_name: str,
    analysis_mode: str,
    chunksize: Optional[int],
    force_recompute: bool = False,
    extraction_mode: Optional[str] = None,
) -> pathlib.Path:
    option_string = f"--net_type {net_type} --project_name {project_name} --analysis_mode {analysis_mode} --chunk_size {chunksize if chunksize is not None else -1}"
    if force_recompute:
        option_string = f"{option_string} --force_recompute"
    if extraction_mode is not None:
        option_string = f"{option_string} --extraction_mode {extraction_mode}"
    file_path = base_slurm_file_generator(
        job_name=f"distcomput_{analysis_mode}_{project_name}",
        partition="prepost",
        qos=None,
        timestr="10:00:00",
        n_gpus=0,
        n_cpus=12,
        python_script="run_distcomput.py",
        script_options=option_string,
    )
    return file_path


def mds_slurm_file_generator(
    net_type: str,
    project_name: str,
    mode: str,
    mds_seed: Optional[int] = None,
) -> pathlib.Path:
    option_string = f"--net_type {net_type} --project_name {project_name} --mode {mode}"
    if mds_seed is not None:
        option_string = f"{option_string} --seed {mds_seed}"
    file_path = base_slurm_file_generator(
        job_name=f"mds_{mode}_{project_name}",
        partition="prepost",
        qos=None,
        timestr="01:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="run_mds.py",
        script_options=option_string,
    )
    return file_path


def mdsviz_slurm_file_generator(
    net_type: str,
    project_name: str,
    mode: str,
) -> pathlib.Path:
    option_string = f"--net_type {net_type} --project_name {project_name} --mode {mode}"
    file_path = base_slurm_file_generator(
        job_name=f"mdsviz_{mode}_{project_name}",
        partition="visu",
        qos=None,
        timestr="01:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="run_mdsviz.py",
        script_options=option_string,
    )
    return file_path


def otheranalysis_slurm_file_generator(
    net_type: str,
    project_name: str,
) -> pathlib.Path:
    option_string = f"--net_type {net_type} --project_name {project_name}"
    file_path = base_slurm_file_generator(
        job_name=f"otheranalysis_{project_name}",
        partition="prepost",
        qos=None,
        timestr="01:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="run_otheranalysis.py",
        script_options=option_string,
    )
    return file_path


def otherviz_slurm_file_generator(
    net_type: str,
    project_name: str,
) -> pathlib.Path:
    option_string = f"--net_type {net_type} --project_name {project_name}"
    file_path = base_slurm_file_generator(
        job_name=f"otherviz_{project_name}",
        partition="visu",
        qos=None,
        timestr="01:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="run_otherviz.py",
        script_options=option_string,
    )
    return file_path


def baseline_slurm_file_generator(
    net_type: str,
    shuffle_type: str,
    source_project_name: str,
    destination_project_name: str,
    baseline_seed: Optional[int] = None,
    extraction_mode: Optional[str] = None,
) -> pathlib.Path:
    option_string = f"--net_type {net_type} --shuffle_type {shuffle_type} --source_name {source_project_name} --destination_name {destination_project_name}"
    if baseline_seed is not None:
        option_string = f"{option_string} --seed {baseline_seed}"
    if extraction_mode is not None:
        option_string = f"{option_string} --extraction_mode {extraction_mode}"
    file_path = base_slurm_file_generator(
        job_name=f"baseline_{destination_project_name}",
        partition="prepost",
        qos=None,
        timestr="02:00:00",
        n_cpus=4,
        n_gpus=0,
        python_script="run_baseline.py",
        script_options=option_string,
    )
    return file_path


def regression_slurm_file_generator(
    net_type: str,
    shuffle_type: str,
    active_name: str,
    baseline_name: str,
) -> pathlib.Path:
    option_string = f"--net_type {net_type} --shuffle_type {shuffle_type} --active_name {active_name} --baseline_name {baseline_name}"
    file_path = base_slurm_file_generator(
        job_name=f"regressions_{baseline_name}",
        partition="prepost",
        qos=None,
        timestr="20:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="run_regressions.py",
        script_options=option_string,
    )
    return file_path


def regviz_slurm_file_generator() -> pathlib.Path:
    # TODO later when regression visualisation will be decided
    option_string = f""
    file_path = base_slurm_file_generator(
        job_name=f"regviz_",
        partition="visu",
        qos=None,
        timestr="04:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="run_regviz.py",
        script_options=option_string,
    )
    return file_path


def panel_slurm_file_generator(
    task: str,
    net_type: str,
    net_hash: str,
    project_name: str,
    wanted_features: str,
    zscore: bool,
    cv: Optional[int],
    extraction_mode: Optional[str] = None,
    mds_seed: Optional[int] = None,
    optimize: Union[bool, int] = False,
    pre_cv: bool = False,
    post_cv: bool = False,
):
    job_name = "panel"
    option_string = f"--task {task} --net_type {net_type} --net_hash {net_hash} --project_name {project_name} --wanted_features {wanted_features} --zscore {zscore}"
    if cv is not None:
        option_string = f"{option_string} --cv {cv}"
        job_name = f"{job_name}_cv"
    if extraction_mode is not None:
        option_string = f"{option_string} --extraction_mode {extraction_mode}"
    if mds_seed is not None:
        option_string = f"{option_string} --seed {mds_seed}"
    if pre_cv:
        option_string = f"{option_string} --pre_cv"
        job_name = f"pre_{job_name}"
    if post_cv:
        option_string = f"{option_string} --post_cv"
        job_name = f"post_{job_name}"
    jobname_complement = ""
    if isinstance(optimize, bool):
        if optimize:
            option_string = f"{option_string} --optimize_net"
            jobname_complement = "Net_"
    elif isinstance(optimize, int):
        option_string = f"{option_string} --optimize_layer {optimize}"
        jobname_complement = f"Layer_{optimize}_"
    file_path = base_slurm_file_generator(
        job_name=f"{job_name}_{jobname_complement}{project_name}_{zscore}",
        partition="prepost",
        qos=None,
        timestr="20:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="run_panel.py",
        script_options=option_string,
    )
    return file_path


def panel_array_file_generator(
    task: str,
    net_type: str,
    net_hash: str,
    project_name: str,
    wanted_features: str,
    zscore: bool,
    cv: int,
    extraction_mode: Optional[str] = None,
    mds_seed: Optional[int] = None,
    optimize: Union[bool, int] = False,
):
    option_string = f"--task {task} --net_type {net_type} --net_hash {net_hash} --project_name {project_name}"
    option_string = f"{option_string} --wanted_features {wanted_features} --zscore {zscore} --parallelized_cv $SLURM_ARRAY_TASK_ID"
    if cv is not None:
        option_string = f"{option_string} --cv {cv}"
    if extraction_mode is not None:
        option_string = f"{option_string} --extraction_mode {extraction_mode}"
    if mds_seed is not None:
        option_string = f"{option_string} --seed {mds_seed}"
    jobname_complement = ""
    if isinstance(optimize, bool):
        if optimize:
            option_string = f"{option_string} --optimize_net"
            jobname_complement = "Net_"
    elif isinstance(optimize, int):
        option_string = f"{option_string} --optimize_layer {optimize}"
        jobname_complement = f"Layer_{optimize}_"
    file_path = base_slurm_file_generator(
        job_name=f"panelarray_{jobname_complement}{project_name}_{zscore}",
        partition="prepost",
        qos=None,
        timestr="20:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="run_panel.py",
        script_options=option_string,
        array_arg=f"0-{cv-1}",
    )
    return file_path


def panelviz_slurm_file_generator(
    project_name: str,
) -> pathlib.Path:
    option_string = f"--project_name {project_name}"
    file_path = base_slurm_file_generator(
        job_name=f"panelviz_{project_name}",
        partition="visu",
        qos=None,
        timestr="04:00:00",
        n_gpus=0,
        n_cpus=4,
        python_script="run_panelviz.py",
        script_options=option_string,
    )
    return file_path


def get_bash_dependency_string(dependency_id_vars: list[str]) -> str:
    dependency_string = ""
    if len(dependency_id_vars) != 0:
        dependency_string = f"--dependency=afterok:${':$'.join(dependency_id_vars)} --kill-on-invalid-dep=yes"
    return dependency_string


def queue_job(
    job_path: pathlib.Path,
    is_required_as_dependency: bool,
    dependency_id_vars: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    logger.info(f"Generating commands for {job_path.name}")
    commands = []
    commands.append(f'echo "Queueing job {job_path.name}"')
    dependency_string = ""
    if dependency_id_vars is not None:
        dependency_string = get_bash_dependency_string(dependency_id_vars)
    if is_required_as_dependency:
        id_var_name = f"{job_path.stem}_id"
        commands.append(
            f'{id_var_name}=`sbatch {dependency_string} {str(job_path.absolute())} | cut -d " " -f 4`'
        )
    else:
        id_var_name = ""
        commands.append(f"sbatch {dependency_string} {str(job_path.absolute())}")
    commands.append(f'echo "Queued job {job_path.name}"')
    logger.info(f"Generated commands for {job_path.name}")
    return id_var_name, commands


def queue_train(
    task: str,
    net_type: str,
    net_hash: str,
    is_required_as_dependency: bool,
    train_dependency_id_vars: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    train_path = train_slurm_file_generator(
        task=task,
        net_type=net_type,
        net_hash=net_hash,
    )
    train_id_var_name, command_lines = queue_job(
        job_path=train_path,
        is_required_as_dependency=is_required_as_dependency,
        dependency_id_vars=train_dependency_id_vars,
    )
    return train_id_var_name, command_lines


def queue_record(
    task: str,
    net_type: str,
    net_hash: str,
    is_required_as_dependency: bool,
    num_samples_per_label: Optional[int] = None,
    recordloader_seed: Optional[int] = None,
    record_dependency_id_vars: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    record_path = record_slurm_file_generator(
        task=task,
        net_type=net_type,
        net_hash=net_hash,
        num_samples_per_label=num_samples_per_label,
        recordloader_seed=recordloader_seed,
    )
    record_id_var_name, command_lines = queue_job(
        job_path=record_path,
        is_required_as_dependency=is_required_as_dependency,
        dependency_id_vars=record_dependency_id_vars,
    )
    return record_id_var_name, command_lines


def queue_train_record(
    task: str,
    net_type: str,
    net_hash: str,
    num_samples_per_label: Optional[int] = None,
    recordloader_seed: Optional[int] = None,
    train_dependency_id_vars: Optional[list[str]] = None,
) -> tuple[str, str, list[str]]:
    commands = []
    train_job_id_var, train_commands = queue_train(
        task=task,
        net_type=net_type,
        net_hash=net_hash,
        is_required_as_dependency=True,
        train_dependency_id_vars=train_dependency_id_vars,
    )
    commands.extend(train_commands)
    record_job_id_var, record_commands = queue_record(
        task=task,
        net_type=net_type,
        net_hash=net_hash,
        is_required_as_dependency=True,
        num_samples_per_label=num_samples_per_label,
        recordloader_seed=recordloader_seed,
        record_dependency_id_vars=[train_job_id_var],
    )
    commands.extend(record_commands)

    return train_job_id_var, record_job_id_var, commands


def queue_precomputRDM(
    net_type: str,
    net_hash: str,
    depth: int,
    extraction_mode: Optional[str],
    zscore: bool,
    precomputRDM_dependency_id_vars: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    commands = ['precomputids=""']
    precomputeRDM_path = precomputRMD_slurm_file_generator(
        net_type=net_type,
        net_hash=net_hash,
        mode="Net",
        extraction_mode=extraction_mode,
        optimize_layer=None,
        zscore=zscore,
    )

    precomputRDM_id_var, precomputRDM_commands = queue_job(
        job_path=precomputeRDM_path,
        is_required_as_dependency=True,
        dependency_id_vars=precomputRDM_dependency_id_vars,
    )
    commands.extend(precomputRDM_commands)
    commands.append(f'precomputids="${precomputRDM_id_var}"')

    for i in range(depth):
        precomputeRDM_path = precomputRMD_slurm_file_generator(
            net_type=net_type,
            net_hash=net_hash,
            mode="Layer",
            extraction_mode=extraction_mode,
            optimize_layer=i,
            zscore=zscore,
        )

        precomputRDM_id_var, precomputRDM_commands = queue_job(
            job_path=precomputeRDM_path,
            is_required_as_dependency=True,
            dependency_id_vars=precomputRDM_dependency_id_vars,
        )
        commands.extend(precomputRDM_commands)
        commands.append(f'precomputids="$precomputids:${precomputRDM_id_var}"')

    return "precomputids", commands


def queue_analysis_without_reg(
    net_type: str,
    project_name: str,
    chunksize: Optional[int] = None,
    mds_seed: Optional[int] = None,
    extraction_mode: Optional[str] = None,
    analysis_dependency_id_vars: Optional[list[str]] = None,
) -> tuple[dict[str, str], list[str]]:
    distcomput_vars = {}
    commands = []
    for analysis_mode in {"Net", "Layer"}:
        distcomput_path = distcomput_slurm_file_generator(
            net_type=net_type,
            project_name=project_name,
            analysis_mode=analysis_mode,
            chunksize=chunksize,
            extraction_mode=extraction_mode,
        )
        distcomput_id_var, distcomput_commands = queue_job(
            job_path=distcomput_path,
            is_required_as_dependency=True,
            dependency_id_vars=analysis_dependency_id_vars,
        )
        commands.extend(distcomput_commands)
        distcomput_vars[analysis_mode] = distcomput_id_var

        mds_path = mds_slurm_file_generator(
            net_type=net_type,
            project_name=project_name,
            mode=analysis_mode,
            mds_seed=mds_seed,
        )
        mds_id_var, mds_commands = queue_job(
            job_path=mds_path,
            is_required_as_dependency=True,
            dependency_id_vars=[distcomput_id_var],
        )
        commands.extend(mds_commands)

        mdsviz_path = mdsviz_slurm_file_generator(
            net_type=net_type,
            project_name=project_name,
            mode=analysis_mode,
        )
        _, mdsviz_commands = queue_job(
            job_path=mdsviz_path,
            is_required_as_dependency=False,
            dependency_id_vars=[mds_id_var],
        )
        commands.extend(mdsviz_commands)

        if analysis_mode == "Net":
            otheranalysis_path = otheranalysis_slurm_file_generator(
                net_type=net_type,
                project_name=project_name,
            )
            otheranalysis_id_var, otheranalysis_commands = queue_job(
                job_path=otheranalysis_path,
                is_required_as_dependency=True,
                dependency_id_vars=[distcomput_id_var],
            )
            commands.extend(otheranalysis_commands)

            otherviz_path = otherviz_slurm_file_generator(
                net_type=net_type,
                project_name=project_name,
            )
            _, otherviz_commands = queue_job(
                job_path=otherviz_path,
                is_required_as_dependency=True,
                dependency_id_vars=[otheranalysis_id_var],
            )
            commands.extend(otherviz_commands)

    return distcomput_vars, commands


def queue_baseline(
    net_type: str,
    shuffle_type: str,
    source_project_name: str,
    source_distcomput_vars: dict[str, str],
    baseline_project_name: str,
    baseline_seed: Optional[int] = None,
    chunksize: Optional[int] = None,
    mds_seed: Optional[int] = None,
    extraction_mode: Optional[str] = None,
    dependency_id_vars: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    commands = []
    baseline_path = baseline_slurm_file_generator(
        net_type=net_type,
        shuffle_type=shuffle_type,
        source_project_name=source_project_name,
        destination_project_name=baseline_project_name,
        baseline_seed=baseline_seed,
        extraction_mode=extraction_mode,
    )
    baseline_id_var, baseline_commands = queue_job(
        job_path=baseline_path,
        is_required_as_dependency=True,
        dependency_id_vars=dependency_id_vars,
    )
    commands.extend(baseline_commands)

    baseline_distcomput_vars, analysis_commands = queue_analysis_without_reg(
        net_type=net_type,
        project_name=baseline_project_name,
        chunksize=chunksize,
        mds_seed=mds_seed,
        extraction_mode=extraction_mode,
        analysis_dependency_id_vars=[baseline_id_var],
    )
    commands.extend(analysis_commands)

    regressions_path = regression_slurm_file_generator(
        net_type=net_type,
        shuffle_type=shuffle_type,
        active_name=source_project_name,
        baseline_name=baseline_project_name,
    )
    _, baseline_regression_commands = queue_job(
        job_path=regressions_path,
        is_required_as_dependency=False,
        dependency_id_vars=[
            baseline_distcomput_vars["Net"],
            source_distcomput_vars["Net"],
        ],
    )
    commands.extend(baseline_regression_commands)

    return baseline_id_var, commands


def queue_parallelized_panel(
    task: str,
    net_type: str,
    net_hash: str,
    project_name: str,
    wanted_features: str,
    zscore: bool,
    cv: Optional[int],
    extraction_mode: Optional[str] = None,
    mds_seed: Optional[int] = None,
    panel_dependency_id_vars: Optional[list[str]] = None,
    optimize: Union[bool, int] = False,
) -> tuple[str, list[str]]:
    commands = []
    pre_cv_path = panel_slurm_file_generator(
        task=task,
        net_type=net_type,
        net_hash=net_hash,
        project_name=project_name,
        wanted_features=wanted_features,
        extraction_mode=extraction_mode,
        mds_seed=mds_seed,
        optimize=optimize,
        zscore=zscore,
        cv=cv,
        pre_cv=True,
    )
    pre_cv_id_var, pre_cv_commands = queue_job(
        job_path=pre_cv_path,
        is_required_as_dependency=True,
        dependency_id_vars=panel_dependency_id_vars,
    )
    commands.extend(pre_cv_commands)

    panel_array_path = panel_array_file_generator(
        task=task,
        net_type=net_type,
        net_hash=net_hash,
        project_name=project_name,
        wanted_features=wanted_features,
        extraction_mode=extraction_mode,
        mds_seed=mds_seed,
        optimize=optimize,
        zscore=zscore,
        cv=cv,  # type:ignore
    )
    panel_array_id_var, panel_array_commands = queue_job(
        job_path=panel_array_path,
        is_required_as_dependency=True,
        dependency_id_vars=[pre_cv_id_var],
    )
    commands.extend(panel_array_commands)

    post_cv_path = panel_slurm_file_generator(
        task=task,
        net_type=net_type,
        net_hash=net_hash,
        project_name=project_name,
        wanted_features=wanted_features,
        extraction_mode=extraction_mode,
        mds_seed=mds_seed,
        optimize=optimize,
        zscore=zscore,
        cv=cv,
        post_cv=True,
    )
    post_cv_id_var, post_cv_commands = queue_job(
        job_path=post_cv_path,
        is_required_as_dependency=True,
        dependency_id_vars=[panel_array_id_var],
    )
    commands.extend(post_cv_commands)

    return post_cv_id_var, commands


def queue_panel(
    task: str,
    net_type: str,
    net_hash: str,
    project_name: str,
    wanted_features: str,
    zscore: bool,
    cv: Optional[int],
    extraction_mode: Optional[str] = None,
    mds_seed: Optional[int] = None,
    panel_dependency_id_vars: Optional[list[str]] = None,
    optimize: bool = False,
    depth: Optional[int] = None,
    parallelize_cv: bool = False,
) -> tuple[str, list[str]]:
    commands = []
    if optimize:
        if depth is None:
            raise ValueError("Expected a depth value for optimization, got None.")
        commands.append('panelids=""')
        if parallelize_cv:
            panel_id_var, panel_commands = queue_parallelized_panel(
                task=task,
                net_type=net_type,
                net_hash=net_hash,
                project_name=project_name,
                wanted_features=wanted_features,
                extraction_mode=extraction_mode,
                mds_seed=mds_seed,
                optimize=optimize,
                zscore=zscore,
                cv=cv,
                panel_dependency_id_vars=panel_dependency_id_vars,
            )
        else:
            panel_path = panel_slurm_file_generator(
                task=task,
                net_type=net_type,
                net_hash=net_hash,
                project_name=project_name,
                wanted_features=wanted_features,
                extraction_mode=extraction_mode,
                mds_seed=mds_seed,
                optimize=True,
                zscore=zscore,
                cv=cv,
            )
            panel_id_var, panel_commands = queue_job(
                job_path=panel_path,
                is_required_as_dependency=True,
                dependency_id_vars=panel_dependency_id_vars,
            )
        commands.extend(panel_commands)
        commands.append(f'panelids="${panel_id_var}"')

        for i in range(depth):
            if parallelize_cv:
                panel_id_var, panel_commands = queue_parallelized_panel(
                    task=task,
                    net_type=net_type,
                    net_hash=net_hash,
                    project_name=project_name,
                    wanted_features=wanted_features,
                    extraction_mode=extraction_mode,
                    mds_seed=mds_seed,
                    optimize=i,
                    zscore=zscore,
                    cv=cv,
                    panel_dependency_id_vars=panel_dependency_id_vars,
                )
            else:
                panel_path = panel_slurm_file_generator(
                    task=task,
                    net_type=net_type,
                    net_hash=net_hash,
                    project_name=project_name,
                    wanted_features=wanted_features,
                    extraction_mode=extraction_mode,
                    mds_seed=mds_seed,
                    optimize=i,
                    zscore=zscore,
                    cv=cv,
                )
                panel_id_var, panel_commands = queue_job(
                    job_path=panel_path,
                    is_required_as_dependency=True,
                    dependency_id_vars=panel_dependency_id_vars,
                )
            commands.extend(panel_commands)
            commands.append(f'panelids="$panelids:${panel_id_var}"')

        panelviz_dependency_id_vars = ["panelids"]

    else:
        if parallelize_cv:
            panel_id_var, panel_commands = queue_parallelized_panel(
                task=task,
                net_type=net_type,
                net_hash=net_hash,
                project_name=project_name,
                wanted_features=wanted_features,
                extraction_mode=extraction_mode,
                mds_seed=mds_seed,
                optimize=optimize,
                zscore=zscore,
                cv=cv,
                panel_dependency_id_vars=panel_dependency_id_vars,
            )
        else:
            panel_path = panel_slurm_file_generator(
                task=task,
                net_type=net_type,
                net_hash=net_hash,
                project_name=project_name,
                wanted_features=wanted_features,
                extraction_mode=extraction_mode,
                mds_seed=mds_seed,
                zscore=zscore,
                cv=cv,
            )
            panel_id_var, panel_commands = queue_job(
                job_path=panel_path,
                is_required_as_dependency=True,
                dependency_id_vars=panel_dependency_id_vars,
            )
        commands.extend(panel_commands)
        panelviz_dependency_id_vars = [panel_id_var]

    panelviz_path = panelviz_slurm_file_generator(project_name)
    _, panelviz_commands = queue_job(
        job_path=panelviz_path,
        is_required_as_dependency=False,
        dependency_id_vars=panelviz_dependency_id_vars,
    )
    commands.extend(panelviz_commands)

    return panel_id_var, commands


def create_jean_zay_queuer_script(
    task: str,
    net_type: str,
    zscore: bool,
    cv: Optional[int],
    num_samples_per_label: Optional[int] = None,
    recordloader_seed: Optional[int] = None,
    chunksize: Optional[int] = None,
    extraction_mode: Optional[str] = None,
    mds_seed: Optional[int] = None,
    baseline_seed: Optional[int] = None,
    bypass_training: bool = False,
    bypass_record: bool = False,
    do_panel: bool = False,
    wanted_features: Optional[str] = None,
    bypass_stability: bool = False,
    optimize: bool = False,
    parallelize_cv: bool = False,
):
    logger.info("Beginning script generation")
    slurm_directory = utils.paths.get_slurm_folder()
    slurm_directory.mkdir(parents=True, exist_ok=True)
    commands = ["#!/bin/bash"]
    if not bypass_stability:
        commands.append('recordids=""')
    first_recordids_append = True
    for config in utils.iterate_grid_combinations(grids.TASK_GRIDS[task][net_type]):
        utils.initialize_config_file(net_type=net_type, config=config)
        net_hash = utils.kwargs_to_id(config)
        panel_dependency_id_vars = []
        precomputRDM_dependency_id_vars = []
        if bypass_record:
            pass
        elif bypass_training:
            record_id_var, queued_jobs_commands = queue_record(
                task=task,
                net_type=net_type,
                net_hash=net_hash,
                is_required_as_dependency=True,
                num_samples_per_label=num_samples_per_label,
                recordloader_seed=recordloader_seed,
            )
            commands.extend(queued_jobs_commands)
            panel_dependency_id_vars = [record_id_var]
            precomputRDM_dependency_id_vars = [record_id_var]
        else:
            _, record_id_var, queued_jobs_commands = queue_train_record(
                task=task,
                net_type=net_type,
                net_hash=net_hash,
                num_samples_per_label=num_samples_per_label,
                recordloader_seed=recordloader_seed,
            )
            commands.extend(queued_jobs_commands)
            if not bypass_stability:
                if first_recordids_append:
                    commands.append(f'recordids="${record_id_var}"')
                    first_recordids_append = False
                else:
                    commands.append(f'recordids="$recordids:${record_id_var}"')
            panel_dependency_id_vars = [record_id_var]
            precomputRDM_dependency_id_vars = [record_id_var]
        model_depth = None
        if optimize:
            model_depth = utils.advanced_num_layers_compute_func(net_type, config)
            precomputRDM_id_var, precomputRDM_commands = queue_precomputRDM(
                net_type=net_type,
                net_hash=net_hash,
                depth=model_depth,
                extraction_mode=extraction_mode,
                precomputRDM_dependency_id_vars=precomputRDM_dependency_id_vars,
                zscore=zscore,
            )
            commands.extend(precomputRDM_commands)
            panel_dependency_id_vars = [precomputRDM_id_var]
        if do_panel:
            if wanted_features is None:
                raise ValueError(
                    "do_panel was set to True without providing a value to wanted_features"
                )
            panel_project_name = f"panel_{net_hash}{'_zscored' if zscore else ''}"
            if extraction_mode is not None:
                panel_project_name = f"{panel_project_name}_{extraction_mode}"
            _, queued_panel_commands = queue_panel(
                task=task,
                net_type=net_type,
                net_hash=net_hash,
                project_name=panel_project_name,
                wanted_features=wanted_features,
                extraction_mode=extraction_mode,
                mds_seed=mds_seed,
                panel_dependency_id_vars=panel_dependency_id_vars,
                optimize=optimize,
                depth=model_depth,
                zscore=zscore,
                cv=cv,
                parallelize_cv=parallelize_cv,
            )
            commands.extend(queued_panel_commands)

    if not bypass_stability:
        # TODO add optimize effect for stability part, need to optimize all the other scripts to use specific_layer option in RDM getter, later work
        source_project_name = f"{task}_{net_type}"
        if extraction_mode is not None:
            source_project_name = f"{source_project_name}_{extraction_mode}"
        if first_recordids_append:
            projinit_dependency_id_vars = None
        else:
            projinit_dependency_id_vars = ["recordids"]

        source_projinit_path = projinit_slurm_file_generator(
            task=task,
            net_type=net_type,
            project_name=source_project_name,
        )
        source_projinit_id_var, projinit_commands = queue_job(
            job_path=source_projinit_path,
            is_required_as_dependency=True,
            dependency_id_vars=projinit_dependency_id_vars,
        )
        commands.extend(projinit_commands)

        source_distcomput_vars, source_analysis_commands = queue_analysis_without_reg(
            net_type=net_type,
            project_name=source_project_name,
            chunksize=chunksize,
            mds_seed=mds_seed,
            extraction_mode=extraction_mode,
            analysis_dependency_id_vars=[source_projinit_id_var],
        )
        commands.extend(source_analysis_commands)
        baseline_id_var = None

        for shuffle_type, prefix in {("Net", "ns"), ("Layer", "ls"), ("Sample", "ss")}:
            baseline_project_name = f"{prefix}_{source_project_name}"
            if baseline_id_var is None:
                dependency_id_vars = [source_projinit_id_var]
            else:
                dependency_id_vars = [source_projinit_id_var, baseline_id_var]
            baseline_id_var, baseline_commands = queue_baseline(
                net_type=net_type,
                shuffle_type=shuffle_type,
                source_project_name=source_project_name,
                source_distcomput_vars=source_distcomput_vars,
                baseline_project_name=baseline_project_name,
                baseline_seed=baseline_seed,
                chunksize=chunksize,
                mds_seed=mds_seed,
                extraction_mode=extraction_mode,
                dependency_id_vars=dependency_id_vars,
            )
            commands.extend(baseline_commands)

    if extraction_mode is not None:
        script_name = f"{task}_{net_type}_{extraction_mode}.sh"
    else:
        script_name = f"{task}_{net_type}.sh"
    with (utils.paths.script_dir / script_name).open("w") as f:
        f.writelines(f"{line}\n" for line in commands)
    logger.info(f"Script created : {script_name}")
