import logging
import pathlib

logger = logging.getLogger()


def get_bash_dependency_string(
    dependency_id_vars: list[str], dependency_array_id_vars: list[str]
) -> str:
    dependency_string = None
    if len(dependency_id_vars) != 0:
        dependency_string = f"afterok:${':$'.join(dependency_id_vars)}"
    if len(dependency_array_id_vars) != 0:
        if dependency_string is None:
            dependency_string = f"aftercorr:${':$'.join(dependency_array_id_vars)}"
        else:
            dependency_string = (
                f"{dependency_string},aftercorr:${':$'.join(dependency_array_id_vars)}"
            )
    if dependency_string is not None:
        return f"--dependency={dependency_string} --kill-on-invalid-dep=yes"
    else:
        return ""


def queue_job(
    job_path: pathlib.Path,
    is_required_as_dependency: bool,
    dependency_id_vars: list[str] = [],
    dependency_array_id_vars: list[str] = [],
) -> tuple[str, list[str]]:
    logger.info(f"Generating commands for {job_path.name}")
    commands = []
    commands.append(f'echo "Queueing job {job_path.name}"')
    dependency_string = get_bash_dependency_string(
        dependency_id_vars, dependency_array_id_vars
    )
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
