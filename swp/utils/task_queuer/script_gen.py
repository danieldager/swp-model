import logging
import pathlib
import sys

from ..paths import get_generated_scripts_dir, get_python_scripts_dir

logger = logging.getLogger()
jz_module = "pytorch-gpu/py3/1.11.0"  # TODO check right env to load


def trim(string: str) -> str:
    r"""Take a string and strips spaces at the beginnings and end of lines"""
    if not string:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = string.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return "\n".join(trimmed)


def base_slurm_file_generator(
    job_name: str,
    partition: str,
    qos: str | None,
    timestr: str,
    n_gpus: int,
    n_cpus: int,
    python_script: str,
    script_options: str,
    array_arg: str | None = None,
) -> pathlib.Path:
    r"""
    Generate a `{job_name}.slurm` file that allow `python_script` execution with `script_options` options.

    The slurm script is meant to be run on Jean-Zay super computer. Hence, it is possible to precise:
    - the `partition` on which to run the script
    - the `qos` which defines basically priority and max times
    - the time limit as a string `timestr`
    - the number of GPUs `n_gpus`
    - the number of CPUs `n_cpus`
    Some additionnal arguments for arrays can be passed through `array_arg`
    """
    logger.info(f"Generating slurm file for {job_name}")
    slurm_directory = get_generated_scripts_dir()
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

    module load {jz_module}
    set -x

    cd {str(get_python_scripts_dir().absolute())}
    srun python {python_script} {script_options}
    """
    with file_path.open("w") as f:
        f.write(trim(file_as_string))
    logger.info(f"Generated slurm file for {job_name}")
    return file_path


def autoarg_slurmarray_file_generator(
    job_name: str,
    partition: str,
    qos: str | None,
    timestr: str,
    n_gpus: int,
    n_cpus: int,
    python_script: str,
    arg_file: pathlib.Path,
    array_stop: int,
    array_start: int = 0,
) -> pathlib.Path:
    r"""
    Generate a `{job_name}.slurm` file that allow `python_script` execution with `script_options` options.

    The slurm script is meant to be run on Jean-Zay super computer. Hence, it is possible to precise:
    - the `partition` on which to run the script
    - the `qos` which defines basically priority and max times
    - the time limit as a string `timestr`
    - the number of GPUs `n_gpus`
    - the number of CPUs `n_cpus`
    Some additionnal arguments for arrays can be passed through `array_arg`
    """
    logger.info(f"Generating slurm file for {job_name}")
    slurm_directory = get_generated_scripts_dir()
    file_path = slurm_directory / f"{job_name}.slurm"
    file_as_string = f"""
    #!/bin/bash
    #SBATCH --job-name={job_name}
    #SBATCH --partition={partition}
    #SBATCH --output={str(slurm_directory.absolute())}/{job_name}_%A_%a.out
    #SBATCH --error={str(slurm_directory.absolute())}/{job_name}_%A_%a.err
    #SBATCH --time={timestr}
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --hint=nomultithread
    #SBATCH --gres=gpu:{n_gpus}
    #SBATCH --cpus-per-task={n_cpus}
    {f'#SBATCH --qos={qos}' if qos is not None else ""}
    #SBATCH --array={array_start}-{array_stop}

    module purge
    conda deactivate

    module load {jz_module}
    set -x

    ARGS=$(sed -n '$SLURM_ARRAY_TASK_ID p' < {str(arg_file.absolute())})
    cd {str(get_python_scripts_dir().absolute())}
    srun python {python_script} $ARGS
    """
    with file_path.open("w") as f:
        f.write(trim(file_as_string))
    logger.info(f"Generated slurm file for {job_name}")
    return file_path
