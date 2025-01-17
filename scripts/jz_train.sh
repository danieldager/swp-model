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