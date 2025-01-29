#!/bin/bash
#SBATCH --job-name=swpm-gen           # Job name
#SBATCH --partition=gpu               # Take a node from the 'gpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --cpus-per-task=2             # Number of CPU cores requested
#SBATCH --gres=gpu:A40:1              # Number and type of GPUs requested
#SBATCH --mem=10G                     # Memory request; MB assumed if not specified
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/%j.log          # Standard output and error log
#SBATCH --nice=1000                   # Priority; higher is lower priority

echo ""
echo ""

# print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running job on $(hostname)"
echo "Generating train and test data"

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p logs/test

# create execution environment
module purge
module load miniconda3/24.3.0-ui7c
eval "$(conda shell.bash hook)"

# create environment only if it doesn't exist
if ! conda env list | grep -q "^swpm "; then
    conda create -n swpm python=3.12 -y
fi

# activate environment and install dependencies
conda activate swpm
pip install -r requirements.txt --quiet

# check if spacy model is already installed
if ! python -c "import spacy; spacy.load('en_core_web_lg')" 2>/dev/null; then
    python -m spacy download en_core_web_lg
else
    echo "SpaCy model already installed, skipping download"
fi

# print environment information
echo "python: $(which python)"
echo "python-version $(python -V)"
# echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"

# check cuda compatability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'DEVICE: {torch.cuda.current_device()}')"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# launch your computation
echo "computation start : $(date)"
echo ""

python scripts/generate_data.py

echo ""
echo "computation end : $(date)"

echo ""
echo ""