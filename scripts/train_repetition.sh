#!/bin/bash
#SBATCH --job-name=swpm-seq2seq       # Job name
#SBATCH --partition=gpu               # Take a node from the 'gpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --cpus-per-task=4             # Ask for 4 CPU cores
#SBATCH --gres=gpu:A40:1                  # Ask for 1 GPU
#SBATCH --mem=10G                     # Memory request; MB assumed if not specified
#SBATCH --time=1:00:00                # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log            # Standard output and error log

echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"

# check cuda compatability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'cuda device: {torch.cuda.current_device()}')"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# create execution environment
module purge                        
module load miniconda3/24.3.0-ui7c
conda create -n swpm python=3.12
conda activate swpm
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# launch your computation
echo "computation start $(date)"
python code/simulations/train_repetition.py
echo "computation end : $(date)"