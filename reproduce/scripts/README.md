# Reproduction Scripts

## Overview

The scripts are designed to be modular and independent, with shared utilities for common functionality. Each script can cache results to avoid regenerating data unnecessarily.

## Scripts

### Individual Analysis Scripts

1. **`behavioral_analysis.py`** - Generate Figure 2A-D

   - Length errors, regressions, position errors, and sonority plots

2. **`phoneme_analysis.py`** - Generate Figure 3A-B

   - Phoneme embeddings dendrogram and feature importance analysis

3. **`ablation_search.py`** - Generate Figure 4

   - Lexical vs sublexical accuracy comparisons

4. **`ablation_behavioral.py`** - Generate Figure 5

   - Effects of ablating specific neurons (49, 31 by default)

5. **`univariate_analysis.py`** - Univariate feature importance analysis

   - How well individual features predict neuron activations

6. **`error_analysis.py`** - Generate Figure 15
   - Error type analysis across all neurons

### Master Script

7. **`run_all.py`** - Run all analyses in sequence
   - Can skip individual analyses with flags
   - Provides summary of completion status

## Usage

### Basic Usage

Run individual scripts with default settings:

```bash
cd reproduce/scripts
python behavioral_analysis.py
python phoneme_analysis.py
python ablation_search.py
# ... etc
```

### With Custom Parameters

All scripts support command-line arguments:

```bash
# Custom model and weights
python behavioral_analysis.py --model_name "CustomModel" --weights_path "/path/to/weights.pth"

# Custom batch size and directories
python phoneme_analysis.py --batch_size 512 --data_dir "/custom/data" --figures_dir "/custom/figures"

# Regenerate cached data
python ablation_search.py --regenerate

# Custom neuron IDs for ablation
python ablation_behavioral.py --neuron_ids 10 20 30
```

### Run All Analyses

```bash
# Run everything with defaults
python run_all.py

# Skip specific analyses
python run_all.py --skip_behavioral --skip_error_analysis

# Regenerate all cached results
python run_all.py --regenerate
```

## Common Arguments

Most scripts support these arguments:

- `--model_name`: Model architecture name (default: "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1")
- `--weights_path`: Path to model weights file
- `--batch_size`: Batch size for evaluation
- `--data_dir`: Directory for cached data files (default: ../data)
- `--figures_dir`: Directory for output figures (default: ../figures)
- `--regenerate`: Force regeneration of cached results
- `--seed`: Random seed (default: 42)

## Data Caching

Scripts automatically cache intermediate results in the `data/` directory:

- `wfe_enriched.csv` - Word Feature Evaluation results
- `ssp_enriched.csv` - Sonority Sequencing Principle results
- `phonemes_embeddings.npy` - Phoneme embeddings
- `wfe_ablated.csv` - Ablation results
- `univariate_importance.csv` - Feature importance results
- `error_analysis.csv` - Error type analysis results

To regenerate any cached data, use the `--regenerate` flag.

## Output Structure

Figures are organized in subdirectories:

```
figures/
├── behavioral/          # Figure 2 outputs
├── phonemes_*.png       # Figure 3 outputs
├── *_lex.png           # Figure 4 outputs
├── neuron_ablations/   # Figure 5 outputs
│   ├── 31/
│   └── 49/
└── error_type_analysis.png  # Figure 15 output
```

## Dependencies

The scripts use the same dependencies as the original notebook:

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- SciPy

Plus the custom `swp` package from the parent directory.

## Help

Each script has built-in help:

```bash
python behavioral_analysis.py --help
python run_all.py --help
```
