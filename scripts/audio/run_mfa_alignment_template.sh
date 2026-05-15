#!/usr/bin/env bash
# =============================================================================
# MFA Alignment Template — NOT an executable pipeline.
# Copy and adapt for your environment before running.
#
# Purpose: Force-align the paradigm corpus with MFA to produce TextGrid files
#          containing phone-level onset/offset times.
#
# Prerequisites:
#   - MFA installed in a separate conda environment (e.g. aligner)
#   - English acoustic model and G2P model downloaded
#   - Corpus prepared with scripts/audio/prepare_mfa_corpus.py
#
# Workflow:
#   1. Prepare corpus   (scripts/audio/prepare_mfa_corpus.py)
#   2. Run G2P          (mfa g2p)
#   3. Run alignment    (mfa align)
#   4. Convert TextGrids (scripts/audio/textgrid_to_phoneme_boundaries.py)
#
# All paths below are examples; adjust to match your MFA installation and
# the dataset you are aligning.
# =============================================================================

# ---------------------------------------------------------------------------
# STEP 0 — Activate your MFA conda environment
# ---------------------------------------------------------------------------
# conda activate aligner        # or whichever env has MFA installed
# mfa version                   # confirm MFA is available

# ---------------------------------------------------------------------------
# STEP 1 — Prepare the MFA corpus (run from swp-model/ repo root)
# ---------------------------------------------------------------------------
# Replace subset_male with the dataset you want to align:
#   subset_male | subset_female | subset_all_speakers

DATASET=data/external/paradigm/processed/subset_male.csv
CORPUS_DIR=data/external/paradigm/mfa/subset_male_corpus
ALIGNED_DIR=data/external/paradigm/mfa/subset_male_aligned

# python scripts/audio/prepare_mfa_corpus.py \
#     --dataset "$DATASET" \
#     --output  "$CORPUS_DIR"

# Optional: use speaker subfolders (MFA handles this transparently):
# python scripts/audio/prepare_mfa_corpus.py \
#     --dataset "$DATASET" \
#     --output  "$CORPUS_DIR" \
#     --speaker-subfolders

# ---------------------------------------------------------------------------
# STEP 2 — Download MFA models (once per installation)
#
# Model names below are placeholders. Run `mfa model list` to see what is
# available in your MFA installation, or check:
# https://mfa-models.readthedocs.io/en/latest/
#
# Common English options (confirm exact names with `mfa model list`):
#   Acoustic: english_us_arpa | english_mfa
#   G2P:      english_us_arpa_g2p
# ---------------------------------------------------------------------------
# mfa model download acoustic   <english_acoustic_model>
# mfa model download g2p        <english_g2p_model>

# ---------------------------------------------------------------------------
# STEP 3 — Generate a pronunciation dictionary via G2P
#
# This creates a dictionary covering all words/transcripts in the corpus.
# Replace <english_g2p_model> with the actual model name.
# ---------------------------------------------------------------------------
DICT_PATH=data/external/paradigm/mfa/subset_male_pronunciation.dict

# mfa g2p \
#     "$CORPUS_DIR" \
#     <english_g2p_model> \
#     "$DICT_PATH" \
#     --clean

# Alternatively, use a pre-built pronunciation dictionary if available:
# DICT_PATH=/path/to/english_us_arpa.dict

# ---------------------------------------------------------------------------
# STEP 4 — Run forced alignment
#
# Replace <english_acoustic_model> with the actual model name.
# --clean removes intermediate files after alignment.
# --output-format textgrid (default; produces one .TextGrid per wav file)
# ---------------------------------------------------------------------------
# mfa align \
#     "$CORPUS_DIR" \
#     "$DICT_PATH" \
#     <english_acoustic_model> \
#     "$ALIGNED_DIR" \
#     --clean

# ---------------------------------------------------------------------------
# STEP 5 — Convert TextGrids to phoneme boundary CSV
#         (run from swp-model/ repo root, in the swpm conda environment)
# ---------------------------------------------------------------------------
# python scripts/audio/textgrid_to_phoneme_boundaries.py \
#     --dataset      "$DATASET" \
#     --textgrid-dir "$ALIGNED_DIR" \
#     --output data/external/paradigm/processed/phoneme_boundaries_mfa_subset_male.csv

# ---------------------------------------------------------------------------
# STEP 6 — Build phoneme embeddings
#         (run from swp-model/ repo root, in the swpm conda environment)
# ---------------------------------------------------------------------------
# python scripts/audio/build_phoneme_embeddings.py \
#     --run         reproduce/data/audio/auristream__9d3f269f \
#     --boundaries  data/external/paradigm/processed/phoneme_boundaries_mfa_subset_male.csv \
#     --layers      embedding block_12 block_24 block_36 block_48
