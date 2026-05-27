# Audio Codec Benchmark

End-to-end pipeline for running neural audio codecs on the lab paradigm
stimulus set, extracting encoder/decoder activations, computing signal
metrics, and running SWP-style statistical analyses.

All commands are run from the **`swp-model/` repo root**.

---

## Overview

| Step | Script | What it does |
|------|--------|-------------|
| 0 | `scripts/audio/setup_paradigm.py` | Build normalized CSV from raw metadata + WAV files |
| Sanity | `scripts/audio/sanity_check.py` | Smoke-test a model before committing to a full run |
| 1 | `scripts/audio/extract.py` | Reconstruction metrics + activation extraction |
| 2 | `reproduce/scripts/audio/analysis.py` | Signal metric analysis (mel distance, SI-SDR) |
| 3 | `reproduce/scripts/audio/activation_analysis.py` | Latent activation analysis (norms, trajectories) |
| 4 | `reproduce/scripts/audio/pca_analysis.py` | PCA geometry + condition distance over time |
| 4b | `reproduce/scripts/audio/pca_summary.py` | 2×2 cross-model/layer summary panels |
| 5 | `scripts/audio/build_xarray.py` | Canonical xarray export `(trials, time, neurons)` |
| 6 | `scripts/audio/run_univariate_encoding.py` | Per-neuron Ridge CV → scores / weights / FI CSVs |
| 7 | `scripts/audio/summarize_univariate_encoding.py` | Aggregate summary CSVs (both layers in ONE call) |
| 8 | `scripts/audio/plot_univariate_encoding_summary.py` | Per-view diagnostic plots → `diagnostic_plots/` |
| 9 | `scripts/audio/summarize_length_controlled_encoding.py` | Cross-view compact summary + canonical figures |
| 10 | `scripts/audio/inspect_encoding_units.py` | Unit-level FI inspection (both layers) |
| 11 | `scripts/audio/build_final_encoding_report.py` | Assemble canonical figures + write `report.md` |
| — | ~~`scripts/audio/compare_univariate_codecs.py`~~ | **LEGACY** — not analysis_view-safe |
| — | ~~`scripts/audio/build_unit_profiles.py`~~ | **LEGACY** — not analysis_view-safe |

See `scripts/audio/PIPELINE.md` for the detailed step-by-step technical reference.

---

## Data: the paradigm stimulus set

### What it is

English word repetition stimuli from the lab paradigm (`en_paradigm.csv`).
Items are single words or pseudowords recorded by a single speaker.

Each item is characterized by four factors:

| Factor | CSV values | Applies to |
|--------|-----------|-----------|
| `lexicality` | `word` / `nonword` *(displayed as "pseudoword")* | all items |
| `length_bin` | `long` / `short` | all items |
| `frequency_bin` | `high` / `low` | words only (NA for pseudowords) |
| `morphology` | `complex` / `simple` | all items |

> **Terminology note:** The raw CSV value for non-word items is `nonword`.
> All scientific displays (figures, labels) use **pseudoword** instead.

Condition codes in the raw CSV encode these factors:
- Words (4-char): `R` + length (`L`/`S`) + frequency (`H`/`L`) + morphology (`C`/`S`)
  — e.g. `RLLC` = word, long, low-freq, complex
- Pseudowords (3-char): `P` + length (`L`/`S`) + morphology (`C`/`S`)
  — e.g. `PLC` = pseudoword, long, complex

### Raw files (not versioned)

Place under `data/external/paradigm/raw/`:
```
data/external/paradigm/raw/
├── metadata.csv          # item metadata (condition codes, filenames)
└── wav/                  # WAV files referenced by metadata.csv
```

### Step 0 — Build the processed CSV

```bash
# Dry-run first to verify condition decoding:
python scripts/audio/setup_paradigm.py \
    --metadata data/external/paradigm/raw/metadata.csv \
    --wav-dir   data/external/paradigm/raw/wav \
    --filename-col Audio_Male \
    --dry-run

# Generate the processed CSV for the male speaker:
python scripts/audio/setup_paradigm.py \
    --metadata data/external/paradigm/raw/metadata.csv \
    --wav-dir   data/external/paradigm/raw/wav \
    --filename-col Audio_Male \
    --output    data/external/paradigm/processed/subset_male.csv
```

Output: `data/external/paradigm/processed/subset_male.csv` — 180 items
(normalized columns: `item_id`, `wav_path`, `original_sample_rate`,
`duration_s`, `speaker`, `condition`, `lexicality`, `length_bin`,
`morphology`, `frequency_bin`).

---

## Dependencies

```bash
pip install -r requirements.txt
```

Audio-specific packages added on this branch:
- `torchaudio` — WAV I/O and mel spectrogram
- `transformers` — HuggingFace EnCodec
- `descript-audio-codec` — DAC
- `descript-audiotools` — required by DAC at runtime
- `xarray` — N-dimensional labeled arrays (Step 5)
- `h5netcdf` — NetCDF4 write backend (Step 5)

Model weights are downloaded automatically on first use:
- EnCodec: `facebook/encodec_24khz` cached by HuggingFace (`~/.cache/huggingface/`)
- DAC: cached under `~/.cache/descript/dac/`

---

## Smoke test

Run before committing to a full extraction:

```bash
# EnCodec (default)
python scripts/audio/sanity_check.py --model encodec --bandwidth 6.0

# DAC 24 kHz
python scripts/audio/sanity_check.py --model dac --model-type 24khz

# DAC 16 kHz
python scripts/audio/sanity_check.py --model dac --model-type 16khz
```

Checks: model loads, `reconstruct()` returns finite audio, `extract_activations()`
works for all three stable layers, activation shapes are correct.

---

## Step 1 — Extraction

Runs each WAV through the model, saves signal metrics and per-layer activations.

### EnCodec

```bash
python scripts/audio/extract.py \
    --model encodec --model-arg bandwidth=6.0 \
    --dataset data/external/paradigm/processed/subset_male.csv \
    --layers encoder_out decoder_in decoder_out \
    --output reproduce/data/audio/
```

### DAC

```bash
python scripts/audio/extract.py \
    --model dac --model-arg model_type=24khz \
    --dataset data/external/paradigm/processed/subset_male.csv \
    --layers encoder_out decoder_in decoder_out \
    --output reproduce/data/audio/
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--regenerate` | off | Overwrite an existing run (otherwise skipped) |
| `--save-audio` | off | Save original + reconstructed WAV files for listening |

With `--save-audio`:
```bash
python scripts/audio/extract.py \
    --model dac --model-arg model_type=24khz \
    --dataset data/external/paradigm/processed/subset_male.csv \
    --layers encoder_out decoder_in decoder_out \
    --output reproduce/data/audio/ \
    --regenerate --save-audio
```

### Output structure

Each run lands under `reproduce/data/audio/{model}__{hash}/`:

```
reproduce/data/audio/
├── encodec__7f7d3b97/
│   ├── manifest.json            # run params + item → layer → .pt path map
│   ├── metrics.csv              # item_id, mel_distance, si_sdr
│   ├── activations/
│   │   ├── infertile_MALE_D__encoder_out.pt
│   │   ├── infertile_MALE_D__decoder_in.pt
│   │   ├── infertile_MALE_D__decoder_out.pt
│   │   └── ...                  # 3 files × 180 items = 540 .pt files
│   └── audio/                   # only when --save-audio was used
│       ├── infertile_MALE_D__original.wav
│       ├── infertile_MALE_D__reconstructed.wav
│       └── ...
└── dac__f104bbdd/
    └── ...
```

The `{hash}` (8-hex-char SHA-256) is deterministic: same model + same dataset +
same layers always produce the same `run_id`. Re-running without `--regenerate`
is a no-op.

### Activation shapes

Both models expose three stable layer names with identical semantics:

| Layer | Type | Shape | Description |
|-------|------|-------|-------------|
| `encoder_out` | 2D | `[D, T_enc]` | Encoder output before quantizer |
| `decoder_in` | 2D | `[D, T_enc]` | Quantizer output fed to decoder |
| `decoder_out` | 2D | `[1, T_audio]` | Reconstructed waveform |

`D` = 128 for EnCodec, 1024 for DAC.
`T_enc` ≈ `T_audio / 320` at 24 kHz (≈ 75 frames for a 1-second item).

---

## Step 2 — Signal metric analysis

```bash
python reproduce/scripts/audio/analysis.py \
    --results reproduce/data/audio/encodec__7f7d3b97/metrics.csv
```

The dataset CSV path is read automatically from `manifest.json` — no extra argument needed.

Output written to `reproduce/figures/audio/{run_id}/`:

| File | Content |
|------|---------|
| `summary_by_factor.csv` | Mean ± std per factor level, both metrics |
| `regression_summary.txt` | OLS results: univariate + multivariate, all items and words-only |
| `mel_distance_by_lexicality.png` | Bar plot (mean ± 95 % CI) |
| `mel_distance_by_length_bin.png` | … |
| `mel_distance_by_morphology.png` | … |
| `mel_distance_by_frequency_bin.png` | Words only |
| `si_sdr_by_*.png` | Same set for SI-SDR |

**Metric notes:**
- `mel_distance`: mean absolute log-mel spectrogram distance — lower is better.
  Comparable across models. Primary cross-model metric.
- `si_sdr`: Scale-Invariant SDR (dB) — measures waveform-level phase alignment.
  Meaningful for near-lossless codecs. Negative values are expected for
  high-compression codecs like DAC — this is not a bug.

---

## Step 3 — Activation analysis

```bash
python reproduce/scripts/audio/activation_analysis.py \
    --run reproduce/data/audio/encodec__7f7d3b97/
```

Loads all `.pt` files from `activations/`, merges paradigm factor columns via
`manifest.json`, computes summary features per item per layer, then runs OLS.

Output written to `reproduce/figures/audio/{run_id}/activations/`:

**Features computed (2D layers: `encoder_out`, `decoder_in`)**

| Feature | Description |
|---------|-------------|
| `mean_norm` | Mean L2 norm of frame vectors across time |
| `trajectory_length` | Cumulative L2 distance of successive frames in channel space |
| `trajectory_length_per_frame` | `trajectory_length / n_frames` — duration-normalized |
| `mean_step_distance` | `trajectory_length / (n_frames − 1)` — per-step average |
| `n_frames` | Number of temporal frames |

**Analyses**

The script runs four OLS models per factor × layer to disentangle duration
effects from genuine latent dynamics:

1. `trajectory_length ~ C(factor)` — baseline (confounded by duration)
2. `trajectory_length ~ C(factor) + n_frames` — frame-controlled
3. `trajectory_length_per_frame ~ C(factor)` — duration-normalized dependent variable
4. `mean_step_distance ~ C(factor)` — per-step normalization

Plus `mean_norm ~ C(factor)` and a cross-layer comparison table.

---

## Step 4 — PCA geometry

Fits a 2-component PCA on codec activations per (run, layer). Supports three modes:
`point` (mean-pooled per item), `trajectory` (frame-level), and `both`.

```bash
# Point + trajectory for all layers
python reproduce/scripts/audio/pca_analysis.py \
    --run reproduce/data/audio/encodec__7f7d3b97/ --mode both

# Trajectory only, single layer
python reproduce/scripts/audio/pca_analysis.py \
    --run reproduce/data/audio/dac__f104bbdd/ \
    --mode trajectory --layers decoder_in
```

Outputs to `reproduce/figures/audio/pca/{run_id}/{layer}/`.
Visual grammar: **color = lexicality** (word=blue, pseudoword=red), **marker/linestyle = length** (short=○/solid, long=△/dashed).
Trajectory mode also produces polar plots and condition-distance curves over normalized time.

```bash
# 2×2 cross-model/layer summary panels (reads existing CSVs, no recomputation)
python reproduce/scripts/audio/pca_summary.py
# Output: reproduce/figures/audio/pca/summary/
```

See `docs/audio_project_state.md` §7a for full interpretive notes.

---

## Models

### EnCodec (`encodec`)

HuggingFace `facebook/encodec_24khz`. Neural audio codec trained for
speech compression.

| Kwarg | Values | Default |
|-------|--------|---------|
| `bandwidth` | 1.5 / 3.0 / 6.0 / 12.0 / 24.0 (kbps) | 6.0 |

```bash
--model encodec --model-arg bandwidth=6.0
```

### DAC (`dac`)

Descript Audio Codec (`descript-audio-codec`). State-of-the-art
universal audio codec.

| Kwarg | Values | Default | Sample rate |
|-------|--------|---------|-------------|
| `model_type` | `16khz` / `24khz` / `44khz` | `24khz` | 16 000 / 24 000 / 44 100 Hz |

```bash
--model dac --model-arg model_type=24khz
```

Both models auto-select device: CUDA > MPS > CPU.

### AuriStream (`auristream`)

Biologically inspired speech language model (Tuckute et al., Interspeech 2025).
Converts waveforms to cochlear token IDs (WavCoch) then extracts layer-wise
continuous hidden states from a GPT-style transformer.

**This is a representation-only model.** It does not reconstruct audio and
produces no signal metrics (mel distance, SI-SDR). Use
`scripts/audio/extract_representations.py` — not `extract.py`.

| Parameter | Value |
|---|---|
| Sample rate | 16 000 Hz, mono |
| Token rate | 200 Hz (5 ms per token) |
| Token count | `L = floor(n_samples / 80)` |
| Hidden dim | 1 280 (AuriStream-1B) |
| Layers | `embedding`, `block_01` … `block_48` |

```bash
python scripts/audio/extract_representations.py \
    --model auristream \
    --dataset data/external/paradigm/processed/subset_male.csv \
    --layers embedding block_12 block_24 block_36 block_48 \
    --output reproduce/data/audio/
```

Output: `reproduce/data/audio/auristream__{hash}/`
- `manifest.json` — run params (token_rate_hz, hop_length, transformers_version, …)
- `extraction_summary.csv` — per-item n_tokens, trailing_samples, represented_duration_s
- `activations/{item_id}__{layer}.pt` — shape `[1280, L]` float32

**Requires** HuggingFace login and acceptance of gated-model terms:
```bash
huggingface-cli login
```

See `docs/auristream_setup.md` for setup history and `docs/audio_project_state.md`
for scientific context and validated run IDs.

---

## AuriStream phoneme embeddings

Builds phoneme-level AuriStream embeddings by mean-pooling the saved temporal
hidden states over phoneme boundary intervals from MFA forced alignment.

This is a four-step pipeline. Steps 1 and 3–4 are run from the `swpm` conda
environment; step 2 is run in a separate MFA conda environment.

### Step A — Prepare MFA corpus

```bash
python scripts/audio/prepare_mfa_corpus.py \
    --dataset data/external/paradigm/processed/subset_male.csv \
    --output  data/external/paradigm/mfa/subset_male_corpus
```

Creates `{item_id}.wav` + `{item_id}.txt` pairs. The transcript is inferred
from the item_id by default (e.g. `infertile_MALE_D` → `infertile`).

### Step B — Run MFA forced alignment (user-run, separate environment)

```bash
# Activate your MFA environment, then run:
# (model names are examples — confirm with `mfa model list` in your installation)

# mfa model download acoustic <english_acoustic_model>
# mfa model download g2p      <english_g2p_model>

# mfa g2p \
#     data/external/paradigm/mfa/subset_male_corpus \
#     <english_g2p_model> \
#     data/external/paradigm/mfa/subset_male_pronunciation.dict

# mfa align \
#     data/external/paradigm/mfa/subset_male_corpus \
#     data/external/paradigm/mfa/subset_male_pronunciation.dict \
#     <english_acoustic_model> \
#     data/external/paradigm/mfa/subset_male_aligned
```

See `scripts/audio/run_mfa_alignment_template.sh` for the full annotated template.

### Step C — Convert TextGrids to boundary CSV

```bash
python scripts/audio/textgrid_to_phoneme_boundaries.py \
    --dataset      data/external/paradigm/processed/subset_male.csv \
    --textgrid-dir data/external/paradigm/mfa/subset_male_aligned \
    --output       data/external/paradigm/processed/phoneme_boundaries_mfa_subset_male.csv
```

Output: `data/external/paradigm/processed/phoneme_boundaries_mfa_{dataset_name}.csv`
with columns: `item_id, phoneme, phoneme_index, start_s, end_s, duration_s, word, tier, textgrid_path`

**Dataset naming convention:** the boundary CSV is named after the dataset stem so that
multiple datasets can coexist:
- `phoneme_boundaries_mfa_subset_male.csv`
- `phoneme_boundaries_mfa_subset_female.csv`
- `phoneme_boundaries_mfa_subset_all_speakers.csv`

The boundary CSV must match the dataset used for the AuriStream extraction run it
will be paired with (e.g. `auristream__9d3f269f` was extracted from `subset_male.csv`).

### Step D — Build phoneme embeddings

```bash
python scripts/audio/build_phoneme_embeddings.py \
    --run        reproduce/data/audio/auristream__9d3f269f \
    --boundaries data/external/paradigm/processed/phoneme_boundaries_mfa_subset_male.csv \
    --layers     embedding block_12 block_24 block_36 block_48
```

Smoke test (3 items):

```bash
python scripts/audio/build_phoneme_embeddings.py \
    --run        reproduce/data/audio/auristream__9d3f269f \
    --boundaries data/external/paradigm/processed/phoneme_boundaries_mfa_subset_male.csv \
    --layers     embedding block_12 block_48 \
    --max-items  3
```

Output: `reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/`
- `metadata_phonemes.csv` — one row per phoneme, aligned with tensors
- `embeddings_{layer}.pt` — shape `[n_phonemes, 1280]` float32
- `manifest.json` — provenance, boundary hash, token selection rule

**Validated run (2026-05-15, `subset_male`):**
1055 phonemes, 180 items, 5 layers, `[1055, 1280]` float32, zero zero-token rows,
token_selection=center. Source run: `auristream__9d3f269f`.

> **Note:** Generated data files are not committed:
> `data/external/paradigm/mfa/`, `phoneme_boundaries_mfa_*.csv`,
> `reproduce/data/audio/*/phoneme_embeddings/`

### Step E — Analyse phoneme embeddings: PCA + norm by position

```bash
# Full run — all 5 layers, with Nafis-comparable focus-phone plots
python scripts/audio/auristream_phoneme_pca_norm.py \
    --embeddings-dir reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/ \
    --dataset        data/external/paradigm/processed/subset_male.csv \
    --layers         embedding block_12 block_24 block_36 block_48 \
    --focus-phones   AH IH ER \
    --overwrite
```

Add `--focus-phone-refit-pca` to also refit PCA on each focus-phone subset
(closer to Nafis's per-phoneme PCA plots).

Output: `reproduce/figures/audio/auristream_phonemes/auristream__9d3f269f/`

Per layer: PCA scatter plots (by position, lexicality, phone label, **phoneme type**:
vowel / consonant / other), **per-phone focus plots** in global PCA space (one per
`--focus-phones` entry, coloured by position), norm-by-position curves (from start / from end,
split by lexicality and length), and aligned CSVs.

Phoneme type is ARPAbet-based: `phoneme_base` strips trailing stress digits (e.g. `AH1` → `AH`);
15-vowel set: AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW. `phoneme_base` and `phoneme_type`
are written into all per-layer CSVs automatically.

**Validated run (2026-05-17, `subset_male`, 5 layers):**

| Layer | PC1 | PC2 | n |
|---|---|---|---|
| `embedding` | 14.4 % | 9.6 % | 1055 |
| `block_12` | 11.5 % | 6.9 % | 1055 |
| `block_24` | 12.6 % | 7.9 % | 1055 |
| `block_36` | 11.6 % | 7.7 % | 1055 |
| `block_48` | 82.9 % | 2.9 % | 1055 |

Note: `block_48` PC1 dominance (82.9 %) is a vector-magnitude effect — see Step G
(`auristream_block48_diagnostics.py`) for full diagnostics.

### Step F — Cross-layer summary panels

Assembles side-by-side comparison panels from the per-layer CSVs produced in Step E.

```bash
python scripts/audio/auristream_phoneme_summary_panels.py \
    --figures-dir reproduce/figures/audio/auristream_phonemes/auristream__9d3f269f/ \
    --overwrite
```

Output: `{figures-dir}/summary_panels/`

| Panel | Description |
|---|---|
| `summary_pca_by_phoneme_type_all_layers.png` | PCA: vowel / consonant / other across all layers |
| `summary_pca_by_position_all_layers.png` | PCA by phoneme position across all layers |
| `summary_pca_by_lexicality_all_layers.png` | PCA by lexicality across all layers |
| `summary_norm_by_position_start_all_layers.png` | Norm by position, shared y-axis |
| `summary_norm_by_position_start_all_layers_free_y.png` | Norm by position, free y-axis (compare shape) |
| `summary_norm_by_position_start_all_layers_normalized.png` | Norm normalised to position 0 |
| `summary_norm_by_position_lexicality_all_layers.png` | Norm by position × lexicality |

### Step G — block_48 PC1 diagnostics

Investigates why `block_48` PC1 explains 82.9 % of variance (vs. 11–13 % for other layers).

```bash
python scripts/audio/auristream_block48_diagnostics.py \
    --figures-dir    reproduce/figures/audio/auristream_phonemes/auristream__9d3f269f/ \
    --embeddings-dir reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/ \
    --layer          block_48
```

**Validated findings (2026-05-18):**
- PC1 ~ L2 norm: Pearson r = 1.000, Spearman ρ = 1.000 — PC1 is the norm axis
- After L2-normalising embeddings: PC1 drops from 82.9 % → 12.8 % (in line with other layers)
- Interpretation: `block_48` PC1 reflects embedding magnitude. For geometric analyses
  (phoneme type, position), use L2-normalised embeddings or PC2+.

Outputs: `{layer}_pc1_diagnostics.csv`, `{layer}_pc1_correlations.csv`,
`{layer}_normalized_pca_comparison.json`, and three scatter PNGs (PC1 vs norm / position / n_tokens).
The `--layer` flag accepts any layer name; `block_48` is the default.

---

## Package structure

```
swp/audio/
├── models/
│   ├── base.py          # AudioModel Protocol + ReconstructionResult TypedDict
│   ├── registry.py      # @register / get_model() lazy registry
│   ├── device.py        # select_device() — CUDA > MPS > CPU fallback
│   ├── encodec.py       # EnCodecModel wrapper (HuggingFace)
│   ├── dac.py           # DACModel wrapper (descript-audio-codec)
│   └── auristream.py    # AuriStreamModel wrapper (WavCoch + AuriStream-1B)
├── hooks/
│   └── manager.py       # HookManager — forward hooks + pre-hooks + extractor_fn
├── datasets/
│   ├── base.py          # ParadigmDataset — reads CSV, loads WAV at target_sr
│   └── utils.py         # load_audio() helper
├── metrics/
│   └── signal.py        # mel_distance(), si_sdr(), compute_all()
├── pipeline/
│   ├── extraction.py               # run_extraction() — codec reconstruction + activations
│   └── representation_extraction.py  # run_representation_extraction() — hidden states only
├── phonemes/
│   ├── boundaries.py               # load_boundaries(), SILENCE_LABELS, REQUIRED_COLUMNS
│   └── pooling.py                  # tokens_for_phoneme(), pool_phoneme(), build_phoneme_embeddings()
└── encoding/
    ├── xarray_builder.py     # Step 5: .pt → xarray.Dataset (trials, time, neurons)
    ├── temporal_binning.py   # relative-time binning for encoding analyses
    ├── design_matrix.py      # effect-coded (−1/+1) design matrix from trial metadata
    ├── univariate_encoder.py # AudioUnivariateEncoder — Ridge nested CV (from Intracranial/encoder.py)
    ├── univariate.py         # orchestration: binning + design matrix + per-neuron Ridge CV
    ├── univariate_summary.py # aggregate CSV summaries from univariate run outputs
    ├── univariate_plots.py   # score / weight / FI figures
    ├── univariate_compare.py # cross-codec comparison CSVs + figures
    └── unit_profiles.py      # unit-level FI profiles, dominance, top-k tables + figures

scripts/audio/
├── setup_paradigm.py                          # Step 0: build processed CSV from raw data
├── sanity_check.py                            # Smoke test for any registered model
├── extract.py                                 # Step 1 CLI (codec reconstruction + activations)
├── extract_representations.py                 # Representation-only extraction CLI (AuriStream, …)
├── prepare_mfa_corpus.py                      # Step A: prepare MFA corpus from paradigm CSV
├── textgrid_to_phoneme_boundaries.py          # Step C: TextGrid → phoneme boundary CSV
├── build_phoneme_embeddings.py                # Step D: mean-pool hidden states → phoneme embeddings
├── auristream_phoneme_pca_norm.py             # Step E: PCA + norm-by-position analysis of phoneme embeddings
├── auristream_block48_diagnostics.py          # PC1 diagnostics for a specific layer (block_48 focus)
├── auristream_phoneme_summary_panels.py       # Cross-layer summary panels from per-layer CSVs
├── auristream_wpe_position_diagnostics.py     # 7 diagnostics: wpe norm slope, random-init control,
│                                              #   phoneme projection, shift/shuffle controls,
│                                              #   directional geometry, layer propagation
├── run_mfa_alignment_template.sh              # Step B: MFA alignment template (not executable)
├── build_xarray.py                            # Step 5 CLI: canonical xarray export
├── run_univariate_encoding.py                 # Step 6 CLI: per-neuron Ridge CV
├── summarize_univariate_encoding.py           # Step 7 CLI: summary CSVs (both layers in ONE call)
├── plot_univariate_encoding_summary.py        # Step 8 CLI: per-view diagnostic plots
├── summarize_length_controlled_encoding.py    # Step 9 CLI: cross-view canonical summary + figures
├── inspect_encoding_units.py                  # Step 10 CLI: unit-level FI inspection
├── build_final_encoding_report.py             # Step 11 CLI: assemble canonical figures + report.md
├── PIPELINE.md                                # Technical step-by-step reference
├── compare_univariate_codecs.py               # LEGACY — not analysis_view-safe
└── build_unit_profiles.py                     # LEGACY — not analysis_view-safe

reproduce/scripts/audio/
├── analysis.py          # Step 2: signal metric OLS + bar plots
├── activation_analysis.py  # Step 3: activation feature OLS + trajectory plots
├── pca_analysis.py      # Step 4: PCA geometry + condition distances
└── pca_summary.py       # Step 4b: 2×2 cross-model/layer summary panels
```

---

## Step 5 — Canonical xarray export

Builds `xarray.Dataset` files with dimensions `(trials, time, neurons)` from the
`.pt` activation files produced by Step 1 (codecs) or `extract_representations.py`
(AuriStream). Required for encoding-model analyses.

Any latent `[D, T]` layer is accepted. `decoder_out` (waveform shape `[1, T_audio]`)
is explicitly rejected. If `--layers` is omitted, all eligible layers from the run
manifest are used automatically.

```bash
# Codec runs (explicit layers)
python scripts/audio/build_xarray.py \
    --run reproduce/data/audio/encodec__7f7d3b97/ \
    --layers encoder_out decoder_in

python scripts/audio/build_xarray.py \
    --run reproduce/data/audio/dac__f104bbdd/ \
    --layers encoder_out decoder_in

# AuriStream run (explicit layers)
python scripts/audio/build_xarray.py \
    --run reproduce/data/audio/auristream__6ee9aeb6/ \
    --layers embedding block_01 block_24 block_48_lnf

# Any run — infer all eligible layers from manifest
python scripts/audio/build_xarray.py \
    --run reproduce/data/audio/auristream__6ee9aeb6/ \
    --overwrite
```

Outputs written to `{run}/xarray/`:

| File | Content |
|------|---------|
| `encoder_out.nc` | `xr.Dataset(dims=(trials, time, neurons))` |
| `encoder_out_qc.json` | QC report (n_trials, n_neurons, padding fraction, NaN checks) |
| `decoder_in.nc` | idem |
| `decoder_in_qc.json` | idem |
| `metadata_trials.csv` | trial-level metadata, aligned with trials dimension |

Time axis is in seconds from word onset. Padded positions are NaN in `activations`
and False in `valid_time`. Activations are raw (no normalisation, no resampling).

The script is idempotent: it refuses to overwrite without `--overwrite`.

---

## Step 6 — Univariate encoding

Per-neuron Ridge regression with nested 5×5 CV. Two analysis sets:
`all_items` (lexicality, length_bin, morphology) and `words_only` (frequency_bin, length_bin, morphology).

```bash
python scripts/audio/run_univariate_encoding.py \
    --xarray reproduce/data/audio/encodec__7f7d3b97/xarray/encoder_out.nc \
    --analysis-set all_items \
    --n-bins 10 --metrics r2 spearman \
    --compute-fi --n-jobs -1 \
    --output reproduce/figures/audio/encoding/encodec__7f7d3b97/encoder_out/univariate/all_items_fi/
```

Key flags: `--n-bins` (default 10), `--compute-fi`, `--max-neurons` (for quick tests).
Outputs: `scores.csv`, `weights.csv`, `feature_importance.csv` (if `--compute-fi`), `config.json`, `qc_check.json`.

---

## Step 7 — Summaries

Aggregates per-neuron CSVs into population-level summary tables.

```bash
# Pass all run dirs for a codec in one call (--inputs accepts multiple paths)
python scripts/audio/summarize_univariate_encoding.py \
    --inputs \
        reproduce/figures/audio/encoding/encodec__7f7d3b97/encoder_out/univariate/all_items_fi/ \
        reproduce/figures/audio/encoding/encodec__7f7d3b97/encoder_out/univariate/words_only_fi/ \
        reproduce/figures/audio/encoding/encodec__7f7d3b97/decoder_in/univariate/all_items_fi/ \
        reproduce/figures/audio/encoding/encodec__7f7d3b97/decoder_in/univariate/words_only_fi/ \
    --output reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/ \
    --overwrite
```

Outputs: `score_summary_by_time.csv`, `weight_summary_by_feature_time.csv`,
`global_feature_ranking.csv`, `top_units_by_feature.csv`.
FI outputs (`fi_summary_by_feature_time.csv`, `global_fi_ranking.csv`, `top_units_by_fi.csv`)
are produced only when at least one input dir contains `feature_importance.csv`.

---

## Step 8 — Diagnostic plots

Output goes to `diagnostic_plots/` — these are per-view diagnostic figures, **not canonical**.
Canonical figures are produced in Steps 9–11.

```bash
# Per-view diagnostic plots (score, weights, FI)
python scripts/audio/plot_univariate_encoding_summary.py \
    --summary-dir reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/ \
    --output      reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/diagnostic_plots/ \
    --fi-dirs \
        reproduce/figures/audio/encoding/encodec__7f7d3b97/encoder_out/univariate/all_items_short_only/ \
        reproduce/figures/audio/encoding/encodec__7f7d3b97/decoder_in/univariate/all_items_short_only/ \
    --overwrite
```

Key outputs in `diagnostic_plots/`: `score_over_time_{model}_{view}_{metric}.png`,
`weights_over_time_{model}_{view}.png`, `encoding_fi_{model}_{layer}_{view}_{metric}.png`.

---

## Steps 9–11 — Finalized encoding analysis pipeline (chantier 2.5)

These steps replace the legacy Steps 9–10 (`compare_univariate_codecs.py`,
`build_unit_profiles.py`) for multi-view all-speakers analyses. They are
**analysis_view-safe**: each analysis_view is tracked as a first-class identifier
and summaries are never aggregated across different experimental conditions.

> **Important — Step 7 correction:** Pass `encoder_out` and `decoder_in` together
> in ONE `summarize_univariate_encoding.py` call. Running them in a loop with
> `--overwrite` would cause the second layer to overwrite the first layer's summaries.

### Step 9 — Cross-view compact summary

Reads `global_fi_ranking.csv` and `fi_summary_by_feature_time.csv` from multiple
univariate summary directories (all-speakers + male-only length-controlled) and
writes comparison figures to `length_controlled_summary/`.

```bash
ENC=reproduce/figures/audio/encoding/encodec__b429aeea
DAC=reproduce/figures/audio/encoding/dac__d466515b
ENC_M=reproduce/figures/audio/encoding/encodec__7f7d3b97
DAC_M=reproduce/figures/audio/encoding/dac__f104bbdd

python scripts/audio/summarize_length_controlled_encoding.py \
    --summaries \
        all_speakers:${ENC}/univariate_summary/ \
        all_speakers:${DAC}/univariate_summary/ \
        male_only:${ENC_M}/univariate_summary_length_controlled/ \
        male_only:${DAC_M}/univariate_summary_length_controlled/ \
    --output reproduce/figures/audio/encoding/length_controlled_summary/ \
    --overwrite
```

Outputs in `length_controlled_summary/`:

| File | Content |
|---|---|
| `fi_global_comparison.csv` | Combined FI ranking across all sources and views |
| `fi_global_heatmap.png` | Heatmap: mean FI per (view × layer) row, feature column |
| `male_only_short_long_fi_summary.png` | Male-only baseline: short_only vs long_only FI contrast |
| `short_only_morphology_vs_lexicality.png` | Main residual effects after length control |
| `long_only_morphology_vs_lexicality.png` | Long-only residual (weaker/mixed) |
| `temporal_fi_short_only.png` | Temporal FI profile — short_only |
| `temporal_fi_long_only.png` | Temporal FI profile — long_only |
| `speaker_effect_summary.png` | Speaker FI in speakerctrl views |
| `summary_readme.md` | Auto-generated summary report |

### Step 10 — Unit inspection

Reads `feature_importance.csv` + `scores.csv` from per-analysis_view run directories.
Must include **both layers** (`encoder_out` and `decoder_in`).

```bash
python scripts/audio/inspect_encoding_units.py \
    --fi-dirs \
        all_speakers:${ENC}/encoder_out/univariate/all_items_all_speakers/ \
        all_speakers:${ENC}/encoder_out/univariate/all_items_short_only/ \
        all_speakers:${ENC}/encoder_out/univariate/all_items_long_only/ \
        all_speakers:${ENC}/decoder_in/univariate/all_items_all_speakers/ \
        all_speakers:${ENC}/decoder_in/univariate/all_items_short_only/ \
        all_speakers:${ENC}/decoder_in/univariate/all_items_long_only/ \
        all_speakers:${DAC}/encoder_out/univariate/all_items_all_speakers/ \
        all_speakers:${DAC}/encoder_out/univariate/all_items_short_only/ \
        all_speakers:${DAC}/encoder_out/univariate/all_items_long_only/ \
        all_speakers:${DAC}/decoder_in/univariate/all_items_all_speakers/ \
        all_speakers:${DAC}/decoder_in/univariate/all_items_short_only/ \
        all_speakers:${DAC}/decoder_in/univariate/all_items_long_only/ \
        male_only:${ENC_M}/encoder_out/univariate/all_items_short_only/ \
        male_only:${ENC_M}/decoder_in/univariate/all_items_short_only/ \
        male_only:${ENC_M}/encoder_out/univariate/all_items_long_only/ \
        male_only:${ENC_M}/decoder_in/univariate/all_items_long_only/ \
        male_only:${DAC_M}/encoder_out/univariate/all_items_short_only/ \
        male_only:${DAC_M}/decoder_in/univariate/all_items_short_only/ \
        male_only:${DAC_M}/encoder_out/univariate/all_items_long_only/ \
        male_only:${DAC_M}/decoder_in/univariate/all_items_long_only/ \
    --output reproduce/figures/audio/encoding/unit_inspection/ \
    --top-k 20 --min-fi-for-dominance 0.001 --min-dominance-margin 0.003 --overwrite
```

Unit-level inspection focuses on `all_items` views. `words_only`/frequency analyses
remain available in per-run summaries and `diagnostic_plots/`.

### Step 11 — Final assembly

Copies the 8 canonical figures and writes `report.md`. Does NOT pull from
`diagnostic_plots/` or `plots_aggregate_legacy/`.

```bash
python scripts/audio/build_final_encoding_report.py \
    --length-controlled-summary reproduce/figures/audio/encoding/length_controlled_summary/ \
    --unit-inspection reproduce/figures/audio/encoding/unit_inspection/ \
    --output reproduce/figures/audio/encoding/final/ \
    --overwrite
```

### Output directory structure (finalized)

```
reproduce/figures/audio/encoding/

  {run_id}/
    {layer}/univariate/{analysis_view}/    ← raw per-view outputs
    univariate_summary/
      diagnostic_plots/                    ← per-view diagnostic PNGs (NOT canonical)
      plots_aggregate_legacy/              ← OLD pre-analysis_view plots (ignore)

  length_controlled_summary/               ← cross-view comparison (Step 9)
  unit_inspection/                         ← unit-level analysis (Step 10)
  final/                                   ← 8 canonical figures + report.md (Step 11)
```

> **diagnostic_plots/ are not canonical.** They are per-view diagnostic figures
> useful for checking individual analysis_views but should not be cited as final results.
>
> **Old `plots/` folders are legacy.** Any `univariate_summary/plots/` directory
> predating chantier 2.5 contains aggregate figures that mix analysis_views (e.g.
> short_only and all_speakers plotted together). Rename to `plots_aggregate_legacy/`
> if not already done; do not use for scientific claims.

### Canonical final figures

| # | Filename | Scientific message |
|---|---|---|
| 01 | `01_fi_global_heatmap.png` | Full picture: length dominates; morphology residual after length control |
| 02 | `02_male_only_short_long_fi_summary.png` | Male-only baseline: short_only vs long_only contrast |
| 03 | `03_short_only_morphology_vs_lexicality.png` | After length control: morphology > lexicality |
| 04 | `04_long_only_morphology_vs_lexicality.png` | Long-only: weaker/mixed; small DAC lex signal |
| 05 | `05_speaker_effect_summary.png` | Speaker identity is a strong acoustic axis |
| 06 | `06_temporal_fi_short_only.png` | Temporal profile of residual factors in short_only |
| 07 | `07_unit_dominance_counts.png` | Fraction of units per view dominated by each feature |
| 08 | `08_lexicality_concentration.png` | Lexicality is sparse/localized at unit level |

---

## Legacy Steps 9–10 (pre-chantier 2.5)

> **These scripts are LEGACY and not analysis_view-safe.** They group only by
> `analysis_set` and will silently mix `short_only`, `long_only`, and `all_speakers`
> views into the same rows. Use Steps 9–11 above for all current analyses.

`compare_univariate_codecs.py` — cross-codec comparison (now superseded by Step 9).
`build_unit_profiles.py` — unit FI profiles (now superseded by Step 10).

---

## Quick reference

```bash
# Steps 0–5: setup → extract → signal analysis → PCA → xarray
python scripts/audio/setup_paradigm.py \
    --metadata data/external/paradigm/raw/metadata.csv \
    --wav-dir   data/external/paradigm/raw/wav \
    --filename-col Audio_Male \
    --output    data/external/paradigm/processed/subset_male.csv

python scripts/audio/sanity_check.py --model encodec

python scripts/audio/extract.py \
    --model encodec --model-arg bandwidth=6.0 \
    --dataset data/external/paradigm/processed/subset_male.csv \
    --layers encoder_out decoder_in \
    --output reproduce/data/audio/

python reproduce/scripts/audio/analysis.py \
    --results reproduce/data/audio/encodec__7f7d3b97/metrics.csv

python reproduce/scripts/audio/activation_analysis.py \
    --run reproduce/data/audio/encodec__7f7d3b97/

python reproduce/scripts/audio/pca_analysis.py \
    --run reproduce/data/audio/encodec__7f7d3b97/ --mode trajectory

python scripts/audio/build_xarray.py \
    --run reproduce/data/audio/encodec__b429aeea/ \
    --layers encoder_out decoder_in

# Steps 6–11: finalized univariate encoding pipeline (all-speakers run)
ENC=reproduce/figures/audio/encoding/encodec__b429aeea
DAC=reproduce/figures/audio/encoding/dac__d466515b
ENC_M=reproduce/figures/audio/encoding/encodec__7f7d3b97
DAC_M=reproduce/figures/audio/encoding/dac__f104bbdd

# Step 6: per-analysis_view Ridge regression (example: all_items_short_only)
python scripts/audio/run_univariate_encoding.py \
    --xarray reproduce/data/audio/encodec__b429aeea/xarray/encoder_out.nc \
    --analysis-set all_items --trial-filter length_bin=short \
    --n-bins 10 --metrics r2 spearman --compute-fi --n-jobs -1 \
    --output ${ENC}/encoder_out/univariate/all_items_short_only/

# Step 7: summaries — encoder_out AND decoder_in in ONE call
python scripts/audio/summarize_univariate_encoding.py \
    --inputs \
        ${ENC}/encoder_out/univariate/all_items_all_speakers/ \
        ${ENC}/encoder_out/univariate/all_items_short_only/ \
        ${ENC}/encoder_out/univariate/all_items_long_only/ \
        ${ENC}/encoder_out/univariate/words_only_all_speakers/ \
        ${ENC}/decoder_in/univariate/all_items_all_speakers/ \
        ${ENC}/decoder_in/univariate/all_items_short_only/ \
        ${ENC}/decoder_in/univariate/all_items_long_only/ \
        ${ENC}/decoder_in/univariate/words_only_all_speakers/ \
    --output ${ENC}/univariate_summary/ --overwrite

# Step 8: per-view diagnostic plots (NOT canonical; output to diagnostic_plots/)
python scripts/audio/plot_univariate_encoding_summary.py \
    --summary-dir ${ENC}/univariate_summary/ \
    --output      ${ENC}/univariate_summary/diagnostic_plots/ --overwrite

# Step 9: cross-view canonical summary
python scripts/audio/summarize_length_controlled_encoding.py \
    --summaries \
        all_speakers:${ENC}/univariate_summary/ \
        all_speakers:${DAC}/univariate_summary/ \
        male_only:${ENC_M}/univariate_summary_length_controlled/ \
        male_only:${DAC_M}/univariate_summary_length_controlled/ \
    --output reproduce/figures/audio/encoding/length_controlled_summary/ --overwrite

# Step 10: unit inspection (both layers, EnCodec + DAC only)
python scripts/audio/inspect_encoding_units.py \
    --fi-dirs \
        all_speakers:${ENC}/encoder_out/univariate/all_items_all_speakers/ \
        all_speakers:${ENC}/encoder_out/univariate/all_items_short_only/ \
        all_speakers:${ENC}/encoder_out/univariate/all_items_long_only/ \
        all_speakers:${ENC}/decoder_in/univariate/all_items_all_speakers/ \
        all_speakers:${ENC}/decoder_in/univariate/all_items_short_only/ \
        all_speakers:${ENC}/decoder_in/univariate/all_items_long_only/ \
        all_speakers:${DAC}/encoder_out/univariate/all_items_all_speakers/ \
        all_speakers:${DAC}/encoder_out/univariate/all_items_short_only/ \
        all_speakers:${DAC}/encoder_out/univariate/all_items_long_only/ \
        all_speakers:${DAC}/decoder_in/univariate/all_items_all_speakers/ \
        all_speakers:${DAC}/decoder_in/univariate/all_items_short_only/ \
        all_speakers:${DAC}/decoder_in/univariate/all_items_long_only/ \
        male_only:${ENC_M}/encoder_out/univariate/all_items_short_only/ \
        male_only:${ENC_M}/decoder_in/univariate/all_items_short_only/ \
        male_only:${ENC_M}/encoder_out/univariate/all_items_long_only/ \
        male_only:${ENC_M}/decoder_in/univariate/all_items_long_only/ \
        male_only:${DAC_M}/encoder_out/univariate/all_items_short_only/ \
        male_only:${DAC_M}/decoder_in/univariate/all_items_short_only/ \
        male_only:${DAC_M}/encoder_out/univariate/all_items_long_only/ \
        male_only:${DAC_M}/decoder_in/univariate/all_items_long_only/ \
    --output reproduce/figures/audio/encoding/unit_inspection/ \
    --top-k 20 --min-fi-for-dominance 0.001 --min-dominance-margin 0.003 --overwrite

# Step 11: final assembly
python scripts/audio/build_final_encoding_report.py \
    --length-controlled-summary reproduce/figures/audio/encoding/length_controlled_summary/ \
    --unit-inspection reproduce/figures/audio/encoding/unit_inspection/ \
    --output reproduce/figures/audio/encoding/final/ --overwrite
```