# Audio Codec Benchmark — `feat/paradigm-audio-codecs`

End-to-end pipeline for running neural audio codecs on the lab paradigm
stimulus set, extracting encoder/decoder activations, computing signal
metrics, and running SWP-style statistical analyses.

All commands are run from the **`swp-model/` repo root**.

---

## Overview

The pipeline has five sequential steps:

| Step | Script | What it does |
|------|--------|-------------|
| 0 | `scripts/audio/setup_paradigm.py` | Build normalized CSV from raw metadata + WAV files |
| Sanity | `scripts/audio/sanity_check.py` | Smoke-test a model before committing to a full run |
| 1 | `scripts/audio/extract.py` | Reconstruction metrics + activation extraction |
| 2 | `reproduce/scripts/audio/analysis.py` | Signal metric analysis (mel distance, SI-SDR) |
| 3 | `reproduce/scripts/audio/activation_analysis.py` | Latent activation analysis (norms, trajectories) |

---

## Data: the paradigm stimulus set

### What it is

English word repetition stimuli from the lab paradigm (`en_paradigm.csv`).
Items are single words or pseudowords recorded by a single speaker.

Each item is characterized by four factors:

| Factor | Values | Applies to |
|--------|--------|-----------|
| `lexicality` | `word` / `nonword` | all items |
| `length_bin` | `long` / `short` | all items |
| `frequency_bin` | `high` / `low` | words only (NA for nonwords) |
| `morphology` | `complex` / `simple` | all items |

Condition codes in the raw CSV encode these factors:
- Words (4-char): `R` + length (`L`/`S`) + frequency (`H`/`L`) + morphology (`C`/`S`)
  — e.g. `RLLC` = word, long, low-freq, complex
- Pseudowords (3-char): `P` + length (`L`/`S`) + morphology (`C`/`S`)
  — e.g. `PLC` = nonword, long, complex

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

---

## Package structure

```
swp/audio/
├── models/
│   ├── base.py          # AudioModel Protocol + ReconstructionResult TypedDict
│   ├── registry.py      # @register / get_model() lazy registry
│   ├── device.py        # select_device() — CUDA > MPS > CPU fallback
│   ├── encodec.py       # EnCodecModel wrapper (HuggingFace)
│   └── dac.py           # DACModel wrapper (descript-audio-codec)
├── hooks/
│   └── manager.py       # HookManager — forward hooks + pre-hooks + extractor_fn
├── datasets/
│   ├── base.py          # ParadigmDataset — reads CSV, loads WAV at target_sr
│   └── utils.py         # load_audio() helper
├── metrics/
│   └── signal.py        # mel_distance(), si_sdr(), compute_all()
└── pipeline/
    └── extraction.py    # run_extraction() — ties everything together

scripts/audio/
├── setup_paradigm.py    # Step 0: build processed CSV from raw data
├── sanity_check.py      # Smoke test for any registered model
└── extract.py           # Step 1 CLI

reproduce/scripts/audio/
├── analysis.py          # Step 2: signal metric OLS + bar plots
└── activation_analysis.py  # Step 3: activation feature OLS + trajectory plots
```

---

## Quick reference

```bash
# Full pipeline — DAC 24 kHz
python scripts/audio/setup_paradigm.py \
    --metadata data/external/paradigm/raw/metadata.csv \
    --wav-dir   data/external/paradigm/raw/wav \
    --filename-col Audio_Male \
    --output    data/external/paradigm/processed/subset_male.csv

python scripts/audio/sanity_check.py --model dac --model-type 24khz

python scripts/audio/extract.py \
    --model dac --model-arg model_type=24khz \
    --dataset data/external/paradigm/processed/subset_male.csv \
    --layers encoder_out decoder_in decoder_out \
    --output reproduce/data/audio/

# Replace dac__HASH with the actual run_id printed at the end of extract.py
python reproduce/scripts/audio/analysis.py \
    --results reproduce/data/audio/dac__HASH/metrics.csv

python reproduce/scripts/audio/activation_analysis.py \
    --run reproduce/data/audio/dac__HASH/
```