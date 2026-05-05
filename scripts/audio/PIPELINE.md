# Audio Encoding Analysis Pipeline — Technical Reference

> **This is a technical companion document**, not the main source of truth.
> The main documentation is:
> - `README_audio.md` — pipeline overview, canonical commands, final figures
> - `docs/audio_project_state.md` — scientific status, findings, caveats

This document provides the step-by-step technical reference for SWP audio
univariate encoding analyses. Run all commands from the `swp-model/` repo root.

---

## Script Classification

### Canonical scripts (use these)

| Script | Role |
|---|---|
| `run_univariate_encoding.py` | Run Ridge regression per analysis_view |
| `summarize_univariate_encoding.py` | Aggregate per-run outputs into summary CSVs |
| `plot_univariate_encoding_summary.py` | Generate per-view diagnostic plots |
| `summarize_length_controlled_encoding.py` | Cross-view compact summary + figures |
| `inspect_encoding_units.py` | Unit-level FI inspection |
| `build_final_encoding_report.py` | Assemble canonical figures + write report.md |

### Legacy scripts (do not use with multi-view summaries)

| Script | Reason |
|---|---|
| `build_unit_profiles.py` | Wraps `unit_profiles.py` which lacks `analysis_view` in `_UNIT_COLS`; silently collapses all views |
| `compare_univariate_codecs.py` | Wraps `univariate_compare.py` which groups only by `analysis_set`; mixes views |

---

## Canonical Execution Order

Variable conventions used throughout this document:

```bash
# DATA_RUN: where raw activations and xarray files are stored
DATA_RUN=reproduce/data/audio/{run_id}        # e.g. reproduce/data/audio/encodec__b429aeea

# FIG_RUN: where encoding figures and summaries are stored  (= FIG_BASE/{run_id})
FIG_RUN=reproduce/figures/audio/encoding/{run_id}   # e.g. reproduce/figures/audio/encoding/encodec__b429aeea

# LAYER: encoder_out or decoder_in
# ANALYSIS_VIEW: e.g. all_items_short_only, all_items_all_speakers

# Shortcuts used in multi-run steps (Steps 3–5):
ENC=reproduce/figures/audio/encoding/encodec__b429aeea   # all-speakers EnCodec
DAC=reproduce/figures/audio/encoding/dac__d466515b       # all-speakers DAC
ENC_M=reproduce/figures/audio/encoding/encodec__7f7d3b97 # male-only EnCodec
DAC_M=reproduce/figures/audio/encoding/dac__f104bbdd     # male-only DAC
```

### Step 0 — Run univariate encoding (skip if outputs already exist)

```bash
# Per analysis_view × layer × run_id:
python scripts/audio/run_univariate_encoding.py \
    --xarray ${DATA_RUN}/xarray/${LAYER}.nc \
    --analysis-set all_items \
    --trial-filter length_bin=short \
    --compute-fi \
    --output ${FIG_RUN}/${LAYER}/univariate/all_items_short_only/ \
    --overwrite
```

### Step 1 — Summarize per-run (both layers in ONE call)

**Important:** Pass encoder_out and decoder_in together. Running per-layer with
`--overwrite` would clobber the first layer's summary.

```bash
python scripts/audio/summarize_univariate_encoding.py \
    --inputs \
        ${FIG_RUN}/encoder_out/univariate/all_items_all_speakers/ \
        ${FIG_RUN}/encoder_out/univariate/all_items_short_only/ \
        ${FIG_RUN}/encoder_out/univariate/all_items_long_only/ \
        ... \
        ${FIG_RUN}/decoder_in/univariate/all_items_all_speakers/ \
        ${FIG_RUN}/decoder_in/univariate/all_items_short_only/ \
        ... \
    --output ${FIG_RUN}/univariate_summary/ \
    --overwrite
```

### Step 2 — Diagnostic plots per view

Output goes to `univariate_summary/diagnostic_plots/` (never to `plots/`).
These figures are diagnostic; canonical figures come from Steps 3–5.

```bash
python scripts/audio/plot_univariate_encoding_summary.py \
    --summary-dir ${FIG_RUN}/univariate_summary/ \
    --output ${FIG_RUN}/univariate_summary/diagnostic_plots/ \
    --fi-dirs ${FIG_RUN}/encoder_out/univariate/all_items_short_only/ ... \
    --overwrite
```

### Step 3 — Cross-view compact summary

Produces `length_controlled_summary/` with all comparison figures including
`long_only_morphology_vs_lexicality.png`, `temporal_fi_long_only.png`, and
`male_only_short_long_fi_summary.png`.

```bash
python scripts/audio/summarize_length_controlled_encoding.py \
    --summaries \
        all_speakers:${ENC}/univariate_summary/ \
        all_speakers:${DAC}/univariate_summary/ \
        male_only:${ENC_M}/univariate_summary_length_controlled/ \
        male_only:${DAC_M}/univariate_summary_length_controlled/ \
    --output reproduce/figures/audio/encoding/length_controlled_summary/ \
    --overwrite
```

### Step 4 — Unit inspection (both layers, encoder + DAC only)

```bash
python scripts/audio/inspect_encoding_units.py \
    --fi-dirs \
        all_speakers:${ENC}/encoder_out/univariate/all_items_short_only/ \
        all_speakers:${ENC}/decoder_in/univariate/all_items_short_only/ \
        ... \
    --output reproduce/figures/audio/encoding/unit_inspection/ \
    --top-k 20 --min-fi-for-dominance 0.001 --min-dominance-margin 0.003 --overwrite
```

### Step 5 — Final assembly

Copies 8 canonical figures to `final/` and writes `report.md`.
Does NOT pull from `diagnostic_plots/` or `plots_aggregate_legacy/`.

```bash
python scripts/audio/build_final_encoding_report.py \
    --length-controlled-summary reproduce/figures/audio/encoding/length_controlled_summary/ \
    --unit-inspection reproduce/figures/audio/encoding/unit_inspection/ \
    --output reproduce/figures/audio/encoding/final/ \
    --overwrite
```

---

## Output Directory Convention

```
{run_id}/
  {layer}/univariate/{analysis_view}/    ← raw per-view outputs
  univariate_summary/                    ← aggregated CSVs
    diagnostic_plots/                    ← per-view diagnostic PNGs (Step 2)
    plots_aggregate_legacy/              ← OLD aggregate plots (pre-analysis_view era)

length_controlled_summary/               ← cross-view comparison figures (Step 3)
unit_inspection/                         ← unit-level inspection outputs (Step 4)
final/                                   ← canonical presentation figures (Step 5)
```

## FI vs Weights

- `global_fi_ranking.csv` and FI plots = **canonical** metric for feature importance.
- `global_feature_ranking.csv` and weight plots = **diagnostic** only.
  Weight magnitude and FI can disagree. Always use FI for scientific claims.

## analysis_set vs analysis_view

- `analysis_set`: feature family (`all_items`, `words_only`)
- `analysis_view`: concrete experimental condition (`all_items_short_only`, etc.)

Never group summaries or plots by `analysis_set` alone when multiple
`analysis_view` values are present. Use `analysis_view` as the primary key.
