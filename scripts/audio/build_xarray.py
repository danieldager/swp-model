#!/usr/bin/env python3
"""Build canonical xarray .nc files from audio codec activation runs.

Converts per-item per-layer .pt activation tensors from a Step 1 extraction
run into xarray.Dataset files with dimensions (trials, time, neurons).

Only 'encoder_out' and 'decoder_in' are supported. 'decoder_out' (waveform)
is rejected with a clear error.

Usage
-----
python scripts/audio/build_xarray.py \\
    --run reproduce/data/audio/encodec__7f7d3b97/ \\
    --layers encoder_out decoder_in

python scripts/audio/build_xarray.py \\
    --run reproduce/data/audio/dac__f104bbdd/ \\
    --layers encoder_out decoder_in \\
    --overwrite

Outputs (written to {run}/xarray/ by default)
---------------------------------------------
    encoder_out.nc          xarray.Dataset with dims (trials, time, neurons)
    encoder_out_qc.json     QC report
    decoder_in.nc
    decoder_in_qc.json
    metadata_trials.csv     trial-level metadata, ordered as in manifest.items
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swp.audio.encoding.xarray_builder import (
    LATENT_LAYERS,
    WAVEFORM_LAYER,
    export_layer,
    load_manifest,
    load_trial_metadata,
)

DEFAULT_LAYERS = ("encoder_out", "decoder_in")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical xarray .nc files from a Step 1 audio activation run.\n"
            "Outputs are written to {run}/xarray/ by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run",
        required=True,
        metavar="RUN_DIR",
        help="Path to run directory (e.g. reproduce/data/audio/encodec__7f7d3b97/)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=list(DEFAULT_LAYERS),
        metavar="LAYER",
        help=(
            f"Layers to export (default: {list(DEFAULT_LAYERS)}). "
            f"Eligible: {sorted(LATENT_LAYERS)}. "
            f"'{WAVEFORM_LAYER}' is not supported and will cause an error."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory (default: {run}/xarray/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .nc files (default: refuse if file exists)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Dtype for the padded activations array (default: float32)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        metavar="DIR",
        help="Repo root for resolving dataset path (default: auto-detected)",
    )
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    run_dir = Path(args.run).resolve()
    if not run_dir.exists():
        print(
            f"[build_xarray] ERROR: run directory not found: {run_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else run_dir / "xarray"
    )

    # Validate layer names before doing any work
    for layer in args.layers:
        if layer == WAVEFORM_LAYER:
            print(
                f"[build_xarray] ERROR: Layer '{layer}' is a reconstructed waveform "
                "and is not supported by the xarray export. "
                "Only 'encoder_out' and 'decoder_in' are supported in Phase A.",
                file=sys.stderr,
            )
            sys.exit(1)
        if layer not in LATENT_LAYERS:
            print(
                f"[build_xarray] ERROR: Unknown layer '{layer}'. "
                f"Eligible layers: {sorted(LATENT_LAYERS)}.",
                file=sys.stderr,
            )
            sys.exit(1)

    manifest = load_manifest(run_dir)
    run_id = manifest["run_id"]
    n_items = len(manifest["items"])

    print(f"[build_xarray] Run dir  : {run_dir}")
    print(f"[build_xarray] Run ID   : {run_id}")
    print(f"[build_xarray] Items    : {n_items}")
    print(f"[build_xarray] Layers   : {args.layers}")
    print(f"[build_xarray] Output   : {output_dir}")
    print(f"[build_xarray] Overwrite: {args.overwrite}")
    print()

    qc_reports: dict[str, dict] = {}

    for layer in args.layers:
        print(f"[build_xarray] ── Layer: {layer} ──────────────────────────────")
        ds, qc = export_layer(
            run_dir=run_dir,
            layer=layer,
            repo_root=repo_root,
            output_dir=output_dir,
            dtype=dtype,
            overwrite=args.overwrite,
        )
        qc_reports[layer] = qc
        n_time = ds.sizes["time"]
        print(
            f"[build_xarray] OK  shape=({qc['n_trials_loaded']}, {n_time}, {qc['n_neurons']})"
            f"  padding={qc['padding_fraction']:.1%}"
        )
        print()

    # Write metadata_trials.csv (layer-independent, aligned with trials dimension)
    meta_df = load_trial_metadata(manifest, repo_root)
    meta_path = output_dir / "metadata_trials.csv"
    if not meta_path.exists() or args.overwrite:
        meta_df.to_csv(meta_path, index=False)
        print(f"[build_xarray] Written: {meta_path}")
    else:
        print(f"[build_xarray] Skipped (exists): {meta_path}")

    # QC summary
    print()
    print("[build_xarray] QC summary:")
    all_ok = True
    for layer, qc in qc_reports.items():
        issues = []
        if qc["has_nan_inside_valid_region"]:
            issues.append("NaN inside valid region")
        if qc["has_non_nan_outside_valid_region"]:
            issues.append("non-NaN outside valid region")
        if not qc["metadata_merge_success"]:
            issues.append("metadata merge incomplete")
        if issues:
            all_ok = False
        status = "OK" if not issues else "WARNING: " + " | ".join(issues)
        print(
            f"  {layer}: "
            f"trials={qc['n_trials_loaded']}/{qc['n_trials_expected']}  "
            f"neurons={qc['n_neurons']}  "
            f"frames=[{qc['min_frames']}, {qc['max_frames']}]  "
            f"median={qc['median_frames']:.0f}  "
            f"padding={qc['padding_fraction']:.1%}  "
            f"{status}"
        )

    print()
    if not all_ok:
        print("[build_xarray] WARNING: QC issues detected — review QC reports above.")
    print(f"[build_xarray] All outputs written to {output_dir}")


if __name__ == "__main__":
    main()