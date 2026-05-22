#!/usr/bin/env python3
"""Build canonical xarray .nc files from audio model activation runs.

Converts per-item per-layer .pt activation tensors from an extraction run
into xarray.Dataset files with dimensions (trials, time, neurons).

Any latent [D, T] layer is accepted. 'decoder_out' (waveform) is rejected.
If --layers is not specified, all eligible layers from the run manifest are used.

Usage
-----
# Codec runs (explicit layers)
python scripts/audio/build_xarray.py \\
    --run reproduce/data/audio/encodec__7f7d3b97/ \\
    --layers encoder_out decoder_in

# AuriStream run (explicit layers)
python scripts/audio/build_xarray.py \\
    --run reproduce/data/audio/auristream__6ee9aeb6/ \\
    --layers embedding block_01 block_24 block_48_lnf

# Any run — all eligible layers from manifest (no --layers needed)
python scripts/audio/build_xarray.py \\
    --run reproduce/data/audio/auristream__6ee9aeb6/ \\
    --overwrite

Outputs (written to {run}/xarray/ by default)
---------------------------------------------
    {layer}.nc          xarray.Dataset with dims (trials, time, neurons)
    {layer}_qc.json     QC report
    metadata_trials.csv trial-level metadata, ordered as in manifest.items
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
    WAVEFORM_LAYERS,
    export_layer,
    load_manifest,
    load_trial_metadata,
)


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
        default=None,
        metavar="LAYER",
        help=(
            "Layers to export. If not specified, all eligible layers from the "
            "run manifest are used (excluding waveform outputs such as 'decoder_out'). "
            "Example: --layers encoder_out decoder_in  or  --layers block_24 block_48_lnf"
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

    manifest = load_manifest(run_dir)
    run_id = manifest["run_id"]
    n_items = len(manifest["items"])

    # Determine layers to export
    if args.layers is not None:
        layers = args.layers
    else:
        # Infer from manifest: use all extracted layers, excluding waveform outputs
        available = manifest["run_params"].get("layers", [])
        layers = [la for la in available if la not in WAVEFORM_LAYERS]
        if not layers:
            print(
                "[build_xarray] ERROR: No eligible layers found in manifest. "
                "Pass --layers explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[build_xarray] --layers not specified; using layers from manifest: {layers}")

    # Reject waveform layers early with a clear message
    for layer in layers:
        if layer in WAVEFORM_LAYERS:
            print(
                f"[build_xarray] ERROR: Layer '{layer}' is a reconstructed waveform "
                "and cannot be exported as a latent representation. "
                "Remove it from --layers.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"[build_xarray] Run dir  : {run_dir}")
    print(f"[build_xarray] Run ID   : {run_id}")
    print(f"[build_xarray] Items    : {n_items}")
    print(f"[build_xarray] Layers   : {layers}")
    print(f"[build_xarray] Output   : {output_dir}")
    print(f"[build_xarray] Overwrite: {args.overwrite}")
    print()

    qc_reports: dict[str, dict] = {}

    for layer in layers:
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