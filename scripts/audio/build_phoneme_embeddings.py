#!/usr/bin/env python3
"""Build phoneme-level embeddings by mean-pooling AuriStream hidden states.

Reads saved [D, L] activation tensors from a representation extraction run and a
phoneme boundary CSV, then writes per-layer [n_phonemes, D] embedding tensors and
aligned metadata.

Does not load audio or run any model — pure post-processing of saved .pt files.

Run from swp-model/ repo root:

    python scripts/audio/build_phoneme_embeddings.py \\
        --run reproduce/data/audio/auristream__9d3f269f \\
        --boundaries data/external/paradigm/processed/phoneme_boundaries_mfa.csv \\
        --layers embedding block_12 block_24 block_36 block_48

Smoke test (3 items, default center-based selection):

    python scripts/audio/build_phoneme_embeddings.py \\
        --run reproduce/data/audio/auristream__9d3f269f \\
        --boundaries data/external/paradigm/processed/phoneme_boundaries_mfa.csv \\
        --layers embedding block_12 block_48 \\
        --max-items 3

Output (inside run directory by default):
    reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/
        metadata_phonemes.csv
        embeddings_embedding.pt
        embeddings_block_12.pt   ...
        manifest.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

import torch

from swp.audio.phonemes.boundaries import load_boundaries
from swp.audio.phonemes.pooling import build_phoneme_embeddings


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build phoneme-level embeddings by mean-pooling AuriStream hidden states."
    )
    parser.add_argument(
        "--run", required=True,
        help="Path to extraction run directory (e.g. reproduce/data/audio/auristream__9d3f269f)",
    )
    parser.add_argument(
        "--boundaries", required=True,
        help=(
            "Path to phoneme boundary CSV with columns: "
            "item_id, phoneme, phoneme_index, start_s, end_s, duration_s"
        ),
    )
    parser.add_argument(
        "--layers", nargs="+", default=None,
        help="Layers to embed. Default: all layers present in run manifest.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory. Default: {run}/phoneme_embeddings/",
    )
    parser.add_argument(
        "--token-selection",
        choices=["center", "start"],
        default="center",
        dest="token_selection",
        help=(
            "Token assignment rule (default: center). "
            "center: include token i if start_s <= (i+0.5)/token_rate_hz < end_s. "
            "start:  include token i if start_s <= i/token_rate_hz < end_s."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output directory.",
    )
    parser.add_argument(
        "--max-items", dest="max_items", type=int, default=None,
        help="Stop after N items (for smoke tests).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    boundaries_path = Path(args.boundaries)

    if not (run_dir / "manifest.json").exists():
        sys.exit(f"manifest.json not found in: {run_dir}")
    if not boundaries_path.exists():
        sys.exit(f"Boundary CSV not found: {boundaries_path}")

    # Read run manifest
    with open(run_dir / "manifest.json") as f:
        run_manifest = json.load(f)
    available_layers: list[str] = run_manifest["run_params"]["layers"]
    token_rate_hz = float(run_manifest["run_params"]["token_rate_hz"])
    layers = sorted(args.layers) if args.layers is not None else sorted(available_layers)

    # Output directory
    output_dir = Path(args.output) if args.output else run_dir / "phoneme_embeddings"
    if output_dir.exists() and not args.overwrite:
        sys.exit(
            f"Output directory already exists: {output_dir}\n"
            "Pass --overwrite to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run            : {run_dir}")
    print(f"Source run ID  : {run_manifest['run_id']}")
    print(f"Boundaries     : {boundaries_path}")
    print(f"Layers         : {layers}")
    print(f"Token selection: {args.token_selection}")
    if args.max_items:
        print(f"Max items      : {args.max_items}")

    # Load and validate boundary CSV
    boundaries_df = load_boundaries(boundaries_path)
    print(
        f"\nBoundary CSV   : {len(boundaries_df)} rows  "
        f"({boundaries_df['item_id'].nunique()} items, "
        f"{boundaries_df['phoneme'].nunique()} unique phones)"
    )

    # Build embeddings
    embeddings, meta_df = build_phoneme_embeddings(
        run_dir=run_dir,
        boundaries_df=boundaries_df,
        layers=layers,
        token_selection=args.token_selection,
        max_items=args.max_items,
    )

    # Write metadata CSV
    meta_df.to_csv(output_dir / "metadata_phonemes.csv", index=False)

    # Write per-layer embedding tensors
    print(f"\nWriting {len(embeddings)} embedding file(s) to {output_dir} …")
    for layer_name in layers:
        tensor = embeddings[layer_name]
        out_path = output_dir / f"embeddings_{layer_name}.pt"
        torch.save(tensor, out_path)
        n_valid = int(meta_df["valid_embedding"].sum())
        print(f"  {layer_name:<20} shape={list(tensor.shape)}  ({n_valid}/{len(meta_df)} valid)")

    # Summary stats from meta_df
    n_phonemes = len(meta_df)
    n_zero = int((~meta_df["valid_embedding"]).sum())
    n_sil = int(meta_df["is_silence"].sum())

    # Write manifest
    manifest_out: dict = {
        "source_run_id": run_manifest["run_id"],
        "source_run_path": str(run_dir.resolve()),
        "boundaries_path": str(boundaries_path.resolve()),
        "boundaries_sha256": _sha256(boundaries_path),
        "layers": layers,
        "token_rate_hz": token_rate_hz,
        "hop_s": round(1.0 / token_rate_hz, 6),
        "token_selection_rule": args.token_selection,
        "max_items": args.max_items,
        "n_phonemes_total": n_phonemes,
        "n_zero_token_phonemes": n_zero,
        "n_silence_phonemes": n_sil,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)

    print(f"\nDone: {output_dir}")
    print(f"  n_phonemes={n_phonemes}  zero_token={n_zero}  silence={n_sil}")


if __name__ == "__main__":
    main()
