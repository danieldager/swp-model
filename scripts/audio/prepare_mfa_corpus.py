#!/usr/bin/env python3
"""Prepare an MFA corpus directory from a processed paradigm CSV.

Creates a directory of {item_id}.wav + {item_id}.txt pairs ready for
`mfa align`. WAV files are copied or symlinked from their original location.
Transcripts are inferred from the item_id by default (see --transcript-col).

Speaker subfolders (Male/, Female/, …) are created when --speaker-subfolders
is set and the dataset has a `speaker` column. Item IDs remain exact stems.

Run from swp-model/ repo root:

    python scripts/audio/prepare_mfa_corpus.py \\
        --dataset data/external/paradigm/processed/subset_male.csv

    python scripts/audio/prepare_mfa_corpus.py \\
        --dataset data/external/paradigm/processed/subset_female.csv \\
        --output  data/external/paradigm/mfa/subset_female_corpus \\
        --speaker-subfolders

Output:
    data/external/paradigm/mfa/{dataset_name}_corpus/
        {item_id}.wav          (or Male/{item_id}.wav with --speaker-subfolders)
        {item_id}.txt
    data/external/paradigm/mfa/{dataset_name}_corpus/mfa_corpus_manifest.csv
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SPEAKER_PAT = re.compile(r"_(?:MALE|FEMALE)", re.IGNORECASE)


def infer_transcript(item_id: str) -> str:
    """Strip speaker/subject suffix from item_id to recover the word/stimulus.

    Examples:
        infertile_MALE_D  →  infertile
        abosh_MALE_D      →  abosh
        Press_FEMALE_C    →  Press
    """
    parts = _SPEAKER_PAT.split(item_id, maxsplit=1)
    return parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an MFA corpus directory from a processed paradigm CSV."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to processed paradigm CSV (e.g. data/external/paradigm/processed/subset_male.csv)",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output corpus directory. "
            "Default: data/external/paradigm/mfa/{dataset_name}_corpus/"
        ),
    )
    parser.add_argument(
        "--transcript-col", default=None, dest="transcript_col",
        help=(
            "Column in the CSV to use as the transcript. "
            "Default: infer from item_id by stripping speaker suffix."
        ),
    )
    parser.add_argument(
        "--mode", choices=["copy", "symlink"], default="copy",
        help="Whether to copy or symlink WAV files (default: copy).",
    )
    parser.add_argument(
        "--speaker-subfolders", action="store_true", dest="speaker_subfolders",
        help=(
            "Organize files into speaker subfolders (e.g. Male/, Female/) "
            "when the dataset has a `speaker` column."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing corpus directory.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        sys.exit(f"Dataset CSV not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    required = {"item_id", "wav_path"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Dataset CSV is missing required columns: {missing}")

    # Validate uniqueness
    if df["item_id"].duplicated().any():
        sys.exit("Dataset CSV contains duplicate item_id values.")

    dataset_name = dataset_path.stem
    output_dir = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "data" / "external" / "paradigm" / "mfa" / f"{dataset_name}_corpus"
    )

    if output_dir.exists() and not args.overwrite:
        sys.exit(
            f"Output directory already exists: {output_dir}\n"
            "Pass --overwrite to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    has_speaker = "speaker" in df.columns
    use_subfolders = args.speaker_subfolders and has_speaker

    print(f"Dataset      : {dataset_path}  ({len(df)} items)")
    print(f"Dataset name : {dataset_name}")
    print(f"Output dir   : {output_dir}")
    print(f"Mode         : {args.mode}")
    print(f"Subfolders   : {use_subfolders}")

    manifest_rows: list[dict] = []
    errors: list[str] = []

    for _, row in df.iterrows():
        item_id: str = str(row["item_id"])
        wav_rel: str = str(row["wav_path"])

        # Resolve WAV source path
        src = Path(wav_rel) if Path(wav_rel).is_absolute() else REPO_ROOT / wav_rel
        if not src.exists():
            errors.append(f"  WAV not found: {src}  (item_id={item_id})")
            continue

        # Determine transcript
        if args.transcript_col:
            if args.transcript_col not in df.columns:
                sys.exit(
                    f"--transcript-col '{args.transcript_col}' not found in CSV. "
                    f"Available columns: {list(df.columns)}"
                )
            transcript = str(row[args.transcript_col]).strip()
        else:
            transcript = infer_transcript(item_id)

        if not transcript:
            errors.append(f"  Empty transcript for item_id={item_id}")
            continue

        # Determine destination subfolder
        if use_subfolders:
            speaker_raw = str(row["speaker"]).strip()
            subfolder = output_dir / speaker_raw.capitalize()
        else:
            subfolder = output_dir
        subfolder.mkdir(parents=True, exist_ok=True)

        # WAV destination
        dst_wav = subfolder / f"{item_id}.wav"
        if dst_wav.exists() and not args.overwrite:
            pass  # leave existing file; overwrite flag already checked above
        if args.mode == "copy":
            shutil.copy2(src, dst_wav)
        else:
            if dst_wav.is_symlink() or dst_wav.exists():
                dst_wav.unlink()
            dst_wav.symlink_to(src.resolve())

        # TXT transcript
        dst_txt = subfolder / f"{item_id}.txt"
        dst_txt.write_text(transcript + "\n", encoding="utf-8")

        manifest_row: dict = {
            "item_id": item_id,
            "transcript": transcript,
            "source_wav": str(src.resolve()),
            "mfa_wav": str(dst_wav.resolve()),
            "mfa_txt": str(dst_txt.resolve()),
        }
        if has_speaker:
            manifest_row["speaker"] = str(row["speaker"])
        manifest_rows.append(manifest_row)

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(e)
        sys.exit("Aborting due to errors above.")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / "mfa_corpus_manifest.csv", index=False)

    print(f"\nCorpus prepared: {output_dir}")
    print(f"  Items written : {len(manifest_rows)}")
    print(f"  Manifest      : {output_dir / 'mfa_corpus_manifest.csv'}")


if __name__ == "__main__":
    main()
