#!/usr/bin/env python3
"""Convert MFA TextGrid outputs to a canonical phoneme boundary CSV.

Searches a directory of TextGrid files recursively, maps each file back to an
item_id by exact stem match, parses the phone tier, and writes a flat CSV
compatible with scripts/audio/build_phoneme_embeddings.py.

Run from swp-model/ repo root:

    python scripts/audio/textgrid_to_phoneme_boundaries.py \\
        --dataset    data/external/paradigm/processed/subset_male.csv \\
        --textgrid-dir data/external/paradigm/mfa/subset_male_aligned

    # Custom output path:
    python scripts/audio/textgrid_to_phoneme_boundaries.py \\
        --dataset    data/external/paradigm/processed/subset_male.csv \\
        --textgrid-dir data/external/paradigm/mfa/subset_male_aligned \\
        --output   data/external/paradigm/processed/phoneme_boundaries_mfa_subset_male.csv

Output columns:
    item_id, phoneme, phoneme_index, start_s, end_s, duration_s,
    word, tier, textgrid_path

The output is sorted by (item_id, phoneme_index) and is ready to be
consumed by scripts/audio/build_phoneme_embeddings.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SPEAKER_PAT = re.compile(r"_(?:MALE|FEMALE)", re.IGNORECASE)

# Phone tier names accepted (case-insensitive search)
_PHONE_TIER_NAMES: frozenset[str] = frozenset({"phones", "phone", "phoneme", "phonemes"})


def _infer_word(item_id: str) -> str:
    parts = _SPEAKER_PAT.split(item_id, maxsplit=1)
    return parts[0]


# ── TextGrid parsing ──────────────────────────────────────────────────────────


def _parse_textgrid(path: Path) -> tuple[list[dict], str]:
    """Parse phone intervals from a MFA long-form TextGrid file.

    Returns:
        intervals: list of {'phoneme': str, 'start_s': float, 'end_s': float}
        tier_name: the tier name that was used

    Tries praatio first (if installed), then falls back to an inline parser
    for MFA long-form TextGrid output.
    """
    # Try praatio (installed by MFA as a dependency)
    try:
        from praatio import textgrid as tgio  # type: ignore
        tg = tgio.openTextgrid(str(path), includeEmptyIntervals=True)
        tier_name_lc = {t.lower(): t for t in tg.tierNames}
        for name in _PHONE_TIER_NAMES:
            if name in tier_name_lc:
                canonical = tier_name_lc[name]
                tier = tg.getTier(canonical)
                intervals = [
                    {"phoneme": e.label, "start_s": e.start, "end_s": e.end}
                    for e in tier.entries
                    if e.label  # skip empty-label intervals
                ]
                return intervals, canonical
        raise ValueError(
            f"No phone tier found in {path}.\n"
            f"Available tiers: {tg.tierNames}\n"
            f"Expected one of: {sorted(_PHONE_TIER_NAMES)}"
        )
    except ImportError:
        pass

    # Inline fallback for MFA long-form TextGrid
    return _parse_textgrid_inline(path)


def _parse_textgrid_inline(path: Path) -> tuple[list[dict], str]:
    """Minimal parser for MFA long-form TextGrid files.

    Handles the standard output produced by MFA (UTF-8-sig encoded,
    long format with indented `item`, `intervals` blocks).
    """
    text = path.read_text(encoding="utf-8-sig")

    # Split into tier blocks
    tier_blocks = re.split(r"\n\s{4}item \[\d+\]:", text)
    for block in tier_blocks[1:]:
        name_m = re.search(r'\bname = "([^"]*)"', block)
        if not name_m:
            continue
        tier_name = name_m.group(1).strip()
        if tier_name.lower() not in _PHONE_TIER_NAMES:
            continue
        if "IntervalTier" not in block:
            continue

        intervals: list[dict] = []
        for ib in re.split(r"\n\s+intervals \[\d+\]:", block)[1:]:
            xmin_m = re.search(r"xmin = ([\d.eE+\-]+)", ib)
            xmax_m = re.search(r"xmax = ([\d.eE+\-]+)", ib)
            text_m = re.search(r'text = "([^"]*)"', ib)
            if xmin_m and xmax_m and text_m:
                label = text_m.group(1).strip()
                if label:  # skip empty-label intervals
                    intervals.append({
                        "phoneme": label,
                        "start_s": float(xmin_m.group(1)),
                        "end_s": float(xmax_m.group(1)),
                    })
        return intervals, tier_name

    # Report available tier names for diagnosis
    available = re.findall(r'\bname = "([^"]*)"', text)
    raise ValueError(
        f"No phone tier found in {path}.\n"
        f"Available tiers: {available}\n"
        f"Expected one of: {sorted(_PHONE_TIER_NAMES)}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MFA TextGrid outputs to a canonical phoneme boundary CSV."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Processed paradigm CSV (used to validate item_ids and infer words).",
    )
    parser.add_argument(
        "--textgrid-dir", required=True, dest="textgrid_dir",
        help="Directory containing MFA-aligned TextGrid files (searched recursively).",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output CSV path. "
            "Default: data/external/paradigm/processed/"
            "phoneme_boundaries_mfa_{dataset_name}.csv"
        ),
    )
    parser.add_argument(
        "--transcript-col", default=None, dest="transcript_col",
        help=(
            "CSV column to use as word/transcript. "
            "Default: infer from item_id by stripping speaker suffix."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing output CSV.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    textgrid_dir = Path(args.textgrid_dir)

    if not dataset_path.exists():
        sys.exit(f"Dataset CSV not found: {dataset_path}")
    if not textgrid_dir.exists():
        sys.exit(f"TextGrid directory not found: {textgrid_dir}")

    dataset_name = dataset_path.stem
    default_output = (
        REPO_ROOT
        / "data" / "external" / "paradigm" / "processed"
        / f"phoneme_boundaries_mfa_{dataset_name}.csv"
    )
    output_path = Path(args.output) if args.output else default_output

    if output_path.exists() and not args.overwrite:
        sys.exit(
            f"Output CSV already exists: {output_path}\n"
            "Pass --overwrite to overwrite."
        )

    # Load dataset
    df = pd.read_csv(dataset_path)
    if "item_id" not in df.columns:
        sys.exit("Dataset CSV must have an 'item_id' column.")
    dataset_item_ids: set[str] = set(df["item_id"].astype(str))

    # Validate --transcript-col
    if args.transcript_col and args.transcript_col not in df.columns:
        sys.exit(
            f"--transcript-col '{args.transcript_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    transcript_map: dict[str, str] = {}
    if args.transcript_col:
        transcript_map = df.set_index("item_id")[args.transcript_col].astype(str).to_dict()

    # Discover TextGrid files
    tg_files = sorted(textgrid_dir.rglob("*.TextGrid"))
    if not tg_files:
        sys.exit(f"No .TextGrid files found under: {textgrid_dir}")

    # Build stem → path map (strict: error on duplicate stems)
    stem_to_path: dict[str, Path] = {}
    for tg in tg_files:
        stem = tg.stem
        if stem in stem_to_path:
            sys.exit(
                f"Duplicate TextGrid stem '{stem}':\n"
                f"  {stem_to_path[stem]}\n  {tg}\n"
                "Ensure each item_id appears at most once."
            )
        stem_to_path[stem] = tg

    # Validate coverage
    tg_item_ids = set(stem_to_path.keys())
    unknown = tg_item_ids - dataset_item_ids
    if unknown:
        sys.exit(
            f"{len(unknown)} TextGrid stem(s) do not match any item_id in the dataset:\n"
            + ", ".join(sorted(unknown))
        )
    missing = dataset_item_ids - tg_item_ids
    if missing:
        print(
            f"  Warning: {len(missing)} dataset item(s) have no TextGrid — "
            "they will be absent from the output CSV:\n  "
            + ", ".join(sorted(missing)[:10])
            + (" …" if len(missing) > 10 else "")
        )

    print(f"Dataset      : {dataset_path}  ({len(dataset_item_ids)} items)")
    print(f"TextGrids    : {len(tg_item_ids)} found in {textgrid_dir}")
    print(f"Output       : {output_path}")

    # Parse TextGrids
    rows: list[dict] = []
    errors: list[str] = []

    for item_id in sorted(tg_item_ids):
        tg_path = stem_to_path[item_id]
        word = (
            transcript_map.get(item_id, _infer_word(item_id))
            if not args.transcript_col
            else transcript_map.get(item_id, _infer_word(item_id))
        )

        try:
            intervals, tier_name = _parse_textgrid(tg_path)
        except Exception as exc:
            errors.append(f"  {item_id}: {exc}")
            continue

        for ph_idx, iv in enumerate(intervals):
            rows.append({
                "item_id": item_id,
                "phoneme": iv["phoneme"],
                "phoneme_index": ph_idx,
                "start_s": round(iv["start_s"], 6),
                "end_s": round(iv["end_s"], 6),
                "duration_s": round(iv["end_s"] - iv["start_s"], 6),
                "word": word,
                "tier": tier_name,
                "textgrid_path": str(tg_path.resolve()),
            })

    if errors:
        print(f"\nParse errors ({len(errors)}):")
        for e in errors:
            print(e)
        sys.exit("Aborting due to parse errors above.")

    if not rows:
        sys.exit("No phone intervals were extracted. Check TextGrid format.")

    out_df = (
        pd.DataFrame(rows)
        .sort_values(["item_id", "phoneme_index"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    n_items = out_df["item_id"].nunique()
    n_phones = len(out_df)
    print(f"\nDone: {output_path}")
    print(f"  Items   : {n_items}")
    print(f"  Phonemes: {n_phones}  (~{n_phones / n_items:.1f} per item)")


if __name__ == "__main__":
    main()
