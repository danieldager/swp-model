#!/usr/bin/env python3
"""Build a normalized processed/subset.csv from raw paradigm metadata and WAV files.

Verified condition code convention for en_paradigm.csv
-------------------------------------------------------
Real words start with R (4 chars):
  Position 0  R          → lexicality    word
  Position 1  L / S      → length_bin    long / short
  Position 2  H / L      → frequency_bin high / low
  Position 3  C / S      → morphology    complex / simple

  Examples:  RLLC = word, long,  low-freq,  complex
             RSHS = word, short, high-freq, simple
             RLHC = word, long,  high-freq, complex

Pseudowords start with P (3 chars):
  Position 0  P          → lexicality    nonword
  Position 1  L / S      → length_bin    long / short
  Position 2  C / S      → morphology    complex / simple
  frequency_bin          → always NA

  Examples:  PLC = nonword, long,  complex
             PLS = nonword, long,  simple
             PSC = nonword, short, complex

WAV filenames come from a dedicated column in the raw metadata (e.g. Audio_Male
or Audio_Female). The filename stem is used as item_id and for WAV file lookup.
The speaker label is inferred from the column name when not given explicitly.

Usage (run from swp-model/ repo root):
    # Dry-run — verify decoding before generating output:
    python scripts/audio/setup_paradigm.py \\
        --metadata data/external/paradigm/raw/metadata.csv \\
        --wav-dir   data/external/paradigm/raw/wav \\
        --filename-col Audio_Male \\
        --dry-run

    # Generate processed CSV for the male speaker:
    python scripts/audio/setup_paradigm.py \\
        --metadata data/external/paradigm/raw/metadata.csv \\
        --wav-dir   data/external/paradigm/raw/wav \\
        --filename-col Audio_Male \\
        --output    data/external/paradigm/processed/subset_male.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torchaudio


# ── Condition-code decoding (en_paradigm.csv convention) ─────────────────────

_COND_LENGTH = {"L": "long", "S": "short"}
_COND_MORPHOLOGY = {"C": "complex", "S": "simple"}
_COND_FREQUENCY = {"H": "high", "L": "low"}

_LEX_NORMALISE = {
    "real": "word",
    "word": "word",
    "nonword": "nonword",
    "pseudo": "nonword",
    "pseudoword": "nonword",
}


def decode_condition(cond: str) -> dict[str, str | None]:
    """Decode a condition code into factor values.

    Real-word codes (R + 3 chars):
      pos 0 = R (lexicality)
      pos 1 = L/S (length_bin)
      pos 2 = H/L (frequency_bin)
      pos 3 = C/S (morphology)

    Pseudoword codes (P + 2 chars):
      pos 0 = P (lexicality)
      pos 1 = L/S (length_bin)
      pos 2 = C/S (morphology)
      frequency_bin → None

    Returns dict with: lexicality, length_bin, frequency_bin, morphology
    """
    cond = cond.strip()
    lex_code = cond[0].upper() if cond else ""

    if lex_code == "R":
        if len(cond) < 4:
            raise ValueError(
                f"Real-word condition code must be 4 chars (e.g. RLHC), got {cond!r}"
            )
        length = _COND_LENGTH.get(cond[1].upper())
        freq = _COND_FREQUENCY.get(cond[2].upper())
        morph = _COND_MORPHOLOGY.get(cond[3].upper())
        if length is None:
            raise ValueError(f"Unknown length code {cond[1]!r} in condition {cond!r}")
        if freq is None:
            raise ValueError(f"Unknown frequency code {cond[2]!r} in condition {cond!r}")
        if morph is None:
            raise ValueError(f"Unknown morphology code {cond[3]!r} in condition {cond!r}")
        return {
            "lexicality": "word",
            "length_bin": length,
            "frequency_bin": freq,
            "morphology": morph,
        }

    elif lex_code == "P":
        if len(cond) < 3:
            raise ValueError(
                f"Pseudoword condition code must be 3 chars (e.g. PLC), got {cond!r}"
            )
        length = _COND_LENGTH.get(cond[1].upper())
        morph = _COND_MORPHOLOGY.get(cond[2].upper())
        if length is None:
            raise ValueError(f"Unknown length code {cond[1]!r} in condition {cond!r}")
        if morph is None:
            raise ValueError(f"Unknown morphology code {cond[2]!r} in condition {cond!r}")
        return {
            "lexicality": "nonword",
            "length_bin": length,
            "frequency_bin": None,
            "morphology": morph,
        }

    else:
        raise ValueError(
            f"Unknown lexicality prefix {lex_code!r} in condition {cond!r}. "
            "Expected R (real word) or P (pseudoword)."
        )


# ── Column detection helpers ─────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def _require_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    col = _find_col(df, candidates)
    if col is None:
        raise ValueError(
            f"Cannot find required column {label!r}. "
            f"Expected one of {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return col


def _speaker_from_col_name(col_name: str) -> str:
    """Infer a speaker label from the filename column name.

    'Audio_Male'   → 'MALE'
    'Audio_Female' → 'FEMALE'
    Otherwise      → col_name (unchanged)
    """
    lower = col_name.lower()
    if "female" in lower:
        return "FEMALE"
    if "male" in lower:
        return "MALE"
    return col_name


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _audio_info(path: Path) -> tuple[int, float]:
    try:
        info = torchaudio.info(path)
        return info.sample_rate, info.num_frames / info.sample_rate
    except Exception:
        waveform, sr = torchaudio.load(path)
        return sr, waveform.shape[-1] / sr


# ── Main builder ─────────────────────────────────────────────────────────────

def build_subset(
    metadata_path: Path,
    wav_dir: Path,
    output_path: Path,
    word_col: str,
    condition_col: str,
    filename_col: str | None,
    speaker_col: str | None,
    speaker_default: str,
    dry_run: bool,
    repo_root: Path,
) -> None:
    df_raw = pd.read_csv(metadata_path)
    print(f"Loaded {len(df_raw)} rows from {metadata_path}")
    print(f"Raw columns: {list(df_raw.columns)}")

    # ── Required columns ─────────────────────────────────────────────────────
    word_c = _require_col(df_raw, [word_col, "Word", "word", "item", "Item"], "word")
    cond_c = _require_col(
        df_raw, [condition_col, "Condition", "condition", "CONDITION"], "condition"
    )

    # Filename column — required for WAV linking when present, else fall back to word
    fn_c = (
        _find_col(df_raw, [filename_col]) if filename_col
        else _find_col(df_raw, ["Audio_Male", "Audio_Female", "filename", "Filename"])
    )
    if fn_c:
        print(f"WAV filename column: {fn_c!r}")
    else:
        print(f"No filename column found; using {word_c!r} as WAV filename stem.")

    # Speaker: explicit column > infer from filename column name > default
    spk_c = (
        _find_col(df_raw, [speaker_col]) if speaker_col
        else _find_col(df_raw, ["Speaker", "speaker"])
    )
    if spk_c:
        speaker_label = None  # will be read per row
        print(f"Speaker column: {spk_c!r}")
    elif fn_c:
        speaker_label = _speaker_from_col_name(fn_c)
        print(f"Speaker inferred from filename column {fn_c!r}: {speaker_label!r}")
    else:
        speaker_label = speaker_default
        print(f"Speaker: {speaker_label!r} (default)")

    # Optional pre-decoded factor columns (per-factor)
    lex_c = _find_col(df_raw, ["Lexicality", "lexicality"])
    size_c = _find_col(df_raw, ["Size", "size", "length_bin"])
    morph_c = _find_col(df_raw, ["Morphology", "morphology"])
    freq_c = _find_col(df_raw, ["frequency_bin", "FrequencyBin", "Frequency_Bin"])
    zipf_c = _find_col(df_raw, ["Zipf_Frequency", "zipf_frequency", "ZipfFrequency"])
    nphones_c = _find_col(df_raw, ["num_phones", "NumPhones", "Length", "length"])

    print(f"\nPer-factor column resolution:")
    print(f"  lexicality   : {'column ' + repr(lex_c) if lex_c else 'decode from condition'}")
    print(f"  length_bin   : {'column ' + repr(size_c) if size_c else 'decode from condition'}")
    print(f"  morphology   : {'column ' + repr(morph_c) if morph_c else 'decode from condition'}")
    print(f"  frequency_bin: {'column ' + repr(freq_c) if freq_c else 'decode from condition'}")

    # ── Condition decoding validation ─────────────────────────────────────────
    print("\n── Condition decoding sample ────────────────────────────────────────")
    seen: set[str] = set()
    decode_errors: list[str] = []
    for _, raw_row in df_raw.iterrows():
        cond = str(raw_row[cond_c]).strip()
        if cond in seen:
            continue
        seen.add(cond)
        try:
            factors = decode_condition(cond)
            print(
                f"  {cond:<6} → lexicality={factors['lexicality']!r:10} "
                f"length_bin={factors['length_bin']!r:8} "
                f"morphology={factors['morphology']!r:10} "
                f"frequency_bin={factors['frequency_bin']!r}"
            )
        except ValueError as exc:
            decode_errors.append(f"  {cond:<6} → DECODE ERROR: {exc}")
            print(decode_errors[-1])

    print("────────────────────────────────────────────────────────────────────")
    if decode_errors:
        print(f"\nWARNING: {len(decode_errors)} condition(s) could not be decoded.")

    if dry_run:
        print("\nVerify the decoding above before running without --dry-run.")
        print("[dry-run] No output written.")
        return

    # Columns supplying canonical values (skip from extra forwarding)
    canonical_src_cols = {c for c in [
        word_c, cond_c, fn_c, lex_c, size_c, morph_c, freq_c,
        zipf_c, nphones_c, spk_c,
    ] if c is not None}

    # ── Build rows ────────────────────────────────────────────────────────────
    rows: list[dict] = []
    errors: list[str] = []

    for row_idx, raw_row in df_raw.iterrows():
        word = str(raw_row[word_c]).strip()
        condition = str(raw_row[cond_c]).strip()

        # WAV resolution
        filename_stem = str(raw_row[fn_c]).strip() if fn_c else word
        if filename_stem.lower().endswith(".wav"):
            filename_stem = filename_stem[:-4]
        wav_candidates = [
            (wav_dir / f"{filename_stem}.wav").resolve(),
            (wav_dir / f"{filename_stem}.WAV").resolve(),
        ]
        wav_abs: Path | None = next((p for p in wav_candidates if p.exists()), None)

        if wav_abs is None:
            errors.append(f"[row {row_idx}] WAV not found for {filename_stem!r} in {wav_dir}")
            continue

        try:
            original_sr, duration_s = _audio_info(wav_abs)
        except Exception as exc:
            errors.append(f"[row {row_idx}] Cannot read audio info for {wav_abs}: {exc}")
            continue

        # ── Per-factor resolution ─────────────────────────────────────────────
        try:
            # Decode condition first (always needed for pseudoword freq=None logic)
            cond_factors = decode_condition(condition)

            lexicality = (
                _LEX_NORMALISE.get(str(raw_row[lex_c]).strip().lower(), str(raw_row[lex_c]).strip().lower())
                if lex_c else cond_factors["lexicality"]
            )
            length_bin = (
                str(raw_row[size_c]).strip().lower()
                if size_c else cond_factors["length_bin"]
            )
            morphology = (
                str(raw_row[morph_c]).strip().lower()
                if morph_c else cond_factors["morphology"]
            )
            # frequency_bin: always None for nonwords
            if lexicality == "nonword":
                frequency_bin: str | None = None
            elif freq_c:
                raw_freq = str(raw_row[freq_c]).strip().lower()
                frequency_bin = raw_freq if raw_freq not in ("", "nan", "none") else None
            else:
                frequency_bin = cond_factors["frequency_bin"]

        except ValueError as exc:
            errors.append(f"[row {row_idx}] {exc}")
            continue

        # Speaker
        speaker = (
            str(raw_row[spk_c]).strip() if spk_c
            else (speaker_label or speaker_default)
        )

        # item_id = filename stem (unique per WAV file)
        item_id = filename_stem

        # ── Canonical columns ─────────────────────────────────────────────────
        canonical: dict = {
            "item_id": item_id,
            "wav_path": str(wav_abs.relative_to(repo_root)),
            "original_sample_rate": original_sr,
            "duration_s": round(duration_s, 6),
            "speaker": speaker,
            "condition": condition,
            "lexicality": lexicality,
            "length_bin": length_bin,
            "frequency_bin": frequency_bin,
            "morphology": morphology,
        }
        if zipf_c:
            canonical["zipf_frequency"] = raw_row[zipf_c]
        if nphones_c:
            canonical["num_phones"] = raw_row[nphones_c]

        # Forward ALL other original columns unchanged
        extra: dict = {
            col: raw_row[col]
            for col in df_raw.columns
            if col not in canonical_src_cols
        }

        rows.append({**canonical, **extra})

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\nProcessed {len(rows)} items successfully, {len(errors)} errors.")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")

    if rows:
        df_out = pd.DataFrame(rows)
        print(f"\nProcessed CSV schema:")
        print(f"  Rows   : {len(df_out)}")
        print(f"  Columns: {list(df_out.columns)}")
        print(f"\nFactor distribution:")
        for col in ["lexicality", "length_bin", "frequency_bin", "morphology"]:
            if col in df_out.columns:
                counts = df_out[col].value_counts(dropna=False)
                print(f"  {col}: {counts.to_dict()}")
        srs = df_out["original_sample_rate"].unique()
        print(f"  sample_rates : {sorted(srs)}")
        dur = df_out["duration_s"]
        print(
            f"  duration_s   : min={dur.min():.2f}s  "
            f"max={dur.max():.2f}s  mean={dur.mean():.2f}s"
        )

    if not rows:
        print("No rows to write. Aborting.", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nWritten: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized processed/subset.csv from raw paradigm data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Path to raw metadata CSV.",
    )
    parser.add_argument(
        "--wav-dir",
        required=True,
        type=Path,
        help="Directory containing raw WAV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/paradigm/processed/subset.csv"),
        help="Output path for processed subset.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--word-col",
        default="Word",
        help="Column name for the word/item identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--condition-col",
        default="Condition",
        help="Column name for the condition code (default: %(default)s).",
    )
    parser.add_argument(
        "--filename-col",
        default=None,
        help=(
            "Column containing WAV filename stems (e.g. Audio_Male, Audio_Female). "
            "The speaker label is inferred from this name when no --speaker-col is given. "
            "If absent, --word-col is used as the filename stem."
        ),
    )
    parser.add_argument(
        "--speaker-col",
        default=None,
        help="Column name for speaker ID; overrides speaker inference from --filename-col.",
    )
    parser.add_argument(
        "--speaker-default",
        default="unknown",
        help="Fallback speaker label when neither --speaker-col nor --filename-col is given.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print condition decoding table and schema info; do not write any output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_subset(
        metadata_path=args.metadata,
        wav_dir=args.wav_dir,
        output_path=args.output,
        word_col=args.word_col,
        condition_col=args.condition_col,
        filename_col=args.filename_col,
        speaker_col=args.speaker_col,
        speaker_default=args.speaker_default,
        dry_run=args.dry_run,
        repo_root=Path.cwd().resolve(),
    )