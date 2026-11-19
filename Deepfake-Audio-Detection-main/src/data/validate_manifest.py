"""Validate manifest paths, labels, split membership, and possible leakage."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.config import load_config, project_path

REQUIRED = {"audio_path", "label", "split", "speaker_id", "source_id", "attack_id"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--config")
    parser.add_argument("--hash-audio", action="store_true", help="Hash files to identify exact duplicates (slower)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = Path(args.manifest) if args.manifest else project_path(cfg["manifest"])
    root = Path(args.root) if args.root else project_path(cfg["data_root"])
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}. Run python -m src.data.prepare_manifest first.")
    df = pd.read_csv(manifest, keep_default_na=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing required columns: {sorted(missing)}")
    errors: list[str] = []
    try:
        labels = set(df["label"].astype(int).unique())
        if not labels.issubset({0, 1}):
            errors.append("label values must be 0 (genuine) or 1 (spoof)")
    except (TypeError, ValueError):
        errors.append("label values must be integer 0 (genuine) or 1 (spoof)")
    if not set(df["split"].unique()).issubset({"train", "validation", "test"}):
        errors.append("split values must be train, validation, or test")
    duplicated = df[df["audio_path"].duplicated(keep=False)]
    if not duplicated.empty:
        errors.append(f"{len(duplicated)} rows have duplicate audio_path values")
    absent = [p for p in df["audio_path"] if not (root / p).is_file()]
    if absent:
        errors.append(f"{len(absent)} audio files are missing; first examples: {absent[:5]}")
    for col in ("speaker_id", "source_id", "attack_id"):
        known = df[df[col] != ""].groupby(col)["split"].nunique()
        overlap = known[known > 1]
        if len(overlap):
            print(f"WARNING: {len(overlap)} {col} values occur in multiple official splits. Preserve the official protocol and disclose this.")
    if args.hash_audio:
        digest_to_splits: dict[str, set[str]] = {}
        for rel, split in zip(df["audio_path"], df["split"]):
            p = root / rel
            if p.is_file():
                hasher = hashlib.sha256()
                with p.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(block)
                digest_to_splits.setdefault(hasher.hexdigest(), set()).add(split)
        cross = sum(len(splits) > 1 for splits in digest_to_splits.values())
        print(f"Exact-content duplicate groups spanning splits: {cross}")
    print("Rows by split/label:")
    print(df.groupby(["split", "label"], dropna=False).size().to_string())
    if errors:
        raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors))
    print("Manifest checks passed. Speaker/source overlap warnings are informational, not silently rewritten.")


if __name__ == "__main__":
    main()
