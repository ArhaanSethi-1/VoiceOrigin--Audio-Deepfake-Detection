"""Precompute local feature cache for manifest rows; no audio is changed."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import load_config, project_path
from src.features import configured_cache, ensure_cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--limit", type=int, help="Only process the first N rows, useful for a quick smoke run")
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = Path(args.manifest) if args.manifest else project_path(cfg["manifest"])
    root = Path(args.root) if args.root else project_path(cfg["data_root"])
    df = pd.read_csv(manifest, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)
    cache = configured_cache(cfg)
    failures: list[dict[str, str]] = []
    for rel in tqdm(df["audio_path"].astype(str), desc="Extracting features"):
        try:
            ensure_cached(rel, root, cache, cfg)
        except Exception as exc:
            failures.append({"audio_path": rel, "error": str(exc)})
    if failures:
        failure_file = cache / "failures.csv"
        failure_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failure_file, index=False)
        raise SystemExit(f"Failed on {len(failures)} files. Details: {failure_file}")
    print(f"Cached features for {len(df):,} clips in {cache}")
    if args.limit:
        print("This was a limited run; rerun without --limit to process the full manifest.")


if __name__ == "__main__":
    main()
