"""Plot one genuine and one spoof log-Mel/MFCC example from the feature cache."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.features import configured_cache, ensure_cached, load_cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--output", default="reports/generated/feature_examples.png")
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = Path(args.manifest) if args.manifest else project_path(cfg["manifest"])
    root = Path(args.root) if args.root else project_path(cfg["data_root"])
    df = pd.read_csv(manifest, keep_default_na=False)
    chosen = []
    for label in (0, 1):
        subset = df[df.label.astype(int) == label]
        if subset.empty:
            raise SystemExit(f"No rows found for label {label}.")
        row = subset.iloc[0]
        cache = ensure_cached(str(row.audio_path), root, configured_cache(cfg), cfg)
        chosen.append(("Genuine" if label == 0 else "Spoof", load_cached(cache, "logmel")[0], load_cached(cache, "mfcc_summary")))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, (label, mel, mfcc) in enumerate(chosen):
        axes[0, col].imshow(mel, origin="lower", aspect="auto", cmap="magma")
        axes[0, col].set_title(f"{label}: normalized log-Mel")
        axes[0, col].set_xlabel("Frame")
        axes[0, col].set_ylabel("Mel band")
        n = len(mfcc) // 2
        axes[1, col].bar(np.arange(n), mfcc[:n], yerr=mfcc[n:], capsize=1)
        axes[1, col].set_title(f"{label}: MFCC mean ± std")
        axes[1, col].set_xlabel("Coefficient")
    fig.tight_layout()
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved feature visualization to {output}")


if __name__ == "__main__":
    main()
