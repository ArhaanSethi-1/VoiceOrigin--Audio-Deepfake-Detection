"""Shared manifest and feature-cache access for training/evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import project_path
from src.features import configured_cache, ensure_cached, load_cached


def read_manifest(config: dict, manifest_override: str | None = None, root_override: str | None = None) -> tuple[pd.DataFrame, Path, Path]:
    manifest = Path(manifest_override) if manifest_override else project_path(config["manifest"])
    root = Path(root_override) if root_override else project_path(config["data_root"])
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest missing: {manifest}. Run python -m src.data.prepare_manifest")
    frame = pd.read_csv(manifest, keep_default_na=False)
    needed = {"audio_path", "label", "split"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"Manifest must include {sorted(needed)}")
    return frame, root, configured_cache(config)


def feature_arrays(frame: pd.DataFrame, root: Path, cache_root: Path, config: dict, kind: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    key = "logmel" if kind == "logmel" else "mfcc_summary"
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"Loading {kind}"):
        path = ensure_cached(str(row.audio_path), root, cache_root, config)
        X.append(load_cached(path, key))
        y.append(int(row.label))
    if not X:
        raise ValueError("No rows selected for this split.")
    return np.stack(X), np.asarray(y, dtype=np.int64)
