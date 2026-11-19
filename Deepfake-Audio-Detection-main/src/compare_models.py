"""Generate side-by-side held-out test metrics for both trained models."""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config, project_path
from src.data_utils import feature_arrays, read_manifest
from src.metrics import calculate_metrics
from src.predict import load_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cnn, cnn_meta, device = load_detector("cnn", cfg)
    mfcc, mfcc_meta, _ = load_detector("mfcc", cfg)
    cfg = cnn_meta.get("config", cfg)
    frame, root, cache = read_manifest(cfg, args.manifest, args.root)
    test = frame[frame.split == "test"]
    if test.empty:
        raise SystemExit("The manifest has no test rows.")
    y = test.label.astype(int).to_numpy()

    mfcc_x, _ = feature_arrays(test, root, cache, cfg, "mfcc")
    mfcc_scores = mfcc.predict_proba(mfcc_x)[:, 1]

    from src.train_cnn import CachedDataset

    loader = DataLoader(CachedDataset(test, root, cache, cfg), batch_size=int(cfg["cnn"]["batch_size"]), shuffle=False)
    cnn.eval()
    cnn_scores = []
    with torch.no_grad():
        for features, _ in loader:
            cnn_scores.extend(torch.sigmoid(cnn(features.to(device))).cpu().numpy().tolist())

    results = {
        "mfcc_baseline": calculate_metrics(y, mfcc_scores, float(mfcc_meta["threshold"])),
        "logmel_cnn": calculate_metrics(y, pd.Series(cnn_scores).to_numpy(), float(cnn_meta["threshold"])),
    }
    output_dir = project_path(cfg["reports"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = pd.DataFrame(results).T
    table.to_csv(output_dir / "model_comparison.csv", index_label="model")
    print(table.to_string())
    print(f"Saved comparison to {output_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
