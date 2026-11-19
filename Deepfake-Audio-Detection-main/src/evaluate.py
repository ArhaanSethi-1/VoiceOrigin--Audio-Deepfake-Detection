"""Evaluate a trained model once on the held-out test partition."""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, project_path
from src.data_utils import feature_arrays, read_manifest
from src.metrics import calculate_metrics, save_plots
from src.models import LogMelCNN
from src.predict import load_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "mfcc"], default="cnn")
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model, metadata, device = load_detector(args.model, cfg)
    cfg = metadata.get("config", cfg)
    frame, root, cache = read_manifest(cfg, args.manifest, args.root)
    test = frame[frame.split == "test"]
    if test.empty:
        raise SystemExit("The manifest has no test rows.")
    y = test.label.astype(int).to_numpy()
    if args.model == "mfcc":
        x, _ = feature_arrays(test, root, cache, cfg, "mfcc")
        scores = model.predict_proba(x)[:, 1]
    else:
        from src.train_cnn import CachedDataset

        settings = cfg["cnn"]
        workers = int(settings.get("num_workers", 0))
        loader = DataLoader(
            CachedDataset(test, root, cache, cfg),
            batch_size=int(settings.get("eval_batch_size", settings["batch_size"])),
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )
        predictions = []
        model.eval()
        with torch.no_grad():
            for features, _ in tqdm(loader, desc="Scoring held-out test clips"):
                predictions.extend(torch.sigmoid(model(features.to(device, non_blocking=True))).cpu().numpy().tolist())
        scores = pd.Series(predictions).to_numpy()
    threshold = float(metadata["threshold"])
    metrics = calculate_metrics(y, scores, threshold)
    outdir = project_path(cfg["reports"]) / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_plots(y, scores, threshold, outdir, "test")
    print(json.dumps(metrics, indent=2))
    print(f"Saved held-out test metrics and plots to {outdir}")


if __name__ == "__main__":
    main()
