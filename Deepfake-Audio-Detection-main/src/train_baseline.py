"""Train the MFCC mean/std + logistic regression comparison baseline."""
from __future__ import annotations

import argparse
import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config, project_path, seed_everything
from src.data_utils import feature_arrays, read_manifest
from src.metrics import best_f1_threshold, calculate_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    frame, root, cache = read_manifest(cfg, args.manifest, args.root)
    train, val = frame[frame.split == "train"], frame[frame.split == "validation"]
    if train.empty or val.empty:
        raise SystemExit("Manifest needs non-empty train and validation partitions.")
    x_train, y_train = feature_arrays(train, root, cache, cfg, "mfcc")
    x_val, y_val = feature_arrays(val, root, cache, cfg, "mfcc")
    settings = cfg["baseline"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(settings["c"]), max_iter=int(settings["max_iter"]), class_weight="balanced", random_state=int(cfg["seed"])),
    )
    model.fit(x_train, y_train)
    val_scores = model.predict_proba(x_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_scores)
    metrics = calculate_metrics(y_val, val_scores, threshold)
    outdir = project_path(cfg["artifacts"]) / "mfcc_baseline"
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outdir / "model.joblib")
    metadata = {"model": "mfcc_logistic_regression", "positive_label": 1, "threshold": threshold,
                "config": cfg, "validation_metrics": metrics}
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved baseline to {outdir}; classification threshold selected on validation data.")


if __name__ == "__main__":
    main()
