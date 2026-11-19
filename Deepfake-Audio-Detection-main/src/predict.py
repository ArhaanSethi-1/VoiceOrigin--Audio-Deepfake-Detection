"""Predict on an audio file with a trained MFCC or log-Mel model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.config import load_config, project_path
from src.features import features_from_waveform, load_waveform
from src.models import LogMelCNN


def load_detector(kind: str, cfg: dict):
    base = project_path(cfg["artifacts"])
    if kind == "mfcc":
        folder = base / "mfcc_baseline"
        model = joblib.load(folder / "model.joblib")
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        return model, meta, None
    if kind == "cnn":
        folder = base / "logmel_cnn"
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LogMelCNN().to(device)
        checkpoint = torch.load(folder / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, meta, device
    raise ValueError("kind must be 'mfcc' or 'cnn'")


def windows(waveform: np.ndarray, sr: int, seconds: float) -> list[np.ndarray]:
    size = max(1, int(sr * seconds))
    if len(waveform) <= size:
        return [waveform]
    step = max(1, size // 2)
    starts = list(range(0, max(1, len(waveform) - size + 1), step))
    last = len(waveform) - size
    if starts[-1] != last:
        starts.append(last)
    return [waveform[start : start + size] for start in starts]


def score_waveform(waveform: np.ndarray, kind: str, cfg: dict, model, device=None) -> tuple[float, list[float]]:
    sr, seconds = int(cfg["sample_rate"]), float(cfg["clip_seconds"])
    pieces = windows(waveform, sr, seconds)
    scores: list[float] = []
    for piece in pieces:
        logmel, mfcc = features_from_waveform(piece, cfg)
        if kind == "mfcc":
            score = float(model.predict_proba(mfcc.reshape(1, -1))[0, 1])
        else:
            x = torch.from_numpy(logmel).unsqueeze(0).to(device)
            with torch.no_grad():
                score = float(torch.sigmoid(model(x)).item())
        scores.append(score)
    return float(np.mean(scores)), scores


def predict_file(audio_path: str | Path, kind: str = "cnn", config_path: str | Path | None = None) -> dict:
    cfg = load_config(config_path)
    model, metadata, device = load_detector(kind, cfg)
    # Use the exact preprocessing settings saved alongside the trained weights.
    cfg = metadata.get("config", cfg)
    waveform = load_waveform(audio_path, int(cfg["sample_rate"]))
    score, chunk_scores = score_waveform(waveform, kind, cfg, model, device)
    threshold = float(metadata["threshold"])
    sides = {chunk_score >= threshold for chunk_score in chunk_scores}
    return {
        "model": kind, "label": "spoof" if score >= threshold else "genuine",
        "spoof_score": score, "threshold": threshold, "chunks_scored": len(chunk_scores),
        "duration_seconds": len(waveform) / int(cfg["sample_rate"]),
        "segment_scores": chunk_scores,
        "segments_disagree": len(sides) > 1,
        "warning": "Research estimate only; a score is not proof of authenticity or manipulation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Audio file to score")
    parser.add_argument("--model", choices=["cnn", "mfcc"], default="cnn")
    parser.add_argument("--config")
    args = parser.parse_args()
    try:
        result = predict_file(args.audio, args.model, args.config)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}. Train the selected model first.") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
