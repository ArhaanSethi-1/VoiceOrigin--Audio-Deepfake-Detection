"""Evaluate noise, bandwidth reduction, and optional MP3 compression robustness."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.config import load_config, project_path
from src.features import load_waveform
from src.metrics import calculate_metrics
from src.predict import load_detector, score_waveform


def noisy(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms < 1e-8:
        return x.copy()
    noise = rng.normal(size=x.shape).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    return (x + noise * (rms / (10 ** (snr_db / 20)) / max(noise_rms, 1e-8))).astype(np.float32)


def mp3_roundtrip(x: np.ndarray, sr: int, bitrate: str) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is not installed; skip MP3 compression condition")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "input.wav"
        mp3 = Path(tmp) / "compressed.mp3"
        decoded = Path(tmp) / "decoded.wav"
        sf.write(wav, x, sr)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(wav), "-b:a", bitrate, str(mp3)], check=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3), "-ar", str(sr), "-ac", "1", str(decoded)], check=True)
        y, _ = sf.read(decoded, dtype="float32")
        return y.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "mfcc"], default="cnn")
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--config")
    parser.add_argument("--limit", type=int, help="Limit test clips for a quick exploratory run")
    parser.add_argument("--include-mp3", action="store_true", help="Requires ffmpeg; tests a 64 kb/s MP3 encode/decode")
    args = parser.parse_args()
    cfg = load_config(args.config)
    detector, meta, device = load_detector(args.model, cfg)
    cfg = meta.get("config", cfg)
    manifest = Path(args.manifest) if args.manifest else project_path(cfg["manifest"])
    root = Path(args.root) if args.root else project_path(cfg["data_root"])
    rows = pd.read_csv(manifest, keep_default_na=False)
    rows = rows[rows.split == "test"]
    if args.limit:
        rows = rows.head(args.limit)
    if rows.empty:
        raise SystemExit("No test examples selected.")
    sr = int(cfg["sample_rate"])
    variants: dict[str, list[float]] = {"clean": [], "noise_20db": [], "noise_10db": [], "bandwidth_8khz": []}
    if args.include_mp3:
        variants["mp3_64kbps"] = []
    labels: list[int] = []
    rng = np.random.default_rng(int(cfg["seed"]))
    skipped_mp3 = False
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Robustness conditions"):
        wave = load_waveform(root / row.audio_path, sr)
        conditions = {
            "clean": wave,
            "noise_20db": noisy(wave, 20, rng),
            "noise_10db": noisy(wave, 10, rng),
            "bandwidth_8khz": librosa.resample(librosa.resample(wave, orig_sr=sr, target_sr=8000), orig_sr=8000, target_sr=sr),
        }
        if args.include_mp3 and not skipped_mp3:
            try:
                conditions["mp3_64kbps"] = mp3_roundtrip(wave, sr, "64k")
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                print(f"Skipping MP3 condition: {exc}")
                skipped_mp3 = True
        labels.append(int(row.label))
        for name, changed in conditions.items():
            variants[name].append(score_waveform(changed, args.model, cfg, detector, device)[0])
    y = np.asarray(labels)
    results = {name: calculate_metrics(y, np.asarray(scores), float(meta["threshold"])) for name, scores in variants.items() if len(scores) == len(y)}
    output = project_path(cfg["reports"]) / args.model / "robustness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"clips": len(y), "conditions": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved robustness results to {output}")


if __name__ == "__main__":
    main()
