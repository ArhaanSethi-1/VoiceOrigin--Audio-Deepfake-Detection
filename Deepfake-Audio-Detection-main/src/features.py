"""Audio loading and deterministic log-Mel/MFCC extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from src.config import project_path


def load_waveform(path: str | Path, sample_rate: int) -> np.ndarray:
    waveform, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Audio is empty or contains invalid values: {path}")
    return waveform.astype(np.float32, copy=False)


def fixed_length(waveform: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    """Center-crop long clips and zero-pad short clips to a stable duration."""
    size = max(1, int(round(sample_rate * seconds)))
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size > size:
        start = (waveform.size - size) // 2
        waveform = waveform[start : start + size]
    elif waveform.size < size:
        left = (size - waveform.size) // 2
        waveform = np.pad(waveform, (left, size - waveform.size - left))
    return waveform


def features_from_waveform(waveform: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    sr = int(cfg["sample_rate"])
    x = fixed_length(waveform, sr, float(cfg["clip_seconds"]))
    common = dict(
        y=x, sr=sr, n_fft=int(cfg["n_fft"]), hop_length=int(cfg["hop_length"]),
        fmin=float(cfg["fmin"]), fmax=float(cfg["fmax"]), center=True,
    )
    mel_power = librosa.feature.melspectrogram(**common, n_mels=int(cfg["n_mels"]), power=2.0)
    logmel_db = librosa.power_to_db(mel_power, ref=1.0, top_db=80.0).astype(np.float32)
    # Fixed-range scaling makes the CNN input stable and gives SpecAugment a
    # meaningful neutral fill value. MFCCs still use the log-power Mel values.
    logmel = np.clip((logmel_db + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)
    mfcc = librosa.feature.mfcc(S=logmel_db, n_mfcc=int(cfg["n_mfcc"]))
    # Compact fixed-size baseline: mean and standard deviation over time for each coefficient.
    mfcc_summary = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)
    return logmel[None, :, :], mfcc_summary


def cache_path(audio_relpath: str, audio_root: Path, cache_root: Path, cfg: dict[str, Any]) -> Path:
    audio = audio_root / audio_relpath
    stat = audio.stat()
    signature = {
        "path": str(audio.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "feature_version": cfg.get("feature_version", 1),
        "sample_rate": cfg["sample_rate"], "clip_seconds": cfg["clip_seconds"],
        "n_fft": cfg["n_fft"], "hop_length": cfg["hop_length"],
        "n_mels": cfg["n_mels"], "n_mfcc": cfg["n_mfcc"], "fmin": cfg["fmin"], "fmax": cfg["fmax"],
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return cache_root / f"{key}.npz"


def ensure_cached(audio_relpath: str, audio_root: Path, cache_root: Path, cfg: dict[str, Any]) -> Path:
    path = cache_path(audio_relpath, audio_root, cache_root, cfg)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = load_waveform(audio_root / audio_relpath, int(cfg["sample_rate"]))
    logmel, mfcc_summary = features_from_waveform(waveform, cfg)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, logmel=logmel.astype(np.float16), mfcc_summary=mfcc_summary)
    temp.replace(path)
    return path


def load_cached(path: Path, kind: str) -> np.ndarray:
    with np.load(path) as data:
        if kind not in data:
            raise KeyError(f"Feature {kind!r} missing from cache file {path}")
        return data[kind].astype(np.float32)


def configured_cache(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["feature_cache"])
