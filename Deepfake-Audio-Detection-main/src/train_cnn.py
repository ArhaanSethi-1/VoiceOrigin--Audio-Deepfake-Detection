"""Train the main log-Mel CNN. The test split is never read during fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import load_config, project_path, seed_everything
from src.features import cache_path, configured_cache, ensure_cached, load_cached
from src.metrics import best_f1_threshold, calculate_metrics
from src.models import LogMelCNN


class CachedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, cache: Path, cfg: dict):
        self.frame, self.root, self.cache, self.cfg = frame.reset_index(drop=True), root, cache, cfg
        self.cache_paths = []
        for audio_relpath in tqdm(self.frame.audio_path.astype(str), desc="Preparing cached feature paths", leave=False):
            cached = cache_path(audio_relpath, self.root, self.cache, self.cfg)
            if not cached.is_file():
                cached = ensure_cached(audio_relpath, self.root, self.cache, self.cfg)
            self.cache_paths.append(cached)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        x = load_cached(self.cache_paths[index], "logmel")
        return torch.from_numpy(x), torch.tensor(int(row.label), dtype=torch.float32)


def add_logmel_power_noise(batch: torch.Tensor, settings: dict) -> torch.Tensor:
    """Approximate additive noise in Mel power; called only on training batches."""
    probability = float(settings.get("noise_probability", 0.0))
    if probability <= 0.0 or len(batch) == 0:
        return batch

    selected = torch.rand(len(batch), device=batch.device) < probability
    if not bool(selected.any()):
        return batch

    # The cached input is log-power mapped to [0, 1]. Undo that mapping,
    # add random Mel-band noise power at a sampled SNR, then map back.
    chosen = batch[selected].clamp(0.0, 1.0)
    db = chosen * 80.0 - 80.0
    power = torch.pow(10.0, db / 10.0)
    signal_power = power.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)

    snr_min = float(settings.get("noise_snr_min_db", 5.0))
    snr_max = float(settings.get("noise_snr_max_db", 20.0))
    if snr_min > snr_max:
        raise ValueError("noise_snr_min_db must be less than or equal to noise_snr_max_db")
    snr = torch.empty((len(chosen), 1, 1, 1), device=batch.device).uniform_(snr_min, snr_max)
    mean_noise_power = signal_power / torch.pow(10.0, snr / 10.0)
    noise_power = -torch.log(torch.rand_like(power).clamp_min(1e-6)) * mean_noise_power

    noisy_db = 10.0 * torch.log10((power + noise_power).clamp_min(1e-8))
    noisy = ((noisy_db + 80.0) / 80.0).clamp_(0.0, 1.0)
    result = batch.clone()
    result[selected] = noisy
    return result


def augment_logmel(batch: torch.Tensor, settings: dict) -> torch.Tensor:
    """Training-only noise, gain, and SpecAugment masks; validation/test stay clean."""
    if not settings.get("enabled", False):
        return batch
    x = add_logmel_power_noise(batch, settings)
    max_shift = float(settings.get("gain_db", 0.0)) / 80.0
    if max_shift:
        shifts = torch.empty((len(x), 1, 1, 1), device=x.device).uniform_(-max_shift, max_shift)
        x = (x + shifts).clamp_(0.0, 1.0)
    fill = 0.5
    max_f = int(settings.get("frequency_mask_bands", 0))
    max_t = int(settings.get("time_mask_frames", 0))
    for sample in x:
        _, mel_bands, frames = sample.shape
        for _ in range(int(settings.get("frequency_masks", 0))):
            width = int(torch.randint(0, max_f + 1, (1,)).item()) if max_f else 0
            if width:
                start = int(torch.randint(0, max(1, mel_bands - width + 1), (1,)).item())
                sample[:, start : start + width, :] = fill
        for _ in range(int(settings.get("time_masks", 0))):
            width = int(torch.randint(0, max_t + 1, (1,)).item()) if max_t else 0
            if width:
                start = int(torch.randint(0, max(1, frames - width + 1), (1,)).item())
                sample[:, :, start : start + width] = fill
    return x


def collect_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels, scores, loss_sum, count = [], [], 0.0, 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += float(criterion(logits, y).item()) * len(y)
            count += len(y)
            labels.extend(y.cpu().numpy().astype(int).tolist())
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(labels), np.asarray(scores), loss_sum / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    manifest = Path(args.manifest) if args.manifest else project_path(cfg["manifest"])
    root = Path(args.root) if args.root else project_path(cfg["data_root"])
    cache = configured_cache(cfg)
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}. Prepare the dataset first.")
    frame = pd.read_csv(manifest, keep_default_na=False)
    train_df, val_df = frame[frame.split == "train"], frame[frame.split == "validation"]
    if train_df.empty or val_df.empty:
        raise SystemExit("Manifest needs non-empty train and validation partitions.")
    settings = cfg["cnn"]
    batch = int(settings["batch_size"])
    eval_batch = int(settings.get("eval_batch_size", batch))
    workers = int(settings.get("num_workers", 0))
    loader_options = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        CachedDataset(train_df, root, cache, cfg), batch_size=batch, shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        CachedDataset(val_df, root, cache, cfg), batch_size=eval_batch, shuffle=False, **loader_options
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogMelCNN().to(device)
    counts = train_df.label.astype(int).value_counts().to_dict()
    n0, n1 = max(counts.get(0, 0), 1), max(counts.get(1, 0), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n0 / n1], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    patience, best_loss, stale = int(settings["patience"]), float("inf"), 0
    outdir = project_path(cfg["artifacts"]) / "logmel_cnn"
    outdir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_path = outdir / "best_model.pt"
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        running, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for x, y in progress:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = augment_logmel(x, cfg.get("augmentation", {}))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip_norm", 1.0)))
            optimizer.step()
            running += float(loss.item()) * len(y)
            seen += len(y)
            progress.set_postfix(loss=f"{running / max(seen, 1):.4f}")
        y_val, val_scores, val_loss = collect_scores(model, val_loader, device)
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), "validation_loss": val_loss}
        history.append(row)
        print(row)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg}, best_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    y_val, val_scores, _ = collect_scores(model, val_loader, device)
    threshold = best_f1_threshold(y_val, val_scores)
    validation_metrics = calculate_metrics(y_val, val_scores, threshold)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "threshold": threshold}, best_path)
    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata = {"model": "logmel_cnn", "positive_label": 1, "threshold": threshold,
                "config": cfg, "validation_metrics": validation_metrics, "device": str(device)}
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_path}")
    print(json.dumps(validation_metrics, indent=2))
    print("Test data was not loaded or used during training.")


if __name__ == "__main__":
    main()
