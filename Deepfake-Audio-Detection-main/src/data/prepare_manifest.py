"""Build a CSV manifest from the official ASVspoof 2019 LA protocols."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config import load_config, project_path

PROTOCOLS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "validation": "ASVspoof2019.LA.cm.dev.trl.txt",
    "test": "ASVspoof2019.LA.cm.eval.trl.txt",
}
AUDIO_DIRS = {"train": "ASVspoof2019_LA_train", "validation": "ASVspoof2019_LA_dev", "test": "ASVspoof2019_LA_eval"}
FIELDS = ["audio_path", "label", "split", "speaker_id", "source_id", "attack_id"]


def locate_audio(root: Path, split: str, utterance_id: str) -> Path:
    expected = root / AUDIO_DIRS[split] / "flac" / f"{utterance_id}.flac"
    if expected.is_file():
        return expected
    matches = list(root.glob(f"**/{utterance_id}.flac"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Audio file not found for utterance {utterance_id}: expected {expected}")
    raise RuntimeError(f"Multiple audio files found for {utterance_id}; fix the dataset layout: {matches}")


def read_protocol(path: Path, split: str, root: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) < 2:
                continue
            speaker, utterance = fields[0], fields[1]
            raw_label = fields[-1].lower()
            if raw_label not in {"bonafide", "spoof"}:
                raise ValueError(f"Unexpected label {raw_label!r} in {path}:{line_number}")
            attack = fields[2] if len(fields) > 2 and fields[2] not in {"-", "--"} else ""
            audio = locate_audio(root, split, utterance)
            rows.append({
                "audio_path": audio.relative_to(root).as_posix(),
                "label": 0 if raw_label == "bonafide" else 1,
                "split": split,
                "speaker_id": speaker,
                # ASVspoof 2019 LA protocol identifies the spoofing system, not a
                # general recording/source corpus identifier.
                "source_id": "",
                "attack_id": attack,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Dataset root containing protocol and audio folders")
    parser.add_argument("--protocol-dir", help="Protocol directory; defaults to <root>/ASVspoof2019_LA_cm_protocols")
    parser.add_argument("--output", help="Output manifest CSV")
    parser.add_argument("--config", help="Project JSON configuration")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.root) if args.root else project_path(config["data_root"])
    protocol_dir = Path(args.protocol_dir) if args.protocol_dir else root / "ASVspoof2019_LA_cm_protocols"
    output = Path(args.output) if args.output else project_path(config["manifest"])
    if not root.exists():
        raise SystemExit(f"Dataset root does not exist: {root}. Download/extract ASVspoof 2019 LA and pass --root.")
    rows: list[dict[str, str | int]] = []
    for split, filename in PROTOCOLS.items():
        protocol = protocol_dir / filename
        if not protocol.is_file():
            raise SystemExit(f"Missing protocol file: {protocol}. Check the dataset archive and --protocol-dir.")
        rows.extend(read_protocol(protocol, split, root))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows to {output}")
    print("Labels: 0=bonafide, 1=spoof; official partitions map to train/validation/test.")


if __name__ == "__main__":
    main()
