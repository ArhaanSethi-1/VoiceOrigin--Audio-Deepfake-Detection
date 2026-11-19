# Deepfake Audio Detection

This project compares two ways to classify speech as genuine or spoofed using the ASVspoof 2019 Logical Access dataset. The first is a small MFCC and logistic-regression baseline. The second is a CNN trained on log-Mel spectrograms. There is also a local Streamlit page for trying an audio clip.

The work was developed with AI coding assistance. I used it to help draft and organize parts of the code and documentation, then ran the training and evaluation and reviewed the outputs. I’m responsible for understanding the project and being able to explain its choices and limitations; I don’t claim every line was written without assistance.

This is a student research project. Its score is not proof that a recording is genuine or manipulated, and results on ASVspoof 2019 do not establish performance on newer voice generators or real-world recordings.

## One recorded CNN run

In this run, early stopping ended training after 15 epochs and kept the checkpoint from epoch 9. The model saw training-only augmentation: some log-Mel examples received simulated noise, plus gain changes and time/frequency masks. On the 71,237-clip held-out test split, balanced accuracy was 91.2%, spoof recall 84.9%, genuine specificity 97.4%, EER 9.8%, and ROC-AUC 95.8%. The saved metrics and plots are local under `reports/generated/cnn/`.

For an exploratory 200-clip noise check, the 10 dB condition changed from 51.5% to 84.4% balanced accuracy after the augmentation update; genuine specificity changed from 3.6% to 82.1%. This is a small stress test, not evidence of broad real-world robustness. The earlier run is saved locally as `reports/generated/cnn/robustness_before_noise_augmentation.json`.

See [MODEL_OUTPUTS.md](MODEL_OUTPUTS.md) for the recorded test metrics, confusion matrices, and the 200-clip robustness summary for both models.

## Project layout

```text
configs/default.json          Shared paths and model/audio settings
src/data/                     ASVspoof protocol manifest preparation and validation
src/features.py               Audio loading, log-Mel/MFCC extraction, local cache
src/precompute_features.py    Feature cache warm-up
src/train_baseline.py         MFCC summary + logistic regression baseline
src/train_cnn.py              Main log-Mel 2D CNN trainer
src/evaluate.py               Held-out test metrics and plots
src/compare_models.py         Side-by-side test comparison
src/robustness.py             Noise, bandwidth, optional MP3 compression checks
src/predict.py                Command-line inference
app.py                        Local Streamlit demo
docs/                         Methodology and technical report template
data/raw/                     Downloaded dataset (ignored by Git)
data/cache/                   Local feature cache (ignored by Git)
artifacts/                    Model checkpoints (ignored by Git)
reports/generated/            Metrics/plots from your runs (ignored by Git)
```

## 1. Set up Python

Python 3.10 or newer is recommended. From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The complete Python runtime dependency list is in both `requirements.txt` and `pyproject.toml`. If you have an NVIDIA GPU or the default PyTorch wheel does not fit your machine, install the matching PyTorch build using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). CPU training works but can take a long time. `ffmpeg` is an optional system executable used only for the MP3 robustness condition; it is not required for training, inference, or the UI.

## 2. Get and place the dataset

The initial dataset is **ASVspoof 2019 Logical Access (LA)**. Follow the official [ASVspoof 2019 page](https://www.asvspoof.org/index2019.html) and its data access/licensing instructions. Do not commit the audio to Git. Extract the files so the configured data root contains a layout similar to:

```text
data/raw/ASVspoof2019_LA/
  ASVspoof2019_LA_cm_protocols/
    ASVspoof2019.LA.cm.train.trn.txt
    ASVspoof2019.LA.cm.dev.trl.txt
    ASVspoof2019.LA.cm.eval.trl.txt
  ASVspoof2019_LA_train/flac/*.flac
  ASVspoof2019_LA_dev/flac/*.flac
  ASVspoof2019_LA_eval/flac/*.flac
```

If your extracted folders differ, change `data_root` in `configs/default.json` or pass `--root`. `prepare_manifest` currently expects the protocol folder inside the data root; use `--protocol-dir` if it is elsewhere. The evaluation protocol must include labels for this project’s metric script. If your release does not include evaluation labels, do not invent them: obtain the authorized key file or create a separately labeled held-out test set and document that deviation.

## 3. Create and check the manifest

```powershell
python -m src.data.prepare_manifest
python -m src.data.validate_manifest
```

Manifest labels are consistently encoded as `0 = genuine/bona fide` and `1 = spoof`. The official train, development, and evaluation partitions map to `train`, `validation`, and `test`. The validator reports speaker/source overlap; it does not silently rewrite official partitions. Review and document those overlaps because the assignment asks you to consider leakage and identity shortcuts.

## 4. Extract and inspect features

Start with a small cache run:

```powershell
python -m src.precompute_features --limit 20
python -m src.data.inspect_features
```

If the sample looks reasonable, cache the full manifest:

```powershell
python -m src.precompute_features
```

The cache is local and ignored by Git. It contains a log-Mel tensor for the CNN and an MFCC mean/standard-deviation vector for the baseline. Audio is decoded mono at 16 kHz, center-cropped or zero-padded to four seconds, and converted to 64 log-Mel bands or 40 MFCC coefficients. These are starter settings in `configs/default.json`, not universal optimum values.

## 5. Train both models

Train the simpler baseline first:

```powershell
python -m src.train_baseline
```

Then train the main CNN:

```powershell
python -m src.train_cnn
```

Both scripts use the training partition for fitting and validation for model selection / threshold choice. The residual CNN uses channel attention, weighted binary cross-entropy for class imbalance, training-only Mel-power noise sampled from 5–20 dB SNR, gain perturbation, and SpecAugment-style masks, plus AdamW, learning-rate reduction, gradient clipping, and early stopping on validation loss. Validation/test features stay clean. Neither trainer loads the test partition. Checkpoints and metadata go under `artifacts/`.

The MFCC baseline uses per-clip coefficient mean and standard deviation with standardization and class-balanced logistic regression. The CNN consumes log-Mel spectrograms. Comparing their held-out metrics answers whether the CNN representation/model improved over this lightweight baseline. The reference Dessa repository and literature lists are useful context, but their reported results should not be copied as your results.

## 6. Final evaluation and robustness

Run test evaluation only after you have finished choosing settings:

```powershell
python -m src.evaluate --model mfcc
python -m src.evaluate --model cnn
python -m src.compare_models
```

The report includes accuracy, balanced accuracy, spoof precision/recall/F1, genuine specificity, false-positive and false-negative rates, macro F1, ROC-AUC, EER, a confusion matrix, and ROC/confusion plots. The operating threshold is selected using validation F1 and then held fixed for test reporting; EER is a separate threshold-free benchmark statistic. `compare_models` writes a side-by-side CSV for the report.

Explore robustness on a small subset first:

```powershell
python -m src.robustness --model cnn --limit 200
```

The script evaluates clean audio, additive noise at two signal-to-noise levels, and 8 kHz bandwidth reduction. Optional MP3 encode/decode tests require `ffmpeg`:

```powershell
python -m src.robustness --model cnn --limit 200 --include-mp3
```

Run the complete robustness pass only after the small run completes. This is not a substitute for an external-source test. If time permits, use a separate, licensed dataset such as ASVspoof 2021 DF or In-the-Wild as an out-of-domain evaluation and keep its data isolated from training and threshold selection. Create a separate CSV with the same manifest columns (`audio_path`, `label`, `split`, `speaker_id`, `source_id`, `attack_id`), set those examples to `split=test`, then evaluate without retraining or changing the threshold:

```powershell
python -m src.evaluate --model cnn --manifest data\ood\manifest.csv --root data\ood
```

The external set must not be used to select the checkpoint, features, or threshold.

## 7. Inference and local demo

For a Windows step-by-step setup, see [RUN_APP.txt](RUN_APP.txt). It explains
that the dataset and trained checkpoint are not included in the GitHub repo and
must be obtained or created before running the CNN demo.

After training the selected model:

```powershell
python -m src.predict path\to\speech.wav --model cnn
streamlit run app.py
```

Long files are divided into overlapping four-second segments and their scores are averaged. The custom VoxTrace-style demo runs locally; uploads are not sent to a hosted service by this code. MP3 robustness checks additionally require an `ffmpeg` executable on your system PATH; it is optional and not a Python package dependency.

## Reproducibility and limitations

- Set the seed and model/audio parameters in `configs/default.json`.
- Keep official partitions and do not tune on test results.
- Preserve dataset provenance, license/access conditions, and preprocessing choices in your report.
- Inspect speaker, source, and attack overlap; strong in-domain metrics may reflect dataset-specific cues.
- Compression/noise experiments are controlled perturbations, not proof of real-world robustness.
- Detector scores can be wrong, particularly for unseen generators, languages, codecs, and recording conditions.

See [the methodology](docs/methodology.md), [design decisions](docs/design_decisions.md), [references](docs/references.md), [the experiment log](docs/experiment_log_template.md), and [the report template](docs/technical_report_template.md) for guidance. Fill in the log and report from your own runs rather than guessing at results.
