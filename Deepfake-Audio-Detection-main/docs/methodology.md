# Methodology and improvement choices

## Task

Binary speech anti-spoofing: label bona fide human speech as `0` and synthetic/converted spoof speech as `1`. The project is detection-only; it does not create or imitate voices.

## Data protocol

Use the official ASVspoof 2019 Logical Access protocol as the primary benchmark. LA targets synthesis and voice-conversion attacks. Preserve official train/development/evaluation membership. Verify exact duplicate paths and report speaker/source/attack overlap rather than silently reshuffling a benchmark. If creating a secondary group-held-out experiment, record the exact grouping rule and keep it distinct from the official result.

The benchmark is intentionally not described as a realistic sample of all current voice generators. A separate corpus or generator holdout is needed to support out-of-domain claims. Respect all dataset access and licensing terms.

## Feature and model comparison

1. MFCC baseline: extract 40 MFCC coefficients from each fixed-length clip, summarize each with time mean and standard deviation, standardize, then fit class-balanced logistic regression.
2. Main model: compute 64-band log-Mel spectrograms scaled to a fixed 0-1 range and train a moderate four-stage residual 2D CNN with squeeze-and-excitation channel attention, mean/max pooling, and dropout.

Training-only augmentation applies Mel-power noise at a sampled 5–20 dB SNR, small gain shifts, and time/frequency masks to the cached log-Mel representation. Validation and test inputs remain unaugmented. This exposes the CNN to controlled recording variation without creating additional synthesized voices or contaminating evaluation partitions. The Mel-power noise is an approximation of additive audio noise and should be assessed with the separate waveform-level robustness evaluation.

Keeping one primary benchmark, one interpretable baseline, and one CNN makes the project reproducible and feasible. The comparison is not a claim that these are state-of-the-art systems. Log-Mel and MFCC outputs should be inspected for preprocessing problems, while predictive usefulness must be judged on validation/test metrics.

## Training and thresholding

Train only on `train`. Select the CNN checkpoint by validation loss and select the operating threshold by validation spoof F1. Freeze that threshold before test evaluation. Report EER independently, since EER uses the score ranking across thresholds. Do not use test results to choose epochs, features, or thresholds.

The training split may be imbalanced. Logistic regression uses class-balanced weights; the CNN uses a positive-class weight based on the training label counts. The CNN also uses gradient clipping, a validation-loss learning-rate scheduler, and early stopping. Report per-class error counts as well as aggregate scores.

## Robustness experiment

Measure the same trained detector on clean held-out test clips and controlled transformations: additive white noise at 20 dB and 10 dB SNR, bandwidth reduction through 8 kHz resampling, and optional MP3 encode/decode at 64 kb/s. Keep the validation-selected threshold fixed for each condition. Report each condition separately. If running a secondary dataset/generator test, do not use it for training or threshold selection.

## Metrics and cautions

Positive class is spoof. Report accuracy, spoof precision/recall/F1, macro F1, ROC-AUC, EER, and the confusion matrix. Precision captures how often a spoof flag is correct; recall captures how many spoof clips are caught. EER is the point where false-accept and false-reject rates are approximately equal. With class imbalance, accuracy by itself can hide poor performance on the minority class.

The prediction score is a model score and is not calibrated certainty. A false positive can wrongly cast suspicion on genuine speech; a false negative can miss a spoof. No single detector output proves provenance.
