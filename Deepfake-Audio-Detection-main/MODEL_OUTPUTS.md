# Model outputs

This file summarizes the recorded held-out test results for the MFCC baseline and Log-Mel CNN. The positive class is **spoof** (`1`); genuine speech is `0`.

## Test-set results

The ASVspoof 2019 LA test split used here contains 71,237 clips: 7,355 genuine and 63,882 spoof. It is class-imbalanced, so accuracy should be read alongside balanced accuracy and per-class measures.

| Model | Accuracy | Balanced accuracy | Spoof precision | Spoof recall | Spoof F1 | Macro F1 | Genuine specificity | ROC-AUC | EER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MFCC + logistic regression | 89.48% | 68.65% | 93.47% | 94.90% | 94.18% | 69.80% | 42.39% | 85.80% | 22.35% |
| Log-Mel CNN | 86.19% | 91.16% | 99.65% | 84.90% | 91.68% | 75.49% | 97.43% | 95.84% | 9.80% |

The decision threshold was selected using validation data and then held fixed for test evaluation: **0.0696** for MFCC and **0.7921** for the CNN. The reported score is not a calibrated probability. EER is calculated at its own operating point and is not the app's decision threshold.

## Confusion matrices

Rows are the actual class (`genuine`, `spoof`) and columns are the predicted class (`genuine`, `spoof`).

**MFCC + logistic regression**

| Actual \ Predicted | Genuine | Spoof |
|---|---:|---:|
| Genuine | 3,118 | 4,237 |
| Spoof | 3,255 | 60,627 |

**Log-Mel CNN**

| Actual \ Predicted | Genuine | Spoof |
|---|---:|---:|
| Genuine | 7,166 | 189 |
| Spoof | 9,649 | 54,233 |

The MFCC model caught a larger share of spoof clips, which raised its spoof recall and ordinary accuracy on this spoof-heavy test set. It also falsely flagged many genuine clips. The CNN caught fewer spoof clips, but correctly left far more genuine clips unflagged; this is why its balanced accuracy and genuine specificity are higher while its ordinary accuracy is lower. Neither model is best on every metric.

## CNN robustness sample

The following is an exploratory check on **200 clips**, not a full robustness benchmark. The CNN's validation-selected threshold was kept fixed.

| Condition | Accuracy | Balanced accuracy | Spoof recall | Genuine specificity |
|---|---:|---:|---:|---:|
| Clean audio | 86.0% | 90.4% | 84.3% | 96.4% |
| Added noise, 20 dB SNR | 88.5% | 88.8% | 88.4% | 89.3% |
| Added noise, 10 dB SNR | 86.0% | 84.4% | 86.6% | 82.1% |
| Bandwidth reduced to 8 kHz | 82.0% | 88.0% | 79.7% | 96.4% |

Small sample size means these values can vary substantially. These controlled changes do not prove generalization to new voice generators, languages, or real-world recordings.

## Reading the measures

- **Accuracy:** fraction of all test clips classified correctly.
- **Balanced accuracy:** average of genuine recall and spoof recall, giving both classes equal weight.
- **Spoof precision:** among clips flagged as spoof, fraction that are actually spoof.
- **Spoof recall:** among actual spoof clips, fraction detected.
- **Genuine specificity:** among actual genuine clips, fraction correctly left unflagged.
- **F1:** combines precision and recall; `Macro F1` averages F1 across both classes.
- **ROC-AUC:** how well scores rank spoof above genuine across thresholds.
- **EER:** equal error rate, where false-accept and false-reject rates are approximately equal; lower is better.

## Reproducing the outputs

From the project folder, after preparing the dataset and training both models:

```powershell
python -m src.evaluate --model mfcc
python -m src.evaluate --model cnn
python -m src.compare_models
python -m src.robustness --model cnn --limit 200
```

Detailed JSON metrics, plots, and robustness output are generated locally under `reports/generated/`. Dataset audio, checkpoints, and generated reports are excluded from Git; this summary contains the recorded values for reference.
