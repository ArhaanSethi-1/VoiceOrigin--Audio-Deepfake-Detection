# Technical report: speech deepfake detection

## Abstract

Summarize the task, dataset(s), compared models, main test results, and principal limitation. Do not claim results before running the pipeline.

## 1. Problem and motivation

- Define the genuine and spoof classes.
- Describe the intended use as a research detector and its limitations.

## 2. Data

- Dataset name, version, access date, license/access conditions, and provenance:
- Number of genuine/spoof clips in each partition:
- Official protocol files used:
- Missing/corrupt files and handling:
- Speaker/source/attack overlap and duplicate checks:
- Any external or out-of-domain evaluation set:

## 3. Preprocessing and features

- Audio decoder, sample rate, mono conversion, clip duration, crop/pad approach:
- Log-Mel parameters (`n_fft`, hop, bands, frequency range):
- MFCC parameters and summary method:
- Training-only log-Mel augmentation and why it is excluded from validation/test:
- Feature inspection observations:
- Cache/config details for reproducibility:

## 4. Models and training

### MFCC baseline

Describe standardization, logistic regression, class weights, and selected settings.

### Log-Mel CNN

Describe residual convolution blocks, channel attention, pooling, dropout, training-only augmentation, loss, optimizer/scheduler, batch size, stopping criterion, device, and selected checkpoint.

### Leakage controls

Explain how the official partitions were preserved, how duplicate/source overlap was checked, and what limitations remain.

## 5. Evaluation protocol

State which data were used for fitting, validation selection, threshold selection, and final test. Confirm that the test set did not influence model decisions.

## 6. Results

| Model/condition | Accuracy | Balanced accuracy | Spoof precision | Spoof recall | Spoof F1 | Macro F1 | ROC-AUC | EER | Genuine specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MFCC baseline, test | | | | | | | | | |
| Log-Mel CNN, test | | | | | | | | | |

Include confusion matrices and ROC plots. Discuss false positives and false negatives in counts and percentages.

## 7. Robustness and generalization

Compare clean test audio to noise, bandwidth reduction, compression, and any external-source evaluation. Explain where performance changes and plausible reasons. Distinguish controlled perturbations from real-world generalization.

## 8. Discussion and limitations

Discuss dataset age/domain, attack coverage, class balance, identity/source shortcuts, language and codec coverage, model uncertainty, compute limits, and the consequences of false decisions.

## 9. Conclusion and next steps

State what the experiments support, what they do not support, and one or two evidence-based improvements.

## Reproducibility appendix

- Python and library versions:
- Hardware:
- Config and seed:
- Commands used:
- Dataset manifest checksum/version:
- Checkpoint identifier:
