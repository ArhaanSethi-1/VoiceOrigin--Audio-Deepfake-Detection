# Design choices

These are the choices used in the current project run. They are practical starting points for this assignment, not claims that these are the best possible detector settings.

## Why compare an MFCC baseline with a CNN?

I wanted a simple model to compare against the neural network. The baseline summarizes MFCC features with their mean and standard deviation, then uses logistic regression. It is quick to train and gives us a reference point.

The CNN uses a log-Mel spectrogram, which keeps more of the changes across time and frequency than the compact MFCC summary. That gives the CNN more information, but it also makes it more complex. The test results tell us whether that extra complexity helped on this dataset.

## Why augment the training data?

The CNN training code sometimes adds noise in the Mel representation. It also changes gain and masks small time and frequency regions. These changes are applied to training batches only. Validation and test examples stay clean so they remain separate from training.

The noise in the Mel representation is an approximation. The separate robustness script adds noise to audio itself, so the two experiments are related but not identical.

## How the data splits are used

The training split fits the model. The validation split is used to select a checkpoint and a decision threshold. The test split is reserved for the final evaluation. Changing settings after looking at test results would make the test score less trustworthy.

The manifest validator can report speaker and source overlap. Seeing overlap does not automatically mean there is leakage, but it is something to report and consider when interpreting results.

## What I would check next

1. Confirm the manifest counts, labels, and partition membership.
2. Compare the baseline and CNN on the same held-out test set.
3. Inspect false positives and false negatives, not just the overall accuracy.
4. Repeat robustness checks on more than the small exploratory sample.
5. If an appropriately licensed external dataset is available, evaluate it without using it to tune the model.

The most important limitation is generalization. ASVspoof 2019 does not represent every current voice generator, language, microphone, or recording condition. A strong result on this benchmark should be described as a result on this benchmark.
