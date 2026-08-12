# BayesianExoAdaptation

The project tests one causal chain, and every experiment described in `TODO.md` exists
to validate a specific link in it:

1. A neural network trained to MAP on source data only becomes overconfident under
   domain shift (calibration breaks on target data). This is a known failure mode of
   deep learning models, particularly problematic in safety-critical, assistive
   contexts where a system needs to know when it doesn't know.
2. A post-hoc Bayesian treatment of the last layer (Laplace approximation) yields a
   predictive epistemic variance that should grow systematically as samples move
   farther from the source distribution — i.e. variance as a shift detector.
3. That same variance signal can drive adaptation: down-weighting high-uncertainty
   samples during entropy-based adaptation (following the U-SFAN paper, see
   `paper/U-SFAN.pdf`) should let the optimization be steered mainly by reliable
   target samples, rather than being misled by confident-but-wrong predictions on
   heavily shifted data.

The original framing for this project targeted power-assist lower-limb exoskeletons,
using a source-free domain adaptation setup: adapt a controller pre-trained on a pool
of subjects (source domain) to a new, unseen subject (target domain) without access to
the original source data.

## A note on the dataset pivot

An earlier version of this project used a lower-limb sEMG dataset (BASAN, 22 subjects —
11 healthy, 11 with diagnosed knee pathologies), which offered a stronger version of
the exoskeleton framing: a real healthy-vs-pathological population shift, rather than a
surrogate. That dataset was set aside after raw-signal data-quality issues (inconsistent
file headers, per-channel sample-count mismatches, unexplained EMG amplitude outliers,
inconsistent goniometer sign conventions, and highly uneven recording durations across
subjects) turned out to demand disproportionate data-engineering effort relative to
their relevance for a course project centered on approximate Bayesian inference rather
than biomedical signal processing.

The project now uses **UCI HAR** instead — pre-engineered features, well-documented,
no comparable data-quality issues found on inspection. The trade-off is an explicit
one: a weaker version of the exoskeleton motivation (healthy subjects performing
different locomotion modes, rather than a real pathological population), in exchange
for a dataset that lets the project's actual focus — the Bayesian machinery — get most
of the attention. This trade-off, and the reasoning behind it, is documented as part of
the project's own limitations rather than left implicit.

## Dataset: UCI HAR (Human Activity Recognition Using Smartphones)

### Overview

Recordings from **30 subjects**, each wearing a waist-mounted smartphone, performing
six activities. The dataset ships 561 pre-engineered features per 2.56s window
(time- and frequency-domain statistics of accelerometer/gyroscope signals), already
computed — no raw-signal processing is needed for the main pipeline.

### Activities

| Label | Activity | Used in main task? |
|---|---|---|
| 1 | WALKING | yes |
| 2 | WALKING_UPSTAIRS | yes |
| 3 | WALKING_DOWNSTAIRS | yes |
| 4 | SITTING | secondary/6-class setting only |
| 5 | STANDING | secondary/6-class setting only |
| 6 | LAYING | secondary/6-class setting only |

The main task is restricted to the three locomotion classes (1-3): this is the actual
intent-recognition problem relevant to a mobility-assistance framing, and it avoids the
near-ceiling accuracy (~96%+) typically reported when all six classes are included,
which would leave little room to demonstrate an uncertainty/adaptation effect.

### Domain shift

There is no healthy-vs-pathological split in this dataset, all 30 subjects are healthy. Domain shift here is **inter-subject
variability**: the project builds its own subject-wise source/target partition (a
fixed source pool vs. each remaining subject as a separate target domain), discarding
the dataset's original train/test split, which was not designed around this axis.

### File organization

The raw data is the official UCI HAR
zip, containing pre-engineered features (`X_*.txt`, `y_*.txt`, `subject_*.txt` under
`train/` and `test/`) plus raw inertial signals (`Inertial Signals/`, not used by the
main pipeline, kept aside for a possible learned-features extension). See `TODO.md`
§0 for the exact file layout and format quirks (whitespace-separated columns, etc.).