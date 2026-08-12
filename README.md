# BayesianExoAdaptation

Source-free domain adaptation with last-layer Laplace uncertainty, applied to
locomotion-mode recognition. Replicates and extends Roy et al., *Uncertainty-guided
Source-free Domain Adaptation* (U-SFAN, `paper/U-SFAN.pdf`).

Pipeline: MAP-train a source classifier -> fit a last-layer Laplace posterior ->use
the resulting epistemic uncertainty (BALD) to down-weight unreliable target samples
during information-maximization adaptation.

## Dataset

**UCI HAR** (Human Activity Recognition Using Smartphones): 30 subjects, waist-mounted
smartphone, 561 pre-engineered accelerometer/gyroscope features per 2.56s window.

- Main task: 3-class locomotion (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`).
  Secondary: full 6-class setting for comparison.
- Domain shift: inter-subject variability. Original train/test split discarded;
  project builds its own subject-wise source pool + per-subject target domains.
- Open-set extension planned via HAPT (postural transitions as OOD classes).
- Data not included in this repo — download from the
  [UCI ML Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
  see `TODO.md` §0 for file layout and format notes.

An earlier version used a lower-limb sEMG dataset (BASAN); set aside due to raw-signal
data-quality issues unrelated to the project's actual focus. See `TODO.md` for details.

## Sanity checks

A synthetic 2D toy (replicating the paper's Fig. 4) and an MNIST vs Fashion-MNIST /
Rotated-MNIST check (replicating Kristiadi, Hein & Hennig, ICML 2020 — ref. [26] in
the U-SFAN paper) validate the Laplace/BALD implementation on unambiguous shifts
before applying it to UCI HAR's subtler inter-subject shift.

## Course context

Developed for the Probabilistic Machine Learning course (University of Trieste,
`PML_notes_full.pdf`). Theoretical basis: Laplace approximation (§7.3), Bayesian
logistic regression (§7.4), entropy/mutual information (§2.3), Bayesian model
averaging (§10.6).

## Status

No code implemented yet — see `TODO.md` for the current plan and task list.