# Project TODO
## 0. Dataset facts (UCI HAR)

- Zip structure:
  ```
  UCI HAR Dataset/
    activity_labels.txt        -- 1-6 -> activity name
    features.txt               -- 561 feature names
    train/  X_train.txt, y_train.txt, subject_train.txt, Inertial Signals/
    test/   X_test.txt,  y_test.txt,  subject_test.txt,  Inertial Signals/
  ```
  Ignore `__MACOSX/`, `.DS_Store`.
- Include: `activity_labels.txt`, `features.txt`, all `X_*/y_*/subject_*.txt`.
  Keep aside (not in main pipeline): `Inertial Signals/` (for §12 stretch).
- Labels: `1=WALKING, 2=WALKING_UPSTAIRS, 3=WALKING_DOWNSTAIRS, 4=SITTING,
  5=STANDING, 6=LAYING`.
- `X_*.txt`: 561 features/window, whitespace-separated (multiple spaces — use
  `sep=r'\s+'` or `np.loadtxt`, not a naive single-space split).
- `y_*.txt`, `subject_*.txt`: aligned by row index with `X_*.txt`. Keep all three
  in lockstep when sorting/filtering.
- 30 subjects total, no overlap between shipped train/test (7352 + 2947 rows).
- Train class counts (locomotion only): WALKING 1226, UPSTAIRS 1073,
  DOWNSTAIRS 986.
- Not yet checked: NaNs, values outside [-1, 1] in `X_*.txt`.

---

## 1. Design decisions

- Main task: 3-class locomotion (`WALKING`, `WALKING_UPSTAIRS`,
  `WALKING_DOWNSTAIRS`). Secondary: full 6-class run for comparison.
- No synthetic shift. Merge train+test, discard original split. Build
  subject-wise source pool (~20 subjects) + per-subject target domains (~10,
  or full LOSO over 30).
- Shift proxy per target subject: centroid distance in feature space; optionally
  cross-check with domain-classifier AUROC.
- Random window-wise split (ignoring subject) as near-zero-shift reference.
- Standardize on source pool only, apply to targets.
- Architecture: `561 → 128 → 64 → 3`.
- Bayesian treatment: last layer only, feature extractor frozen at β_MAP.
- Adaptation: update β, freeze θ. Recompute uncertainty every minibatch.
  Down-weighting applies only to the conditional-entropy term, never L_div.
- Fixed hyperparameters: γ = 0.5, τ = 0.4 (check what τ does empirically here).
- No target labels used for model selection.
- One config file. 5 seeds minimum for any reported correlation/ranking.
- Run §3b (MNIST) before investing in §7/§8's HAR correlation numbers.

---

## 1b. Implementation requirements (Laplace/Hessian)

- Prior precision: τ_prior = weight_decay × N_source (PyTorch weight_decay
  convention). Do not use weight_decay alone.
- Validate the exact Hessian two ways before building on top of it:
  1. K=2 reduction matches the binary formula (notes §7.4).
  2. Predictive distribution matches Metropolis-Hastings/HMC on a small
     synthetic problem.
- If KFAC is implemented: add √τ to each factor's eigenvalues, not τI to the
  Kronecker product directly.
- Normalize epistemic signal on an absolute scale (log K or source quantiles),
  never per-batch.
- Verify class balance within the actual chosen source subject pool, not just
  the global count.

---

## 2. Data pipeline

- [ ] Parser for `X_*/y_*/subject_*.txt`, whitespace-aware.
- [ ] Check NaNs / value range.
- [ ] Merge train+test into one pool.
- [ ] Map labels via `activity_labels.txt`; 3-class and 6-class loader paths.
- [ ] Class balance check (global and within chosen source pool).
- [ ] Subject-wise source/target partition.
- [ ] Shift proxy per target subject.
- [ ] PCA sanity check: per-subject centroids, source-vs-target scatter,
  near-zero-shift reference.
- [ ] Fit scaler on source pool only.

---

## 3. Synthetic track

- [ ] 2D, 3-class toy replicating paper Fig. 4 (mild vs strong shift, decision surfaces by certainty, epistemic/aleatoric maps).
- [ ] Semantic check: far-from-source point → epistemic-dominant; near-boundary
  in-distribution point → aleatoric-dominant.
- [ ] Optional: `make_classification()` sweep with ground-truth shift magnitude.

---

## 3b. MNIST / Rotated-MNIST sanity check

- [ ] Part 1: MAP + last-layer Laplace trained on MNIST, evaluated on MNIST
  test vs Fashion-MNIST test. Compare predictive entropy / BALD epistemic
  uncertainty between the two.
- [ ] Part 2 (optional, time-boxed): train on upright MNIST, evaluate on test
  digits rotated across a sweep of angles (0°–180°). Mean epistemic uncertainty
  vs. rotation angle.
- [ ] Needs a small CNN feature extractor (not the HAR MLP). Laplace/Hessian/BALD
  code unchanged — operates only on the last linear layer.
- [ ] Same Hessian validations as §1b apply here.
- [ ] Cite Kristiadi, Hein & Hennig (ICML 2020) — ref. [26] in the U-SFAN paper.

---

## 4. Source MAP training

- [ ] `561 → 128 → 64 → 3` MLP, subject-wise train/val split, CE + weight decay.
- [ ] Track macro-recall and raw accuracy on source val.
- [ ] Evaluate per target subject; check accuracy range (no saturation, no
  near-chance collapse).

---

## 5. Exact last-layer Laplace

- [ ] Exact GGN Hessian: $H = \sum_n \Lambda_n \otimes \phi_n\phi_n^T + \tau I$,
  $\Lambda_n = \mathrm{diag}(p_n) - p_n p_n^T$.
- [ ] Validation 1: K=2 reduction vs. notes §7.4.
- [ ] Validation 2: vs. Metropolis-Hastings/HMC on a small synthetic problem.
- [ ] MC predictive + BALD, with a convergence check on number of MC samples.

---

## 6. KFAC

- [ ] Implement Kronecker-factored approximation; state explicitly that the
  exact sum doesn't factor (independence assumption).
- [ ] Prior via √τ on eigenvalues.
- [ ] Measure error vs. exact Hessian: predictive variance, ECE, BALD-ranking
  correlation (Spearman/Kendall).

---

## 7. Uncertainty decomposition

- [ ] BALD epistemic/aleatoric split, tested against MCMC (§5).
- [ ] Semantic checks: SITTING vs STANDING (6-class setting) → aleatoric-dominant;
  hardest target subject → epistemic-dominant.
- [ ] Absolute normalization of epistemic signal.
- [ ] Check effect of τ on epistemic/aleatoric balance.

---

## 8. Calibration check

- [ ] Reliability diagrams, MAP vs Laplace, source + every target subject.
- [ ] ECE, NLL, Brier — multi-seed mean ± std.
- [ ] AUROC for misclassification detection.
- [ ] Accuracy-vs-coverage curve.
- [ ] Spearman(shift proxy, epistemic uncertainty), multi-seed.
- [ ] Plots: ECE vs shift, epistemic vs shift with seed bands, reliability
  diagrams at low/mid/high shift, accuracy-vs-coverage.

---

## 9. Adaptation + ablation

- [ ] IM loss:
  $$ \mathcal{L}_{\text{IM}} = H\Big[\tfrac{1}{N}\sum_i p(y|x_i)\Big] - \tfrac{1}{N}\sum_i H[p(y|x_i)] $$
- [ ] 6 arms, same seeds/data/optimizer, only the objective changes:
  - (a) no adaptation
  - (b) entropy only (expect collapse)
  - (c) IM, no weighting (SHOT-IM)
  - (d) IM + MAP predictive entropy weight
  - (e) IM + Laplace predictive entropy weight (paper-faithful)
  - (f) IM + standardized BALD epistemic weight (deviation, label as such)
- [ ] Per-shift-tercile or continuous regression breakdown.
- [ ] Log per-arm trajectories, reliability diagrams, multi-seed mean ± std.

---

## 10. Open-set (HAPT)

- [ ] Source = steady-state activities, target = source + 6 postural transitions
  as target-private/OOD.
- [ ] Report OS accuracy (incl. unknown class) and OS* (shared classes only).
- [ ] Fallback: drop `LAYING` from 6-class source, keep in target.

---

## 11. Report

- [ ] Anchor to course notes: §7.3, §7.4, §2.3, §10.6, ch. 8.
- [ ] Cite Kristiadi, Hein & Hennig (ICML 2020) for §3b.
- [ ] Limitations: posterior staleness, LA local/unimodal, arm (f) deviation,
  dataset-pivot rationale (BASAN → HAR), standard HAR limitations (healthy
  subjects, no cost asymmetry, offline classification, phone-mounted sensor).
- [ ] State model-selection protocol (no target labels used, ever).

---

## 12. Stretch

- [ ] Full-network KFAC vs. last-layer-only.
- [ ] MC dropout / small ensemble comparison.
- [ ] Learned feature extractor on `Inertial Signals/` vs. hand-engineered 561
  features.
- [ ] Cost-sensitive rejection.