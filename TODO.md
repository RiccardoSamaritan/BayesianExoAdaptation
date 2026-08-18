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

- [x] Parser for `X_*/y_*/subject_*.txt`, whitespace-aware.
- [x] Check NaNs / value range.
- [x] Merge train+test into one pool.
- [x] Map labels via `activity_labels.txt`; 3-class and 6-class loader paths.
- [x] Class balance check (global and within chosen source pool).
- [x] Subject-wise source/target partition.
- [x] Shift proxy per target subject.
- [x] PCA sanity check: per-subject centroids, source-vs-target scatter,
  near-zero-shift reference.
- [x] Fit scaler on source pool only.

---

## 3. Synthetic track

- [x] 2D, 3-class toy replicating paper Fig. 4 (mild vs strong shift, decision surfaces by certainty, epistemic/aleatoric maps).
- [x] Semantic check: far-from-source point → epistemic-dominant; near-boundary in-distribution point → aleatoric-dominant.
- [x] Optional: `make_classification()` sweep with ground-truth shift magnitude.

---

## 3b. MNIST / Rotated-MNIST sanity check

- [x] Part 1: MAP + last-layer Laplace trained on MNIST, evaluated on MNIST test vs Fashion-MNIST test. Compare predictive entropy / BALD epistemic uncertainty between the two.
  `src/mnist_check.py`, notebook `notebooks/mnist_check.ipynb`. MNIST test acc 0.978;
  mean epistemic 0.05 (MNIST) vs 0.55 (Fashion-MNIST, 11x); AUROC using epistemic
  uncertainty as an OOD score = 0.987 (vs 0.946 for plain MAP softmax entropy).
- [x] Part 2 (optional, time-boxed): train on upright MNIST, evaluate on test digits rotated across a sweep of angles (0°–180°). Mean epistemic uncertainty vs. rotation angle.
  Same notebook, Part 2 section; reuses Part 1's trained model/Laplace posterior
  (no retraining). Epistemic uncertainty rises sharply from 0° (0.05) to a peak
  around 100° (0.44), then partially recedes towards 180° (0.26) as some digits
  (e.g. an 8) start resembling a plausible rotated digit again rather than an
  unrecognizable shape — mirrored by accuracy dropping from 0.98 to a trough near
  100–120° (0.12) before partially recovering to 0.31 at 180°. Non-monotonic
  by design, not a bug (see markdown note in the notebook).
- [x] Needs a small CNN feature extractor (not the HAR MLP). Laplace/Hessian/BALD code unchanged — operates only on the last linear layer.
  `mc.SmallCNN` (2 conv blocks + FC, 128-d features) wrapped via the new
  `FeatureClassifier.from_feature_extractor` in `bayesian.py`; `LastLayerLaplace`/
  BALD code untouched.
- [x] Same Hessian validations as §1b apply here. (Laplace/BALD code identical to the
  already-validated synthetic-track code; posterior covariance confirmed PSD here too.)
- [x] Cite Kristiadi, Hein & Hennig (ICML 2020) — ref. [26] in the U-SFAN paper.
  (cited in `mnist_check.ipynb`'s intro.)

---

## 4. Source MAP training

- [x] `561 → 128 → 64 → 3` MLP, subject-wise train/val split, CE + weight decay.
- [x] Track macro-recall and raw accuracy on source val.
- [x] Evaluate per target subject; check accuracy range (no saturation, no
  near-chance collapse).

---

## 5. Exact last-layer Laplace

- [x] Exact GGN Hessian: $H = \sum_n \Lambda_n \otimes \phi_n\phi_n^T + \tau I$,
  $\Lambda_n = \mathrm{diag}(p_n) - p_n p_n^T$.
- [x] Validation 1: K=2 reduction vs. notes §7.4.
- [x] Validation 2: vs. Metropolis-Hastings/HMC on a small synthetic problem.
- [x] MC predictive + BALD, with a convergence check on number of MC samples.

---

## 6. KFAC - NON LA FACCIAMO !!!!!

- [ ] Implement Kronecker-factored approximation; state explicitly that the
  exact sum doesn't factor (independence assumption).
- [ ] Prior via √τ on eigenvalues.
- [ ] Measure error vs. exact Hessian: predictive variance, ECE, BALD-ranking
  correlation (Spearman/Kendall).

---

## 7. Uncertainty decomposition

- [x] BALD epistemic/aleatoric split, tested against MCMC (§5).
  `notebooks/har_uncertainty_decomposition.ipynb`. Reuses the §5 artifact
  (`data/har_bald_artifact.npz`, M_FIXED=5000) and cites §5's MCMC validation
  (`test_laplace_vs_mcmc.ipynb`: Pearson corr. vs. MCMC = 0.94 total, 0.72
  epistemic, 0.95 aleatoric) rather than re-running MCMC on the real 195-dim
  HAR head.
- [x] Semantic checks: SITTING vs STANDING (6-class setting) → aleatoric-dominant;
  hardest target subject → epistemic-dominant.
  Same notebook. Hardest target (subject 20, acc=0.653) has both the lowest
  accuracy and the highest epistemic uncertainty of the cohort (8.9x the
  source-val reference). A separate 6-class model (SITTING/STANDING aren't in
  the 3-class locomotion task) shows SITTING+STANDING aleatoric elevated 5.7x
  over clearly-separable classes vs. only 2.5x for epistemic -- confusion
  there is driven by aleatoric, not epistemic, uncertainty.
- [x] Absolute normalization of epistemic signal.
  `normalize_epistemic()` added to `src/bayesian.py` (`log_k` and
  `source_quantile` modes, both computed from fixed references, never
  per-batch). Source-quantile z-score separates easy targets (z~0-2) from
  hard ones (z>20) on one fixed scale.
- [x] Check effect of τ on epistemic/aleatoric balance.
  Same notebook, §6. τ = `LastLayerLaplace.predictive`'s `temperature`
  (the same τ=0.4 fixed in §1) -- swept 0.1 to 4.0: epistemic fraction falls
  monotonically from ~0.76-0.87 (source/hardest target) at τ=0.1 to ~0.02-0.03
  at τ=4.0. The fixed τ=0.4 keeps epistemic near half the signal for the
  hardest target (vs. only 23% at the untouched τ=1 default).

---

## 8. Calibration check

- [x] Reliability diagrams, MAP vs Laplace, source + every target subject.
  `notebooks/har_calibration.ipynb`, §4. 5-seed model init/training (fixed
  seed=42 source/target split, per TODO.md §1 "5 seeds minimum"), predictions
  pooled across seeds per subject; marker size = bin count so sparse bins
  read as noisy rather than as signal.
- [x] ECE, NLL, Brier — multi-seed mean ± std.
  Same notebook, §5. Laplace improves mean ECE/NLL on 7/11 subjects --
  consistently the harder, higher-shift ones (e.g. subject 20: NLL
  1.55 -> 1.03); negligible overcorrection on the easiest, already-
  well-calibrated ones (source val, 15, 30).
- [x] AUROC for misclassification detection.
  Same notebook, §6. Epistemic alone is not uniformly the best detector --
  it is a shift detector first (tracks difficulty, §7), a misclassification
  detector second; total entropy / MAP entropy do comparably or better on
  the hardest subjects where errors are mostly aleatoric.
- [x] Accuracy-vs-coverage curve.
  Same notebook, §7. Rejecting by Laplace total entropy recovers near-
  source accuracy at reduced coverage even on the highest-shift target
  (subject 19: 0.77 at full coverage -> ~1.0 at low coverage).
- [x] Spearman(shift proxy, epistemic uncertainty), multi-seed.
  Same notebook, §8. ρ = 0.84 ± 0.03 across 5 seeds (all p<0.01). One
  disagreement worth noting: subject 20 has lower shift-proxy distance than
  subject 19 but higher epistemic uncertainty and lower accuracy -- epistemic
  tracks true difficulty better than the crude centroid-distance proxy here.
- [x] Plots: ECE vs shift, epistemic vs shift with seed bands, reliability
  diagrams at low/mid/high shift, accuracy-vs-coverage.
  Same notebook, §9 (+ §4/§7 for the reliability/coverage plots).

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