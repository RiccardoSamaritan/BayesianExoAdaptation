# Project TODO 
Uncertainty-guided SFDA (U-SFAN replication + extension), applied to locomotion-mode
recognition from pre-engineered smartphone accelerometer/gyroscope features. This
supersedes v3 (BASAN sEMG), abandoned due to raw-signal data-quality issues (variable
header format, per-row column mismatches, unexplained EMG artifacts, inconsistent
goniometer sign conventions, wildly uneven recording durations) that were consuming
disproportionate effort relative to their relevance for a Probabilistic ML exam project.
v2, written earlier for this same dataset, is the direct ancestor of this file — the
overall plan is largely unchanged; what's new here is the set of dataset facts confirmed
by directly inspecting the actual downloaded zip, not just its documentation.

Nothing below has been implemented yet — this is a plan, not a log.

---

## 0. Dataset facts (confirmed by direct inspection of the zip, not just docs)

- **UCI HAR** (Human Activity Recognition Using Smartphones). Source zip structure:
  ```
  UCI HAR Dataset/
    activity_labels.txt        -- maps 1-6 to activity names
    features.txt               -- 561 feature names, one per line
    features_info.txt          -- descriptive docs on how features were computed
    README.txt
    train/  X_train.txt, y_train.txt, subject_train.txt, Inertial Signals/
    test/   X_test.txt,  y_test.txt,  subject_test.txt,  Inertial Signals/
  ```
  Plus `__MACOSX/` and `.DS_Store` junk from how the zip was packaged — ignore, don't
  commit.
- **Files to include in the project**: `activity_labels.txt`, `features.txt`,
  and the six `X_*.txt`/`y_*.txt`/`subject_*.txt` files under `train/` and `test/`.
  **Files to keep aside but not commit to the main pipeline**: the `Inertial Signals/`
  folders (raw accelerometer/gyroscope signal, not the 561 engineered features) —
  not needed for the main plan, but exactly what a §12 stretch goal (learned feature
  extractor vs. hand-engineered features) would need, so don't delete them from disk.
- **Activity label mapping** (`activity_labels.txt`): `1=WALKING, 2=WALKING_UPSTAIRS,
  3=WALKING_DOWNSTAIRS, 4=SITTING, 5=STANDING, 6=LAYING`. The main task uses classes
  1-3 only; 4-6 are the static postures excluded from the main 3-class task (§1).
- **`X_*.txt`**: one row per 2.56s window, 561 columns = the pre-engineered features
  named in `features.txt`, already normalized to roughly [-1, 1] (per the dataset's
  own documentation — **verify this claim directly** rather than trusting it, see §1).
- **`y_*.txt`**: one integer label (1-6) per row, aligned by row index (not by an
  explicit key) with the corresponding row of `X_*.txt` and `subject_*.txt`. All
  three files for a given split must be loaded and kept in lockstep — sorting or
  filtering one without applying the identical operation to the other two will
  silently desynchronize them.
- **Separator quirk confirmed**: `X_*.txt` uses **multiple spaces**, not tabs and not
  single spaces, as the column separator (fixed-width-style formatting). A naive
  `sep=' '` will produce spurious empty columns. Use `sep=r'\s+'` (pandas) or
  `np.loadtxt` (which handles repeated whitespace natively without extra arguments).
- **Subject counts confirmed**: 30 distinct subjects total across train+test
  (train: 21 subjects, test: 9 subjects — an uneven split). **Confirmed zero subject
  overlap between the shipped train and test splits** — but this project ignores that
  split anyway (§1) and rebuilds its own subject-wise source/target partition, so the
  zero-overlap fact is reassuring about data hygiene but not something the pipeline
  depends on.
- **Row counts confirmed**: 7352 rows in train, 2947 in test → 10299 total.
- **Class balance in the 3 locomotion classes looks reasonable** from a first count
  (train: 1226/1073/986 for WALKING/UPSTAIRS/DOWNSTAIRS respectively, roughly a 1.2:1
  ratio between the largest and smallest) — **not yet verified after restricting to
  only NORMAL... wait, there is no NORMAL/ABNORMAL split here, verify after the
  subject-wise source/target repartition** in §1, since the global count doesn't
  guarantee balance within whatever subject subset ends up as the source pool.
- **Not yet checked**: NaNs, or values outside the documented [-1, 1] range, in
  `X_train.txt`/`X_test.txt`. Do this before trusting the "clean data" framing that
  motivated switching back to this dataset — quick with `np.isnan().sum()` and
  `.min()`/`.max()`, but hasn't actually been run yet.

---

## 1. Locked design decisions

- **Main task: 3-class locomotion** — `WALKING`, `WALKING_UPSTAIRS`,
  `WALKING_DOWNSTAIRS` only. Reasons: (i) the full 6-class task is expected to
  saturate (static-vs-dynamic is an easy separation, typical reported accuracy on
  all 6 classes is ~96%+), leaving no room to demonstrate an uncertainty/adaptation
  effect; (ii) these 3 classes are the actual locomotion-mode-recognition problem,
  the others (sitting/standing/laying) aren't relevant to a mobility-assistance
  framing.
- **Secondary: 6-class setting**, run once, expected to show little/no adaptation
  benefit due to saturation. Report as an explicit negative-result comparison, not
  hidden.
- **No synthetic domain shift.** Real inter-subject shift only. Concatenate
  `X_train`+`X_test` (and `y_*`, `subject_*` in lockstep) into one pool of 10299
  windows / 30 subjects, discard the original train/test split entirely, and build
  a subject-wise source/target partition:
  - Source = a fixed pool of subjects (e.g. 20).
  - Target = each of the remaining subjects individually, evaluated as its own
    domain (≈10 target domains), OR full leave-one-subject-out over all 30 if time
    allows — decide based on compute budget once the pipeline runs end-to-end once.
- **Shift proxy per target subject**: since there's no ground-truth shift magnitude
  here (unlike the synthetic track), use at least one of centroid distance in
  feature space or AUROC of a source-vs-target domain classifier; check the two
  agree in ranking before trusting either alone.
- **Lower-bound reference**: a random window-wise split (ignoring subject identity)
  as a near-zero-shift control, to compare against the real subject-wise shift
  magnitudes.
- **Standardization fit on the source pool only**, applied to target subjects —
  even though the raw features are already roughly normalized per the dataset docs,
  re-fitting a scaler on the chosen source pool (rather than trusting a global
  normalization computed by the dataset's original authors over all subjects
  including what will become target) avoids a subtle form of leakage.
- **Architecture**: `561 → 128 → 64 → 3`, matching the feature dimensionality here
  (much larger than the ~27 features BASAN would have had — this MLP is sized for
  a genuinely high-dimensional pre-engineered feature bank, not hand-derived
  time-domain statistics from 4 raw channels).
- **Bayesian treatment: last layer only.** Feature extractor deterministic at
  β_MAP. Same textbook-equivalence argument as always: this is Bayesian
  multi-class logistic regression with φ(x) = g_β(x) learned. With a 64-dim last
  hidden layer and K=3 classes, the augmented last-layer parameter count is
  64·(3+1) + ... i.e. K·(F+1) = 3·65 = 195 params — small enough to invert the
  exact Hessian directly, no Kronecker factorization needed for tractability (§6).
- During target adaptation: update β (feature extractor), freeze θ (classifier
  head). If both are frozen, nothing adapts and all ablation arms collapse.
- Posterior q(θ) stays fixed at the β_MAP features throughout adaptation — an
  approximation to flag explicitly in Limitations (§11), not hide.
- Uncertainty recomputed every minibatch during adaptation, not frozen at the start.
- Down-weighting applies only to the conditional-entropy term of the IM loss,
  never to the diversity term L_div.
- Hyperparameters fixed a priori from the paper: γ = 0.5, τ = 0.4. Check what τ
  actually does on this data (as was starting to be investigated on BASAN) rather
  than inheriting it blindly.
- No target labels used for model selection, ever.
- One config file, all seeds, no hardcoded values scattered around.
- Multiple seeds (5 recommended) before any correlation or ablation-ranking claim
  is treated as final — this matters more, not less, than on BASAN, since with
  ~10-30 target domains any single-seed correlation estimate will be noisy.

---

## 1b. Conventions & bug traps

Unchanged in substance from the BASAN version of this plan — none of these are
dataset-specific, they're properties of the Laplace/Hessian math itself:

- **Prior precision convention**: τ_prior = weight_decay × N_source (PyTorch
  `weight_decay` convention specifically). Do not assume τ = weight_decay alone —
  re-derive and test the correspondence rather than trusting it by inspection.
- **Validate the exact Hessian two ways before building anything on top of it:**
  1. Reduce to K=2 classes, check against the exact binary formula in notes §7.4.
  2. On a small synthetic problem, compare the Laplace predictive distribution
     against real Metropolis-Hastings/HMC samples of the true posterior.
- **Kronecker + prior interaction** (relevant only if/when KFAC is implemented,
  §6): adding τI to a Kronecker-factored V ⊗ U breaks the factorization; add √τ
  to each factor's eigenvalues instead.
- **Absolute normalization of the epistemic signal**, never per-batch — use log K
  or source-quantile-based normalization.
- **Check class balance within whatever subject subset ends up as the source
  pool**, not just globally (§0) — a global count can look fine while a particular
  20-subject source pool ends up skewed.

---

## 2. Data pipeline

- [ ] Load `X_*.txt` with a whitespace-aware parser (`np.loadtxt` or
  `pd.read_csv(sep=r'\s+', header=None)`), `y_*.txt` and `subject_*.txt` similarly,
  keeping train and test arrays separate at first so a lockstep-alignment bug would
  show up as a shape mismatch.
- [ ] Verify no NaNs and that values fall in the documented range before trusting
  the "already normalized" claim (§0).
- [ ] Concatenate train+test into a single pool (X, y, subject_id), discarding the
  original split.
- [ ] Map integer labels to activity names via `activity_labels.txt`; restrict to
  the 3 locomotion classes for the main task, keep a separate 6-class loader path
  for the secondary comparison (§1).
- [ ] Check class balance within the 3-class task, both globally and within
  whatever source subject pool gets chosen — don't assume the global count from
  §0 transfers unchanged to a specific 20-subject subset.
- [ ] Build the subject-wise source/target partition (§1): fixed source pool vs.
  per-subject target domains (or full LOSO).
- [ ] Compute the shift proxy(ies) per target subject; check rank agreement if
  using more than one proxy.
- [ ] Sanity check before touching any model: PCA (fit on source pool only) of
  per-subject centroids, source-vs-target scatter, and the near-zero-shift random
  split as reference — confirm the subject-wise split actually looks like a
  distribution shift rather than assuming it does.
- [ ] Fit the standardization scaler on the source pool only, apply to all target
  subjects.

---

## 3. Synthetic track (optional, for intuition and slides)

- [ ] 2D, 3-class toy replicating paper Fig. 4: mild vs. strong shift, decision
  surfaces shaded by predictive certainty, epistemic/aleatoric maps.
- [ ] Semantic sanity check on the toy: far-from-source point should be
  epistemic-dominant; near-boundary in-distribution point should be
  aleatoric-dominant. Debug BALD here, on a controlled problem, before trusting it
  on real data.
- [ ] `make_classification()` sweep with ground-truth shift magnitude, if time
  allows.

---

## 4. Source MAP training

- [ ] `561 → 128 → 64 → 3` MLP, subject-wise source train/val split (not
  window-wise), cross-entropy + weight decay to MAP.
- [ ] Track macro-averaged recall (balanced accuracy) alongside raw accuracy on
  source val — even though §0's global class counts look reasonably balanced,
  verify it holds for the actual chosen source pool before trusting raw accuracy
  alone.
- [ ] Evaluate on each target subject separately; check the resulting accuracy
  range gives enough headroom to show an adaptation effect (watch out for
  saturation — this was the original concern for this dataset before BASAN was
  tried, and is exactly why the task is restricted to 3 classes in §1).
- [ ] If the full 6-class setting is run as the secondary comparison, expect (and
  explicitly report, don't just note) accuracy near ceiling with little room for
  the adaptation story to show anything.

---

## 5. Exact last-layer Laplace (reference implementation, do this before KFAC)

- [ ] Exact GGN Hessian of the last layer: $H = \sum_n \Lambda_n \otimes
  \phi_n\phi_n^T + \tau I$ with $\Lambda_n = \mathrm{diag}(p_n) - p_n p_n^T$. At
  K=3, F=64 (195 augmented params), this should be small enough to build and
  invert directly.
- [ ] Validation 1: K=2 reduction against the binary formula in notes §7.4.
- [ ] Validation 2: compare against Metropolis-Hastings/HMC on a small synthetic
  problem with matching structure.
- [ ] MC predictive distribution and BALD decomposition, with a convergence check
  (mean BALD vs. number of MC samples) before fixing S in the config.

---

## 6. KFAC as an object of study, not a necessity

At 195 augmented last-layer parameters, the exact Hessian is trivially invertible
directly — Kronecker factorization isn't needed for tractability here either
(same situation as it would have been on BASAN, just with a bigger exact matrix
that's still easy).

- [ ] Implement KFAC anyway, framed as "what does this approximation cost?" State
  explicitly that the exact sum does not factor into a single Kronecker product —
  KFAC adds an independence assumption.
- [ ] Fix the prior via √τ on each factor's eigenvalues.
- [ ] Measure the approximation error against the exact Hessian from §5:
  predictive variance, ECE, and Spearman/Kendall rank correlation of the BALD
  ordering (since the down-weighting only consumes the ranking).

---

## 7. Uncertainty decomposition + semantic validation

- [ ] BALD epistemic/aleatoric split, unit-tested against MCMC (§5) before
  trusting it on real data.
- [ ] Write down explicitly that the IM loss's marginal/conditional entropy terms
  (§9) and BALD share the functional form H[p̄] − H[p] but average over different
  things — data vs. weights.
- [ ] Semantic validation candidates on this dataset:
  - `SITTING` vs `STANDING` (in the 6-class setting) is the well-known confusion
    for this dataset — a waist-mounted IMU struggles to distinguish these two
    static postures. In-distribution, ambiguous by sensor limitation rather than
    lack of data → should show up **aleatoric-dominant**.
  - The hardest held-out target subject (lowest MAP accuracy, from §4) → should
    show up **epistemic-dominant**.
  - If BASAN's finding (real inter-subject shift producing only a small, if
    statistically real, epistemic signal — most uncertainty being aleatoric even
    at the hardest subject) recurs here too, that would be a useful cross-dataset
    consistency point for the report rather than a concern.
- [ ] Absolute normalization of the epistemic signal (source quantiles or log K),
  never per-batch.
- [ ] Check what τ actually does to the epistemic/aleatoric balance on this data,
  rather than assuming the paper's default transfers unchanged.

---

## 8. Calibration check

- [ ] Reliability diagrams, MAP vs. Laplace, source and every target subject.
- [ ] ECE, NLL, Brier score — MAP vs. Laplace, multi-seed mean ± std.
- [ ] AUROC for misclassification detection.
- [ ] Accuracy-vs-coverage curve (reject predictions above an epistemic-uncertainty
  threshold).
- [ ] Spearman correlation between shift proxy (§2) and mean epistemic
  uncertainty across target subjects, multi-seed, converged MC sample count,
  reported as a range/CI rather than a single point estimate.
- [ ] Plots: ECE vs. shift proxy, epistemic uncertainty vs. shift proxy with seed
  bands, reliability diagrams at low/mid/high shift, accuracy-vs-coverage.

**Check before moving on:** does epistemic uncertainty correlate with the shift
proxy, and is the Laplace-based model better calibrated than MAP on more-shifted
subjects? If the signal is weak, rule out an implementation bug via §5's
validations before concluding the real-data effect is genuinely small.

---

## 9. Adaptation + ablation (6 arms)

- [ ] IM objective (not plain entropy minimization, which collapses to one
  class):
  $$ \mathcal{L}_{\text{IM}} = \underbrace{H\Big[\tfrac{1}{N}\sum_i p(y \mid x_i)\Big]}_{\text{marginal, maximize}} \;-\; \underbrace{\tfrac{1}{N}\sum_i H[p(y \mid x_i)]}_{\text{conditional, minimize}} $$
- [ ] Down-weighting multiplies only the conditional-entropy term.
- [ ] Same seeds/data/optimizer/schedule across all arms — only the objective
  changes:
  - (a) no adaptation (reuse frozen-model numbers from §4/§8)
  - (b) entropy only — expect collapse; watch the predicted class histogram
  - (c) full IM, no weighting = SHOT-IM
  - (d) IM + weight from MAP predictive entropy (no Bayes) — the crucial control
    proving any gain comes from epistemic uncertainty specifically, not
    re-weighting per se
  - (e) IM + weight from Laplace predictive entropy = paper-faithful U-SFAN
  - (f) IM + weight from standardized BALD epistemic component — deviation from
    the paper, label it as such
- [ ] Per-shift-level breakdown (low/mid/high tercile by shift proxy), or a
  continuous regression if the number of target domains is too small for terciles
  to be meaningful (decide based on how many target domains §1/§2 end up using).
- [ ] Log trajectories per arm; reliability diagrams post-adaptation; multiple
  seeds, mean ± std.

**Check:** does (b) collapse while (c) doesn't? Does (e) beat (d)? Does (e) or (f)
beat (c), especially on more-shifted target subjects?

---

## 10. Open-set

This is where UCI HAR has a real advantage over BASAN: the extended **HAPT**
dataset (UCI ML repo #341, same 30 subjects, same protocol) adds 6 postural
transitions (stand-to-sit, sit-to-stand, sit-to-lie, lie-to-sit, stand-to-lie,
lie-to-stand) as additional classes, plus raw inertial signals. This was the
original v2 plan and is unaffected by the BASAN detour.

- [ ] If HAPT is used: source = steady-state activities only, target = source
  activities + the 6 transitions as target-private/OOD classes. Report OS
  accuracy (mean per-class accuracy including the unknown class) and OS*
  (shared classes only), not conflated.
- [ ] Fallback if HAPT integration is too costly: drop `LAYING` from the 6-class
  source set, keep it in the target set, as a weaker open-set substitute using
  only the base UCI HAR files already in the project.

---

## 11. Report & presentation

- [ ] Anchor explicitly to course notes: §7.3 (Laplace approximation and its
  local/unimodal caveats), §7.4 (Bayesian logistic regression — the last-layer LA
  *is* this), §2.3 (entropy/KL/mutual information), §10.6 (Bayesian NNs, predictive
  as model averaging), ch. 8 (sampling, used as MCMC ground truth for the Laplace
  validation).
- [ ] Slide on why last-layer-only: tractability + textbook-equivalence argument.
- [ ] Limitations section, to include at minimum:
  - Posterior staleness (q(θ) fixed at β_MAP while β moves during adaptation).
  - LA is local and unimodal.
  - Any deviation from the paper (e.g. arm (f) in §9).
  - **Framing honesty**: this dataset trades a weaker exoskeleton/clinical framing
    (healthy subjects only, phone-worn IMU rather than a purpose-built sensor) for
    substantially lower data-engineering risk — say explicitly that this was a
    deliberate choice after BASAN's raw-signal issues proved disproportionately
    costly relative to a methodology-focused exam project, not an oversight.
  - Standard HAR limitations: healthy young subjects only, no cost asymmetry
    between confusion types, offline windowed classification rather than causal
    real-time, waist/phone-mounted sensor rather than a purpose-built exoskeleton
    sensor placement.
- [ ] State the model-selection protocol (no target labels used, ever) as one
  explicit sentence.

---

## 12. Eventually / stretch

- [ ] Full-network KFAC vs. last-layer-only.
- [ ] MC dropout / small deep ensemble comparison against the Laplace approach.
- [ ] Since the raw `Inertial Signals/` folders were kept aside (§0) rather than
  discarded, a learned feature extractor (e.g. small 1D-conv over raw
  accelerometer/gyroscope windows) instead of the 561 hand-engineered features is
  a natural extension — compare whether a learned representation changes the
  strength of any shift-vs-uncertainty correlation found in §8.
- [ ] Cost-sensitive rejection reflecting that not all confusions are equally
  costly in a mobility-assistance setting.