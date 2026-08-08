# Project TODO — v2

Uncertainty-guided SFDA (U-SFAN replication + extension), applied to locomotion-mode
recognition as a surrogate for lower-limb exoskeleton intent recognition.

Ordered by dependency: each section needs the one before it. Check gates at the end of
sections — don't skip them, they're where bugs surface cheaply.

---

## 0. Locked decisions (write these in the report, don't drift from them)

Fix these now so the experiments stay comparable and so hyperparameters can't be
retro-fitted to the target.

- [ ] **Main closed-set task: 3 locomotion classes only** — `WALKING`, `WALKING_UPSTAIRS`,
      `WALKING_DOWNSTAIRS`. Reasons: (i) 6-class UCI HAR is saturated (~96%), same situation
      as Office31 in the paper where the gain is "moderate"; (ii) these 3 classes *are* the
      exoskeleton intent-recognition problem — static postures are irrelevant to a lower-limb
      exo. So K = 3.
- [ ] **Secondary: 6-class setting**, run once, expected to show little/no gain. Report it as
      a negative result consistent with the paper's saturated benchmarks. Don't hide it.
- [ ] **Open-set setting: HAPT postural transitions as target-private classes** (see §9).
- [ ] **Domain shift = subject identity.** No synthetic shift on the real data. Subject-wise
      splits only, never window-wise.
- [ ] **Bayesian treatment: last layer only.** Feature extractor deterministic at β_MAP when
      computing the Hessian. Reasons for a slide: tractable Hessian; cheap MC sampling; the
      last layer with learned features is *exactly* Bayesian multi-class logistic regression
      (notes §7.4), so the whole apparatus is textbook.
- [ ] **During target adaptation: update β (feature extractor), FREEZE θ (classifier head).**
      This is the paper's setup (Fig. 3b). ⚠️ If both are frozen there is nothing to optimize
      and all ablation arms become identical. This was missing from v1.
- [ ] **Posterior q(θ) stays fixed** at the one computed on β_MAP features, even as β moves.
      This is an approximation, not an implementation detail → goes in Limitations (§11).
- [ ] **Uncertainty recomputed every minibatch** during adaptation (paper Algo. 1: z = g_β[T](x)
      changes as β moves, so w_i must change too). Cost is one extra forward pass + M cheap
      last-layer samples.
- [ ] **Weight applies only to the conditional-entropy term**, not to the diversity term.
      L_U-SFAN = (1−γ)·L_ent^ug + γ·L_div, exactly as Eq. 7. L_div is a batch-level statistic,
      per-sample weights don't belong in it.
- [ ] **Hyperparameters fixed a priori** (from the paper): γ = 0.5, τ = 0.4,
      prior precision = weight decay = 5e-4. Any other value must be chosen on a **source**
      validation split.
- [ ] **No target labels are used for anything except final evaluation.** No early stopping on
      target accuracy, no LR picked by looking at target ECE. Write this sentence in the report
      — it's the first thing that gets questioned.
- [ ] One config file, all seeds, no hardcoded values scattered around.
- [ ] Seeds: 5 for the main sweeps, 3 if the LOSO grid gets too slow.

---

## 0b. Conventions & bug traps (read before writing any Hessian code)

These four cause silent failures, not crashes.

- [ ] **Prior precision vs loss reduction.** The exact last-layer Hessian (notes §7.4) sums
      over data:
      $$ \mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + \sum_{n=1}^{N} s_n(1-s_n)\,\phi(x_n)\phi(x_n)^{T} $$
      If training uses `reduction='mean'` + weight decay λ, then the prior precision that
      matches is τ ≈ 2λN (or λN, depending on whether the penalty is λ‖w‖² or ½λ‖w‖²).
      **Not λ.** Getting this wrong gives variances that are uselessly tiny or absurdly huge,
      with no error message. Write down the convention explicitly and unit-test it.
- [ ] **Kronecker structure does not survive adding the prior.** Adding τI to V ⊗ U breaks the
      factorization. Use Ritter et al.'s fix: add √τ to the eigenvalues of each factor
      separately. Adding τ "to the diagonal" after factorizing is wrong.
- [ ] **Overlapping windows break the i.i.d. assumption behind the sum above.** UCI HAR windows
      are 2.56 s with 50% overlap, so every sample is counted twice → inflated effective N →
      artificially narrow posterior → you *understate* exactly the uncertainty you're studying.
      Fix in §1.
- [ ] **Normalize the epistemic signal on an absolute scale, not per batch.** Per-batch
      standardization makes weights relative *within* a batch: even on an unshifted target half
      the samples get down-weighted, and weights stop being comparable across shift levels
      (which is the thing being measured). Use either log K (BALD's upper bound) or quantiles
      of BALD computed on a held-out **source** split. This replaces the v1 "standardize
      per-batch or temperature" item.

---

## 1. Data: UCI HAR + HAPT

- [ ] Download UCI HAR v1 (561 pre-engineered features, 30 subjects, 2.56 s windows @ 50 Hz).
- [ ] Download HAPT (UCI dataset 341) — same 30 subjects, adds the 6 postural transitions and
      ships the raw inertial signals. Needed for §9. Check format differences early, it's the
      one thing that can eat a weekend.
- [ ] **De-overlap:** windows are ordered in time per subject → keep every other window.
      ~10.3k → ~5.1k samples, plenty for a small MLP, and overlap goes to zero without
      touching raw signals. Log this in the report.
- [ ] **Refit standardization on the source pool only**, apply to target. The shipped features
      are already scaled to [-1,1] using statistics over *all* subjects including test — slight
      leakage, and it partly washes out the inter-subject shift you want to study.
- [ ] Restrict to the 3 locomotion classes for the main task; keep the 6-class version as a
      separate loader flag.
- [ ] **Class balance check.** Log the class histogram per subject. Gait data is imbalanced.
      ⚠️ L_div pushes the batch marginal toward uniform K⁻¹·1; if the target class distribution
      isn't uniform this is actively wrong and will be blamed on the uncertainty. Mitigations,
      pick one and state it: source-estimated class prior in place of uniform, or balanced
      batch sampling. Report the vanilla version too, as a confounder.
- [ ] **Design: subject pool as source, individual held-out subjects as target domains.**
      Source = fixed pool (e.g. 20 subjects). Target = each of the remaining subjects,
      separately → ~10 target domains (or full LOSO for 30). This replaces "pick two subject
      groups", which would mean choosing the split that maximizes the effect = circular.
- [ ] **Shift proxy per target subject** — this is the continuous x-axis that replaces synthetic
      shift magnitude. Compute at least two of: centroid distance in feature space, MMD,
      AUROC of a source-vs-target domain classifier. Check they agree (rank correlation between
      proxies); if they don't, that's itself worth a paragraph.
- [ ] **Lower-bound reference: random window-wise split** ignoring subjects → shift ≈ 0.
      Adaptation should do nothing here. If the method *hurts* in this regime, that matters.
- [ ] Plot before touching any model: PCA of the 30 per-subject feature centroids, plus
      source-vs-target scatter for the most- and least-shifted subject. Sanity check that
      subject identity actually looks like a distribution shift.
- [ ] Identify the **hardest held-out subject** (lowest LOSO MAP accuracy) — needed in §6.

---

## 2. Synthetic track (keep, but slimmed — it's for intuition and slides)

- [ ] **2D, 3-class toy replicating Fig. 4.** Source blobs + mild shift + strong shift, decision
      surfaces shaded by predictive certainty, MAP row vs uncertainty-guided row. This is the
      one figure that carries the whole argument; costs an afternoon. Do this *before* the
      high-dimensional sweep.
- [ ] `make_classification()` wrapper with n_informative / n_redundant / n_repeated, n_classes,
      class_sep, n_clusters_per_class, flip_y exposed.
- [ ] Domain-shift generator on top: translation vector with scalar magnitude knob. Add
      covariance scaling and/or rotation if translation alone turns out too easy.
- [ ] Sweep 5–10 shift magnitudes (none → large). Ground-truth shift magnitude is the point of
      this track — it's what the real data can't give you.

---

## 3. Source MAP training

- [ ] MLP `561 → 128 → 64 → K`. Features are already engineered, so nothing deeper is needed.
      Last hidden layer = 64 dims → last layer has 64·K + K params (195 for K = 3).
- [ ] Gaussian prior on weights = L2 (notes §10.6). Record the precision — it comes back in
      §4, and see §0b for the N convention.
- [ ] Train on source only, cross-entropy + weight decay, to MAP. Label smoothing α = 0.1
      if replicating Eq. 1 faithfully.
- [ ] Save: MAP weights, loss/acc curves, last-hidden features. Note: **source** features are
      what the Hessian needs (the posterior conditions on source data); target features are
      needed at prediction time only.
- [ ] Baseline accuracy + confusion matrix — source, and per target subject. Non-Bayesian
      reference point.

**Check:** is 3-class accuracy on held-out subjects in a range with room to move (roughly
80–92%)? If it's already ≥97%, the task is still saturated → weaken the model (fewer features,
e.g. accelerometer-only or time-domain-only subset) rather than proceeding.

---

## 4. Exact last-layer Laplace (reference implementation, do this first)

- [ ] GGN / Hessian of the negative log-posterior at MAP, last layer only. Softmax case: the
      per-sample output block is Λ_n = diag(p_n) − p_n p_nᵀ (not diagonal), so
      $$ \mathbf{H} = \sum_n \Lambda_n \otimes \phi_n\phi_n^{T} + \tau \mathbf{I} $$
      With 195 params this is a 195×195 matrix — build it and invert it directly.
- [ ] q(θ) = N(θ_MAP, H⁻¹).
- [ ] **Validation 1:** reduce to 2 classes and check the Hessian against the binary formula in
      the notes (§7.4) term by term.
- [ ] **Validation 2:** on a small subsample, compare LA predictive variance against
      Metropolis-Hastings / HMC over the same ~195 parameters (notes ch. 8). This is the test
      that catches a wrong prior precision. Do it before building anything on top.
- [ ] MC predictive: S = 50–200 samples, logits scaled by 1/τ with τ = 0.4 before softmax,
      predictive = average softmax (Bayesian model averaging, notes §10.6).
- [ ] Convergence check: mean BALD vs number of MC samples. Cheap plot, justifies the choice
      of S and supports the "computationally lightweight" claim.

---

## 5. KFAC as an object of study (not as a necessity)

Framing: with a 195-parameter last layer the Kronecker factorization buys nothing
computationally — Ritter et al. need it for a 256-dim bottleneck with 345 classes. So the
question becomes *how much accuracy does it cost?*, which is a better project contribution
than borrowing it without reason.

- [ ] Implement the Kronecker approximation H ≈ V ⊗ U (activation covariance ⊗ pre-activation
      gradient covariance). ⚠️ State clearly in the report that Σ_n Λ_n ⊗ φ_nφ_nᵀ does **not**
      factor exactly — KFAC adds an independence assumption (expectation of product ≈ product
      of expectations). v1 conflated the exact sum with the factored form.
- [ ] Prior precision via √τ on each factor's eigenvalues (§0b).
- [ ] Invert the two factors separately; matrix-normal sampling (sample the factors, combine
      via the Kronecker structure — don't sample the full joint).
- [ ] **Measure the approximation error vs the exact Hessian:** predictive variance
      (per-sample scatter + summary), ECE, rank correlation of BALD ordering
      (Spearman/Kendall — the ordering is what the down-weighting actually consumes),
      wall-clock. Plot as a function of layer width if time allows.

**Check:** does KFAC preserve the BALD *ranking* even where it distorts the magnitudes? That's
the practically relevant question, since the weights are a monotone function of BALD.

---

## 6. Uncertainty decomposition + validation on real semantics

- [ ] Split total uncertainty into epistemic/aleatoric via BALD:
      $$ I[y, \theta \mid x] = H\Big[\mathbb{E}_{q(\theta)}\, p(y \mid x, \theta)\Big] - \mathbb{E}_{q(\theta)}\Big[H[p(y \mid x, \theta)]\Big] $$
      entropy of the averaged prediction minus expected entropy of per-sample predictions
      = epistemic.
- [ ] ⚠️ **Write the conceptual note now, before it causes confusion:** L_IM (§8) and BALD have
      the *same functional form* H[p̄] − H[p] but average over different things — data vs
      weights. They are two different mutual informations, I[x;y] vs I[y;θ|x] (notes §2.3).
      Use different symbols in code and in the report.
- [ ] **Validation on real semantics (highest information-per-cost experiment in the project).**
      In the 6-class setting:
      - `SITTING` vs `STANDING` → the classic UCI HAR confusion. In-distribution, ambiguous
        because of the sensor (both static, waist-mounted IMU), not because of missing data →
        must come out **aleatoric-dominant**, and must not shrink as source data grows.
      - Hardest held-out subject (from §1) → must come out **epistemic-dominant**.
      If this pattern doesn't appear, there's a bug — find it before building the
      down-weighting on top.
- [ ] Absolute normalization of the epistemic signal (§0b): log K, or source-derived quantiles.
      Prefer the source quantiles — "high uncertainty" then means "high relative to what the
      model has seen", which is the semantics the weighting needs.
- [ ] Per-sample weights: w_i = exp(−·) for both variants that get compared in §8
      (LA predictive entropy → paper-faithful; standardized BALD → your variant).

---

## 7. Calibration check

- [ ] Reliability diagrams, MAP vs LA. Use adaptive/equal-mass binning, and check bin counts.
- [ ] Metrics, MAP vs LA, on source and on every target subject: accuracy, **ECE**, **NLL**,
      **Brier**. ECE alone is easy to game.
- [ ] **AUROC for misclassification detection** — does the uncertainty separate correct from
      incorrect predictions? More directly answers "is this uncertainty useful?" than ECE.
- [ ] **Accuracy vs coverage curve** (reject when epistemic uncertainty exceeds a threshold).
      This is the operational metric for the exoskeleton framing and the bridge to the
      reject-option / neural predictive monitoring example in notes ch. 1.
- [ ] Log per target subject: shift proxy, MAP acc/ECE, LA acc/ECE, mean epistemic uncertainty.
- [ ] **Spearman correlation between shift proxy and mean epistemic uncertainty** across target
      subjects (n ≈ 10–30 real domains). Same on the synthetic sweep against true shift
      magnitude.
- [ ] Multiple seeds, mean ± std, never single runs.
- [ ] Plots: ECE vs shift proxy (MAP + LA, same axes); epistemic uncertainty vs shift proxy with
      seed bands; reliability diagrams at low/mid/high shift; accuracy-vs-coverage.

**Check before moving on:** does epistemic uncertainty rise with the shift proxy (or at least
correlate strongly), and is LA better calibrated than MAP on shifted subjects? If not, suspects
in order: prior precision convention (§0b), per-batch normalization sneaking back in, MAP
under/overfit, shift proxy not capturing anything, bug in the Kronecker math (test against the
exact Hessian from §4 — that's why §4 comes first).

---

## 8. Adaptation + ablation (6 arms)

- [ ] IM objective, not plain entropy minimization (plain entropy collapses to one class,
      known failure mode):
      $$ \mathcal{L}_{\text{IM}} = \underbrace{H\Big[\tfrac{1}{N}\sum_i p(y \mid x_i)\Big]}_{\text{marginal, maximize}} \;-\; \underbrace{\tfrac{1}{N}\sum_i H[p(y \mid x_i)]}_{\text{conditional, minimize}} $$
      Fix the sign convention once in code and assert on it.
- [ ] Down-weighting multiplies the per-sample **conditional entropy** term only (§0).
- [ ] Same seeds / data / optimizer / schedule across all arms — only the objective changes:
  - **(a)** no adaptation (reuse frozen-model numbers from §7)
  - **(b)** entropy only — expect collapse; watch the predicted class histogram to confirm
  - **(c)** full IM, no weighting = SHOT-IM
  - **(d)** IM + weight from the **MAP** predictive entropy (no Bayes) ← *the crucial control*.
        This is the paper's own ablation (Tab. 1: 71.2 vs 71.8). Without it you have not shown
        that the gain comes from epistemic uncertainty rather than from re-weighting per se.
        Missing in v1.
  - **(e)** IM + weight from **LA predictive entropy** = U-SFAN, paper-faithful
  - **(f)** IM + weight from **standardized BALD epistemic** ← your variant
- [ ] Isolation logic to state explicitly: (c)→(d) isolates re-weighting, (d)→(e) isolates the
      Bayesian treatment, (e)→(f) isolates *which* uncertainty. Deviating from the paper at (f)
      must be labelled as a deviation, not passed off as replication — the paper uses total
      predictive entropy, not the epistemic component.
- [ ] Optional 7th arm if time: class-prior-aware L_div (§1) on the best-performing weighting.
- [ ] Log trajectories per arm: target accuracy, ECE, marginal entropy, predicted class
      histogram over training.
- [ ] Reliability diagrams post-adaptation, all arms.
- [ ] Multiple seeds, mean ± std on final acc/ECE/NLL.
- [ ] Summary table + bar chart, ranking obvious. Separate panel showing (b) collapsing.
- [ ] **Per-shift-level breakdown**, not just averages — the paper's whole claim is that the
      advantage grows with shift severity (Office-Home vs DomainNet). Split target subjects
      into low/mid/high shift terciles by proxy.

**Check:** does (b) collapse while (c) doesn't? Does (e) beat (d)? Does (e) or (f) beat (c) on
ECE and/or accuracy, especially in the high-shift tercile? If not: weighting scale (absolute
vs relative), γ balance, adaptation LR/schedule, or the L_div/imbalance interaction.

---

## 9. Open-set (HAPT)

The setting where the Bayesian treatment should matter most (paper Fig. 1, Fig. 2b), and where
the exoskeleton framing is strongest: postural transitions are the highest fall-risk moments,
exactly when the controller must know that it doesn't know.

- [ ] Source: steady-state activities only. Target: same + the 6 postural transitions
      (stand-to-sit, sit-to-stand, sit-to-lie, lie-to-sit, stand-to-lie, lie-to-stand) as
      target-private / OOD classes.
- [ ] Fallback if HAPT preprocessing gets expensive: keep 6-class UCI HAR, drop `LAYING` from
      the source, keep it in the target. Works, much less interesting.
- [ ] Metrics: OS accuracy = mean per-class accuracy over K shared classes + the unknown class;
      also report OS* (shared classes only) so the two aren't conflated.
- [ ] Re-run arms (a), (c), (e), (f) here. Full 6-arm grid only if time allows.
- [ ] Plot: epistemic uncertainty distribution, shared vs target-private samples. This is the
      OOD-detection story of Fig. 2b on real data.

---

## 10. Report & presentation

- [ ] **Anchor to the course notes explicitly** — this is what makes it a PML project rather
      than an ECCV reimplementation:
      - §7.3 Laplace approximation: mode-seeking, local, unimodal, and the caveats stated there
      - §7.4 Bayesian logistic regression: the last-layer LA **is** this, with φ(x) = g_β(x)
        learned. Same Hessian, same approximation. Not an analogy.
      - §2.3 entropy / KL / mutual information: the IM loss and BALD both live here
      - §10.6 Bayesian NNs: the predictive as Bayesian model averaging over an ensemble
      - ch. 8 sampling: used as ground truth for the LA validation in §4
      - ch. 1: uncertainty for decisions, reject option, neural predictive monitoring — the
        motivating frame for §7's accuracy-vs-coverage curve
- [ ] Slide on *why last-layer only* (from §0), with the tractability + textbook-equivalence
      argument.
- [ ] **Limitations section, written honestly.** Must include:
  - Posterior staleness: q(θ) was computed at β_MAP but β moves during adaptation, so the
    posterior is no longer valid for the new features. The paper does the same; say so.
  - LA is local and unimodal — no guarantee about the rest of the posterior.
  - Deviation from the paper at arm (f).
  - **Exoskeleton surrogacy.** What transfers: inter-subject variability in locomotion-mode
    recognition from a body-worn IMU, unlabelled target shift, need for a reject option.
    What doesn't: no exoskeleton in the loop (the device alters the very kinematics being
    classified, and assistance creates a feedback loop that isn't modelled); waist-mounted
    phone rather than thigh/shank sensors; healthy young subjects only, while the clinical
    target population has pathological gait; offline windowed classification rather than
    causal real-time; and no cost asymmetry — misreading "stair descent" costs vastly more
    than misreading "stair ascent".
- [ ] State the model-selection protocol (§0) in one explicit sentence.

---

## 11. Eventually / stretch

- [ ] Full-network KFAC instead of last-layer-only, compared against the last-layer version.
      This is where the Kronecker factorization actually earns its keep.
- [ ] Alternative uncertainty estimators for comparison: MC dropout, small deep ensemble.
      Rehearse the paper's six arguments for LA over MC dropout (§3.2, "How does this differ")
      and check empirically which hold in this setting.
- [ ] Raw-signal model (1D CNN on the HAPT inertial signals) instead of the 561 engineered
      features — closer to a real deployment, and makes the last-layer restriction meaningful
      rather than cosmetic.
- [ ] Cost-sensitive rejection: asymmetric loss reflecting that not all locomotion-mode
      confusions are equally dangerous.