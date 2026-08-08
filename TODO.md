# Project TODO

Uncertainty-guided SFDA (U-SFAN replication + extension), applied to sEMG-based
exercise/locomotion-mode recognition, source = healthy subjects, target = subjects with
diagnosed knee pathologies. This supersedes v2 (written against UCI HAR/HAPT before the
dataset changed to BASAN).

---

## 0. Dataset facts (documentation, not results — read before writing the parser)

- **BASAN sEMG** (UCI ML repo #278). 22 male subjects: 11 NORMAL (healthy), 11
  ABNORMAL (diagnosed knee pathology — per the literature, a mix of ACL injury,
  meniscus injury, and sciatic nerve injury, but per-subject etiology is not present
  in the raw files here, only the group label).
- 4 sEMG channels — RF (Rectus Femoris), BF (Biceps Femoris), VM (Vastus Medialis),
  ST (Semitendinosus), in mV — plus 1 goniometer channel FX (knee flexion angle, deg).
- Sampling rate: 1000 Hz, 14-bit resolution (per BASAN documentation; not stated
  inside the files themselves).
- 3 exercises per subject, filename-encoded as `{subject_id}{A|N}{exercise}.txt`:
  `mar` = marcha (walking/gait), `pie` = depie (standing, knee flexion), `sen` =
  sentado (seated, knee extension). 11 subjects × 3 exercises × 2 groups = 66 files.
- **File format quirks to handle, not assume away:**
  - 6-line metadata header ("File Name: ..." + 5× "Channel k: ..." description
    lines), CRLF line endings, tab-separated data after a blank line.
  - A subset of files (9 of 66, per a prior look at the raw data — verify this count
    yourself) carry a 6th column ("Digitals combined ..." in the header) that was
    found to be a constant event marker with no information. Confirm this is still
    true before dropping it silently; don't assume a fixed column count across all
    files.
  - Don't hardcode "skip N header lines" — detect the first line that parses as
    the expected number of tab-separated floats instead, since header length was
    observed to vary between files.
- No per-subject injury-etiology mapping is available in this repo's data folder —
  relevant for the open-set decision in §9.
- No postural-transition data exists in BASAN (unlike the HAPT dataset from the v2
  plan) — the old open-set idea does not port over as-is (§9).

---

## 1. Locked design decisions

- **Task: 3-class exercise recognition** — `WALKING`, `KNEE_FLEXION_STANDING`,
  `LEG_EXTENSION_SEATED`. This is directly the intent-recognition problem, not a
  surrogate for one.
- **Source = NORMAL pool, target = ABNORMAL, one target domain per subject** (11
  target domains). This is a real clinically-meaningful shift, not a stand-in.
- **No synthetic shift on real data.** Subject-wise splits only, never window-wise.
  Standardization fit on NORMAL only, applied to ABNORMAL.
- **Windows: 200 ms (200 samples @ 1000 Hz), non-overlapping.** Decide this up front
  rather than fixing an overlap problem later — non-overlap keeps the i.i.d.
  assumption behind the last-layer Hessian sum valid without a correction step.
- **Feature set: hand-engineered, not learned**, at least for the main track.
  Classic Hudgins time-domain set per EMG channel — MAV, RMS, WL, ZC, SSC, VAR —
  giving 6 × 4 = 24 features, plus a small number of goniometer summary stats
  (e.g. mean angle, range, mean |velocity|) for the FX channel. Expect a feature
  dimension in the 25-30 range, much smaller than UCI HAR's 561 — architecture
  should be scaled down accordingly (a small MLP, not the 561→128→64 one from the
  old plan).
- **Bayesian treatment: last layer only.** Feature extractor deterministic at
  β_MAP. Same textbook-equivalence argument as before (notes §7.4): this is
  Bayesian multi-class logistic regression with φ(x) = g_β(x) learned.
- **Check class balance before trusting any accuracy number.** Gait/exercise data
  from a fixed protocol is not guaranteed to be balanced across exercises or across
  subjects (recordings may differ in duration). If one exercise is underrepresented,
  a class-weighted loss (inverse frequency) is likely necessary, and **macro-averaged
  recall (balanced accuracy), not raw accuracy**, should be the primary source-val
  metric to watch — raw accuracy can look fine while a minority class is essentially
  unlearned. Verify this concretely rather than assuming it's fine or assuming it's
  broken.
- During target adaptation: update β (feature extractor), freeze θ (classifier
  head) — paper's setup (Fig. 3b). If both are frozen there is nothing to adapt and
  all ablation arms collapse to the same thing.
- Posterior q(θ) stays fixed at the β_MAP features throughout adaptation — an
  approximation, not an implementation detail, goes in Limitations (§10).
- Uncertainty recomputed every minibatch during adaptation (not frozen at the
  start) — features move as β moves.
- Down-weighting applies only to the conditional-entropy term of the IM loss,
  never to the diversity term L_div.
- Hyperparameters fixed a priori from the paper: γ = 0.5, τ = 0.4. Any deviation
  should be justified on a source validation split only, never by looking at target
  performance.
- No target labels used for model selection, ever — no early stopping on target
  accuracy, no LR chosen by looking at target ECE.
- One config file, all seeds, no hardcoded values scattered around.
- Multiple seeds (5 recommended) for every reported number — no single-seed result
  should be treated as final, especially for any correlation computed across only
  ~11 target domains.

---

## 1b. Conventions & bug traps to watch for

- **Prior precision convention.** The exact last-layer Hessian (notes §7.4) sums
  over data:
  $$ \mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + \sum_{n=1}^{N} s_n(1-s_n)\,\phi(x_n)\phi(x_n)^{T} $$
  If training uses `reduction='mean'` cross-entropy plus weight decay λ, the prior
  precision consistent with the summed Hessian is τ ≈ λ·N (with PyTorch's
  `weight_decay` convention specifically — re-derive if using a different framework
  or a different weight-decay convention). Do not assume τ = λ. Write the
  correspondence down explicitly and test it (see below) rather than trusting it by
  inspection — a wrong convention gives predictive variances that are silently
  wrong (too tight or too wide), not an error.
- **Validate the Hessian two ways before building anything on top of it:**
  1. Reduce to K=2 classes and check against the exact binary formula in the notes
     (§7.4) term by term.
  2. On a small synthetic problem (small N, small parameter count), compare the
     Laplace predictive distribution against real Metropolis-Hastings or HMC
     samples of the true posterior. This is the test that actually catches a wrong
     prior-precision convention — the binary-reduction check alone won't catch a
     scaling error that affects both sides equally.
- **Kronecker structure does not survive adding the prior naively.** If/when KFAC
  is implemented (§5), adding τI to a Kronecker-factored V ⊗ U breaks the
  factorization; add √τ to each factor's eigenvalues instead.
- **Class imbalance can break MAP training itself, not just the adaptation
  diversity term.** The v2 plan (written for UCI HAR) only flagged imbalance as an
  L_div concern during adaptation. On a smaller, protocol-driven dataset like this
  one, check whether an unweighted classifier fails to learn a minority class
  entirely before assuming the imbalance is a minor issue — verify per-class
  recall, not just overall accuracy, on the source validation set.
- **Per-subject sample counts may be very uneven** if recordings have different
  durations rather than a fixed protocol length. Report window counts alongside
  every per-subject number, and don't treat subjects with very different amounts of
  data as equally reliable point estimates.
- **Absolute normalization of the epistemic signal**, never per-batch. Per-batch
  standardization makes weights relative within a batch — even an unshifted target
  would have half its samples down-weighted, and weights stop being comparable
  across subjects/shift levels. Use log K (BALD's upper bound) or quantiles of BALD
  computed on a held-out source split.

---

## 2. Data pipeline

- [ ] Write the parser: handle the header quirks in §0, extract subject id / group
  / exercise from the filename, verify all 66 files parse and produce physiologically
  sane ranges (EMG in mV, knee angle in a plausible degree range).
- [ ] Windowing + feature extraction: 200ms non-overlapping windows, Hudgins
  time-domain features per EMG channel, goniometer summary stats. Save the
  resulting feature table.
- [ ] Check class balance (windows per exercise, per group, per subject) before
  doing anything else with the features.
- [ ] Sanity check before touching any model: PCA (fit on NORMAL only) of
  per-subject centroids, plus a source-vs-target scatter. Confirm NORMAL → ABNORMAL
  actually looks like a distribution shift rather than assuming it does. Also
  compute within-NORMAL leave-one-subject-out centroid distances as a "near-zero
  shift" reference to compare the ABNORMAL shift magnitudes against.
- [ ] Subject-wise train/val split within NORMAL (not window-wise) for source
  model selection.
- [ ] Shift proxy per target subject: at least centroid distance in feature space;
  ideally also AUROC of a source-vs-target domain classifier, checked for rank
  agreement with the centroid-distance proxy.
- [ ] Identify candidate hardest/easiest target subjects for the §6 semantic
  validation, and check whether any observed difficulty is a target-side class
  imbalance artifact or a genuine shift effect before using a subject as a clean
  test case.

---

## 3. Synthetic track (optional, for intuition and slides)

- [ ] 2D, 3-class toy replicating paper Fig. 4: mild vs strong shift, decision
  surfaces shaded by predictive certainty, epistemic/aleatoric maps. Useful as an
  "ideal case" reference to contrast against whatever the real data shows in §7 —
  do this before or alongside the real-data track, not as an afterthought.
- [ ] Semantic sanity check on the toy: a point far from all source data should be
  epistemic-dominant; a point near a decision boundary but inside the source
  support should be aleatoric-dominant. If this doesn't hold, there's a bug in the
  BALD implementation — find it here, on a controlled problem, before trusting it
  on real data.
- [ ] `make_classification()` sweep with ground-truth shift magnitude, if time
  allows — lower priority than the real-data track.

---

## 4. Source MAP training

- [ ] Small MLP sized to the actual feature dimension (see §1 — expect ~25-30
  input features, not 561), subject-wise source train/val split, class-weighted
  cross-entropy if §2's balance check shows it's needed.
- [ ] Track macro-averaged recall (balanced accuracy) on source val as the primary
  metric, not just raw accuracy.
- [ ] Evaluate on each ABNORMAL subject as a separate target domain; check whether
  the resulting accuracy range gives enough headroom to show an adaptation effect
  (a saturated near-ceiling baseline, or a collapsed near-chance baseline, both make
  the rest of the project hard to interpret).
- [ ] If one exercise class remains hard to classify even after balancing, consider
  whether there's a content-level explanation (e.g. a dynamic/cyclic movement vs.
  static held postures within a fixed-length window) before assuming it's purely a
  data or hyperparameter problem — check per-class feature variance as one way to
  test this.

---

## 5. Exact last-layer Laplace (reference implementation, do this before KFAC)

- [ ] Exact GGN Hessian of the last layer: for K classes and augmented feature dim
  D (features + 1 for the bias), $H = \sum_n \Lambda_n \otimes \phi_n\phi_n^T + \tau I$
  with $\Lambda_n = \mathrm{diag}(p_n) - p_n p_n^T$. With a small feature dimension
  here, this matrix should be small enough to build and invert directly — no need
  for Kronecker factorization for tractability reasons (see §6).
- [ ] Validation 1: reduce to K=2 and check against the binary formula in notes §7.4.
- [ ] Validation 2: compare against Metropolis-Hastings/HMC on a small synthetic
  problem with the same structure (see §1b).
- [ ] MC predictive distribution and BALD decomposition, with a convergence check
  (mean BALD vs. number of MC samples) before fixing S in the config.

---

## 6. KFAC as an object of study, not a necessity

With the small feature dimension expected here, the last-layer parameter count will
likely be small enough that Kronecker factorization buys nothing computationally —
Ritter et al. need it for a much larger bottleneck/class-count combination. So:

- [ ] Implement KFAC anyway, but frame it as "what does this approximation cost?"
  rather than "this is necessary." State explicitly that the exact sum does not
  factor into a single Kronecker product — KFAC adds an independence assumption.
- [ ] Fix the prior via √τ on each factor's eigenvalues, not τI added after
  factorizing.
- [ ] Measure the approximation error against the exact Hessian from §5: predictive
  variance, ECE, and — most relevant for the downstream weighting — Spearman/Kendall
  rank correlation of the BALD ordering, since the down-weighting only consumes the
  ranking, not the exact magnitude.

---

## 7. Uncertainty decomposition + semantic validation

- [ ] BALD epistemic/aleatoric split, unit-tested against MCMC (§5) before trusting
  it on real data.
- [ ] Write down explicitly (for the report, and to avoid confusing the two in
  code) that the IM loss's marginal/conditional entropy terms (§9) and BALD have
  the same functional form H[p̄] − H[p] but average over different things — data vs.
  weights. Two different mutual informations.
- [ ] Semantic validation on real data: check whether the hardest target subject
  (by MAP accuracy) shows elevated epistemic uncertainty relative to an easy target
  subject and relative to the source validation set. Also check whether the
  hardest-to-classify exercise class (if one exists after §4) shows elevated
  *aleatoric* (not epistemic) uncertainty in-distribution — that would support a
  genuine-ambiguity explanation over a shift explanation for that class.
- [ ] Absolute normalization of the epistemic signal (source quantiles or log K),
  never per-batch.
- [ ] Check whether the temperature τ (paper default 0.4) meaningfully changes the
  balance between epistemic and aleatoric signal on this data — don't just inherit
  the paper's value without checking what it does here.

---

## 8. Calibration check

- [ ] Reliability diagrams, MAP vs Laplace, source and every target subject.
- [ ] ECE, NLL, Brier score — MAP vs Laplace, multi-seed mean ± std, not
  single-run numbers.
- [ ] AUROC for misclassification detection (does the uncertainty separate correct
  from incorrect predictions?).
- [ ] Accuracy-vs-coverage curve (reject predictions above an epistemic-uncertainty
  threshold) — the operational metric for the clinical/assistive framing.
- [ ] Spearman correlation between shift proxy (§2) and mean epistemic uncertainty
  across target subjects, with multiple seeds and a converged MC sample count
  before treating any single correlation value as final — with only ~11 target
  domains, a single-seed estimate should be expected to be noisy.
- [ ] Plots: ECE vs. shift proxy, epistemic uncertainty vs. shift proxy with seed
  bands, reliability diagrams at low/mid/high shift, accuracy-vs-coverage.

**Check before moving on:** does epistemic uncertainty correlate with the shift
proxy across target subjects, and is the Laplace-based model better calibrated than
MAP on the more-shifted subjects? If the signal is weak or noisy even after
multiple seeds, first rule out an implementation issue via §5's validations (if
those pass on synthetic data, the machinery itself is trustworthy) before
concluding the real-data effect is genuinely small — both are legitimate, but they
support different conclusions in the report.

---

## 9. Adaptation + ablation (6 arms)

- [ ] IM objective (not plain entropy minimization, which is known to collapse to
  one class):
  $$ \mathcal{L}_{\text{IM}} = \underbrace{H\Big[\tfrac{1}{N}\sum_i p(y \mid x_i)\Big]}_{\text{marginal, maximize}} \;-\; \underbrace{\tfrac{1}{N}\sum_i H[p(y \mid x_i)]}_{\text{conditional, minimize}} $$
- [ ] Down-weighting multiplies only the conditional-entropy term.
- [ ] Same seeds/data/optimizer/schedule across all arms — only the objective
  changes:
  - (a) no adaptation (reuse frozen-model numbers from §4/§8)
  - (b) entropy only — expect collapse; watch the predicted class histogram
  - (c) full IM, no weighting = SHOT-IM
  - (d) IM + weight from MAP predictive entropy (no Bayes) — the crucial control;
    without this arm there's no evidence the gain (if any) comes from epistemic
    uncertainty specifically rather than re-weighting per se
  - (e) IM + weight from Laplace predictive entropy = paper-faithful U-SFAN
  - (f) IM + weight from standardized BALD epistemic component — a deviation from
    the paper (which uses total predictive entropy), label it as such
- [ ] Given a class imbalance in the source data (§2), consider whether L_div's
  push toward a uniform batch marginal is appropriate here, or whether a
  source-estimated class prior should replace it — report the vanilla version too
  as a confounder either way.
- [ ] With only 11 target domains, think about whether a per-shift-tercile
  breakdown is statistically meaningful at this sample size, or whether a
  continuous regression (accuracy/ECE vs. shift proxy) is more honest than binning.
- [ ] Log trajectories per arm (target accuracy, ECE, marginal entropy, predicted
  class histogram); reliability diagrams post-adaptation, all arms; multiple seeds,
  mean ± std.

**Check:** does (b) collapse while (c) doesn't? Does (e) beat (d)? Does (e) or (f)
beat (c), especially on the more-shifted target subjects? If not, suspects in
order: weighting scale/normalization, γ balance, adaptation LR/schedule, or the
L_div/imbalance interaction.

---

## 10. Open-set — needs a decision before starting

BASAN has no postural-transition data and no per-subject etiology file is present
in this repo, so the original (UCI HAR/HAPT-based) open-set plan does not port over.
Options, in order of preference:

- [ ] Check whether per-subject etiology (which of the 11 ABNORMAL subjects has
  ACL vs. meniscus vs. sciatic nerve injury) can be recovered from the original
  dataset documentation or associated publications. If yes, this is the most
  clinically meaningful open-set structure: train source excluding one etiology,
  treat it as target-private.
- [ ] If etiology can't be recovered, consider dropping open-set entirely and
  noting it as future work requiring that metadata.
- [ ] Leave-one-exercise-out as a last-resort substitute (train on 2 of 3
  exercises, treat the 3rd as target-private) — flag clearly that this conflates a
  label-space shift with the healthy/pathological shift, so it's a weaker and less
  clean setting than the other two options.

---

## 11. Report & presentation

- [ ] Anchor explicitly to course notes: §7.3 (Laplace approximation, and its
  local/unimodal caveats), §7.4 (Bayesian logistic regression — the last-layer LA
  *is* this, with learned features), §2.3 (entropy/KL/mutual information — both the
  IM loss and BALD live here), §10.6 (Bayesian NNs, predictive as model averaging),
  ch. 8 (sampling, used as MCMC ground truth for the Laplace validation).
- [ ] Slide on why last-layer-only: tractability + the textbook-equivalence
  argument.
- [ ] Limitations section, to include at minimum:
  - Posterior staleness (q(θ) fixed at β_MAP while β moves during adaptation).
  - LA is local and unimodal — no guarantee about the rest of the posterior.
  - Any deviation from the paper (e.g. arm (f) in §9).
  - Dataset-specific limitations: single joint (knee), fixed EMG electrode
    placement that may not match a given exoskeleton's sensors, subject sex/age
    coverage limited to whatever BASAN actually documents, the ABNORMAL population's
    etiologies may not represent the population a real assistive device would see,
    offline windowed classification rather than causal real-time, no cost asymmetry
    between confusion types, small number of target domains (11) limiting the
    statistical power of any shift-vs-uncertainty correlation.
  - Whichever open-set option was taken in §10, and why.
- [ ] State the model-selection protocol (no target labels used, ever) as one
  explicit sentence.

---

## 12. Eventually / stretch

- [ ] Full-network KFAC vs. last-layer-only.
- [ ] MC dropout / small deep ensemble comparison against the Laplace approach —
  check empirically which of the paper's stated advantages of LA over MC dropout
  actually hold in this setting.
- [ ] Since raw EMG is available (unlike UCI HAR's pre-engineered features), a
  learned feature extractor (e.g. small 1D-conv over raw windows) instead of the
  hand-engineered Hudgins set is a natural extension — worth comparing whether a
  learned representation changes the strength of any shift-vs-uncertainty
  correlation found in §8.
- [ ] Cost-sensitive rejection reflecting that not all confusions are equally
  costly in a clinical/assistive setting.