# Project TODO

Uncertainty-guided SFDA (U-SFAN replication + extension), applied to **real** sEMG-based
exercise/locomotion-mode recognition, source = healthy subjects, target = subjects with
diagnosed knee pathologies. This is v3: v2 was written against UCI HAR/HAPT (IMU,
pre-engineered features, surrogate exoskeleton framing) before the dataset changed to
BASAN. Everything below reflects the real dataset, real code already in the repo, and
real numbers already obtained — not a plan written in the abstract.

Status legend: [x] done and checked, [~] done but flagged as provisional/noisy,
[ ] not started.

---

## 0. Locked decisions

- [x] **Dataset: BASAN sEMG** (UCI ML repo #278). 22 male subjects, 4 sEMG channels
      (RF, BF, VM, ST) + 1 goniometer channel (FX, knee flexion angle), sampled at
      1000 Hz, 14-bit. 11 NORMAL (healthy) + 11 ABNORMAL (6 ACL, 4 meniscus, 1 sciatic
      nerve injury — per-subject mapping not available in our files, see §9).
- [x] **Task: 3-class exercise recognition** — `WALKING` (marcha), `KNEE_FLEXION_STANDING`
      (depie), `LEG_EXTENSION_SEATED` (sentado). This is not a surrogate for anything —
      it's directly the intent-recognition problem a lower-limb assistive device would
      need to solve, on real EMG.
- [x] **Source = NORMAL pool, target = ABNORMAL, one target domain per subject** (11
      target domains, not up to 30 like the old HAR plan — smaller dataset, smaller n
      for any correlation across domains, keep this in mind for statistical claims).
      This replaces the old "surrogate exoskeleton" framing: NORMAL→ABNORMAL is a real,
      clinically meaningful distribution shift, not a stand-in for one.
- [x] **No synthetic shift on real data.** Subject-wise splits only, never window-wise.
      Standardization (`StandardScaler`) fit on NORMAL only, applied to ABNORMAL —
      done in `experiments/train_map.py`.
- [x] **Windows: 200 ms (200 samples @ 1000 Hz), non-overlapping.** Chosen from the
      start (not retrofitted like the HAR overlap fix), so the i.i.d. assumption behind
      the Hessian sum (§0b) holds without a correction step.
- [x] **27 hand-engineered features**, not 561 like HAR: classic Hudgins time-domain set
      (MAV, RMS, WL, ZC, SSC, VAR) × 4 EMG channels = 24, plus 3 goniometer summary
      stats (mean angle, range, mean |velocity|). Implemented in `data/basan_features.py`.
- [x] **Architecture rescaled accordingly**: `27 → 32 → 16 → 3` (config.py), not
      `561 → 128 → 64 → K`. Last-layer augmented params: K·(F+1) = 3·17 = 51, even
      smaller than the HAR estimate of 195 — makes the "KFAC is unnecessary here,
      it's an object of study" argument in §5 even stronger.
- [x] **Bayesian treatment: last layer only.** Feature extractor deterministic at
      β_MAP. Same textbook-equivalence argument as before (notes §7.4): this is
      Bayesian multi-class logistic regression with φ(x) = g_β(x) learned.
- [x] **Class-weighted loss is NOT optional here — it's load-bearing.** NORMAL source
      pool is imbalanced (WALKING ≈ 17% of train windows vs ≈ 46% for
      LEG_EXTENSION_SEATED). Confirmed empirically: unweighted CE gives WALKING recall
      ≈ 0.31 (chance-level, three-way); inverse-frequency weighting brings it to 0.557.
      **Primary source-val metric is macro-averaged recall (balanced accuracy), not
      raw accuracy** — raw accuracy is misleading under this imbalance and actually
      *drops* when the fix is applied (0.766 → 0.697) because it stops getting a free
      ride from majority classes.
- [ ] During target adaptation: update β, freeze θ (unchanged from v2, not yet
      implemented for this dataset — §8).
- [ ] Posterior q(θ) stays fixed at β_MAP features during adaptation (unchanged
      approximation, goes in Limitations, §10).
- [ ] Uncertainty recomputed every minibatch during adaptation (§8, not yet built).
- [ ] Weight applies only to the conditional-entropy term, never L_div (§8).
- [x] **Hyperparameters**: γ = 0.5, τ = 0.4 (paper values). τ = 0.4 is *not* just
      inherited without checking — confirmed empirically on real target subjects that
      τ = 0.4 exposes more epistemic signal than τ = 1.0 (mean epistemic fraction
      0.044 vs 0.012 across target subjects). Worth a sentence in the report: the
      paper's choice does something real here, not just cosmetic.
- [ ] No target labels used for model selection — holds so far (early stopping and
      class weights are computed on source only); keep enforcing this in §8.
- [x] One config file (`config/config.py`), all seeds, no hardcoded values scattered
      around.
- [ ] Seeds: currently only seed 0 has been run end-to-end. **5 seeds minimum before
      any correlation number goes in the report** — see §7, the single-seed Spearman
      estimate was unstable (−0.42 to −0.65 depending on temperature/MC sample count).

---

## 0b. Conventions & bug traps

- [x] **Prior precision convention validated, not just assumed.** τ_prior =
      weight_decay × N_source. Validated two ways in `tests/`:
      1. `test_hessian_binary.py` — the K-class softmax Hessian reduces to the exact
         binary logistic-regression formula from notes §7.4 (max abs diff ~3.5e-15).
      2. `test_laplace_vs_mcmc.py` — Laplace predictive (total entropy, epistemic,
         aleatoric) agrees with true Metropolis-Hastings posterior samples on a small
         problem (K=3, 9 params) within tolerance (mean |diff| < 0.03 nats).
      Both pass. Convention is documented directly in `laplace/hessian.py`'s docstring.
- [x] **Overlap avoided by construction**, not corrected after the fact — 200ms
      windows are non-overlapping from the first line of `basan_features.py`.
- [ ] **Kronecker + prior interaction** (√τ on eigenvalues, not τI added post-hoc) —
      not yet relevant since KFAC (§5) hasn't been implemented; exact Hessian is used
      throughout so far because at 51 params it's trivial to invert directly.
- [ ] **Absolute normalization of the epistemic signal** (source quantiles, not
      per-batch) — not yet needed since down-weighting (§8) hasn't been built, but
      keep the rule when it is.
- [x] **New bug trap specific to this dataset: class imbalance breaks MAP training
      itself, not just L_div.** The v2 TODO only flagged imbalance as an L_div
      concern during adaptation. On BASAN it's more fundamental — an unweighted MAP
      classifier fails to learn the minority class at all. Anything built on top
      (Hessian, BALD, adaptation) inherits a broken feature extractor if this isn't
      fixed first. Fixed in `experiments/train_map.py` via inverse-frequency
      `CrossEntropyLoss(weight=...)`; re-verify after every architecture change.
- [ ] **Per-subject window count is highly uneven** (187 to 820 windows per ABNORMAL
      subject — a 4.4× range) because recordings have different durations, not
      because of a fixed protocol length. This means per-subject accuracy estimates
      have very different sample sizes; report confidence intervals or at least
      window counts alongside every per-subject number, and don't treat a subject
      with 187 windows and one with 820 as equally reliable point estimates.

---

## 1. Data pipeline — DONE, documented here for the report

- [x] **Parser** (`data/basan_parser.py`): handles the 6-line metadata header (File
      Name + 5× Channel description lines), CRLF line endings, tab-delimited data.
      Robust to a dataset quirk found during implementation: 9 of 66 files have a 6th
      "Digitals combined" column (verified constant, e.g. always 31 — an event marker
      with no information) which is detected and dropped rather than hardcoded away.
      Filename encodes subject id, group (A/N), exercise (mar/pie/sen) via regex.
- [x] All 66 files parse correctly (11 subjects × 3 exercises × 2 groups, no missing
      files). Verified duration (15–25s per recording), EMG range (mV), knee angle
      range (up to ~105°) are all physiologically sane.
- [x] **Feature extraction** (`data/basan_features.py`): 200ms non-overlapping
      windows → 6592 total windows, 27 features. Hudgins TD set per EMG channel,
      3 summary stats for the goniometer channel.
- [x] **Sanity check before touching any model** (`experiments/basan_sanity_check.py`):
      PCA (fit on NORMAL only) of per-subject centroids. **Honest finding: NORMAL and
      ABNORMAL centroids partially overlap** — ABNORMAL-vs-source-mean distances range
      1.98–5.05, NORMAL-internal (leave-one-out) distances range 1.01–3.41. The ranges
      overlap. This is reported as-is: the shift is a continuum tied to pathology
      severity/type, not a clean binary separation. This is a *more* honest and
      arguably more realistic finding than a clean cluster split would have been —
      say so explicitly in the report rather than treating it as a problem to fix.
- [x] Class balance logged (§0). WALKING underrepresented in NORMAL source pool.
- [ ] **Shift proxy, second method.** Centroid distance is implemented; still need
      an AUROC-of-a-domain-classifier proxy (source-vs-target logistic regression on
      the 27 features) to check rank agreement with centroid distance, per the
      original multi-proxy-agreement idea. Cheap to add, do it before §7's
      shift-vs-uncertainty correlation is treated as final.
- [x] Subject-wise split enforced for source train/val (`subject_split()` in
      `train_map.py`), not window-wise.
- [x] Hardest / easiest target subjects identified for §6: subject 2 (MAP acc 0.364,
      near chance) and subject 3 (MAP acc 0.847). Checked subject 2's low accuracy is
      **not** a target-side class-imbalance artifact (its own exercise counts are
      55/64/68, fairly balanced) — so the difficulty is a genuine shift/pathology
      effect, which is what makes it a clean test case for §6.

---

## 2. Synthetic track — DONE

- [x] 2D, 3-class toy replicating paper Fig. 4 (`experiments/toy_fig4.py`). Mild and
      strong shift both reproduce the paper's qualitative story: under strong shift,
      target points fall under the wrong MAP decision region ("flipped" boundary,
      matching Fig. 4b); epistemic uncertainty (BALD) forms the expected V-shaped
      bands radiating away from the source blobs (matches Fig. 2b's OOD-detection
      story); aleatoric uncertainty concentrates at in-distribution decision
      boundaries.
- [x] Semantic validation on the toy: far-from-source point → 76% of total
      uncertainty is epistemic; near-boundary in-distribution point → 14% epistemic
      (86% aleatoric). Matches the expected pattern exactly.
- [ ] `make_classification()` sweep with ground-truth shift magnitude — not done,
      lower priority now that real data is the main track and already shows a
      (noisy) real correlation. Keep as an appendix figure if time allows, mainly
      useful as an "ideal case" contrast against the real-data noise in §7.

---

## 3. Source MAP training — DONE for seed 0, needs multi-seed

- [x] `experiments/train_map.py`: subject-wise train/val split within NORMAL,
      class-weighted cross-entropy, early stopping on val loss.
- [x] Seed 0 result: source val macro-recall 0.790 (WALKING 0.557, the other two
      ≈0.90–0.91). Target (ABNORMAL) subject accuracy: mean 0.688, range
      0.364–0.847, std 0.130.
- [x] **No saturation problem** — opposite situation from the original UCI HAR
      concern. Plenty of headroom to show an adaptation effect; the risk here is the
      opposite one (task hard enough that the *baseline* needs to be trustworthy
      before building the Bayesian story on top).
- [ ] Run 5 seeds, report mean ± std for source val macro-recall and per-subject
      target accuracy. Single-seed numbers above are provisional.
- [ ] Open question for the report, not yet resolved: WALKING remains the hardest
      class even after re-weighting (recall 0.557 vs ~0.90 for the static exercises).
      Plausible physiological explanation to state explicitly: WALKING is the only
      *dynamic, cyclic* movement in the protocol — a fixed 200ms window can land in
      different phases of the gait cycle, adding within-class variability that
      KNEE_FLEXION_STANDING and LEG_EXTENSION_SEATED (held static postures) don't
      have. This is a content-level limitation, not obviously a bug — but confirm by
      checking per-window feature variance for WALKING vs the other two classes
      before asserting it in the report.

---

## 4. Exact last-layer Laplace — DONE and validated

- [x] `laplace/hessian.py`: exact GGN Hessian for the K-class softmax last layer,
      self-contained augmented-weight convention (bias folded into φ via a constant
      column), documented and asserted prior-precision convention.
- [x] **Validation 1** (`tests/test_hessian_binary.py`): K=2 case reduces to the exact
      binary formula in notes §7.4. Passes at machine precision.
- [x] **Validation 2** (`tests/test_laplace_vs_mcmc.py`): compared against real
      Metropolis-Hastings sampling (K=3, 9 params, tuned to ~33% acceptance rate).
      Laplace total/epistemic/aleatoric entropy all agree with MCMC within
      tolerance. This is the test that would have caught a wrong τ_prior — it didn't
      fire, so the convention in §0b is trustworthy, not just plausible.
- [x] Applied to the real trained model (`experiments/laplace_on_real_data.py`):
      posterior fit on 1823 source windows, 51 augmented params, Hessian condition
      number 6.3e3 (healthy, no near-singularity).
- [ ] Convergence check (mean BALD vs number of MC samples) — not yet run on real
      data; do this before finalizing S in the config.

---

## 5. KFAC as an object of study

Even more clearly unnecessary here than in the original HAR plan: 51 total augmented
last-layer parameters. The exact Hessian is a 51×51 matrix, inverted directly in
milliseconds. Framing is unchanged from v2 — implement KFAC anyway and measure what
it costs in accuracy, not because it's needed.

- [ ] Not started. Same plan as v2 §5: implement the Kronecker approximation, note
      explicitly that the exact sum does not factor (KFAC adds an independence
      assumption), fix the prior via √τ on eigenvalues, and measure predictive
      variance / ECE / BALD-ranking (Spearman/Kendall) error against the exact
      Hessian already implemented in §4.

---

## 6. Uncertainty decomposition — partially validated, needs stabilizing

- [x] BALD decomposition implemented and unit-tested against MCMC (§4).
- [x] **Real-data semantic check, first pass** (`experiments/laplace_on_real_data.py`):
      computed epistemic fraction per target subject. Result: **direction is
      correct but the effect is subtle.**
      - Hardest subject (2, acc 0.364): epistemic fraction 0.058
      - Easiest subject (3, acc 0.847): epistemic fraction 0.032
      - Source val (in-distribution reference): epistemic fraction 0.029
      - Spearman(MAP accuracy, epistemic fraction) across 11 target subjects:
        rho ≈ −0.5 to −0.65 depending on temperature and MC sample count (single
        seed, n=11 — **not stable yet**, see below).
      - Epistemic fraction is small everywhere (3–8% of total uncertainty); most of
        the predictive uncertainty on this dataset is aleatoric, at every shift
        level including the hardest subject. This is a real and reportable finding
        in itself — the per-subject shift here (same protocol, same sensors, healthy
        vs. pathological knee) is much milder than the synthetic toy's deliberately
        large translation, so a much smaller effect size is actually expected, not
        a sign the machinery is broken (confirmed working via §4's validations).
      - τ = 0.4 exposes more epistemic signal than τ = 1.0 (mean epistemic fraction
        0.044 vs 0.012) — worth keeping as an explicit ablation in the report rather
        than a footnote.
- [ ] **Not yet done, blocking before this becomes a reported result:** re-run with
      (a) more MC samples (convergence-checked per §4, not just the config default),
      (b) multiple seeds for the MAP model itself, (c) report the Spearman
      correlation with a confidence interval or at least the range across seeds,
      not a single point estimate. The single-seed number swung from −0.42 to −0.65
      just from changing temperature/sample count — that swing needs to be
      characterized, not hidden behind one reported value.
- [ ] The old UCI-HAR-specific validation (SITTING vs STANDING as the
      aleatoric-dominant in-distribution confusion) doesn't transfer — no equivalent
      static-posture confusion pair exists in this 3-class task. Replacement
      candidate: check whether **WALKING** (the hardest, most variable class per §3)
      shows elevated *aleatoric* (not epistemic) uncertainty specifically on
      in-distribution (NORMAL val) data — that would support the "WALKING is
      harder because of genuine within-class variability, not distribution shift"
      explanation from §3.
- [ ] Absolute normalization of epistemic signal (source quantiles) — not yet
      needed until §8 is built, but decide before then.

---

## 7. Calibration check — not started with proper rigor

Everything here is unchanged in spirit from v2, but nothing has been run yet beyond
the provisional single-seed numbers in §6.

- [ ] Reliability diagrams, MAP vs LA, source and each target subject.
- [ ] ECE, NLL, Brier — MAP vs LA, multi-seed mean ± std.
- [ ] AUROC for misclassification detection (does uncertainty separate correct from
      incorrect predictions?).
- [ ] Accuracy-vs-coverage curve (reject on high epistemic uncertainty) — the
      operational metric for the clinical/assistive framing, and now more literally
      justified than in the HAR plan: this is actually healthy-vs-pathological data.
- [ ] Second shift proxy (domain-classifier AUROC, §1) and rank agreement with
      centroid distance.
- [ ] **Formal multi-seed Spearman correlation** between shift proxy and mean
      epistemic uncertainty — supersedes the provisional §6 numbers.
- [ ] Plots: ECE vs shift proxy, epistemic-vs-shift-proxy with seed bands,
      reliability diagrams at low/mid/high shift tercile, accuracy-vs-coverage.

**Check before moving on:** with 5 seeds and a converged MC sample count, does the
Spearman correlation stabilize to something reportable (consistent sign, reasonable
CI), or does it stay noisy? If it stays noisy even after that, the honest conclusion
may be "the effect is real but small on this dataset, unlike the synthetic case" —
that is a legitimate and reportable finding, not a failure, provided §4's validations
(which passed) rule out an implementation bug as the explanation.

---

## 8. Adaptation + ablation (6 arms) — not started

Unchanged in structure from v2 (see arms (a)–(f) and the isolation logic — (c)→(d)
isolates re-weighting, (d)→(e) isolates the Bayesian treatment, (e)→(f) isolates
which uncertainty). Dataset-specific notes:

- [ ] Class-imbalance interaction with L_div (§0b in v2) applies here too, and is
      now confirmed to matter even more than expected (§0): consider a source-
      estimated class prior instead of uniform in L_div, given how skewed WALKING is.
- [ ] Given only 11 target domains (not ~30), consider whether per-shift-tercile
      breakdown (low/mid/high) is statistically meaningful at this n, or whether a
      continuous regression (accuracy/ECE gain vs. shift proxy) is more honest than
      binning into terciles of ~3-4 subjects each.
- [ ] Everything else (freeze θ / update β, recompute uncertainty per minibatch,
      weight only the conditional-entropy term, fixed γ/τ) — carries over unchanged
      from v2 §8, not yet implemented.

---

## 9. Open-set — needs a new plan, HAPT-based idea does not apply

- [x] **Confirmed: BASAN has no postural-transition data** (per the dataset's own
      documentation) and **no per-subject injury-etiology file** is present in this
      repo (checked — only the two `A_TXT`/`N_TXT` folders and the raw signal files
      exist; the 6/4/1 ACL/meniscus/sciatic breakdown is only available in the
      literature, not per-subject-ID here). The v2 HAPT-transitions plan cannot be
      ported as-is.
- [ ] **Decision needed** on which replacement, if any:
      (a) Drop open-set entirely, note it as future work requiring access to the
          per-subject etiology labels (e.g. by contacting the dataset authors or
          finding a version of BASAN with subject-level metadata).
      (b) Leave-one-exercise-out as a weaker substitute: train source on 2 of the 3
          exercises, treat the 3rd as target-private in the ABNORMAL target set.
          Not the same phenomenon (it's a label-space shift within a healthy-vs-
          pathological shift, conflating two things at once), but exercises fastest
          to implement with existing code.
      (c) If per-subject etiology can be obtained from the original paper/dataset
          documentation (worth 15 minutes of checking before giving up on it): use
          etiology as the OOD structure, e.g. train source excluding sciatic-nerve
          subjects, treat that etiology as target-private. This is the most
          clinically meaningful version if the labels can be found.
      Recommendation: try (c) first (cheap to check), fall back to (a) if the
      per-subject mapping isn't recoverable, treat (b) as a last resort since it
      muddies the shift being studied.

---

## 10. Report & presentation

- [x] Anchoring to course notes (unchanged from v2 — §7.3, §7.4, §2.3, §10.6, ch. 8)
      still fully applies; the validations in §4 are now the concrete artifact that
      makes "ch. 8 sampling used as ground truth" literally true rather than aspirational.
- [ ] **Limitations section — substantially different from v2, write fresh:**
      - Posterior staleness and LA's local/unimodal nature — unchanged from v2.
      - **The exoskeleton framing is stronger here than in the HAR plan** (real
        pathological subjects, not healthy-subjects-only), but new limitations
        specific to this dataset: single joint (knee) rather than whole-limb
        multi-joint intent recognition; 4 fixed EMG electrode sites that may not
        match a given exoskeleton's sensor placement; all subjects male (per BASAN
        documentation), so no claims about generalization across sex; the ABNORMAL
        population (ACL/meniscus/sciatic nerve) doesn't necessarily represent the
        typical exoskeleton user population (e.g. stroke, spinal cord injury);
        offline windowed classification, not causal real-time; no cost asymmetry
        between confusion types; small n (11 target domains) limits the statistical
        power of any shift-vs-uncertainty correlation claim (§7).
      - State plainly if §7's correlation stays weak-but-consistent after
        stabilization: the honest interpretation is that inter-subject shift in this
        dataset is real but mild relative to the synthetic case, not that the method
        doesn't work — supported by the fact that the same machinery, validated
        against MCMC in §4, shows the large expected effect on synthetic data.
      - Open-set section's fate (§9) — state clearly which option was taken and why.
- [ ] Model-selection protocol sentence — unchanged requirement from v2.

---

## 11. Eventually / stretch

- [ ] Full-network KFAC vs last-layer (unchanged from v2).
- [ ] MC dropout / small ensemble comparison (unchanged from v2).
- [ ] Given real raw EMG is already available (unlike UCI HAR's pre-engineered
      features), a stretch worth more here than in v2: compare the 27 hand-engineered
      Hudgins features against a learned 1D-conv feature extractor on raw sEMG
      windows, and check whether the epistemic-uncertainty-vs-shift correlation
      strengthens with learned features — directly tests whether §7's weak signal is
      a feature-engineering ceiling rather than a property of the shift itself.
- [ ] Cost-sensitive rejection (unchanged from v2, arguably more defensible now
      given the genuinely clinical population).