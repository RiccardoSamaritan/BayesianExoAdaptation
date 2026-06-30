# Project TODO

## Baseline / MAP training

- [ ] One config file for all seeds, no hardcoded values scattered around
- [ ] Data gen wrapper around `make_classification()` — need to expose: n_informative/redundant/repeated, n_classes, class_sep, clusters_per_class, flip_y
- [ ] Domain shift generator on top of that. Probably translation vector with a scalar magnitude knob, maybe also try covariance scaling / rotation if translation alone looks too easy
- [ ] Sweep over shift magnitudes (5-10 steps, none -> large). Don't just pick one shift value and call it a day
- [ ] Plot source vs target before touching any model — PCA or raw scatter if low-dim. Sanity check that shift actually looks like a shift
- [ ] Small MLP, 2-3 hidden layers, nothing too difficult
- [ ] Gaussian prior on weights -> note down the precision used, it comes back later in the Hessian (see KFLA section, don't forget this)
- [ ] Train on source only, cross-entropy loss + weight decay, to MAP
- [ ] Save: MAP weights, loss/acc curves, last-hidden-layer features for both source and target (needed as design matrix for the Hessian)
- [ ] Baseline accuracy + confusion matrix, source and across shift levels, just as a non-Bayesian reference point before getting into the Bayesian stuff

## KFLA

Decision: last-layer only, not full network. Feature extractor frozen at MAP. 
Reasons: tractable Hessian, MC sampling is cheap since we only sample the last layer. Important to write this reasoning in some presentation slide.

- [ ] Hessian of neg-log-posterior at MAP, last layer only. Sum of outer products of last-layer features weighted by softmax/Bernoulli variance at MAP, + prior precision on the diagonal
- [ ] Kronecker factorization — split into the two smaller matrices (activation covariance / pre-activation gradient covariance) instead of forming the full thing
- [ ] Invert the two factors separately, way cheaper than inverting the full Hessian
- [ ] q(w) = N(w_hat, H^-1) from the above
- [ ] MC sampling from q(w), S=50-200ish, use the matrix-normal trick (sample the two factors, combine via Kronecker structure — don't sample from the full joint, no need)
- [ ] Predictive dist = average softmax over the S samples (Bayesian model averaging)
- [ ] Split total uncertainty into epistemic/aleatoric via BALD:
  $$ I[y, w \mid x] = H\Big[\mathbb{E}_{q(w)}\, p(y \mid x, w)\Big] - \mathbb{E}_{q(w)}\Big[H[p(y \mid x, w)]\Big] $$
  entropy of the averaged prediction minus expected entropy of per-sample predictions = epistemic
- [ ] Sanity check the decomposition on a couple of toy points — far from training data should be epistemic-heavy, near a boundary but in-distribution should be aleatoric-heavy
- [ ] Standardize the epistemic signal before it goes anywhere near an exponential later (per-batch or temperature). Otherwise it'll just saturate to 0/1 and the down-weighting becomes pointless


## Calibration check

- [ ] Reliability diagrams, MAP vs KFLA
- [ ] ECE (Expected Calibration Error), same comparison, source + every shift level
- [ ] Log per shift level: MAP acc/ECE, KFLA acc/ECE, mean epistemic variance on target
- [ ] Spearman correlation between shift magnitude and epistemic variance, across the sweep
- [ ] Multiple seeds (5-10), report mean ± std, not single runs
- [ ] Plots: ECE vs shift (MAP+KFLA same axes), variance vs shift with seed bands, a few reliability diagrams at low/mid/high shift

Check before moving on: does variance go up with shift (or at least correlate strongly), and is KFLA better calibrated than MAP on shifted data? If yes -> next section. If no, suspects are: variance scale/standardization, MAP under/overfit, shift too weak/too strong to see anything, or a bug somewhere in the Kronecker math.

## Adaptation + ablation

- [ ] IM objective, not plain entropy min (plain entropy collapses to one class, known failure mode):
  $$ \mathcal{L}_{\text{IM}} = \underbrace{H\Big[\tfrac{1}{N}\sum_i p(y \mid x_i)\Big]}_{\text{marginal, maximize}} \;-\; \underbrace{\tfrac{1}{N}\sum_i H[p(y \mid x_i)]}_{\text{conditional, minimize}} $$
  marginal term = entropy of the batch-averaged prediction, conditional term = the usual per-sample entropy
- [ ] Down-weighting: multiply each sample's gradient contribution by e^(-sigma^2(x)), sigma^2 = standardized epistemic uncertainty from KFLA
- [ ] Decide: recompute uncertainty on the fly during adaptation, or freeze it at the start? Either is fine but pick one and say why (responsiveness vs stability/cost tradeoff)
- [ ] 4-arm ablation, same seeds/data/optimizer everywhere, only the objective changes:
  - (a) no adaptation (reuse the frozen model numbers from before)
  - (b) entropy-only — expect collapse here, watch the predicted class distribution to confirm
  - (c) full IM, no down-weighting
  - (d) full thing — IM + down-weighting
- [ ] Log trajectories per arm: target acc, ECE, marginal entropy over training
- [ ] Reliability diagrams post-adaptation, all 4 arms
- [ ] Multiple seeds again, mean ± std on final acc/ECE
- [ ] Summary table/bar chart, all 4 arms, make the ranking obvious. Separate panel showing (b) collapsing would help

Check: does (b) actually collapse while (c) doesn't, and does (d) beat (c) on ECE and/or accuracy, especially at higher shift? If yes, core result is basically done. If no — look at the down-weighting scale, maybe need a weighting hyperparameters between the marginal/conditional terms, or just the LR/schedule for adaptation.

---

## Real data

- [ ] Get + preprocess UCI HAR. Already feature-engineered so shouldn't need much beyond normalization
- [ ] Subject-based source/target split, try to pick subjects where something is plausibly different between groups (intensity, device placement, whatever) to mimic a real shift
- [ ] No ground truth shift magnitude here, so need a proxy — centroid distance between subject groups, or accuracy of a domain classifier, something along those lines
- [ ] Re-run MAP + KFLA + calibration check on this
- [ ] Re-run the ablation
- [ ] Write up whether the synthetic conclusions hold here too, or where they break down — even negative results are worth reporting honestly

## Eventually

- [ ] Full-network KFLA instead of last-layer-only, compare against the last-layer version