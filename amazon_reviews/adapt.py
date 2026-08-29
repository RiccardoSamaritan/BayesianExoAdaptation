"""Laplace fit + MC-convergence check + BALD decomposition + 2-arm
source-free adaptation for the amazon_reviews sentiment pipeline --
extracted from the cells originally written inline in
`sentiment_calibration.ipynb` (single seed) so that both it and
`sentiment_multiseed.ipynb` (5 seeds) call the same code instead of each
carrying its own copy.

Depends on `code_v2.src.bayesian_model` (LastLayerLaplace, extract,
augment, head_weights) and `code_v2.src.digits_adapt` (adapt_target,
already verified compatible with `MLPClassifier`'s
`.features(x)`/`.g`/`.h` interface -- no changes made there).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from code_v2.src.bayesian_model import LastLayerLaplace, augment, extract, head_weights  # noqa: E402
from code_v2.src.digits_adapt import adapt_target  # noqa: E402

M_VALUES = [50, 100, 250, 500, 1000, 2000, 3000, 4000]
M_REFERENCE = 5000
RELATIVE_THRESHOLD = 0.01
ABSOLUTE_THRESHOLD_FRAC = 0.02
STABILITY_WINDOW = 3
CONVERGENCE_RNG_SEED = 123
PREDICTIVE_FINAL_SEED = 456

ARM_WEIGHT_MODE = {"shot_im": "none", "u_sfan": "uncertainty"}
ADAPT_BASE_KWARGS = dict(gamma=0.5, temperature=0.4, lr=1e-2, M=100)


def extract_features(model, X: torch.Tensor, y: torch.Tensor, batch_size: int = 256) -> tuple:
    """`extract()` wants a DataLoader; this is the (X, y) tensor -> (Phi, y_np) convenience
    wrapper used everywhere in this module and in the notebooks."""
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    Phi, y_np, _ = extract(model, loader, device="cpu")
    return Phi, y_np


def fit_laplace_and_check_convergence(model, Phi_aug_train: np.ndarray, tau_prior: float,
                                       Phi_aug_eval: dict, eval_domains: list,
                                       verbose: bool = True) -> dict:
    """Fits the last-layer Laplace posterior on the source's own train-split
    features, then runs the project's standard MC-convergence protocol
    (sweep M_VALUES at a fixed seed, compare against an independent
    M_REFERENCE, require both relative (1%) and absolute (2% of the max
    observed) deviation under threshold for 3 consecutive M values,
    simultaneously across every domain in `eval_domains`) to pick
    `M_FIXED`. Same protocol as `code_v2/notebooks/09_digits_bald.ipynb`
    and `sentiment_calibration.ipynb`, reused here instead of re-derived.

    Returns dict(laplace, M_FIXED, convergence, ref_epi)."""
    W_aug = head_weights(model)
    laplace = LastLayerLaplace.fit(W_aug, Phi_aug_train, tau_prior=tau_prior)
    if verbose:
        print(f"laplace: K={laplace.K} Dp={laplace.Dp}  tau_prior={tau_prior}")

    convergence = {f"{d}_{stat}": [] for d in eval_domains for stat in ["total", "epi", "ale"]}
    convergence["M"] = []
    for M in M_VALUES:
        convergence["M"].append(M)
        for domain in eval_domains:
            rng = np.random.default_rng(CONVERGENCE_RNG_SEED)
            pred = laplace.predictive_batched(Phi_aug_eval[domain], M=M, rng=rng)
            convergence[f"{domain}_total"].append(pred["total"].mean())
            convergence[f"{domain}_epi"].append(pred["epistemic"].mean())
            convergence[f"{domain}_ale"].append(pred["aleatoric"].mean())

    ref_epi = {}
    for domain in eval_domains:
        rng = np.random.default_rng(CONVERGENCE_RNG_SEED)
        pred_ref = laplace.predictive_batched(Phi_aug_eval[domain], M=M_REFERENCE, rng=rng)
        ref_epi[domain] = pred_ref["epistemic"].mean()

    epi_max = max(list(ref_epi.values()) + sum([convergence[f"{d}_epi"] for d in eval_domains], []))
    absolute_threshold = ABSOLUTE_THRESHOLD_FRAC * epi_max

    def check_point(i):
        return all(abs(convergence[f"{d}_epi"][i] - ref_epi[d]) / ref_epi[d] < RELATIVE_THRESHOLD
                   and abs(convergence[f"{d}_epi"][i] - ref_epi[d]) < absolute_threshold
                   for d in eval_domains)

    point_ok = [check_point(i) for i in range(len(convergence["M"]))]
    M_FIXED = None
    for i, M in enumerate(convergence["M"]):
        if i + STABILITY_WINDOW <= len(convergence["M"]) and all(point_ok[i:i + STABILITY_WINDOW]):
            M_FIXED = M
            break
    if M_FIXED is None:
        M_FIXED = M_REFERENCE
    if verbose:
        print(f"M_FIXED = {M_FIXED}" + ("" if M_FIXED != M_REFERENCE or point_ok[-1] else
                                        " (nessuna finestra di stabilità trovata, fallback a M_REFERENCE)"))

    return dict(laplace=laplace, M_FIXED=M_FIXED, convergence=convergence, ref_epi=ref_epi)


def compute_predictive_results(laplace, Phi_aug_eval: dict, y_eval: dict, M_FIXED: int,
                                seed: int = PREDICTIVE_FINAL_SEED) -> dict:
    """BALD decomposition (total/aleatoric/epistemic) for every domain in
    `Phi_aug_eval`, at the given `M_FIXED`. Returns dict[domain] = dict(y=..., probs=...,
    total=..., aleatoric=..., epistemic=...)."""
    predictive_results = {}
    for domain, Phi_aug in Phi_aug_eval.items():
        rng = np.random.default_rng(seed)
        pred = laplace.predictive_batched(Phi_aug, M=M_FIXED, rng=rng)
        predictive_results[domain] = dict(y=y_eval[domain], **pred)
    return predictive_results


def run_three_arm_adaptation(model, laplace, X_target_by_domain: dict, y_target_by_domain: dict,
                              adapt_steps: int, seed: int, verbose: bool = True) -> dict:
    """Runs both weight_mode arms (shot_im/u_sfan) on every domain in
    `X_target_by_domain`, same seed shared across arms AND across target
    domains (so seed variability is never a confound in a cross-target
    comparison).

    Returns dict[domain][arm] = dict(hist, probs_post, acc_pre, acc_post,
    n_classes_used)."""
    import copy

    adaptation_results = {}
    for domain, X_target in X_target_by_domain.items():
        y_target = y_target_by_domain[domain]
        with torch.no_grad():
            logit_pre = model(X_target).numpy()
        acc_pre = (logit_pre.argmax(1) == y_target.numpy()).mean()

        adaptation_results[domain] = {}
        for arm, weight_mode in ARM_WEIGHT_MODE.items():
            m = copy.deepcopy(model)
            hist = adapt_target(m, laplace, X_target, weight_mode=weight_mode, steps=adapt_steps,
                                seed=seed, **ADAPT_BASE_KWARGS)
            m.eval()
            with torch.no_grad():
                probs_post = torch.softmax(m(X_target), dim=-1).numpy()
            acc_post = (probs_post.argmax(axis=1) == y_target.numpy()).mean()
            n_classes_used = len(np.unique(probs_post.argmax(axis=1)))
            adaptation_results[domain][arm] = dict(
                hist=hist, probs_post=probs_post, acc_pre=acc_pre, acc_post=acc_post,
                n_classes_used=n_classes_used)
            if verbose:
                print(f"{domain}/{arm} (seed={seed}): pre={100*acc_pre:.2f}%  post={100*acc_post:.2f}%  "
                      f"delta={100*(acc_post-acc_pre):+.2f}pp  classi={n_classes_used}/2")
    return adaptation_results
