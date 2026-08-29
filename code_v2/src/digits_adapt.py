"""Information-maximization (IM) source-free adaptation with uncertainty-guided
down-weighting (U-SFAN), ported here from the main project's `src/im_adapt.py`
(not modified there) because `code_v2/src/` had no adaptation code of its
own before this file -- the MNIST pedagogical notebooks (00, 01, 03-05)
only ever *measure* shift (AUROC, rejection curve), they never adapt a
model to it (notebook 02, `synthetic_track.ipynb`, does adapt, via the main
project's own `src/im_adapt.py` directly, not this port).

    L_ent = -E_x sum_k p_k(x) log p_k(x)                (per-sample entropy)
    L_div = D_KL(p_hat || uniform) - log K = -H[p_hat]  (batch diversity)
    L     = (1 - gamma) * L_ent_weighted + gamma * L_div

Only L_ent is ever down-weighted per sample. `weight_mode="none"` reproduces
the conventional SHOT-IM baseline; `weight_mode="uncertainty"` reproduces
U-SFAN, using w_i = exp(-H_i) with H_i the total predictive entropy of the
frozen Bayesian head's MC predictive (recomputed every step from the
*current*, adapting features).

Bridging torch/numpy: unlike `src/im_adapt.py` (whose `LastLayerLaplace` is
a torch class, so everything stays in torch), `code_v2/src/bayesian_model.py`'s
`LastLayerLaplace` is numpy-only. The adaptation loss itself (needs
gradients w.r.t. `model.g`) stays in torch; the per-step uncertainty weight
(needs no gradient) is computed by converting the current features to numpy,
calling `LastLayerLaplace.predictive` there, and converting the resulting
per-sample entropy back to a torch tensor to weight the loss. `model.h`
stays frozen throughout, exactly as in `src/im_adapt.py`."""
import numpy as np
import torch
import torch.nn.functional as F

from .bayesian_model import LastLayerLaplace


def _entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(p * (p + eps).log()).sum(dim=-1)


def im_loss(logits: torch.Tensor, weights: torch.Tensor = None, gamma: float = 0.5):
    """logits: (N, K) unnormalized target predictions (already temperature-scaled).
    weights: optional (N,) non-negative per-sample weight on the entropy term.
    Returns (loss, ent_term, div_term), the latter two detached for logging."""
    probs = F.softmax(logits, dim=-1)
    per_sample_ent = _entropy(probs)
    if weights is None:
        weights = torch.ones_like(per_sample_ent)
    ent_term = (weights * per_sample_ent).mean()
    p_hat = probs.mean(dim=0)
    div_term = -_entropy(p_hat)
    loss = (1 - gamma) * ent_term + gamma * div_term
    return loss, ent_term.detach(), div_term.detach()


def adapt_target(model, laplace: LastLayerLaplace, X_target: torch.Tensor,
                 weight_mode: str = "uncertainty", gamma: float = 0.5, temperature: float = 0.4,
                 lr: float = 1e-2, steps: int = 300, M: int = 100, seed: int = 0) -> dict:
    """Adapts `model.g` in place on unlabeled `X_target` via (weighted) IM
    loss. `model.h` stays frozen. `model`: an instance with `.features(x)`
    and `.h` (e.g. `SmallCNN32`). `laplace`: already fit on the source
    (numpy `LastLayerLaplace`, `bayesian_model.py`). Returns the loss
    trajectory for diagnostics."""
    assert weight_mode in ("none", "uncertainty")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    for p in model.h.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.g.parameters(), lr=lr)

    history = {"loss": [], "ent": [], "div": [], "mean_weight": []}
    for _ in range(steps):
        opt.zero_grad()
        logits = model(X_target) / temperature
        if weight_mode == "uncertainty":
            with torch.no_grad():
                phi = model.features(X_target).cpu().numpy()
                phi_aug = np.concatenate([phi, np.ones((phi.shape[0], 1))], axis=1)
                pred = laplace.predictive(phi_aug, M=M, rng=rng)
                weights = torch.exp(-torch.from_numpy(pred["total"]).float())
        else:
            weights = None
        loss, ent, div = im_loss(logits, weights=weights, gamma=gamma)
        loss.backward()
        opt.step()
        history["loss"].append(loss.item())
        history["ent"].append(ent.item())
        history["div"].append(div.item())
        history["mean_weight"].append(1.0 if weights is None else weights.mean().item())

    for p in model.h.parameters():
        p.requires_grad_(True)
    return history
