"""Information-maximization (IM) source-free adaptation, a.k.a. SHOT-IM
[Liang et al. 2020], with the uncertainty-guided down-weighting proposed by
Roy et al., U-SFAN (`paper/U-SFAN.pdf`, Eq. 2-3-7).

    L_ent = -E_x sum_k p_k(x) log p_k(x)                (per-sample entropy)
    L_div = D_KL(p_hat || uniform) - log K = -H[p_hat]  (batch diversity)
    L     = (1 - gamma) * L_ent_weighted + gamma * L_div

Only L_ent is ever down-weighted per sample (TODO.md Sec. 1: "Down-weighting
applies only to the conditional-entropy term, never L_div."). weight_mode
"none" reproduces the conventional MAP/SHOT-IM baseline (top row of Fig. 4);
"uncertainty" reproduces U-SFAN (bottom row), using w_i = exp(-H_i) with H_i
the entropy of the frozen Bayesian head's MC predictive mean (Eq. 6-7) --
recomputed every step from the *current* (adapting) features, per TODO.md
Sec. 1. The feature extractor beta is updated; the head theta (MAP or
Laplace posterior) stays frozen throughout, per the paper's Sec. 3
("Problem Definition") and Fig. 3b.
"""
import torch
import torch.nn.functional as F

from .bayesian import FeatureClassifier, LastLayerLaplace, entropy


def im_loss(logits: torch.Tensor, weights: torch.Tensor = None, gamma: float = 0.5):
    """logits: (N, K) unnormalized target predictions (already temperature-scaled).
    weights: optional (N,) non-negative per-sample weight on the entropy term.
    Returns (loss, ent_term, div_term), the latter two detached for logging."""
    probs = F.softmax(logits, dim=-1)
    per_sample_ent = entropy(probs)
    if weights is None:
        weights = torch.ones_like(per_sample_ent)
    ent_term = (weights * per_sample_ent).mean()
    p_hat = probs.mean(dim=0)
    div_term = -entropy(p_hat)
    loss = (1 - gamma) * ent_term + gamma * div_term
    return loss, ent_term.detach(), div_term.detach()


def adapt_target(model: FeatureClassifier, laplace: LastLayerLaplace, X_target: torch.Tensor,
                  weight_mode: str = "uncertainty", gamma: float = 0.5, temperature: float = 0.4,
                  lr: float = 1e-2, steps: int = 300, M: int = 100, seed: int = 0) -> dict:
    """Adapts model.g in place on unlabeled X_target via (weighted) IM loss.
    model.h stays frozen. Returns the loss trajectory for diagnostics."""
    assert weight_mode in ("none", "uncertainty")
    torch.manual_seed(seed)
    for p in model.h.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.g.parameters(), lr=lr)

    history = {"loss": [], "ent": [], "div": [], "mean_weight": []}
    for _ in range(steps):
        opt.zero_grad()
        logits = model(X_target) / temperature
        if weight_mode == "uncertainty":
            with torch.no_grad():
                pred = laplace.predictive(model, X_target, M=M)
                weights = torch.exp(-pred["total_entropy"])
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
