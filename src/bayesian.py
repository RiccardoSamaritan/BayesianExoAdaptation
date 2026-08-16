"""Last-layer Laplace approximation and BALD uncertainty decomposition.

Architecture-agnostic: only the feature extractor `g` changes between the
synthetic toy (Sec. 3, TODO.md) and the real HAR pipeline (Sec. 4-7). The
head/Laplace/BALD machinery here is written once and reused by both.

Notation follows Roy et al., U-SFAN (`paper/U-SFAN.pdf`), Sec. 3.1-3.2 and
`TODO.md` Sec. 1b/5/7: f = h o g, only h (the last linear layer) gets a
Bayesian treatment via an exact Gauss-Newton Hessian,
    H = sum_n (Lambda_n kron phi_n phi_n^T) + tau_prior * I,
    Lambda_n = diag(p_n) - p_n p_n^T,
with phi_n the (bias-augmented) feature vector of point n and p_n the MAP
softmax output. This is exact (no KFAC) because the toy's feature dimension
keeps K * (D+1) small enough to invert directly.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureClassifier(nn.Module):
    """f = h o g. `g` is an MLP feature extractor (params beta), `h` is the
    linear head (params theta) that gets the Bayesian treatment."""

    def __init__(self, in_dim: int, hidden_dims: list, n_classes: int):
        super().__init__()
        layers = []
        d = in_dim
        for hd in hidden_dims:
            layers += [nn.Linear(d, hd), nn.ReLU()]
            d = hd
        self.g = nn.Sequential(*layers)
        self.h = nn.Linear(d, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(self.features(x))


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy along the last dimension, nats."""
    return -(p * (p + eps).log()).sum(dim=-1)


def _augment(phi: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(phi.shape[0], 1, dtype=phi.dtype, device=phi.device)
    return torch.cat([phi, ones], dim=1)


@dataclass
class LastLayerLaplace:
    theta_map: torch.Tensor  # flat (K*Dp,), row-major over (K, Dp)
    cov: torch.Tensor        # (K*Dp, K*Dp) posterior covariance H^-1
    K: int
    Dp: int                  # feature dim + 1 (bias)

    @staticmethod
    def fit(model: FeatureClassifier, X: torch.Tensor, y: torch.Tensor,
            tau_prior: float) -> "LastLayerLaplace":
        """Exact GGN Hessian of the head at its MAP solution, fully
        vectorized (no per-sample python loop -- see module docstring)."""
        model.eval()
        with torch.no_grad():
            phi_aug = _augment(model.features(X))                # (N, Dp)
            W_aug = torch.cat([model.h.weight, model.h.bias[:, None]], dim=1)  # (K, Dp)
            logits = phi_aug @ W_aug.T                            # (N, K)
            p = F.softmax(logits, dim=1)                          # (N, K)
            K, Dp = p.shape[1], phi_aug.shape[1]

            lambda_all = torch.diag_embed(p) - torch.einsum("nk,nl->nkl", p, p)  # (N,K,K)
            phi_outer = torch.einsum("nd,nf->ndf", phi_aug, phi_aug)            # (N,Dp,Dp)
            H = torch.einsum("nkl,ndf->kdlf", lambda_all, phi_outer)            # (K,Dp,K,Dp)
            H = H.reshape(K * Dp, K * Dp)
            H = H + tau_prior * torch.eye(K * Dp, dtype=H.dtype)

            theta_map = W_aug.reshape(-1)
            cov = torch.linalg.inv(H)
        return LastLayerLaplace(theta_map=theta_map, cov=cov, K=K, Dp=Dp)

    def sample_heads(self, M: int, generator: torch.Generator = None) -> torch.Tensor:
        """M samples of the flattened head params ~ N(theta_map, cov), (M, K*Dp)."""
        cov_sym = 0.5 * (self.cov + self.cov.T)
        L = torch.linalg.cholesky(cov_sym)
        eps = torch.randn(M, self.K * self.Dp, generator=generator, dtype=self.cov.dtype)
        return self.theta_map.unsqueeze(0) + eps @ L.T

    def predictive(self, model: FeatureClassifier, X: torch.Tensor, M: int = 200,
                   temperature: float = 1.0, generator: torch.Generator = None) -> dict:
        """MC-integrated predictive posterior (Eq. 6 of the paper) + BALD
        decomposition (TODO.md Sec. 7): total_entropy = H[E_theta p(y|x,theta)],
        aleatoric = E_theta[H[p(y|x,theta)]], epistemic = total - aleatoric
        (mutual information between prediction and head parameters)."""
        model.eval()
        with torch.no_grad():
            phi_aug = _augment(model.features(X))                # (N, Dp)
            thetas = self.sample_heads(M, generator=generator).reshape(M, self.K, self.Dp)
            logits = torch.einsum("mkd,nd->mnk", thetas, phi_aug) / temperature  # (M,N,K)
            probs = F.softmax(logits, dim=-1)                     # (M,N,K)

            mean_probs = probs.mean(dim=0)                        # (N,K)
            total_ent = entropy(mean_probs)                       # (N,)
            aleatoric = entropy(probs).mean(dim=0)                # (N,)
            epistemic = (total_ent - aleatoric).clamp(min=0.0)
        return dict(mean_probs=mean_probs, total_entropy=total_ent,
                    aleatoric=aleatoric, epistemic=epistemic)


def map_entropy(model: FeatureClassifier, X: torch.Tensor) -> torch.Tensor:
    """Predictive entropy of the plain point-estimate (MAP) softmax --
    used for the 'MAP (conventional)' row and the Ent.-weighting ablation."""
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(X), dim=-1)
        return entropy(probs)
