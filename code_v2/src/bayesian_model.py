"""Generazione del modello bayesiano source + decomposizione BALD.

Usato dai notebook 2 e 3. La filosofia (paper U-SFAN, Sec. 3.1):

    f = h ∘ g.  Il feature extractor g (una CNN, parametri β) resta
    DETERMINISTICO e congelato. Il trattamento bayesiano riguarda SOLO
    l'ultimo layer lineare h (parametri θ): last-layer Laplace approximation.

Divisione dei ruoli:
  - torch  -> solo la CNN: training MAP + estrazione delle feature φ = g(x).
  - numpy  -> tutta la parte bayesiana. Una volta congelato g, estraiamo le φ
              come array numpy e riusiamo ESATTAMENTE le funzioni del notebook 1
              (`weight_space_hessian`, `softmax`): l'unica differenza è che ora
              le φ vengono da una CNN invece che da feature grezze.

Nota "source-free": la source (MNIST) serve solo qui, per un singolo forward
pass che costruisce l'Hessiana. Dopo il fit, la posterior N(θ_MAP, H^-1) è
autosufficiente e la source può essere scartata.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .laplace_core import softmax, weight_space_hessian


# ======================================================================
# Feature extractor g (torch) + testa h
# ======================================================================

class SmallCNN(nn.Module):
    """CNN piccola per immagini 1x28x28: 2 blocchi conv -> feature 128-d -> testa lineare.

    `features(x)` restituisce φ = g(x) (le 128-d prima della testa); `forward`
    aggiunge la testa lineare h. Solo h riceverà il trattamento bayesiano."""

    def __init__(self, n_classes: int = 10, feature_dim: int = 128):
        super().__init__()
        self.g = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, feature_dim), nn.ReLU(),
        )
        self.h = nn.Linear(feature_dim, n_classes)
        self.feature_dim = feature_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(self.features(x))


class MLPClassifier(nn.Module):
    """Feature extractor MLP per feature tabellari (es. i 561 attributi di UCI-HAR):
    input -> hidden -> feature -> testa lineare. Stessa interfaccia di SmallCNN
    (`features` + `h`), così l'intera pipeline bayesiana si riusa invariata."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, feature_dim: int = 64):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim), nn.ReLU(),
        )
        self.h = nn.Linear(feature_dim, n_classes)
        self.feature_dim = feature_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(self.features(x))


def train_map(model, loader, epochs: int = 3, lr: float = 1e-3,
              weight_decay: float = 1e-3, device: str = "cpu", log_every: int = 200) -> float:
    """Addestra il modello a una soluzione MAP (CE + weight decay).

    Ritorna `tau_prior = weight_decay * N_train`: la precisione del prior
    gaussiano N(0, tau^-1 I) implicata dal weight decay, sulla scala della
    verosimiglianza SOMMATA (vedi notebook 1). Servirà al termine +tau*I
    dell'Hessiana."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = len(loader.dataset)
    for ep in range(epochs):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            if log_every and i % log_every == 0:
                print(f"  epoch {ep+1}/{epochs}  batch {i:4d}  loss={loss.item():.4f}")
    return weight_decay * n_train


@torch.no_grad()
def extract(model: SmallCNN, loader, device: str = "cpu"):
    """Estrae feature φ=g(x) (numpy), etichette e logit MAP dal modello congelato.
    Ritorna (Phi [N, feature_dim], y [N], logits_map [N, K])."""
    model.to(device).eval()
    Phi, Y, L = [], [], []
    for x, y in loader:
        z = model.features(x.to(device))
        Phi.append(z.cpu().numpy())
        L.append(model.h(z).cpu().numpy())
        Y.append(np.asarray(y))
    return np.concatenate(Phi), np.concatenate(Y), np.concatenate(L)


def head_weights(model: SmallCNN) -> np.ndarray:
    """La testa lineare come matrice W aumentata col bias: forma (K, feature_dim+1),
    ultima colonna = bias. Coerente con l'augmentation delle feature (φ, 1)."""
    W = model.h.weight.detach().cpu().numpy()      # (K, D)
    b = model.h.bias.detach().cpu().numpy()[:, None]  # (K, 1)
    return np.concatenate([W, b], axis=1)


def augment(Phi: np.ndarray) -> np.ndarray:
    """Aggiunge la colonna di 1 alle feature (per assorbire il bias nella testa)."""
    return np.concatenate([Phi, np.ones((Phi.shape[0], 1))], axis=1)


# ======================================================================
# Last-layer Laplace (numpy) — riusa weight_space_hessian del notebook 1
# ======================================================================

@dataclass
class LastLayerLaplace:
    theta_map: np.ndarray   # (K*Dp,) pesi della testa appiattiti, row-major (K, Dp)
    cov: np.ndarray         # (K*Dp, K*Dp) covarianza posteriore = H^-1
    K: int
    Dp: int                 # feature_dim + 1 (bias)

    @staticmethod
    def fit(W_aug: np.ndarray, Phi_aug: np.ndarray, tau_prior: float) -> "LastLayerLaplace":
        """Costruisce la posterior N(θ_MAP, H^-1) sull'ultimo layer.

        W_aug:   (K, Dp) pesi MAP della testa (bias incluso).
        Phi_aug: (N, Dp) feature source aumentate (φ, 1).
        L'Hessiana è quella del notebook 1:  H = Σ Λ_n ⊗ φφᵀ + τ I."""
        K, Dp = W_aug.shape
        H = weight_space_hessian(Phi_aug, W_aug, tau=tau_prior)   # (K*Dp, K*Dp)
        cov = np.linalg.inv(H)
        cov = 0.5 * (cov + cov.T)     # simmetrizza contro il rumore di inversione
        return LastLayerLaplace(theta_map=W_aug.reshape(-1), cov=cov, K=K, Dp=Dp)

    def sample_heads(self, M: int, rng: np.random.Generator) -> np.ndarray:
        """M campioni θ ~ N(θ_MAP, cov), forma (M, K*Dp)."""
        L = np.linalg.cholesky(self.cov)
        eps = rng.standard_normal((M, self.K * self.Dp))
        return self.theta_map[None, :] + eps @ L.T

    def predictive(self, Phi_aug: np.ndarray, M: int = 200,
                   rng: np.random.Generator = None) -> dict:
        """Predittiva MC integrata sulla posterior + decomposizione BALD.

        Per ogni punto: campiona M teste, calcola p(y|x,θ_m), poi
            total     = H[ (1/M) Σ_m p_m ]           (entropia della media)
            aleatoric = (1/M) Σ_m H[p_m]              (media delle entropie)
            epistemic = total - aleatoric             (mutual information, BALD)
        Ritorna probs medie (N,K) e le tre entropie (N,)."""
        if rng is None:
            rng = np.random.default_rng(0)
        N = Phi_aug.shape[0]
        thetas = self.sample_heads(M, rng).reshape(M, self.K, self.Dp)  # (M,K,Dp)
        # logits[m,n,k] = φ_n · θ_{m,k}
        # optimize=True: senza, questo einsum misura ~4x più lento su N e M
        # grandi (es. 27s vs 6.8s per M=2000, N=26032) -- stesso risultato,
        # solo un percorso di contrazione più efficiente.
        logits = np.einsum("mkd,nd->mnk", thetas, Phi_aug, optimize=True)  # (M,N,K)
        probs_m = softmax(logits)                                      # (M,N,K)
        p_mean = probs_m.mean(axis=0)                                  # (N,K)
        total = _entropy(p_mean)                                       # (N,)
        aleatoric = _entropy(probs_m).mean(axis=0)                     # (N,)
        epistemic = total - aleatoric
        return dict(probs=p_mean, total=total, aleatoric=aleatoric, epistemic=epistemic)

    def predictive_batched(self, Phi_aug: np.ndarray, M: int = 200,
                           rng: np.random.Generator = None, batch_size: int = None,
                           target_bytes: int = 300_000_000) -> dict:
        """Come `predictive`, ma a pezzi su N: `predictive` materializza un
        tensore denso (M, N, K) -- per M ed N grandi insieme (es. M=5000 su
        N=26.032 immagini, K=10) sono ~10.4 GB in un solo array, sufficienti
        a mandare in swap pesante o in OOM una macchina con poca RAM (osservato
        direttamente: kernel Jupyter crashato). Qui Phi_aug è processato in
        blocchi di `batch_size` righe, tenendo il tensore denso a
        `batch_size * M * K` elementi indipendentemente da N. Ogni blocco usa
        theta campionati indipendentemente (stesso `rng`, che avanza fra un
        blocco e l'altro): non introduce bias nelle statistiche per-punto
        restituite (`total`/`aleatoric`/`epistemic` per ciascun punto restano
        medie Monte Carlo sui propri M campioni, indipendentemente da come i
        campioni sono raggruppati fra i punti)."""
        if rng is None:
            rng = np.random.default_rng(0)
        N = Phi_aug.shape[0]
        if batch_size is None:
            # dimensiona il blocco così che il tensore denso (batch_size, M, K)
            # resti sotto target_bytes, indipendentemente da quanto M è grande
            batch_size = max(1, int(target_bytes / (M * self.K * 8)))
        probs_parts, total_parts, ale_parts, epi_parts = [], [], [], []
        for start in range(0, N, batch_size):
            chunk = self.predictive(Phi_aug[start:start + batch_size], M=M, rng=rng)
            probs_parts.append(chunk["probs"])
            total_parts.append(chunk["total"])
            ale_parts.append(chunk["aleatoric"])
            epi_parts.append(chunk["epistemic"])
        return dict(probs=np.concatenate(probs_parts), total=np.concatenate(total_parts),
                    aleatoric=np.concatenate(ale_parts), epistemic=np.concatenate(epi_parts))


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Entropia di Shannon lungo l'ultimo asse (nats)."""
    return -(p * np.log(p + eps)).sum(axis=-1)


# ======================================================================
# Validazione: posterior vera via MCMC (Metropolis-Hastings) — notebook 2
# ======================================================================

def _log_posterior(w_flat, Phi, y, K, tau):
    """log-posterior (a meno di costante) di un modello lineare softmax:
    log-verosimiglianza CE sommata + prior gaussiano N(0, tau^-1 I)."""
    N, Dp = Phi.shape
    W = w_flat.reshape(K, Dp)
    logits = Phi @ W.T
    z = logits - logits.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    ll = log_p[np.arange(N), y].sum()
    lp = -0.5 * tau * (w_flat @ w_flat)
    return ll + lp


def mcmc_posterior(Phi, y, K, tau, n_samples=4000, burn_in=1000, step=0.05,
                   rng=None):
    """Campiona la posterior VERA con Metropolis-Hastings a passo gaussiano.
    Solo per problemi piccoli (pochi parametri): serve come riferimento
    esatto contro cui giudicare la Laplace.
    Ritorna (samples [n_samples, K*Dp], acceptance_rate)."""
    if rng is None:
        rng = np.random.default_rng(0)
    Dp = Phi.shape[1]
    dim = K * Dp
    w = np.zeros(dim)
    lp = _log_posterior(w, Phi, y, K, tau)
    out, accepted = [], 0
    total = n_samples + burn_in
    for t in range(total):
        w_prop = w + step * rng.standard_normal(dim)
        lp_prop = _log_posterior(w_prop, Phi, y, K, tau)
        if np.log(rng.random()) < lp_prop - lp:
            w, lp = w_prop, lp_prop
            accepted += 1
        if t >= burn_in:
            out.append(w.copy())
    return np.array(out), accepted / total
