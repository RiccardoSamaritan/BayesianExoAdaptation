"""Fondamenti della Laplace approximation — blocchi puri usati dai notebook 0 e 1.

Tutto qui segue le dispense del corso, §7.3:

    approssima  p(z) = (1/Z) f(z)  con una Gaussiana centrata in un modo z0,
    con precisione  A = -∇∇ log f(z0)   (match della curvatura).

Nessun I/O, nessun plotting, nessun codice di dataset: solo la matematica, in
numpy esplicito (niente autograd), così ogni funzione è verificabile a mano e
la derivazione dell'Hessiana resta visibile — che è il punto dei due notebook.

Convenzione dei pesi (coerente con il resto del progetto): W ha forma (K, D),
appiattita in ordine row-major, cioè il peso della classe k, feature d sta in
posizione  k*D + d.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


# ======================================================================
# Caso 1-D (Notebook 0)
# ======================================================================

def gaussian_pdf(z, mean: float, precision: float):
    """Densità N(z | mean, 1/precision). `precision` è A = 1/varianza."""
    z = np.asarray(z, dtype=float)
    return np.sqrt(precision / (2.0 * np.pi)) * np.exp(-0.5 * precision * (z - mean) ** 2)


def numeric_second_derivative(g, x: float, h: float = 1e-5) -> float:
    """Derivata seconda di g in x per differenze finite centrate."""
    return (g(x + h) - 2.0 * g(x) + g(x - h)) / (h * h)


def find_mode_1d(log_f, z_init: float, bounds=(1e-8, 1e3)) -> float:
    """Trova il modo z0 di f massimizzando log f(z) (= minimizzando -log f)."""
    res = minimize_scalar(lambda z: -log_f(z), bounds=bounds, method="bounded")
    if not res.success:
        raise RuntimeError(f"ricerca del modo fallita: {res.message}")
    return float(res.x)


@dataclass
class Laplace1D:
    """Approssimazione di Laplace 1-D di p(z) = (1/Z) f(z)."""
    mode: float          # z0
    precision: float     # A = -d²/dz² log f (z0)

    @property
    def variance(self) -> float:
        return 1.0 / self.precision

    def pdf(self, z):
        """La Gaussiana approssimante q(z) = N(z | z0, A^-1)."""
        return gaussian_pdf(z, self.mode, self.precision)

    def log_normalizer(self, log_f_at_mode: float) -> float:
        """Stima di log Z dalla Laplace (dispense §7.3):
            Z ≈ f(z0) * sqrt(2π / A)
        quindi  log Z ≈ log f(z0) + ½ log(2π) - ½ log A .
        Il vero Z va confrontato con una quadratura numerica nel notebook."""
        return log_f_at_mode + 0.5 * np.log(2.0 * np.pi) - 0.5 * np.log(self.precision)


def fit_laplace_1d(log_f, z_init: float, bounds=(1e-8, 1e3), h: float = 1e-5) -> Laplace1D:
    """Pipeline completa del §7.3: trova il modo, poi la precisione dalla
    curvatura A = -d²/dz² log f(z0) (differenze finite sul log)."""
    z0 = find_mode_1d(log_f, z_init, bounds=bounds)
    A = -numeric_second_derivative(log_f, z0, h=h)
    if A <= 0:
        raise ValueError("A <= 0: il punto trovato non è un massimo (curvatura non negativa).")
    return Laplace1D(mode=z0, precision=A)


# ======================================================================
# Caso multiclasse / softmax (Notebook 1)
# ======================================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numericamente stabile lungo l'ultimo asse."""
    logits = np.asarray(logits, dtype=float)
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_sum(w_flat: np.ndarray, Phi: np.ndarray, y: np.ndarray, K: int) -> float:
    """Cross-entropy SOMMATA (non media) di un modello lineare softmax.

    w_flat: (K*D,) pesi appiattiti row-major, W = w_flat.reshape(K, D).
    Phi:    (N, D) feature. y: (N,) etichette intere in [0, K).
    Usiamo la SOMMA (non la media) perché è la sua Hessiana che vale
    esattamente  Σ_n Λ_n ⊗ φ_nφ_nᵀ  (vedi `weight_space_hessian`)."""
    N, D = Phi.shape
    W = w_flat.reshape(K, D)
    logits = Phi @ W.T                       # (N, K)
    z = logits - logits.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_p[np.arange(N), y].sum())


def softmax_hessian_block(p: np.ndarray) -> np.ndarray:
    """Il blocco Λ = diag(p) - p pᵀ  (Hessiana della CE rispetto ai logit).

    Per la softmax questa è ESATTA, non una approssimazione: la cross-entropy
    è convessa nei logit e la sua Hessiana è proprio diag(p) - ppᵀ. È il
    risultato chiave del notebook 1."""
    p = np.asarray(p, dtype=float)
    return np.diag(p) - np.outer(p, p)


def weight_space_hessian(Phi: np.ndarray, W: np.ndarray, tau: float = 0.0) -> np.ndarray:
    """Hessiana nello spazio dei pesi di un modello lineare softmax:

        H = Σ_n Λ_n ⊗ (φ_n φ_nᵀ)  +  τ I,     Λ_n = diag(p_n) - p_n p_nᵀ.

    Per un ultimo layer LINEARE questa è l'Hessiana ESATTA della CE sommata
    (nessuna approssimazione Gauss-Newton): è per questo che la Laplace
    sull'ultimo layer può essere esatta. Il termine τ I è il contributo del
    prior gaussiano N(0, τ^-1 I) sui pesi.

    Phi: (N, D)  W: (K, D). Ritorna H di forma (K*D, K*D), ordine row-major
    su (k, d), cioè H[k*D+d, l*D+f] = Λ[k,l] · (φφᵀ)[d,f]."""
    N, D = Phi.shape
    K = W.shape[0]
    logits = Phi @ W.T
    P = softmax(logits)                                  # (N, K)
    Lam = np.einsum("nk,nl->nkl", P, -P)                 # -p_k p_l
    idx = np.arange(K)
    Lam[:, idx, idx] += P                                # + diag(p)  ->  diag(p) - ppᵀ
    H = np.einsum("nkl,nd,nf->kdlf", Lam, Phi, Phi)      # (K, D, K, D)
    H = H.reshape(K * D, K * D)
    if tau:
        H = H + tau * np.eye(K * D)
    return H


def numeric_hessian(fun, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Hessiana per differenze finite di `fun`: R^n -> R, in x. O(n²) valutazioni.
    Serve nel notebook 1 per confermare che l'Hessiana analitica
    (`weight_space_hessian`) coincide con quella vera della CE."""
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy(); xpp[i] += h; xpp[j] += h
            xpm = x.copy(); xpm[i] += h; xpm[j] -= h
            xmp = x.copy(); xmp[i] -= h; xmp[j] += h
            xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
            H[i, j] = (fun(xpp) - fun(xpm) - fun(xmp) + fun(xmm)) / (4.0 * h * h)
            H[j, i] = H[i, j]
    return H
