"""Metriche di shift e di detection OOD — usate dai notebook 4 e 5.

- `mmd_rbf`        : Maximum Mean Discrepancy con kernel RBF, misura di shift fra
                    due insiemi di feature (proxy di distanza fra distribuzioni,
                    calcolata nello spazio delle feature, non sull'input grezzo).
- `auroc`         : area sotto la ROC, per valutare uno score di incertezza come
                    rivelatore OOD (rank-based, senza dipendenze esterne).
- `rejection_curve`: accuratezza in funzione della copertura, rifiutando i punti a
                    più alta incertezza.
- `reliability_bins`, `expected_calibration_error`: calibrazione (ECE, reliability
                    diagram). Portate qui, invariate nella matematica, da
                    src/calibration.py del progetto principale (che non ha un
                    equivalente in code_v2) -- usate dai notebook digits.
"""
import numpy as np


def median_gamma(X: np.ndarray, Y: np.ndarray, max_n: int = 500,
                 rng: np.random.Generator = None) -> float:
    """Euristica della mediana per il parametro γ del kernel RBF
    k(a,b)=exp(-γ‖a-b‖²): γ = 1 / (2 · mediana delle distanze²)."""
    if rng is None:
        rng = np.random.default_rng(0)
    Z = np.concatenate([X, Y])
    if len(Z) > max_n:
        Z = Z[rng.choice(len(Z), max_n, replace=False)]
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0])
    return 1.0 / (2.0 * med + 1e-12)


def _rbf(A, B, gamma):
    d2 = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None,
            rng: np.random.Generator = None) -> float:
    """MMD² (stima biased) fra X e Y con kernel RBF. 0 ⇔ stesse distribuzioni;
    cresce con lo shift. Se `gamma` è None usa l'euristica della mediana."""
    if gamma is None:
        gamma = median_gamma(X, Y, rng=rng)
    Kxx = _rbf(X, X, gamma).mean()
    Kyy = _rbf(Y, Y, gamma).mean()
    Kxy = _rbf(X, Y, gamma).mean()
    return float(Kxx + Kyy - 2.0 * Kxy)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC rank-based. `labels`: 1 = positivo (es. OOD), 0 = negativo (in-dist).
    `scores`: più alto = più "positivo". Equivale alla U di Mann-Whitney."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # media dei ranghi per i pareggi
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def rejection_curve(uncertainty: np.ndarray, correct: np.ndarray, n_points: int = 20):
    """Accuratezza vs copertura. A ogni soglia teniamo la frazione di punti a più
    BASSA incertezza (i più "fidati") e misuriamo l'accuratezza su quelli.
    Ritorna (coverage [n_points], accuracy [n_points])."""
    unc = np.asarray(uncertainty, float)
    cor = np.asarray(correct, bool)
    order = np.argsort(unc)                 # dal più fidato al meno
    cor_sorted = cor[order]
    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    accs = []
    N = len(unc)
    for c in coverages:
        k = max(1, int(round(c * N)))
        accs.append(cor_sorted[:k].mean())
    return coverages, np.array(accs)


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """Confidenza/accuratezza per bin, per un reliability diagram. Il bin b copre
    confidenza in [edges[b], edges[b+1)). Bin vuoti -> NaN in bin_acc/bin_conf
    (così possono essere saltati nel plot/nell'aggregazione)."""
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, n_bins - 1)

    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        bin_count[b] = mask.sum()
        if mask.any():
            bin_acc[b] = correct[mask].mean()
            bin_conf[b] = confidences[mask].mean()
    return dict(bin_edges=bin_edges, bin_acc=bin_acc, bin_conf=bin_conf, bin_count=bin_count)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE standard: somma_b (n_b / N) * |acc_b - conf_b|."""
    bins = reliability_bins(probs, y, n_bins=n_bins)
    n = len(y)
    weights = bins["bin_count"] / n
    gaps = np.abs(bins["bin_acc"] - bins["bin_conf"])
    return float(np.nansum(weights * np.nan_to_num(gaps)))
