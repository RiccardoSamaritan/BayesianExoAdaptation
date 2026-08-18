"""Calibration and uncertainty-quality metrics (TODO.md Sec. 8).

All functions take plain numpy arrays: `probs` (N, K) already-averaged
predictive probabilities and `y` (N,) integer labels in [0, K). The same
functions are used for the MAP softmax and the Laplace MC-mean predictive --
the only difference between the two calibration curves in the notebook is
which `probs` array is passed in.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """Confidence-binned accuracy/confidence for a reliability diagram.
    Bin b covers confidence in [edges[b], edges[b+1]). Empty bins get NaN in
    bin_acc/bin_conf (so they can be skipped when plotting/aggregating)."""
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
    """Standard ECE: sum_b (n_b / N) * |acc_b - conf_b|."""
    bins = reliability_bins(probs, y, n_bins=n_bins)
    n = len(y)
    weights = bins["bin_count"] / n
    gaps = np.abs(bins["bin_acc"] - bins["bin_conf"])
    return float(np.nansum(weights * np.nan_to_num(gaps)))


def nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true class."""
    p_true = probs[np.arange(len(y)), y]
    return float(-np.log(p_true + eps).mean())


def brier_score(probs: np.ndarray, y: np.ndarray, K: int) -> float:
    """Mean multiclass Brier score: mean_n sum_k (p_nk - onehot_nk)^2."""
    onehot = np.eye(K)[y]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def misclassification_auroc(probs: np.ndarray, y: np.ndarray, uncertainty: np.ndarray) -> float:
    """AUROC for detecting misclassified points by thresholding `uncertainty`
    (higher uncertainty -> more likely wrong). NaN if every point is correct
    or every point is wrong (AUROC undefined -- only one class present)."""
    preds = probs.argmax(axis=1)
    wrong = (preds != y).astype(int)
    if wrong.sum() == 0 or wrong.sum() == len(wrong):
        return float("nan")
    return float(roc_auc_score(wrong, uncertainty))


def accuracy_vs_coverage(probs: np.ndarray, y: np.ndarray, uncertainty: np.ndarray,
                          n_points: int = 20) -> tuple:
    """Selective-prediction curve: sort by ascending uncertainty (most
    confident first), then report accuracy on the most-confident `coverage`
    fraction of points, for a sweep of coverage fractions in (0, 1]."""
    preds = probs.argmax(axis=1)
    correct = (preds == y).astype(float)
    order = np.argsort(uncertainty)
    correct_sorted = correct[order]

    n = len(y)
    coverages = np.linspace(1.0 / n, 1.0, n_points)
    accs = np.array([correct_sorted[:max(1, int(round(c * n)))].mean() for c in coverages])
    return coverages, accs
