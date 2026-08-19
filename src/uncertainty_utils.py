"""Uncertainty quantification utilities for normalization and analysis.

Provides functions for normalizing epistemic uncertainty against source pool
reference distribution (§7, TODO.md).
"""
import numpy as np


def compute_epistemic_normalizer(source_epistemic: np.ndarray,
                                 quantile: float = 0.95) -> float:
    """Compute reference quantile of epistemic uncertainty on source pool.

    This defines the "rare but seen" threshold: epistemic values beyond this
    quantile are higher than what the model experienced on its own source data.

    Args:
        source_epistemic: array of epistemic values from source validation set
        quantile: quantile level (default 0.95 = 95th percentile)

    Returns:
        Reference value (epistemic at given quantile)

    Rationale:
        The 95th percentile captures the "high-but-not-extreme" epistemic range
        seen on source data. Values normalized by this threshold > 1 indicate
        target epistemic exceeds even the rare source cases.
    """
    if quantile <= 0 or quantile >= 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    return np.quantile(source_epistemic, quantile)


def normalize_epistemic(epistemic: np.ndarray, normalizer: float) -> np.ndarray:
    """Normalize epistemic uncertainty by reference quantile.

    Args:
        epistemic: raw epistemic uncertainty values
        normalizer: reference value (from compute_epistemic_normalizer)

    Returns:
        Normalized epistemic (epistemic / normalizer)

    Interpretation:
        - Values < 1: epistemic lower than rare source cases
        - Values ≈ 1: epistemic comparable to rare source cases
        - Values > 1: epistemic exceeds even rare source cases (strong OOD signal)
    """
    return epistemic / normalizer


def epistemic_fraction(epistemic: np.ndarray, total_entropy: np.ndarray) -> np.ndarray:
    """Compute epistemic fraction of total uncertainty.

    Args:
        epistemic: epistemic uncertainty (model uncertainty)
        total_entropy: total predictive entropy

    Returns:
        epistemic / total_entropy (element-wise)

    Interpretation:
        - Values near 0: aleatoric-dominated (irreducible data noise)
        - Values near 1: epistemic-dominated (reducible model uncertainty)
    """
    # Avoid division by zero (shouldn't happen with proper BALD, but be defensive)
    return np.where(total_entropy > 1e-12, epistemic / total_entropy, 0.0)
