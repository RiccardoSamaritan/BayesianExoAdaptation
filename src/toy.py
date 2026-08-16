"""Synthetic 2D, 3-class toy track (TODO.md Sec. 3): replicates Fig. 4 of
Roy et al., U-SFAN (`paper/U-SFAN.pdf`) on a small, fully controllable
problem before trusting the same Laplace/BALD/IM machinery on real HAR data.

Three classes (red / green / blue), source centres fixed; two target shift
regimes:
  - "mild":   every centre nudged a little, target stays inside the source
              decision cells.
  - "strong": the blue cluster is dragged across the source decision
              boundary into red's territory (red/green only nudged mildly) --
              this is the failure mode conventional SHOT-IM cannot handle
              (paper Sec. 4.1: "most blue target points fall under the red
              decision surface").
"""
import numpy as np
import torch
import torch.nn as nn

from bayesian import FeatureClassifier

N_CLASSES = 3
CLASS_NAMES = ["red", "green", "blue"]
CLASS_COLORS = np.array([
    [0.80, 0.10, 0.10],   # red
    [0.10, 0.60, 0.20],   # green
    [0.10, 0.25, 0.85],   # blue
])

SOURCE_CENTERS = np.array([
    [-2.0, 0.0],   # red
    [1.0, 1.6],    # green
    [1.0, -1.6],   # blue
])
SOURCE_STD = 0.5

MILD_SHIFT = np.array([
    [0.3, 0.2],
    [-0.2, 0.3],
    [0.2, -0.3],
])

STRONG_SHIFT_CENTERS = np.array([
    [-1.7, 0.2],   # red, mild nudge
    [0.8, 1.9],    # green, mild nudge
    [-1.7, -1.75], # blue, dragged into a region the source model confidently
                   # (and wrongly) calls "red" by ReLU extrapolation, close
                   # enough to red's basin to be genuinely contested but far
                   # enough from actual red training density that the Laplace
                   # posterior still flags it as markedly more uncertain
])
STRONG_SHIFT_STD = np.array([0.5, 0.5, 0.4])


def _make_blobs(centers, stds, n_per_class, seed):
    rng = np.random.RandomState(seed)
    X, y = [], []
    stds = np.broadcast_to(stds, (len(centers),))
    for k, (c, s) in enumerate(zip(centers, stds)):
        X.append(rng.normal(loc=c, scale=s, size=(n_per_class, 2)))
        y.append(np.full(n_per_class, k))
    return np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int64)


def make_source(n_per_class: int = 150, seed: int = 0):
    return _make_blobs(SOURCE_CENTERS, SOURCE_STD, n_per_class, seed)


def make_target(shift: str, n_per_class: int = 150, seed: int = 1):
    assert shift in ("mild", "strong")
    if shift == "mild":
        centers = SOURCE_CENTERS + MILD_SHIFT
        stds = SOURCE_STD
    else:
        centers = STRONG_SHIFT_CENTERS
        stds = STRONG_SHIFT_STD
    return _make_blobs(centers, stds, n_per_class, seed)


def make_classification_sweep(n_levels: int = 10, n_per_class: int = 150,
                               max_shift: float = 3.5, seed: int = 0):
    """Ground-truth shift-magnitude sweep (TODO.md Sec. 3, optional item):
    the blue cluster is dragged radially outward from the 3-class centroid,
    through its own source position and beyond, by a growing fraction of
    `max_shift`; red and green get a small fixed nudge (same as the "mild"
    case) throughout, so the only thing that changes across the sweep is a
    single, known scalar shift magnitude. Moving radially *outward* (rather
    than towards another class's centre) keeps the sweep monotonic: blue
    only ever gets farther from every source cluster, never passes back
    through one. Returns a list of dicts with X, y and the ground-truth
    shift magnitude (Euclidean centre displacement) for each level."""
    centroid = SOURCE_CENTERS.mean(axis=0)
    direction = SOURCE_CENTERS[2] - centroid
    direction = direction / np.linalg.norm(direction)
    fractions = np.linspace(0.0, 1.0, n_levels)
    levels = []
    for i, frac in enumerate(fractions):
        centers = SOURCE_CENTERS + MILD_SHIFT
        displacement = frac * max_shift * direction
        centers[2] = SOURCE_CENTERS[2] + MILD_SHIFT[2] + displacement
        X, y = _make_blobs(centers, SOURCE_STD, n_per_class, seed=seed + i)
        levels.append(dict(X=X, y=y, shift_magnitude=float(np.linalg.norm(displacement))))
    return levels


def build_model(hidden_dims=(32, 16), seed: int = 0) -> FeatureClassifier:
    torch.manual_seed(seed)
    return FeatureClassifier(in_dim=2, hidden_dims=list(hidden_dims), n_classes=N_CLASSES)


def train_map(model: FeatureClassifier, X: np.ndarray, y: np.ndarray,
              weight_decay: float = 1e-2, lr: float = 1e-2, epochs: int = 500):
    """Trains f = h o g by plain (summed) cross-entropy + weight decay.
    Returns tau_prior = weight_decay * N for the Laplace fit, matching the
    PyTorch weight_decay convention noted in TODO.md Sec. 1b."""
    Xt = torch.as_tensor(X)
    yt = torch.as_tensor(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss(reduction="mean")
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ce(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    tau_prior = weight_decay * X.shape[0]
    return tau_prior


def data_bounds(*point_arrays, margin: float = 0.6):
    """Single source of truth for the plotting window. Returns (xlim, ylim)
    enclosing every point in the given (N, 2) arrays, padded by `margin`.
    Feed it every set that appears on the canvas (source, all targets, probe
    points) so the axis limits, the prediction grid and imshow's `extent` all
    come from one computed value -- nothing is hardcoded and no cluster can
    silently fall off the edge when a shift is made larger."""
    pts = np.concatenate([np.asarray(a, dtype=float).reshape(-1, 2)
                          for a in point_arrays], axis=0)
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def grid_2d(xlim, ylim, n: int = 200):
    """Dense evaluation grid over the given window. `xlim`/`ylim` are required
    (compute them once with `data_bounds`) rather than defaulted, so the grid
    always matches the data actually being plotted."""
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    return XX, YY, grid


def blended_surface(probs: np.ndarray, fade: np.ndarray = None) -> np.ndarray:
    """probs: (H*W, K) or (H,W,K). Blend class colours by predicted
    probability, fading to white as confidence drops (paper Fig. 4: 'decision
    boundaries are shaded... proportional to the strength of colours'). If
    `fade` (H,W) in [0,1] is given (e.g. normalized epistemic uncertainty),
    it additionally washes the image out towards white -- this is what makes
    the uncertainty-guided row visibly 'know what it doesn't know' far from
    the source support."""
    shape = probs.shape[:-1]
    base = probs.reshape(-1, probs.shape[-1]) @ CLASS_COLORS
    conf = probs.reshape(-1, probs.shape[-1]).max(axis=-1, keepdims=True)
    img = conf * base + (1 - conf) * 1.0
    if fade is not None:
        f = fade.reshape(-1, 1)
        img = (1 - f) * img + f * 1.0
    return np.clip(img.reshape(*shape, 3), 0.0, 1.0)