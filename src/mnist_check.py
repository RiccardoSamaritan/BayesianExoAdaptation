"""MNIST vs Fashion-MNIST sanity check (TODO.md Sec. 3b, Part 1).

Reuses the exact-Hessian last-layer Laplace + BALD code from `bayesian.py`
completely unchanged -- only the feature extractor `g` differs from the
synthetic toy (a small CNN here instead of an MLP), exactly as prescribed
by TODO.md Sec. 3b ("Needs a small CNN feature extractor (not the HAR MLP).
Laplace/BALD code unchanged -- operates only on the last linear layer.").

Replicates the OOD-detection experiment of Kristiadi, Hein & Hennig
(ICML 2020) -- ref. [26] in the U-SFAN paper -- at a scale small enough to
run on CPU: a source model trained on MNIST should be well-calibrated
(low epistemic uncertainty) on in-distribution MNIST test data, and flag
Fashion-MNIST test data (same image shape, entirely different content) as
epistemically uncertain, even though the deterministic MAP network alone
gives no such signal.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from .bayesian import FeatureClassifier

N_CLASSES = 10
FEATURE_DIM = 128


class SmallCNN(nn.Module):
    """Two conv blocks + one FC layer, 28x28x1 -> FEATURE_DIM. Deliberately
    small (fast to train on CPU); the point of this check is the last-layer
    Laplace treatment, not CNN accuracy."""

    def __init__(self, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.fc = nn.Sequential(nn.Linear(32 * 7 * 7, feature_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = z.flatten(start_dim=1)
        return self.fc(z)


def build_model(feature_dim: int = FEATURE_DIM, seed: int = 0) -> FeatureClassifier:
    torch.manual_seed(seed)
    return FeatureClassifier.from_feature_extractor(SmallCNN(feature_dim), feature_dim, N_CLASSES)


def load_datasets(data_root: str, n_train: int = 10000, n_test: int = 2000, seed: int = 0):
    """Returns (X_train, y_train, X_mnist_test, y_mnist_test, X_fmnist_test,
    y_fmnist_test) as float32 tensors in [0, 1], shape (N, 1, 28, 28).
    `n_train`/`n_test` subsample each split (with a fixed seed) to keep this
    a fast sanity check rather than a full training run; downloads MNIST and
    FashionMNIST into `data_root` via torchvision on first use."""
    to_tensor = transforms.ToTensor()
    mnist_train = datasets.MNIST(root=data_root, train=True, download=True, transform=to_tensor)
    mnist_test = datasets.MNIST(root=data_root, train=False, download=True, transform=to_tensor)
    fmnist_test = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=to_tensor)

    rng = np.random.RandomState(seed)

    def subsample(ds, n):
        n = min(n, len(ds))
        idx = rng.choice(len(ds), size=n, replace=False)
        X = torch.stack([ds[i][0] for i in idx])
        y = torch.tensor([ds[i][1] for i in idx], dtype=torch.long)
        return X, y

    X_train, y_train = subsample(mnist_train, n_train)
    X_mnist_test, y_mnist_test = subsample(mnist_test, n_test)
    X_fmnist_test, y_fmnist_test = subsample(fmnist_test, n_test)
    return X_train, y_train, X_mnist_test, y_mnist_test, X_fmnist_test, y_fmnist_test


def train_map(model: FeatureClassifier, X: torch.Tensor, y: torch.Tensor,
              weight_decay: float = 1e-4, lr: float = 1e-3, epochs: int = 5,
              batch_size: int = 128, seed: int = 0):
    """Minibatch MAP training (CNN + linear head), plain cross-entropy +
    weight decay. Returns tau_prior = weight_decay * N for the Laplace fit
    (TODO.md Sec. 1b convention, same as `toy.train_map`)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss(reduction="mean")
    n = X.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss, correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
            correct += (logits.argmax(dim=1) == yb).sum().item()
        print(f"  epoch {epoch + 1}/{epochs}  loss={total_loss / n:.4f}  train_acc={correct / n:.4f}")
    model.eval()
    tau_prior = weight_decay * n
    return tau_prior
