"""Loads the MNIST/USPS/SVHN numpy caches produced by
code_v2/notebooks/07_digits_data.ipynb and applies the shared preprocessing
every domain needs before they can be fed to the same model:

1. Grayscale: MNIST and USPS are already single-channel; only SVHN (RGB)
   is converted, via torchvision.transforms.Grayscale, so a model built for
   one input channel never sees three.
2. Resize to a common resolution (32x32, SVHN's native size): MNIST (28x28)
   and USPS (16x16) are upsampled to it, so all three domains produce
   identically-shaped tensors.
3. Normalization: mean/std are computed once from the *source* domain's
   training split only (`compute_source_stats`) and then applied, unchanged,
   to every domain and split (`load_domain`) -- this is standard practice in
   unsupervised domain adaptation: normalizing every domain to the source's
   own pixel statistics keeps a difference in raw pixel intensity/contrast
   from being mistaken for the covariate shift the model is actually meant
   to adapt to.

This module depends only on numpy/torch/torchvision/PIL and the .npz caches
in code_v2/data/digits/ -- no other code_v2/src module imports from here.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "digits"
DOMAINS = ["mnist", "usps", "svhn"]
IMG_SIZE = 32
N_CLASSES = 10


def _load_raw(domain: str, split: str) -> tuple:
    """Returns (X, y): X is (N, H, W) uint8 for mnist/usps, (N, H, W, 3)
    uint8 for svhn; y is (N,) int64, already in [0, 9] for every domain."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}, must be one of {DOMAINS}")
    npz_path = DATA_DIR / f"{domain}_{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found -- run "
                                f"code_v2/notebooks/07_digits_data.ipynb first")
    npz = np.load(npz_path)
    return npz["X"], npz["y"]


def _preprocess_pipeline(domain: str) -> transforms.Compose:
    ops = []
    if domain == "svhn":
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.append(transforms.Resize((IMG_SIZE, IMG_SIZE)))
    ops.append(transforms.ToTensor())  # -> float32 in [0, 1], shape (1, IMG_SIZE, IMG_SIZE)
    return transforms.Compose(ops)


def _to_tensor_unnormalized(X_raw: np.ndarray, domain: str) -> torch.Tensor:
    """Grayscale (SVHN only) + resize to IMG_SIZE, NOT yet normalized.
    Returns (N, 1, IMG_SIZE, IMG_SIZE) float32 in [0, 1]."""
    pipeline = _preprocess_pipeline(domain)
    mode = "RGB" if domain == "svhn" else "L"
    out = torch.empty(len(X_raw), 1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    for i, img in enumerate(X_raw):
        pil_img = Image.fromarray(img, mode=mode)
        out[i] = pipeline(pil_img)
    return out


def compute_source_stats(source_domain: str) -> tuple:
    """Mean/std of the (grayscale, resized, but not yet normalized) pixel
    values of `source_domain`'s TRAINING split only -- never the test split,
    never another domain. Pass the result into every `load_domain` call
    below, for every domain, so all of them share this one normalization."""
    X_raw, _ = _load_raw(source_domain, "train")
    X_tensor = _to_tensor_unnormalized(X_raw, source_domain)
    return X_tensor.mean().item(), X_tensor.std().item()


def load_domain(domain: str, split: str, mean: float, std: float) -> tuple:
    """Returns (X, y): X is (N, 1, IMG_SIZE, IMG_SIZE) float32, grayscale,
    resized to IMG_SIZE, normalized with the given (mean, std) -- computed
    once from the source domain via `compute_source_stats`, reused as-is
    for every domain/split so nothing is normalized against its own
    statistics. y is (N,) int64 in [0, 9]."""
    X_raw, y_raw = _load_raw(domain, split)
    X_tensor = _to_tensor_unnormalized(X_raw, domain)
    X_tensor = (X_tensor - mean) / std
    return X_tensor, torch.tensor(y_raw, dtype=torch.long)


if __name__ == "__main__":
    print("Image counts per domain/split (code_v2/data/digits/*.npz):")
    print("-" * 60)
    for domain in DOMAINS:
        for split in ["train", "test"]:
            X_raw, y_raw = _load_raw(domain, split)
            print(f"  {domain:6s} {split:5s}: {X_raw.shape[0]:6d} images, "
                  f"native shape {X_raw.shape[1:]}, "
                  f"labels {sorted(set(y_raw.tolist()))}")
