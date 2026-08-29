"""MAP training of code_v2/src/digits_model.py's SmallCNN32 on SVHN.

SVHN is the source domain, chosen for the most pronounced shift toward
MNIST: real, cluttered street-view digit photographs vs. clean
handwritten/scanned digits -- a harder, more informative unsupervised
domain-adaptation setting than MNIST<->USPS, whose source-only accuracy is
already high enough to leave little room for adaptation to show an effect.

Weight decay: `WEIGHT_DECAY = 1e-3` below, passed straight into `AdamW`
(same value `bayesian_model.py::train_map` itself uses as its own default).
Kept explicit here because it is what a later `tau_prior = weight_decay *
N_source` last-layer Laplace fit needs (same convention as
`har_train.py`/`code_v2/notebooks/03_bayesian_source_mnist.ipynb`, both of
which derive `tau_prior` the same way) -- this file only trains and saves
the checkpoint; fitting the Laplace posterior is left to the notebook that
uses it.

Not built on top of `bayesian_model.py::train_map` directly: `train_map`
runs a fixed epoch budget with no validation split and no early stopping,
which is fine for MNIST in `03_bayesian_source_mnist.ipynb` (3 epochs is
enough there) but not for SVHN, which needs more epochs to converge and
clearly overfits past its best point if left running (observed at ~24
epochs in an earlier version of this experiment, before it was moved
here). This file therefore holds out an internal validation split from
SVHN train and stops on validation loss, then reuses `train_map`'s own
optimizer convention (`AdamW`, same `weight_decay`) for the actual steps.

Device: uses Apple's MPS backend when available -- `SmallCNN32`, like the
architecture used here before the move to `code_v2`, hits an unreasonably
slow conv path in this machine's CPU-only PyTorch build (measured directly
on the previous architecture: ~24x slower per training step than MPS).
Falls back to CPU automatically if MPS is unavailable.
"""
import os
import sys
import time
from pathlib import Path

# Repo root on sys.path with the fully-qualified `code_v2.src.xxx` form below
# (not the bare `src.xxx` the notebooks use) -- the repo also has its own
# top-level `src/` package (main project), which has an `__init__.py` and
# therefore always wins name resolution for a bare `src` once the repo root
# is anywhere on sys.path, shadowing `code_v2/src` silently. Fully qualifying
# the import avoids that collision regardless of the caller's cwd.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from code_v2.src.digits_data import compute_source_stats, load_domain
from code_v2.src.digits_model import SmallCNN32

SOURCE_DOMAIN = "svhn"
TARGET_DOMAINS = ["mnist", "usps"]

WEIGHT_DECAY = 1e-3
LR = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 5
VAL_FRACTION = 0.1
SEED = 2019

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "source_svhn"


def resolve_device(device_name: str | None = None) -> torch.device:
    """Resolve the preferred training device, with CUDA > MPS > CPU.

    Users may override via `--device cuda|mps|cpu` or the environment variable
    `BAYESIAN_EXO_DEVICE`.
    """
    override = (device_name or os.environ.get("BAYESIAN_EXO_DEVICE") or "").strip().lower()
    if override:
        if override in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available on this machine.")
            return torch.device("cuda")
        if override == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available on this machine.")
            return torch.device("mps")
        if override == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported device '{device_name}'. Use cuda, mps, or cpu.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


for arg in sys.argv[1:]:
    if arg.startswith("--device="):
        DEVICE = resolve_device(arg.split("=", 1)[1])
        break
else:
    DEVICE = resolve_device()


def evaluate(model: nn.Module, loader: DataLoader) -> tuple:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b)
            loss_sum += criterion(logits, y_b).item() * X_b.shape[0]
            correct += (logits.argmax(dim=1) == y_b).sum().item()
            total += X_b.shape[0]
    return correct / total, loss_sum / total


def train_source_model(seed: int = SEED, source_domain: str = SOURCE_DOMAIN, verbose: bool = True) -> dict:
    """Trains one SVHN source model end-to-end for a given `seed` -- the
    internal train/val split, the weight init, and the minibatch order all
    re-derive from it. Used by both `main()` below (single run, `seed=SEED
    =2019`, saves a checkpoint) and `code_v2/notebooks/15_digits_multiseed.ipynb`
    (5 independent calls, one per seed, nothing saved to disk -- kept in
    memory for that notebook's own aggregation).

    Returns a dict with `model`, `mean`, `std`, `weight_decay`,
    `n_source_train`, `source_test_acc`, `target_test_acc`, `train_time`,
    `n_epochs_run` -- everything a caller needs to fit a Laplace posterior
    afterwards without re-doing any of this."""
    target_domains = [d for d in ["mnist", "usps", "svhn"] if d != source_domain]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if verbose:
        print(f"Device: {DEVICE}")
        print(f"Computing {source_domain} (source) normalization stats from its training split...")
    mean, std = compute_source_stats(source_domain)
    if verbose:
        print(f"  mean={mean:.4f}  std={std:.4f}")

    X_train_full, y_train_full = load_domain(source_domain, "train", mean, std)
    X_test, y_test = load_domain(source_domain, "test", mean, std)
    X_train_full, y_train_full = X_train_full.to(DEVICE), y_train_full.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
    if verbose:
        print(f"{source_domain} train: {X_train_full.shape[0]} images   "
              f"{source_domain} test: {X_test.shape[0]} images")

    full_train_ds = TensorDataset(X_train_full, y_train_full)
    n_val = int(len(full_train_ds) * VAL_FRACTION)
    n_train = len(full_train_ds) - n_val
    train_ds, val_ds = random_split(full_train_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))
    if verbose:
        print(f"Internal split ({source_domain} train only): {n_train} train / {n_val} val "
              f"(val_fraction={VAL_FRACTION}, seed={seed}) -- {source_domain} test untouched until the final report")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    model = SmallCNN32(n_classes=10, feature_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        print("Training configuration:")
        print(f"  optimizer=AdamW  lr={LR}  weight_decay={WEIGHT_DECAY}")
        print(f"  batch_size={BATCH_SIZE}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        train_acc, train_loss = evaluate(model, train_loader)
        val_acc, val_loss = evaluate(model, val_loader)
        if verbose:
            print(f"epoch {epoch + 1:3d}/{MAX_EPOCHS}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} (patience={PATIENCE}, "
                          f"best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    train_time = time.time() - t0
    n_epochs_run = epoch + 1

    svhn_test_acc, _ = evaluate(model, test_loader)

    if verbose:
        print(f"Evaluating (no adaptation) on {target_domains}, normalized with "
              f"{source_domain}-train stats (mean={mean:.4f}, std={std:.4f})...")
    target_acc = {}
    for domain in target_domains:
        X_t, y_t = load_domain(domain, "test", mean, std)
        X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=False)
        acc, _ = evaluate(model, loader)
        target_acc[domain] = acc
        if verbose:
            print(f"  {domain} test accuracy (no adaptation): {100 * acc:.2f}%")

    return dict(
        model=model, mean=mean, std=std, weight_decay=WEIGHT_DECAY, n_source_train=n_train,
        feature_dim=128, n_classes=10, seed=seed, source_domain=source_domain,
        target_domains=target_domains, source_test_acc=svhn_test_acc, target_test_acc=target_acc,
        train_time=train_time, n_epochs_run=n_epochs_run,
    )


def main():
    ckpt_path = CHECKPOINT_DIR / "model.pt"
    if ckpt_path.exists():
        print(f"{ckpt_path} esiste già -- training saltato (nessun nuovo modello addestrato). "
              f"Cancella il file, o usa un altro CHECKPOINT_DIR, per riaddestrare da zero.")
        return

    result = train_source_model(seed=SEED, source_domain=SOURCE_DOMAIN, verbose=True)
    model = result["model"]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "source_domain": SOURCE_DOMAIN,
        "source_mean": result["mean"],
        "source_std": result["std"],
        "weight_decay": result["weight_decay"],
        "n_source_train": result["n_source_train"],
        "feature_dim": result["feature_dim"],
        "n_classes": result["n_classes"],
        "seed": SEED,
        "svhn_test_acc": result["source_test_acc"],
        "target_test_acc": result["target_test_acc"],
    }, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total training time: {result['train_time']:.1f} sec ({result['train_time'] / 60:.1f} min)")
    print(f"{SOURCE_DOMAIN} (source) test accuracy:  {100 * result['source_test_acc']:.2f}%")
    for domain in TARGET_DOMAINS:
        print(f"{domain} test accuracy (no adaptation): {100 * result['target_test_acc'][domain]:.2f}%")


if __name__ == "__main__":
    main()
