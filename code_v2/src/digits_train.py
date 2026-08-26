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
`har_train.py`/`code_v2/notebooks/02_bayesian_source_mnist.ipynb`, both of
which derive `tau_prior` the same way) -- this file only trains and saves
the checkpoint; fitting the Laplace posterior is left to the notebook that
uses it.

Not built on top of `bayesian_model.py::train_map` directly: `train_map`
runs a fixed epoch budget with no validation split and no early stopping,
which is fine for MNIST in `02_bayesian_source_mnist.ipynb` (3 epochs is
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

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


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


def main():
    ckpt_path = CHECKPOINT_DIR / "model.pt"
    if ckpt_path.exists():
        print(f"{ckpt_path} esiste già -- training saltato (nessun nuovo modello addestrato). "
              f"Cancella il file, o usa un altro CHECKPOINT_DIR, per riaddestrare da zero.")
        return

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")

    print(f"Computing {SOURCE_DOMAIN} (source) normalization stats from its training split...")
    mean, std = compute_source_stats(SOURCE_DOMAIN)
    print(f"  mean={mean:.4f}  std={std:.4f}")

    X_train_full, y_train_full = load_domain(SOURCE_DOMAIN, "train", mean, std)
    X_test, y_test = load_domain(SOURCE_DOMAIN, "test", mean, std)
    X_train_full, y_train_full = X_train_full.to(DEVICE), y_train_full.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
    print(f"{SOURCE_DOMAIN} train: {X_train_full.shape[0]} images   "
          f"{SOURCE_DOMAIN} test: {X_test.shape[0]} images")

    full_train_ds = TensorDataset(X_train_full, y_train_full)
    n_val = int(len(full_train_ds) * VAL_FRACTION)
    n_train = len(full_train_ds) - n_val
    train_ds, val_ds = random_split(full_train_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    print(f"Internal split (SVHN train only): {n_train} train / {n_val} val "
          f"(val_fraction={VAL_FRACTION}, seed={SEED}) -- SVHN test untouched until the final report")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    model = SmallCNN32(n_classes=10, feature_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining configuration:")
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
        print(f"epoch {epoch + 1:3d}/{MAX_EPOCHS}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} (patience={PATIENCE}, "
                      f"best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    total_time = time.time() - t0

    svhn_test_acc, _ = evaluate(model, test_loader)

    print(f"\nEvaluating (no adaptation) on {TARGET_DOMAINS}, normalized with "
          f"{SOURCE_DOMAIN}-train stats (mean={mean:.4f}, std={std:.4f})...")
    target_acc = {}
    for domain in TARGET_DOMAINS:
        X_t, y_t = load_domain(domain, "test", mean, std)
        X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=False)
        acc, _ = evaluate(model, loader)
        target_acc[domain] = acc

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "model.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "source_domain": SOURCE_DOMAIN,
        "source_mean": mean,
        "source_std": std,
        "weight_decay": WEIGHT_DECAY,
        "n_source_train": n_train,
        "feature_dim": 128,
        "n_classes": 10,
        "seed": SEED,
        "svhn_test_acc": svhn_test_acc,
        "target_test_acc": target_acc,
    }, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total training time: {total_time:.1f} sec ({total_time / 60:.1f} min)")
    print(f"{SOURCE_DOMAIN} (source) test accuracy:  {100 * svhn_test_acc:.2f}%")
    for domain in TARGET_DOMAINS:
        print(f"{domain} test accuracy (no adaptation): {100 * target_acc[domain]:.2f}%")


if __name__ == "__main__":
    main()
