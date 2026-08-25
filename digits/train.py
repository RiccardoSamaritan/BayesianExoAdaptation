"""MAP training of digits/model.py's source model on SVHN.

SVHN is the source domain (digits/data_utils.py's DOMAINS list), chosen for
the most pronounced shift toward MNIST: real, cluttered street-view digit
photographs vs. clean handwritten/scanned digits -- a harder, more
informative unsupervised-domain-adaptation setting than SHOT's own
MNIST<->USPS tasks (u2m/m2u), whose source-only baselines already sit at
90-96% (`results.md`, "UDA results on Digits" table) -- too easy to leave
much room for adaptation to show an effect. SVHN's own source-only s2m
baseline is far lower, which is exactly why it is the more useful source
here.

Weight decay: `WEIGHT_DECAY = 1e-3` below, passed straight into AdamW.
Chosen to match the exact value SHOT hardcodes for this benchmark
(`digit/uda_digit.py::lr_scheduler`, line 27: `param_group['weight_decay']
= 1e-3` -- code-verified, not assumed). Kept explicit here because it is
what a later `tau_prior = weight_decay * N_source` last-layer Laplace fit
will need (same convention as `src/har_train.py`) -- this file only trains
and saves the checkpoint; fitting the Laplace posterior is left to whatever
notebook uses it next.

Two deliberate deviations from SHOT's own `train_source()`
(`digit/uda_digit.py`):

1. SHOT selects its best snapshot by accuracy on the SVHN *test* split
   (`cal_acc(dset_loaders['source_te'], ...)`, `source_te` being the actual
   SVHN test set). This file instead holds out an internal validation split
   from the SVHN *training* set only (`VAL_FRACTION` below) and never looks
   at the test split until the one-time final report -- the test split
   stays a genuine held-out set here.
2. `src/har_train.py::train_source_model` (the project's existing
   source-training helper) does full-batch gradient descent -- correct and
   cheap for HAR's ~2,000-row tabular dataset, but SVHN's 73,257 32x32
   images through a CNN cannot fit in a single forward+backward pass on
   this machine. Training below therefore uses mini-batches via
   `DataLoader`, and is not built on top of `train_source_model`.

Device: unlike the HAR pipeline (established as CPU-only, no CUDA on that
environment), this file uses Apple's MPS backend when available, verified
as a deliberate, measured choice rather than a default: this
machine's installed PyTorch (2.4.0, arm64, no MKLDNN/AVX -- expected on
Apple Silicon, but with no MPS-independent fast conv path either) hits an
absurdly slow fallback for some of this network's specific conv shapes --
benchmarked directly, a single 64->128 channel, kernel-5, stride-2 conv
(the exact shape in `DTNBase`'s second layer) on a batch of 128 32x32
images took 452 ms on CPU. The identical op on MPS took 2.6 ms (~170x), and
a full training step (forward+backward+optimizer, whole model) went from
1830 ms/batch on CPU to 75 ms/batch on MPS (~24x) -- the difference between
a ~13-hour and a ~30-minute run for the epoch budget below. Falls back to
CPU automatically if MPS is unavailable.
"""
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from digits.data_utils import compute_source_stats, load_domain
from digits.model import build_digit_model

SOURCE_DOMAIN = "svhn"
TARGET_DOMAINS = ["mnist", "usps"]

WEIGHT_DECAY = 1e-3  # digit/uda_digit.py::lr_scheduler, line 27 -- see module docstring
LR = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 30  # matches SHOT's own --max_epoch default for this benchmark
PATIENCE = 5
VAL_FRACTION = 0.1
SEED = 2019

CHECKPOINT_DIR = Path(__file__).resolve().parent / "models" / "source_svhn"

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

    model = build_digit_model(bottleneck_dim=256, n_classes=10, classifier_type="bn", layer_type="wn")
    model = model.to(DEVICE)
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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "model.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "source_domain": SOURCE_DOMAIN,
        "source_mean": mean,
        "source_std": std,
        "weight_decay": WEIGHT_DECAY,
        "n_source_train": n_train,
        "bottleneck_dim": 256,
        "n_classes": 10,
        "classifier_type": "bn",
        "layer_type": "wn",
        "seed": SEED,
    }, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")

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

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total training time: {total_time:.1f} sec ({total_time / 60:.1f} min)")
    print(f"{SOURCE_DOMAIN} (source) test accuracy:  {100 * svhn_test_acc:.2f}%")
    for domain in TARGET_DOMAINS:
        print(f"{domain} test accuracy (no adaptation): {100 * target_acc[domain]:.2f}%")


if __name__ == "__main__":
    main()
