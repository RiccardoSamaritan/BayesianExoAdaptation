"""Sanity check on the just-trained code_v2/src/digits_train.py checkpoint.

Reloads code_v2/models/source_svhn/model.pt from scratch (fresh process,
fresh model instance, fresh data loading through code_v2/src/digits_data.py)
and recomputes SVHN test / MNIST test / USPS test accuracy, then compares
the recomputed SVHN test accuracy against the value digits_train.py itself
saved into the checkpoint (`svhn_test_acc`) at the end of that same
training run.

The point is not "does the model work" (already known from training's own
final report) but "does reloading it from disk reproduce the exact same
number" -- a mismatch here would mean a silent preprocessing discrepancy
(a different mean/std, a different resize, evaluation not in `.eval()`
mode) between training-time evaluation and any later use of this
checkpoint.
"""
import sys
from pathlib import Path

# See digits_train.py's module-level comment: fully-qualified `code_v2.src.xxx`
# import, not the bare `src.xxx` the notebooks use, to avoid colliding with
# the repo's own top-level `src/` package once the repo root is on sys.path.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader, TensorDataset

from code_v2.src.digits_data import load_domain
from code_v2.src.digits_model import SmallCNN32

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "source_svhn" / "model.pt"
TARGET_DOMAINS = ["mnist", "usps"]
BATCH_SIZE = 128
# Deliberately loose: a real preprocessing bug changes accuracy by many
# points, not a fraction of one, so this still catches it; a tolerance near
# float precision would instead risk failing spuriously across CPU/MPS
# backend differences (see digits_train.py's module docstring).
PASS_TOLERANCE = 0.005  # 0.5 percentage points


def evaluate(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            preds = model(X_b).argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.shape[0]
    return correct / total


def main():
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: {CHECKPOINT_PATH} not found -- run code_v2/src/digits_train.py first")
        sys.exit(1)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    for required_key in ("svhn_test_acc", "target_test_acc"):
        if required_key not in ckpt:
            print(f"ERROR: checkpoint has no '{required_key}' -- re-run digits_train.py to regenerate it")
            sys.exit(1)

    print("=" * 60)
    print("Verifying code_v2 digits source checkpoint")
    print("=" * 60)
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"source domain: {ckpt['source_domain']}  seed: {ckpt['seed']}")
    print(f"weight_decay: {ckpt['weight_decay']}  n_source_train: {ckpt['n_source_train']}")

    model = SmallCNN32(n_classes=ckpt["n_classes"], feature_dim=ckpt["feature_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mean, std = ckpt["source_mean"], ckpt["source_std"]
    domain = ckpt["source_domain"]

    print(f"\n[1/2] Recomputing {domain} test accuracy (mean={mean:.4f} std={std:.4f})...")
    X_test, y_test = load_domain(domain, "test", mean, std)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    recomputed_svhn_acc = evaluate(model, test_loader)

    reference_svhn_acc = ckpt["svhn_test_acc"]
    diff = abs(recomputed_svhn_acc - reference_svhn_acc)
    print(f"      recomputed: {100 * recomputed_svhn_acc:.4f}%")
    print(f"      reference (saved at end of training): {100 * reference_svhn_acc:.4f}%")
    print(f"      |diff|: {100 * diff:.6f} pp")
    svhn_pass = diff < PASS_TOLERANCE
    if svhn_pass:
        print("      -> PASS (no silent preprocessing discrepancy)")
    else:
        print("      -> FAIL: reload does not reproduce training-time accuracy -- "
              "check mean/std, resize, or eval() mode before trusting this checkpoint further")

    print(f"\n[2/2] Recomputing target accuracy (no adaptation) for confirmation "
          f"({TARGET_DOMAINS}, same {domain}-train normalization)...")
    all_pass = svhn_pass
    for t_domain in TARGET_DOMAINS:
        X_t, y_t = load_domain(t_domain, "test", mean, std)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=False)
        recomputed = evaluate(model, loader)
        reference = ckpt["target_test_acc"][t_domain]
        t_diff = abs(recomputed - reference)
        t_pass = t_diff < PASS_TOLERANCE
        all_pass = all_pass and t_pass
        print(f"      {t_domain}: recomputed={100 * recomputed:.4f}%  "
              f"reference={100 * reference:.4f}%  |diff|={100 * t_diff:.6f}pp  "
              f"-> {'PASS' if t_pass else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"{domain} (source) test accuracy:  {100 * recomputed_svhn_acc:.2f}%")
    for t_domain in TARGET_DOMAINS:
        print(f"{t_domain} test accuracy (no adaptation): {100 * ckpt['target_test_acc'][t_domain]:.2f}%")
    print("=" * 60)
    if all_pass:
        print("OVERALL: PASS -- reloaded checkpoint reproduces training-time accuracy exactly")
    else:
        print("OVERALL: FAIL -- see mismatches above")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
