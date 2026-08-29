"""Sanity check on the just-trained amazon_reviews/train.py checkpoint.

Same convention as code_v2/src/digits_verify.py: reload the checkpoint
from scratch (fresh process, fresh model instance, fresh data
loading/vectorization), recompute electronics (source) test accuracy and
the three target accuracies, and compare against the values train.py
itself saved into the checkpoint at the end of that same run. The point
is not "does the model work" but "does reloading it from disk reproduce
the exact same number" -- a mismatch would mean a silent discrepancy
between training-time evaluation and any later use of this checkpoint
(e.g. a different train/val/test split, a re-fit TF-IDF instead of the
saved one, evaluation not in `.eval()` mode).
"""
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_domain, stratified_split, transform_tfidf  # noqa: E402
from model import build_model  # noqa: E402

CHECKPOINT_DIR = Path(__file__).resolve().parent / "models" / "source_electronics"
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
# Deliberately loose, same rationale as digits_verify.py: a real
# discrepancy (wrong split, re-fit vocabulary) changes accuracy by many
# points, not a fraction of one.
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


def to_tensor(X_sparse, y):
    return torch.tensor(X_sparse.toarray(), dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def main():
    ckpt_path = CHECKPOINT_DIR / "model.pt"
    transformer_path = CHECKPOINT_DIR / "tfidf_transformer.pkl"
    if not ckpt_path.exists() or not transformer_path.exists():
        print(f"ERROR: {ckpt_path} or {transformer_path} not found -- run amazon_reviews/train.py first")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for required_key in ("source_test_acc", "target_test_acc", "vocab", "seed"):
        if required_key not in ckpt:
            print(f"ERROR: checkpoint has no '{required_key}' -- re-run amazon_reviews/train.py to regenerate it")
            sys.exit(1)

    with open(transformer_path, "rb") as f:
        transformer = pickle.load(f)
    vocab = ckpt["vocab"]

    print("=" * 60)
    print("Verifying amazon_reviews sentiment source checkpoint")
    print("=" * 60)
    print(f"checkpoint: {ckpt_path}")
    print(f"source domain: {ckpt['source_domain']}  seed: {ckpt['seed']}")
    print(f"weight_decay: {ckpt['weight_decay']}  n_source_train: {ckpt['n_source_train']}")
    print(f"vocabolario: {len(vocab)} termini")

    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    domain = ckpt["source_domain"]
    print(f"\n[1/2] Recomputing {domain} test accuracy "
          f"(stesso split stratificato, seed={ckpt['seed']})...")
    counts_source, labels_source = load_domain(domain)
    _, _, test_idx = stratified_split(labels_source, VAL_FRACTION, TEST_FRACTION, ckpt["seed"])
    X_test_sparse = transform_tfidf([counts_source[i] for i in test_idx], vocab, transformer)
    X_test, y_test = to_tensor(X_test_sparse, labels_source[test_idx])
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)
    recomputed_source_acc = evaluate(model, test_loader)

    reference_source_acc = ckpt["source_test_acc"]
    diff = abs(recomputed_source_acc - reference_source_acc)
    print(f"      recomputed: {100 * recomputed_source_acc:.4f}%")
    print(f"      reference (saved at end of training): {100 * reference_source_acc:.4f}%")
    print(f"      |diff|: {100 * diff:.6f} pp")
    source_pass = diff < PASS_TOLERANCE
    print("      -> PASS" if source_pass else
          "      -> FAIL: reload does not reproduce training-time accuracy -- "
          "check split/seed, TF-IDF transformer, or eval() mode before trusting this checkpoint further")

    target_domains = list(ckpt["target_test_acc"].keys())
    print(f"\n[2/2] Recomputing target accuracy (no adaptation) for confirmation "
          f"({target_domains}, same electronics-train TF-IDF vocabulary/IDF)...")
    all_pass = source_pass
    for t_domain in target_domains:
        counts_t, labels_t = load_domain(t_domain)
        X_t_sparse = transform_tfidf(counts_t, vocab, transformer)
        X_t, y_t = to_tensor(X_t_sparse, labels_t)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)
        recomputed = evaluate(model, loader)
        reference = ckpt["target_test_acc"][t_domain]
        t_diff = abs(recomputed - reference)
        t_pass = t_diff < PASS_TOLERANCE
        all_pass = all_pass and t_pass
        print(f"      {t_domain:>12s}: recomputed={100 * recomputed:.4f}%  "
              f"reference={100 * reference:.4f}%  |diff|={100 * t_diff:.6f}pp  "
              f"-> {'PASS' if t_pass else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"{domain} (source) test accuracy: {100 * recomputed_source_acc:.2f}%")
    for t_domain in target_domains:
        print(f"{t_domain} test accuracy (no adaptation): {100 * ckpt['target_test_acc'][t_domain]:.2f}%")
    print("=" * 60)
    if all_pass:
        print("OVERALL: PASS -- reloaded checkpoint reproduces training-time accuracy exactly")
    else:
        print("OVERALL: FAIL -- see mismatches above")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
