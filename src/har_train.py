"""Source MAP training for UCI HAR locomotion classification (TODO.md §4).

Architecture: 561 -> 128 -> 64 -> 3 MLP (FeatureClassifier from bayesian.py).
Task: 3-class locomotion (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS).
Training: AdamW + cross-entropy (with class weights if imbalanced) on source pool.
Evaluation: accuracy + macro-recall on source val, per-subject accuracy on targets.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score

from .bayesian import FeatureClassifier
from .loader import HARDataset


def subject_train_val_split(subject_ids: np.ndarray, val_fraction: float = 0.2,
                            seed: int = 0) -> tuple:
    """Split subjects into train/val sets (for source pool internal validation).

    Args:
        subject_ids: array of subject IDs to split
        val_fraction: fraction of subjects to use for validation
        seed: random seed

    Returns:
        (train_subjects, val_subjects) as sorted arrays
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(subject_ids)
    n_val = max(1, int(len(subject_ids) * val_fraction))
    val_subjects = np.sort(shuffled[:n_val])
    train_subjects = np.sort(shuffled[n_val:])
    return train_subjects, val_subjects


def check_class_balance(y: np.ndarray, n_classes: int, split_name: str = ""):
    """Check class distribution and return class weights if imbalanced.

    Args:
        y: label array
        n_classes: number of classes
        split_name: name for logging (e.g., "source train")

    Returns:
        class_weights: torch tensor of shape (n_classes,), or None if balanced
    """
    counts = np.array([(y == k).sum() for k in range(n_classes)])
    total = len(y)
    freqs = counts / total

    print(f"\nClass distribution ({split_name}):")
    print(f"  Counts: {counts}")
    print(f"  Frequencies: {freqs}")

    # Consider balanced if max/min ratio < 1.5
    imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")

    if imbalance_ratio > 1.5:
        # Inverse frequency weighting
        weights = total / (n_classes * counts)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        print(f"  Using class weights (inverse frequency): {weights}")
        return weights_tensor
    else:
        print(f"  Classes reasonably balanced, no weighting needed")
        return None


def train_source_model(model: FeatureClassifier, X_train: torch.Tensor, y_train: torch.Tensor,
                      X_val: torch.Tensor, y_val: torch.Tensor,
                      weight_decay: float = 0.01, lr: float = 1e-3, epochs: int = 100,
                      class_weights: torch.Tensor = None, patience: int = 20) -> dict:
    """Train source MAP model with early stopping on validation loss.

    Args:
        model: FeatureClassifier instance
        X_train, y_train: training data
        X_val, y_val: validation data
        weight_decay: L2 regularization (tau_prior = weight_decay * N_train)
        lr: learning rate
        epochs: maximum epochs
        class_weights: optional class weights for CE loss
        patience: early stopping patience

    Returns:
        dict with training history and tau_prior
    """
    n_train = X_train.shape[0]
    tau_prior = weight_decay * n_train

    print(f"\nTraining configuration:")
    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  tau_prior (weight_decay * N_train): {tau_prior:.2f}")
    print(f"  Max epochs: {epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Train samples: {n_train}, Val samples: {X_val.shape[0]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        'train_loss': [], 'train_acc': [], 'train_recall': [],
        'val_loss': [], 'val_acc': [], 'val_recall': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Train metrics
            train_preds = logits.argmax(dim=1).cpu().numpy()
            train_acc = (train_preds == y_train.cpu().numpy()).mean()
            train_recall = recall_score(y_train.cpu().numpy(), train_preds,
                                       average='macro', zero_division=0)

            # Val metrics
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = (val_preds == y_val.cpu().numpy()).mean()
            val_recall = recall_score(y_val.cpu().numpy(), val_preds,
                                     average='macro', zero_division=0)

        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['train_recall'].append(train_recall)
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc)
        history['val_recall'].append(val_recall)

        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={loss.item():.4f} train_acc={train_acc:.3f} train_recall={train_recall:.3f} | "
                  f"val_loss={val_loss.item():.4f} val_acc={val_acc:.3f} val_recall={val_recall:.3f} | "
                  f"patience={patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    history['tau_prior'] = tau_prior
    history['best_epoch'] = len(history['val_loss']) - patience_counter

    return history


def evaluate_per_subject(model: FeatureClassifier, dataset: HARDataset,
                        subject_ids: np.ndarray, split_name: str = "target") -> dict:
    """Evaluate model separately on each subject.

    Args:
        model: trained FeatureClassifier
        dataset: full HARDataset (will be filtered per subject)
        subject_ids: array of subject IDs to evaluate
        split_name: name for logging

    Returns:
        dict mapping subject_id -> {acc, recall, n_samples}
    """
    model.eval()
    results = {}

    print(f"\nPer-subject evaluation ({split_name}):")
    print("=" * 60)

    for sid in sorted(subject_ids):
        ds_subj = dataset.filter_subjects([sid])
        X = torch.tensor(ds_subj.X, dtype=torch.float32)
        y = torch.tensor(ds_subj.y, dtype=torch.long)

        with torch.no_grad():
            preds = model(X).argmax(dim=1).cpu().numpy()
            acc = (preds == y.numpy()).mean()
            recall = recall_score(y.numpy(), preds, average='macro', zero_division=0)

        results[int(sid)] = {
            'accuracy': acc,
            'recall_macro': recall,
            'n_samples': len(y)
        }

        print(f"  Subject {sid:2d}: acc={acc:.3f}, recall={recall:.3f}, n={len(y)}")

    # Summary statistics
    accs = [r['accuracy'] for r in results.values()]
    recalls = [r['recall_macro'] for r in results.values()]

    print("=" * 60)
    print(f"Summary ({split_name}):")
    print(f"  Accuracy:     mean={np.mean(accs):.3f}, std={np.std(accs):.3f}, "
          f"min={np.min(accs):.3f}, max={np.max(accs):.3f}")
    print(f"  Macro-recall: mean={np.mean(recalls):.3f}, std={np.std(recalls):.3f}, "
          f"min={np.min(recalls):.3f}, max={np.max(recalls):.3f}")

    # Sanity checks
    if np.min(accs) > 0.95:
        print("  ⚠ WARNING: All subjects near saturation (>95% acc) - task may be too easy")
    if np.max(accs) < 0.40:
        print("  ⚠ WARNING: All subjects near chance (~33%) - model may not have learned")
    if np.max(accs) - np.min(accs) < 0.05:
        print("  ⚠ WARNING: Very small accuracy range - limited shift diversity")

    return results
