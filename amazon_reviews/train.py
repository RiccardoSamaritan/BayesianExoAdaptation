"""MAP training di `model.build_model()` su electronics come source.

Perché electronics come source: è uno dei quattro domini del Multi-Domain
Sentiment Dataset che la letteratura (Blitzer et al. 2007; lavori
successivi di domain adaptation su questo stesso dataset) riporta
tipicamente come più "neutro"/meno gergale, con shift più contenuto verso
gli altri domini -- ma questo script verifica empiricamente le tre
accuracy source-only (electronics -> books/dvd/kitchen) invece di
assumerlo, e la scelta del target per un eventuale adattamento successivo
andrà fatta sul dominio con lo shift osservato più marcato (accuracy più
bassa), non sul dominio previsto dalla sola premessa.

TF-IDF (vocabolario + IDF) fittato SOLO sul train split di electronics
(`data_utils.fit_tfidf`), applicato invariato al val/test split di
electronics e a tutti gli altri tre domini -- stesso principio di
`compute_source_stats` in `code_v2/src/digits_data.py`.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import DOMAINS, fit_tfidf, load_domain, stratified_split, transform_tfidf  # noqa: E402
from model import build_model  # noqa: E402

SOURCE_DOMAIN = "electronics"
TARGET_DOMAINS = [d for d in DOMAINS if d != SOURCE_DOMAIN]

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

WEIGHT_DECAY = 1e-3
LR = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8

CHECKPOINT_DIR = Path(__file__).resolve().parent / "models" / f"source_{SOURCE_DOMAIN}"


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


def to_tensor(X_sparse, y) -> tuple:
    X = torch.tensor(X_sparse.toarray(), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def train_source_model(seed: int = SEED, source_domain: str = SOURCE_DOMAIN, verbose: bool = True) -> dict:
    """Trains one source model end-to-end for a given `seed` -- everything
    that depends on the seed (the train/val/test split, the TF-IDF
    vocabulary fitted on that split, the weight init, the minibatch order)
    is re-derived from it, not just the optimizer's random draws. Used by
    both `main()` below (single run, `seed=SEED=42`, saves a checkpoint)
    and `amazon_reviews/sentiment_multiseed.ipynb` (5 independent calls,
    one per seed, nothing saved to disk -- kept in memory for that
    notebook's own aggregation).

    Returns a dict with `model`, `vocab`, `transformer`, `train_idx`,
    `test_idx`, `counts_source`, `labels_source`, `source_test_acc`,
    `target_test_acc`, `train_time`, `load_time`, `n_epochs_run` --
    everything a caller needs to fit a Laplace posterior afterwards
    without re-doing any of this."""
    target_domains = [d for d in DOMAINS if d != source_domain]
    torch.manual_seed(seed)
    np.random.seed(seed)

    t_load0 = time.time()
    if verbose:
        print(f"Caricamento dominio source ({source_domain}, seed={seed})...")
    counts_source, labels_source = load_domain(source_domain)
    train_idx, val_idx, test_idx = stratified_split(labels_source, VAL_FRACTION, TEST_FRACTION, seed)
    train_counts = [counts_source[i] for i in train_idx]

    vocab, transformer, X_train_sparse = fit_tfidf(train_counts, max_features=5000)
    X_val_sparse = transform_tfidf([counts_source[i] for i in val_idx], vocab, transformer)
    X_test_sparse = transform_tfidf([counts_source[i] for i in test_idx], vocab, transformer)
    load_time = time.time() - t_load0
    if verbose:
        print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}  "
              f"vocabolario={len(vocab)} termini  (caricamento+vettorizzazione: {load_time:.1f}s)")

    X_train, y_train = to_tensor(X_train_sparse, labels_source[train_idx])
    X_val, y_val = to_tensor(X_val_sparse, labels_source[val_idx])
    X_test, y_test = to_tensor(X_test_sparse, labels_source[test_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        print(f"Training: optimizer=AdamW lr={LR} weight_decay={WEIGHT_DECAY} "
              f"batch_size={BATCH_SIZE} max_epochs={MAX_EPOCHS} patience={PATIENCE}")

    best_val_loss, best_state, patience_counter = float("inf"), None, 0
    t_train0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        val_acc, val_loss = evaluate(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                if verbose:
                    print(f"Early stopping all'epoca {epoch + 1} (best val_loss={best_val_loss:.4f})")
                break
    model.load_state_dict(best_state)
    train_time = time.time() - t_train0
    n_epochs_run = epoch + 1

    source_test_acc, _ = evaluate(model, test_loader)

    if verbose:
        print("Valutazione (nessun adattamento) sugli altri domini come target...")
    target_acc = {}
    for domain in target_domains:
        counts_t, labels_t = load_domain(domain)
        X_t_sparse = transform_tfidf(counts_t, vocab, transformer)
        X_t, y_t = to_tensor(X_t_sparse, labels_t)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)
        acc, _ = evaluate(model, loader)
        target_acc[domain] = acc
        if verbose:
            print(f"  {domain:>12s} (target, nessun adattamento): {100*acc:.2f}%")

    return dict(
        model=model, vocab=vocab, transformer=transformer,
        counts_source=counts_source, labels_source=labels_source,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        source_domain=source_domain, target_domains=target_domains, seed=seed,
        weight_decay=WEIGHT_DECAY, source_test_acc=source_test_acc, target_test_acc=target_acc,
        load_time=load_time, train_time=train_time, n_epochs_run=n_epochs_run,
    )


def main():
    result = train_source_model(seed=SEED, source_domain=SOURCE_DOMAIN, verbose=True)
    model, vocab, transformer = result["model"], result["vocab"], result["transformer"]
    source_test_acc, target_acc = result["source_test_acc"], result["target_test_acc"]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "source_domain": SOURCE_DOMAIN,
        "vocab": vocab,
        "weight_decay": result["weight_decay"],
        "n_source_train": len(result["train_idx"]),
        "feature_dim": model.feature_dim,
        "n_classes": 2,
        "seed": SEED,
        "source_test_acc": source_test_acc,
        "target_test_acc": target_acc,
    }, CHECKPOINT_DIR / "model.pt")
    import pickle
    with open(CHECKPOINT_DIR / "tfidf_transformer.pkl", "wb") as f:
        pickle.dump(transformer, f)
    print(f"Salvato checkpoint in {CHECKPOINT_DIR}")

    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"Tempo caricamento+vettorizzazione: {result['load_time']:.1f}s")
    print(f"Tempo training: {result['train_time']:.1f}s ({result['n_epochs_run']} epoche)")
    print(f"{SOURCE_DOMAIN} (source) test accuracy: {100*source_test_acc:.2f}%")
    for domain in TARGET_DOMAINS:
        print(f"{domain:>12s} (target, nessun adattamento): {100*target_acc[domain]:.2f}%")
    worst_domain = min(target_acc, key=target_acc.get)
    print(f"\nShift più marcato: {worst_domain} (accuracy più bassa fra i target, "
          f"{100*target_acc[worst_domain]:.2f}%)")


if __name__ == "__main__":
    main()
