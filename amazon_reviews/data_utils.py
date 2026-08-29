"""Caricamento e vettorizzazione del Multi-Domain Sentiment Dataset
(Blitzer, Dredze & Pereira, ACL 2007), 4 domini: books, dvd, electronics,
kitchen.

Fonte: dataset Hugging Face "katossky/multi-domain-sentiment", verificato
prima dell'uso (`huggingface_hub.HfApi().dataset_info(...)`) -- espone per
dominio `positive.review` / `negative.review` / `unlabeled.review`, come
documentato dagli autori.

**Scoperta in fase di verifica, non assunta**: i file NON contengono testo
grezzo. Ogni riga è già nel formato "processed_acl" originale di Blitzer et
al.: una lista di token unigramma/bigramma con il relativo conteggio nel
documento (es. `cable_modem:1 power:3 ... #label#:positive`), seguita da un
tag `#label#:positive` o `#label#:negative` -- niente frasi da tokenizzare,
niente rating a stelle da sogliare (l'etichetta binaria è già esplicita nel
file). Di conseguenza la vettorizzazione qui sotto usa `TfidfTransformer`
(che opera su una matrice di conteggi già tokenizzati) invece di
`TfidfVectorizer` (che si aspetta testo grezzo) -- stesso risultato (pesi
TF-IDF, vocabolario limitato ai termini più frequenti, unigrammi+bigrammi
perché già presenti come tali nel formato sorgente), a partire dai
conteggi che il dataset fornisce direttamente.

`positive.review`/`negative.review` sono bilanciati (1000+1000 per
dominio) e sono le uniche porzioni etichettate usate qui;
`unlabeled.review` (contiene comunque tag `#label#`, derivati da rating
continui più rumorosi secondo gli autori) non è usato in questo modulo.
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

REPO_ID = "katossky/multi-domain-sentiment"
DOMAINS = ["books", "dvd", "electronics", "kitchen"]
_LABEL_RE = re.compile(r"#label#:(positive|negative)")


def _parse_review_file(path: Path) -> list:
    """Ritorna una lista di (dict token->conteggio, label binaria) per ogni
    riga non vuota del file. latin-1: encoding usato dal dump originale."""
    docs = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split(" ")
            label_match = _LABEL_RE.match(tokens[-1])
            if label_match is None:
                raise ValueError(f"riga senza tag #label# riconoscibile: ...{line[-60:]}")
            label = 1 if label_match.group(1) == "positive" else 0
            counts = {}
            for tok in tokens[:-1]:
                term, _, count_str = tok.rpartition(":")
                counts[term] = counts.get(term, 0) + int(count_str)
            docs.append((counts, label))
    return docs


def load_domain(domain: str) -> tuple:
    """Scarica (se non in cache) e carica le porzioni etichettate
    (positive.review + negative.review) di `domain`. Ritorna
    (counts_list, labels): `counts_list` è una lista di dict
    token->conteggio (uno per recensione), `labels` un array (N,) in
    {0, 1} (1 = positive)."""
    if domain not in DOMAINS:
        raise ValueError(f"dominio sconosciuto {domain!r}, deve essere uno di {DOMAINS}")
    docs = []
    for split in ["positive.review", "negative.review"]:
        path = hf_hub_download(repo_id=REPO_ID, filename=f"{domain}/{split}", repo_type="dataset")
        docs.extend(_parse_review_file(Path(path)))
    counts_list = [d[0] for d in docs]
    labels = np.array([d[1] for d in docs], dtype=np.int64)
    return counts_list, labels


def stratified_split(labels: np.ndarray, val_fraction: float, test_fraction: float, seed: int) -> tuple:
    """Split a 3 vie (train/val/test), stratificato per etichetta.
    `val_fraction`/`test_fraction` sono frazioni del totale."""
    idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(
        idx, test_size=val_fraction + test_fraction, stratify=labels, random_state=seed)
    rel_test_fraction = test_fraction / (val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=rel_test_fraction, stratify=labels[rest_idx], random_state=seed)
    return train_idx, val_idx, test_idx


def build_vocabulary(counts_list: list, max_features: int = 5000) -> dict:
    """Vocabolario dei `max_features` termini più frequenti (per conteggio
    totale) in `counts_list` -- da chiamare SOLO sul source (train split),
    mai su target o su porzioni di validazione/test, stesso principio già
    usato per `compute_source_stats` in `code_v2/src/digits_data.py`."""
    freq = Counter()
    for counts in counts_list:
        freq.update(counts)
    vocab_terms = [term for term, _ in freq.most_common(max_features)]
    return {term: i for i, term in enumerate(vocab_terms)}


def _counts_to_matrix(counts_list: list, vocab: dict) -> sp.csr_matrix:
    """Matrice sparsa (N, len(vocab)) di conteggi grezzi. Termini non nel
    vocabolario (fittato sul source) sono scartati -- comportamento OOV
    standard quando si applica un vocabolario fisso a un dominio diverso
    da quello su cui è stato fittato."""
    rows, cols, data = [], [], []
    for i, counts in enumerate(counts_list):
        for term, c in counts.items():
            j = vocab.get(term)
            if j is not None:
                rows.append(i)
                cols.append(j)
                data.append(c)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(counts_list), len(vocab)))


def fit_tfidf(train_counts_list: list, max_features: int = 5000) -> tuple:
    """Fitta vocabolario + pesi IDF SOLO su `train_counts_list` (il train
    split del source). Ritorna (vocab, transformer, X_train_tfidf)."""
    vocab = build_vocabulary(train_counts_list, max_features)
    X_counts = _counts_to_matrix(train_counts_list, vocab)
    transformer = TfidfTransformer()
    X_tfidf = transformer.fit_transform(X_counts)
    return vocab, transformer, X_tfidf


def transform_tfidf(counts_list: list, vocab: dict, transformer: TfidfTransformer) -> sp.csr_matrix:
    """Applica un vocabolario + transformer già fittati (sul source) a
    un'altra porzione di dati (val/test del source, o un dominio target) --
    mai rifittato, stesse statistiche invariate."""
    X_counts = _counts_to_matrix(counts_list, vocab)
    return transformer.transform(X_counts)


if __name__ == "__main__":
    print(f"{'dominio':>12s} {'n. recensioni':>14s} {'positive':>10s} {'negative':>10s}")
    print("-" * 50)
    for domain in DOMAINS:
        counts_list, labels = load_domain(domain)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        print(f"{domain:>12s} {len(labels):14d} {n_pos:10d} {n_neg:10d}")

    print("\nVocabolario TF-IDF (fittato solo su electronics):")
    counts_list, labels = load_domain("electronics")
    train_idx, val_idx, test_idx = stratified_split(labels, val_fraction=0.15, test_fraction=0.15, seed=42)
    train_counts = [counts_list[i] for i in train_idx]
    vocab, transformer, X_train = fit_tfidf(train_counts, max_features=5000)
    print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  dimensione effettiva del vocabolario fittato: {len(vocab)} termini")
    print(f"  X_train shape={X_train.shape}, densità={X_train.nnz / (X_train.shape[0]*X_train.shape[1]):.4%}")
