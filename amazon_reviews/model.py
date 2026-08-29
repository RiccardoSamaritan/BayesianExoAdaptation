"""Riusa `MLPClassifier` da `code_v2/src/bayesian_model.py` (la sola delle
due copie del progetto -- `src/bayesian.py` non la espone -- verificato
prima di scegliere) per il source model sentiment. Nessuna nuova classe:
stessa interfaccia `.features(x)`/`.h`/`.g` già usata da tutta la pipeline
bayesiana (Laplace sull'ultimo layer inclusa)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code_v2.src.bayesian_model import MLPClassifier  # noqa: E402

IN_DIM = 5000       # dimensione dei vettori TF-IDF (data_utils.fit_tfidf, max_features=5000)
HIDDEN = 128
FEATURE_DIM = 64    # K*Dp = 2*(64+1) = 130: Hessiana piccolissima, nessun problema di scala
N_CLASSES = 2


def build_model() -> MLPClassifier:
    return MLPClassifier(in_dim=IN_DIM, n_classes=N_CLASSES, hidden=HIDDEN, feature_dim=FEATURE_DIM)
