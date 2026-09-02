"""Feature extractor per l'esperimento digits (MNIST/USPS/SVHN, 32x32 grayscale).

`bayesian_model.py::SmallCNN` è tarata esplicitamente per immagini 1x28x28
(`nn.Linear(32 * 7 * 7, feature_dim)`, 28 -> 14 -> 7 con due MaxPool2d(2)) --
non modificata qui per non rischiare di rompere i notebook 03/04/05 che la
usano così com'è. `SmallCNN32` sotto è la stessa architettura (stessa
filosofia: 2 blocchi conv -> feature -> testa lineare, solo `h` riceve il
trattamento bayesiano), ridimensionata per l'input 32x32 di questo esperimento
(32 -> 16 -> 8, quindi `32 * 8 * 8`), niente di più: nessuna eredità dalla
precedente architettura ispirata a SHOT (DTNBase, digit/network.py) usata
nella versione di questo esperimento prima di essere spostato qui -- qui si
riusa direttamente lo stile di `code_v2`.

Le funzioni condivise (`train_map`, `extract`, `head_weights`, `augment`,
`LastLayerLaplace`) restano quelle di `bayesian_model.py`, importate da lì,
non duplicate: operano su qualunque modello con l'interfaccia
`.features(x)` / `.h`, che `SmallCNN32` rispetta."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SmallCNN32(nn.Module):
    """CNN piccola per immagini 1x32x32: 2 blocchi conv -> feature -> testa lineare.

    `features(x)` restituisce φ = g(x); `forward` aggiunge la testa lineare h.
    Solo h riceve il trattamento bayesiano (stessa filosofia di SmallCNN)."""

    def __init__(self, n_classes: int = 10, feature_dim: int = 128):
        super().__init__()
        self.g = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x8x8
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, feature_dim), nn.ReLU(),
        )
        self.h = nn.Linear(feature_dim, n_classes)
        self.feature_dim = feature_dim

    def features(self, x):
        return self.g(x)

    def forward(self, x):
        return self.h(self.features(x))


# Import posticipato (non in cima al file): evita un ciclo a livello di modulo,
# dato che bayesian_model.py potrebbe in futuro voler importare le classi
# modello di questo file per type hints -- oggi non lo fa, ma qui l'ordine
# rende esplicito che la dipendenza va in una sola direzione.
from .bayesian_model import LastLayerLaplace  # noqa: E402


def load_svhn_source(models_dir: Path, device: str = "cpu") -> dict:
    """Ricarica il checkpoint source SVHN + la posterior di Laplace fittata
    (+ l'M_FIXED scelto dallo sweep di convergenza, se quel passo è già stato
    eseguito) prodotti da `digits_train.py` / dai Notebook 09-10. Blocco di
    reload comune a ogni notebook a valle del training source (10, 11, 12,
    14): riduce ~20 righe quasi identiche per notebook a una sola chiamata.

    Ritorna un dict: model (SmallCNN32, in eval), laplace (LastLayerLaplace),
    mean/std (normalizzazione source), ckpt (checkpoint grezzo, per qualunque
    campo non già scompattato qui sopra), M_fixed (int, o None se
    models_dir/mc_convergence.npz non esiste ancora -- cioè il Notebook 10
    non è ancora stato eseguito)."""
    ckpt = torch.load(models_dir / "model.pt", map_location="cpu", weights_only=False)
    lap_data = np.load(models_dir / "svhn_laplace.npz")
    mean, std = ckpt["source_mean"], ckpt["source_std"]

    model = SmallCNN32(n_classes=ckpt["n_classes"], feature_dim=ckpt["feature_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    laplace = LastLayerLaplace(theta_map=lap_data["theta_map"], cov=lap_data["cov"],
                               K=int(lap_data["K"]), Dp=int(lap_data["Dp"]))

    M_fixed = None
    mc_conv_path = models_dir / "mc_convergence.npz"
    if mc_conv_path.exists():
        M_fixed = int(np.load(mc_conv_path)["M_FIXED"])

    return dict(model=model, laplace=laplace, mean=mean, std=std, ckpt=ckpt, M_fixed=M_fixed)


def evaluate(model: nn.Module, loader) -> tuple:
    """Accuratezza e loss (CE) di `model` su `loader`, in eval/no_grad.
    Condivisa da `digits_train.py` (training/early stopping) e
    `digits_verify.py` (sanity check dopo il reload, che usa solo
    l'accuratezza e scarta la loss) -- prima ciascuno la ridefiniva."""
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
