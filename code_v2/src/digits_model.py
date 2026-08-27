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
