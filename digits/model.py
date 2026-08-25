"""Digit-recognition backbone for the MNIST/USPS/SVHN experiment (digits/),
copied from SHOT's own digit-benchmark network, not invented from scratch:

    repo:   https://github.com/tim-learn/SHOT
    commit: f7d555a0d53b525b885e5ef2a887a267a5be3c36 (cloned 2026-08-25)
    file:   digit/network.py -- DTNBase, feat_bottleneck, feat_classifier

SHOT picks the backbone by task, not uniformly (digit/uda_digit.py::train_source):
`if args.dset == 's2m': netF = network.DTNBase()`, while `u2m`/`m2u` (both
MNIST<->USPS, no SVHN involved) use the smaller `LeNetBase`. This project's
source domain is SVHN (digits/train.py), i.e. the s2m-style setup, so
DTNBase is the matching family -- LeNetBase would be the wrong choice here,
SHOT itself never uses it for any task involving SVHN.

One deliberate deviation from the verbatim source: digits/data_utils.py
already converts every domain (MNIST/USPS/SVHN alike) to a single grayscale
channel for cross-domain shape consistency (see that module's docstring),
so DTNBase's first conv layer is changed from `nn.Conv2d(3, 64, ...)` to
`nn.Conv2d(1, 64, ...)`. SHOT's own DTNBase expects raw 3-channel SVHN
images because it never grayscales SVHN. Every other detail (channel
counts, kernel sizes, strides, paddings, dropout rates, BatchNorm
placement) is unchanged.

`feat_bottleneck`/`feat_classifier` below are `digit/network.py`'s own
versions, copied verbatim -- note SHOT's digit benchmark applies an extra
`nn.Dropout(p=0.5)` after the bottleneck's BatchNorm1d when `type="bn"`,
which its Office-31 benchmark's `object/network.py` version of the same two
class names does not: the two are genuinely different in SHOT's own repo,
not just renamed, so this file's versions are not interchangeable with any
other benchmark's copy of the same class names.

`FeatureClassifier` (the g/h wrapper with the `.features()` convention that
`src/bayesian.py::LastLayerLaplace` reads) is imported from `src/`, not
redefined here, per this experiment's own convention: new architecture code
lives in digits/, but the Bayesian/adaptation/calibration machinery it
plugs into stays the single copy already validated on HAR and Office-31.
"""
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm

from src.bayesian import FeatureClassifier


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class DTNBase(nn.Module):
    """digit/network.py::DTNBase, first conv changed 3->1 input channels
    (see module docstring)."""

    def __init__(self):
        super().__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU(),
        )
        self.in_features = 256 * 4 * 4

    def forward(self, x):
        x = self.conv_params(x)
        return x.view(x.size(0), -1)


class feat_bottleneck(nn.Module):
    """digit/network.py::feat_bottleneck, verbatim."""

    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super().__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x


class feat_classifier(nn.Module):
    """digit/network.py::feat_classifier, verbatim."""

    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super().__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        return self.fc(x)


class DigitFeatureExtractor(nn.Module):
    """g = DTNBase (conv backbone) -> feat_bottleneck. `features(x)` returns
    the penultimate (post-bottleneck) tensor -- what `FeatureClassifier.features()`
    (src/bayesian.py) actually calls is `self.g(x)`, i.e. this module's
    `forward`; `features()` here is provided only as a readable alias."""

    def __init__(self, backbone: DTNBase, bottleneck: feat_bottleneck):
        super().__init__()
        self.backbone = backbone
        self.bottleneck = bottleneck

    def forward(self, x):
        return self.bottleneck(self.backbone(x))

    def features(self, x):
        return self.forward(x)


def build_digit_model(bottleneck_dim: int = 256, n_classes: int = 10,
                      classifier_type: str = "bn", layer_type: str = "wn") -> FeatureClassifier:
    """Returns a fresh (untrained) `FeatureClassifier` with `g` =
    `DigitFeatureExtractor(DTNBase(), feat_bottleneck(...))` and `h` a linear
    head, built via `FeatureClassifier.from_feature_extractor` (imported
    from src/bayesian.py, not redefined here). `classifier_type`/`layer_type`
    default to SHOT's own digit-benchmark defaults (`--classifier bn
    --layer wn`, digit/uda_digit.py), matching this file's `feat_bottleneck`
    (`type=classifier_type`) and the head's weight normalization
    (`type=layer_type`, applied by wrapping `.h` with `weightNorm` after
    `from_feature_extractor` builds the plain `nn.Linear`, then re-applying
    `init_weights` -- matching `feat_classifier`'s own init order above)."""
    backbone = DTNBase()
    bottleneck = feat_bottleneck(feature_dim=backbone.in_features, bottleneck_dim=bottleneck_dim,
                                 type=classifier_type)
    g = DigitFeatureExtractor(backbone, bottleneck)

    model = FeatureClassifier.from_feature_extractor(g, feature_dim=bottleneck_dim, n_classes=n_classes)
    model.h.apply(init_weights)
    if layer_type == "wn":
        model.h = weightNorm(model.h, name="weight")
    return model
