# digits/ -- MNIST / USPS / SVHN source-free adaptation

Self-contained experiment: all new code lives here, importing the shared
Bayesian/adaptation/calibration machinery from `src/` (`bayesian.py`,
`im_adapt.py`, `calibration.py`) rather than duplicating it.

`digits/data/` and `digits/models/` are gitignored (downloaded/trained
artifacts, ~623MB total) -- regenerate them locally with the two commands
below.

## 1. Get the datasets

Run `notebooks/digits_download.ipynb` top to bottom.

Downloads MNIST (60,000 train + 10,000 test), USPS (~7,291 train + 2,007
test) and SVHN (~73,257 train + ~26,032 test) via `torchvision.datasets`,
and dumps each split to a numpy `.npz` cache under `digits/data/` at its
native resolution/channel count (uniform preprocessing -- grayscale,
resize to 32x32, source-only normalization -- happens later, at load time,
in `digits/data_utils.py`). Idempotent: re-running it will not re-download
anything already on disk with a matching checksum.

Note: `ufldl.stanford.edu` (SVHN's host) has been observed to stall
mid-download in this environment. If the SVHN cell hangs with no progress
for more than ~1-2 minutes, interrupt it and retry the stalled `.mat` file
directly with a resumable, stall-aware download instead (the notebook's
SVHN cell has the full command) -- then re-run that cell, which will detect
the already-downloaded, checksummed `.mat` file and skip straight to the
`.npz` conversion.

## 2. Get the source checkpoint

```
python digits/train.py
```

Trains `digits/model.py`'s source model to a MAP solution on SVHN (~25
minutes on this project's Apple Silicon MPS backend; noticeably slower on
CPU-only, see `train.py`'s module docstring for the measured slowdown and
why MPS is used when available). Saves the checkpoint to
`digits/models/source_svhn/model.pt` and prints total training time plus
no-adaptation accuracy on SVHN test, MNIST test and USPS test.
