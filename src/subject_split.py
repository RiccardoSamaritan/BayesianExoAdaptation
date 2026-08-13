from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from loader import HARDataset


@dataclass
class DomainSplit:
    source_subjects: np.ndarray
    target_subjects: np.ndarray
    scaler: StandardScaler
    X_source: np.ndarray
    y_source: np.ndarray
    subject_id_source: np.ndarray


def split_subjects(ds: HARDataset, n_source: int = 20, seed: int = 0):
    """Randomly assign subjects to a source pool vs. target set."""
    rng = np.random.RandomState(seed)
    all_subjects = np.unique(ds.subject_id)
    shuffled = rng.permutation(all_subjects)
    source_subjects = np.sort(shuffled[:n_source])
    target_subjects = np.sort(shuffled[n_source:])
    return source_subjects, target_subjects


def split_and_scale(ds: HARDataset, n_source: int = 20, seed: int = 0) -> DomainSplit:
    source_subjects, target_subjects = split_subjects(ds, n_source, seed)
    ds_source = ds.filter_subjects(source_subjects)

    scaler = StandardScaler().fit(ds_source.X)
    X_source = scaler.transform(ds_source.X)

    return DomainSplit(
        source_subjects=source_subjects,
        target_subjects=target_subjects,
        scaler=scaler,
        X_source=X_source,
        y_source=ds_source.y,
        subject_id_source=ds_source.subject_id,
    )


def centroid_distance_shift_proxy(ds: HARDataset, split: DomainSplit) -> dict:
    """Centroid distance from each target subject to the source pool's mean,
    in the source-fit standardized space. Returns {subject_id: distance}."""
    source_mean = split.X_source.mean(axis=0)
    distances = {}
    for sid in split.target_subjects:
        ds_t = ds.filter_subjects([sid])
        X_t = split.scaler.transform(ds_t.X)
        centroid = X_t.mean(axis=0)
        distances[int(sid)] = float(np.linalg.norm(centroid - source_mean))
    return distances


def near_zero_shift_reference(split: DomainSplit) -> dict:
    """Leave-one-subject-out centroid distances WITHIN the source pool --
    the 'this subject is not actually shifted' baseline to compare target
    distances against."""
    distances = {}
    for sid in split.source_subjects:
        mask_held_out = split.subject_id_source == sid
        mask_rest = ~mask_held_out
        centroid_held_out = split.X_source[mask_held_out].mean(axis=0)
        centroid_rest = split.X_source[mask_rest].mean(axis=0)
        distances[int(sid)] = float(np.linalg.norm(centroid_held_out - centroid_rest))
    return distances


def random_window_split_reference(ds: HARDataset, split: DomainSplit, seed: int = 0) -> float:
    """Near-zero-shift control: ignore subject identity entirely, split windows
    randomly into two halves of the SOURCE pool, measure centroid distance.
    This should be small -- if it isn't, something is wrong upstream."""
    rng = np.random.RandomState(seed)
    n = split.X_source.shape[0]
    idx = rng.permutation(n)
    half = n // 2
    centroid_a = split.X_source[idx[:half]].mean(axis=0)
    centroid_b = split.X_source[idx[half:]].mean(axis=0)
    return float(np.linalg.norm(centroid_a - centroid_b))
