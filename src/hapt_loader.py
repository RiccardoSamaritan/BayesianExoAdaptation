"""HAPT (Human Activities and Postural Transitions) dataset loader.

Mirrors src/loader.py but supports all 12 activity classes (6 static + 6 transitions).
Reuses parsing logic from loader.py since HAPT format is identical to UCI HAR.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Import from loader.py (reuse dataclass)
from .loader import HARDataset


# Activity mappings for HAPT (12 classes)
ACTIVITY_ID_TO_NAME_HAPT = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING',
    7: 'STAND_TO_SIT',
    8: 'SIT_TO_STAND',
    9: 'SIT_TO_LIE',
    10: 'LIE_TO_SIT',
    11: 'STAND_TO_LIE',
    12: 'LIE_TO_STAND',
}

# Subsets of activities
LOCOMOTION_IDS_HAPT = [1, 2, 3]  # Same as UCI HAR
STATIC_IDS_HAPT = [4, 5, 6]  # SITTING, STANDING, LAYING
TRANSITION_IDS_HAPT = [7, 8, 9, 10, 11, 12]  # Postural transitions


def _load_hapt_feature_names(hapt_dir: Path) -> list:
    """Load HAPT feature names (different format than HAR).

    HAPT features.txt format: one feature name per line (no numbering).
    HAR features.txt format: "1 tBodyAcc-mean()-X" (number + space + name).
    """
    names = []
    with open(hapt_dir / "features.txt") as f:
        for line in f:
            name = line.strip()
            if name:  # Skip empty lines
                names.append(name)

    assert len(names) == 561, f"Expected 561 feature names, got {len(names)}"
    return names


class HAPTDataset(HARDataset):
    """HAPT dataset (extends HARDataset to support 12 activity classes).

    Overrides activity_name property to use HAPT's 12-class mapping.
    Overrides filter methods to return HAPTDataset instances.
    """

    @property
    def activity_name(self) -> np.ndarray:
        return np.array([ACTIVITY_ID_TO_NAME_HAPT[int(yi)] for yi in self.y])

    def filter_activities(self, activity_ids) -> "HAPTDataset":
        """Filter to keep only specified activities (returns HAPTDataset)."""
        mask = np.isin(self.y, list(activity_ids))
        return HAPTDataset(
            X=self.X[mask], y=self.y[mask], subject_id=self.subject_id[mask],
            feature_names=self.feature_names,
        )

    def filter_subjects(self, subject_ids) -> "HAPTDataset":
        """Filter to keep only specified subjects (returns HAPTDataset)."""
        mask = np.isin(self.subject_id, list(subject_ids))
        return HAPTDataset(
            X=self.X[mask], y=self.y[mask], subject_id=self.subject_id[mask],
            feature_names=self.feature_names,
        )


def _load_hapt_split(hapt_dir: Path, split: str):
    """Load HAPT split (Train or Test).

    HAPT structure differs slightly from UCI HAR:
    - Files are capitalized: Train/X_train.txt vs train/X_train.txt
    - Subject file is named: subject_id_train.txt vs subject_train.txt
    """
    split_dir = hapt_dir / split.capitalize()

    X = np.loadtxt(split_dir / f"X_{split}.txt")
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=int)
    subject_id = np.loadtxt(split_dir / f"subject_id_{split}.txt", dtype=int)

    n = X.shape[0]
    assert y.shape[0] == n, f"{split}: X has {n} rows but y has {y.shape[0]}"
    assert subject_id.shape[0] == n, f"{split}: X has {n} rows but subject_id has {subject_id.shape[0]}"

    return X, y, subject_id


def load_hapt(data_dir: str | Path, merge_train_test: bool = True) -> HAPTDataset:
    """Load HAPT dataset from disk.

    Args:
        data_dir: Path to data/ directory (expects data/HAPT/ subdirectory)
        merge_train_test: If True, merge train and test sets

    Returns:
        HAPTDataset instance with all 12 activity classes

    Notes:
        - HAPT format is identical to UCI HAR (561 features, whitespace delimited)
        - Same 30 subjects (1-30)
        - Extends HAR with 6 transition classes (7-12)
    """
    data_dir = Path(data_dir)
    hapt_dir = data_dir / "HAPT"

    if not hapt_dir.exists():
        raise FileNotFoundError(
            f"HAPT directory not found at {hapt_dir}. "
            f"Expected structure: data/HAPT/Train/ and data/HAPT/Test/"
        )

    # Load feature names (HAPT has different features.txt format)
    feature_names = _load_hapt_feature_names(hapt_dir)

    if merge_train_test:
        # Load train and test
        X_train, y_train, subj_train = _load_hapt_split(hapt_dir, "train")
        X_test, y_test, subj_test = _load_hapt_split(hapt_dir, "test")

        # Merge
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test])
        subject_id = np.concatenate([subj_train, subj_test])
    else:
        # Load train only
        X, y, subject_id = _load_hapt_split(hapt_dir, "train")

    return HAPTDataset(
        X=X,
        y=y,
        subject_id=subject_id,
        feature_names=feature_names,
    )


def validate_hapt(dataset: HAPTDataset) -> dict:
    """Validate HAPT dataset (extends loader.validate for 12 classes).

    Args:
        dataset: HAPTDataset instance from load_hapt()

    Returns:
        dict with validation statistics
    """
    report = {}

    # Basic stats
    report['n_rows'] = dataset.X.shape[0]
    report['n_features'] = dataset.X.shape[1]
    report['n_nan'] = np.isnan(dataset.X).sum()
    report['n_inf'] = np.isinf(dataset.X).sum()

    # Value range
    report['min_val'] = dataset.X.min()
    report['max_val'] = dataset.X.max()
    report['n_outside_[-1,1]'] = ((dataset.X < -1) | (dataset.X > 1)).sum()

    # Subjects
    report['n_subjects'] = len(np.unique(dataset.subject_id))
    report['subject_id_range'] = (dataset.subject_id.min(), dataset.subject_id.max())

    # Activities (12 classes for HAPT)
    report['activity_ids_present'] = sorted(np.unique(dataset.y).tolist())

    # Class counts
    class_counts = {}
    for act_id in range(1, 13):  # 1-12 for HAPT
        act_name = ACTIVITY_ID_TO_NAME_HAPT[act_id]
        count = (dataset.y == act_id).sum()
        if count > 0:
            class_counts[act_name] = int(count)
    report['class_counts'] = class_counts

    return report
