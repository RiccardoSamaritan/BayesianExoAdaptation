from dataclasses import dataclass
from pathlib import Path
import numpy as np

ACTIVITY_ID_TO_NAME = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
LOCOMOTION_IDS = (1, 2, 3) 


@dataclass
class HARDataset:
    X: np.ndarray            
    y: np.ndarray            
    subject_id: np.ndarray  
    feature_names: list   

    @property
    def activity_name(self) -> np.ndarray:
        return np.array([ACTIVITY_ID_TO_NAME[yi] for yi in self.y])

    def filter_activities(self, activity_ids) -> "HARDataset":
        mask = np.isin(self.y, list(activity_ids))
        return HARDataset(
            X=self.X[mask], y=self.y[mask], subject_id=self.subject_id[mask],
            feature_names=self.feature_names,
        )

    def filter_subjects(self, subject_ids) -> "HARDataset":
        mask = np.isin(self.subject_id, list(subject_ids))
        return HARDataset(
            X=self.X[mask], y=self.y[mask], subject_id=self.subject_id[mask],
            feature_names=self.feature_names,
        )


def _load_split(root: Path, split: str):
    X = np.loadtxt(root / split / f"X_{split}.txt")
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=int)
    subject_id = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)

    n = X.shape[0]
    assert y.shape[0] == n, f"{split}: X has {n} rows but y has {y.shape[0]}"
    assert subject_id.shape[0] == n, f"{split}: X has {n} rows but subject_id has {subject_id.shape[0]}"
    return X, y, subject_id


def _load_feature_names(root: Path) -> list:
    names = []
    with open(root / "features.txt") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                names.append(parts[1])
    assert len(names) == 561, f"Expected 561 feature names, got {len(names)}"
    return names


def load_har(root_dir: str, merge_train_test: bool = True) -> HARDataset:
    root = Path(root_dir)
    feature_names = _load_feature_names(root)

    X_train, y_train, subj_train = _load_split(root, "train")
    X_test, y_test, subj_test = _load_split(root, "test")

    if merge_train_test:
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        subject_id = np.concatenate([subj_train, subj_test], axis=0)
        return HARDataset(X=X, y=y, subject_id=subject_id, feature_names=feature_names)
    else:
        raise NotImplementedError("Project design discards the original split -- "
                                   "use merge_train_test=True.")


def validate(ds: HARDataset) -> dict:
    report = {}
    report["n_rows"] = ds.X.shape[0]
    report["n_features"] = ds.X.shape[1]
    report["n_nan"] = int(np.isnan(ds.X).sum())
    report["n_inf"] = int(np.isinf(ds.X).sum())
    report["min_val"] = float(np.nanmin(ds.X))
    report["max_val"] = float(np.nanmax(ds.X))
    report["n_outside_[-1,1]"] = int(((ds.X < -1.0) | (ds.X > 1.0)).sum())
    report["n_subjects"] = int(np.unique(ds.subject_id).size)
    report["subject_id_range"] = (int(ds.subject_id.min()), int(ds.subject_id.max()))
    report["activity_ids_present"] = sorted(int(a) for a in np.unique(ds.y))
    report["class_counts"] = {ACTIVITY_ID_TO_NAME[a]: int((ds.y == a).sum())
                               for a in sorted(np.unique(ds.y))}
    return report