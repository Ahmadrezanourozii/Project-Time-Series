"""Export the Data/splits fold structure to small committed metadata files.

Run locally (needs the 489 MB Data/ tree). Produces:
  metadata/fold_assignments.json  fold -> split -> [subject_id, ...]
  metadata/labels.csv             subject_id,label for the 100 used subjects

These let baseline.folds.load_fold_assignment work on machines without
Data/ (i.e. Kaggle), where only the cached window npz is available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.config import N_FOLDS, PROJECT_ROOT
from baseline.folds import load_fold_assignment
from baseline.labels import label_from_subject_id, load_labels

METADATA_DIR = PROJECT_ROOT / "metadata"


def main() -> None:
    assignments: dict[str, dict[str, list[str]]] = {}
    all_test_subjects: list[str] = []
    used_subjects: set[str] = set()

    for fold_idx in range(1, N_FOLDS + 1):
        df = load_fold_assignment(fold_idx)  # asserts 60/20/20 + uniqueness
        by_split = {
            split: sorted(df.loc[df["split"] == split, "subject_id"])
            for split in ("train", "validation", "test")
        }
        assignments[str(fold_idx)] = by_split
        all_test_subjects.extend(by_split["test"])
        used_subjects.update(df["subject_id"])

    if len(all_test_subjects) != len(set(all_test_subjects)):
        raise ValueError("Test subjects overlap across folds")
    if set(all_test_subjects) != used_subjects:
        raise ValueError("Union of test splits does not cover all subjects exactly once")

    labels = load_labels()
    label_rows = []
    for subject_id in sorted(used_subjects):
        label = labels.loc[subject_id, "label"]
        if label != label_from_subject_id(subject_id):
            raise ValueError(f"Label mismatch for {subject_id}")
        label_rows.append(f"{subject_id},{label}")

    METADATA_DIR.mkdir(exist_ok=True)
    (METADATA_DIR / "fold_assignments.json").write_text(json.dumps(assignments, indent=1))
    (METADATA_DIR / "labels.csv").write_text("subject_id,label\n" + "\n".join(label_rows) + "\n")
    print(f"Wrote {len(used_subjects)} subjects across {N_FOLDS} folds to {METADATA_DIR}")


if __name__ == "__main__":
    main()
