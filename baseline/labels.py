"""PD/HC subject-level label loading."""

from __future__ import annotations

import pandas as pd

from .config import DEMOGRAPHICS_CSV

CONTROL_LABELS = {"CO", "CONTROL", "CONTROLS", "HEALTHY CONTROL", "HEALTHY CONTROLS"}


def load_labels(csv_path=DEMOGRAPHICS_CSV) -> pd.DataFrame:
    """Load PD/HC labels.

    Inputs: csv_path to the demographics CSV (columns include 'ID', 'Group').
    Output: DataFrame indexed by subject_id with a single column 'label' in {'HC', 'PD'}.
    """
    df = pd.read_csv(csv_path)
    group = df["Group"].astype(str).str.strip().str.upper()
    group = group.where(~group.isin(CONTROL_LABELS), "HC")
    out = pd.DataFrame({"subject_id": df["ID"], "label": group}).set_index("subject_id")
    unexpected = set(out["label"].unique()) - {"HC", "PD"}
    if unexpected:
        raise ValueError(f"Unexpected group labels: {unexpected}")
    return out


def label_from_subject_id(subject_id: str) -> str:
    """Infer PD/HC directly from the filename convention (Co -> HC, Pt -> PD).

    Used as a sanity check against the demographics CSV at startup.
    """
    if "Pt" in subject_id:
        return "PD"
    if "Co" in subject_id:
        return "HC"
    raise ValueError(f"Cannot infer label from subject_id={subject_id!r}")
