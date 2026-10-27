"""Metadata parsing and leakage-free, patient-level splitting for NIH ChestX-ray14."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from cxr.labels import LABELS

NIH_COLUMNS = {
    "Image Index": "image",
    "Finding Labels": "findings",
    "Follow-up #": "follow_up",
    "Patient ID": "patient_id",
    "Patient Age": "age",
    "Patient Gender": "sex",
    "View Position": "view",
}


def parse_nih_entries(entries: pd.DataFrame) -> pd.DataFrame:
    """Turn Data_Entry_2017.csv into one tidy row per image with multi-hot label columns."""
    df = entries.rename(columns=NIH_COLUMNS)[list(NIH_COLUMNS.values())].copy()
    findings = df["findings"].str.split("|")
    for label in LABELS:
        df[label] = findings.apply(lambda f, lab=label: int(lab in f)).astype(np.uint8)
    # A handful of ages are recorded in months/typos (e.g. 413); treat > 100 as missing.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[df["age"] > 100, "age"] = np.nan
    return df


def assign_splits(
    df: pd.DataFrame,
    test_images: set[str] | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Split by patient so no patient appears in more than one split.

    Splitting by image (what most tutorials do) leaks follow-up studies of the same patient
    into the test set and inflates AUROC by several points.
    If the official NIH test list is given we keep it (it is patient-disjoint) so results are
    comparable with the literature; validation is carved out of train_val by patient.
    """
    df = df.copy()
    groups = df["patient_id"].to_numpy()
    if test_images is not None:
        is_test = df["image"].isin(test_images).to_numpy()
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        _, test_idx = next(gss.split(df, groups=groups))
        is_test = np.zeros(len(df), dtype=bool)
        is_test[test_idx] = True

    df["split"] = np.where(is_test, "test", "train")
    trainval = df.index[~is_test]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    _, val_pos = next(gss.split(trainval, groups=groups[~is_test]))
    df.loc[trainval[val_pos], "split"] = "val"

    check_no_patient_leakage(df)
    return df


def check_no_patient_leakage(df: pd.DataFrame) -> None:
    patients = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[a] & patients[b]
        if overlap:
            raise AssertionError(f"Patient leakage between {a} and {b}: {len(overlap)} patients")


def subsample_patients(df: pd.DataFrame, fraction: float, seed: int = 42) -> pd.DataFrame:
    """Keep a random fraction of patients (all their images) for quick experiments."""
    if fraction >= 1.0:
        return df
    rng = np.random.default_rng(seed)
    patients = df["patient_id"].unique()
    keep = rng.choice(patients, size=max(1, int(len(patients) * fraction)), replace=False)
    return df[df["patient_id"].isin(keep)].reset_index(drop=True)
