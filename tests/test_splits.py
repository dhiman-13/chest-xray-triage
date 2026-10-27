import pandas as pd
import pytest

from cxr.data.splits import (
    assign_splits,
    check_no_patient_leakage,
    parse_nih_entries,
    subsample_patients,
)
from cxr.labels import LABELS


def _entries(n_patients=60, per_patient=3):
    rows = []
    for p in range(n_patients):
        for k in range(per_patient):
            rows.append({
                "Image Index": f"{p:08d}_{k:03d}.png",
                "Finding Labels": "Effusion|Mass" if (p + k) % 4 == 0 else "No Finding",
                "Follow-up #": k, "Patient ID": p, "Patient Age": 413 if p == 0 else 40 + p % 40,
                "Patient Gender": "M" if p % 2 else "F", "View Position": "PA",
            })  # fmt: skip
    return pd.DataFrame(rows)


def test_parse_multi_hot_and_age_cleaning():
    df = parse_nih_entries(_entries())
    first = df.iloc[0]
    assert first["Effusion"] == 1 and first["Mass"] == 1 and first["Hernia"] == 0
    assert set(LABELS) <= set(df.columns)
    assert df.loc[df.patient_id == 0, "age"].isna().all()  # 413 years is a data error


def test_patient_level_split_has_no_leakage():
    df = assign_splits(parse_nih_entries(_entries()), val_frac=0.2, test_frac=0.2, seed=1)
    assert set(df["split"]) == {"train", "val", "test"}
    check_no_patient_leakage(df)  # raises on overlap


def test_official_test_list_is_respected():
    df = parse_nih_entries(_entries())
    test_images = set(df.loc[df.patient_id < 10, "image"])
    out = assign_splits(df, test_images=test_images)
    assert set(out.loc[out.split == "test", "image"]) == test_images


def test_leakage_check_detects_overlap():
    df = pd.DataFrame({"patient_id": [1, 1, 2], "split": ["train", "test", "val"]})
    with pytest.raises(AssertionError):
        check_no_patient_leakage(df)


def test_subsample_keeps_whole_patients():
    df = parse_nih_entries(_entries())
    sub = subsample_patients(df, 0.5)
    assert sub.groupby("patient_id").size().eq(3).all()
