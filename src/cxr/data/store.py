"""On-disk format of a processed dataset.

    <dir>/images.npy           uint8 memmap (N, S, S); row i belongs to metadata row with idx == i
    <dir>/metadata.csv         labels, patient id, demographics, split
    <dir>/reference_stats.csv  image statistics of training images (baseline for drift monitoring)

A single memmap instead of 112k PNGs matters on Colab: it has 2 CPU cores, and PNG decoding
would starve the GPU. Reading a memmap row is a memcpy.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cxr.labels import LABELS
from cxr.preprocess import STAT_COLUMNS, image_stats

IMAGES_FILE = "images.npy"
METADATA_FILE = "metadata.csv"
REFERENCE_FILE = "reference_stats.csv"


def write_dataset(
    out_dir: Path,
    meta: pd.DataFrame,
    images: Iterable[np.ndarray],
    size: int,
    n_reference: int = 5000,
    seed: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True).copy()
    meta["idx"] = np.arange(len(meta))

    arr = np.lib.format.open_memmap(
        out_dir / IMAGES_FILE, mode="w+", dtype=np.uint8, shape=(len(meta), size, size)
    )
    for i, img in enumerate(tqdm(images, total=len(meta), desc="writing images")):
        arr[i] = img
    arr.flush()

    meta.to_csv(out_dir / METADATA_FILE, index=False)

    train_idx = meta.loc[meta["split"] == "train", "idx"].to_numpy()
    rng = np.random.default_rng(seed)
    sample = rng.choice(train_idx, size=min(n_reference, len(train_idx)), replace=False)
    ref = pd.DataFrame([image_stats(np.asarray(arr[i])) for i in sample], columns=STAT_COLUMNS)
    ref.to_csv(out_dir / REFERENCE_FILE, index=False)

    summary = {
        "n_images": int(len(meta)),
        "n_patients": int(meta["patient_id"].nunique()),
        "size": size,
        "splits": meta["split"].value_counts().to_dict(),
        "prevalence": {lab: float(meta[lab].mean()) for lab in LABELS},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / METADATA_FILE)


def open_images(data_dir: str | Path) -> np.ndarray:
    return np.load(Path(data_dir) / IMAGES_FILE, mmap_mode="r")
