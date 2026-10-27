"""Prepare NIH ChestX-ray14 for training.

Reads PNGs straight out of the Kaggle archive (`kaggle datasets download -d nih-chest-xrays/data`)
without extracting it, so the 42 GB zip plus a ~7 GB memmap fits on Colab's local disk.

    python -m cxr.data.prepare_nih --source /content/raw/data.zip --out data/processed
    python -m cxr.data.prepare_nih --source /path/to/extracted/folder --out data/processed
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from cxr.data.splits import assign_splits, parse_nih_entries, subsample_patients
from cxr.data.store import write_dataset
from cxr.preprocess import STORE_SIZE, decode_image, resize_for_storage


class Source:
    """Uniform access to the NIH files whether they are in a zip or an extracted folder."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.is_zip = self.path.suffix == ".zip"
        if self.is_zip:
            with zipfile.ZipFile(self.path) as zf:
                names = zf.namelist()
        else:
            names = [str(p.relative_to(self.path)) for p in self.path.rglob("*") if p.is_file()]
        self.by_basename = {os.path.basename(n): n for n in names}

    def read(self, basename: str) -> bytes:
        name = self.by_basename[basename]
        if self.is_zip:
            with zipfile.ZipFile(self.path) as zf:
                return zf.read(name)
        return (self.path / name).read_bytes()

    def has(self, basename: str) -> bool:
        return basename in self.by_basename


# Worker state: each process opens its own zip handle once.
_ZIP: zipfile.ZipFile | None = None
_ROOT: Path | None = None
_SIZE = STORE_SIZE


def _init_worker(source: str, is_zip: bool, size: int) -> None:
    global _ZIP, _ROOT, _SIZE
    _SIZE = size
    if is_zip:
        _ZIP = zipfile.ZipFile(source)
    else:
        _ROOT = Path(source)


def _load(member: str) -> np.ndarray:
    data = _ZIP.read(member) if _ZIP is not None else (_ROOT / member).read_bytes()
    return resize_for_storage(decode_image(data), _SIZE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Kaggle zip or extracted folder")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--size", type=int, default=STORE_SIZE)
    ap.add_argument("--subset", type=float, default=1.0, help="fraction of patients to keep")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Source(args.source)
    entries = pd.read_csv(io.BytesIO(src.read("Data_Entry_2017.csv")))
    df = parse_nih_entries(entries)
    df = df[df["image"].map(src.has)]
    df = subsample_patients(df, args.subset, args.seed)

    test_images = None
    if src.has("test_list.txt"):
        test_images = set(src.read("test_list.txt").decode().split())
        print(f"Using official NIH test split ({len(test_images)} images)")
    df = assign_splits(df, test_images, val_frac=args.val_frac, seed=args.seed)

    members = [src.by_basename[name] for name in df["image"]]
    with Pool(args.workers, initializer=_init_worker, initargs=(str(src.path), src.is_zip, args.size)) as pool:
        images = pool.imap(_load, members, chunksize=64)
        write_dataset(Path(args.out), df, images, args.size, seed=args.seed)


if __name__ == "__main__":
    main()
