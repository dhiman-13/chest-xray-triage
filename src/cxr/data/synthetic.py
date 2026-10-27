"""Synthetic chest-X-ray-like dataset in the same on-disk format as the real one.

Each finding paints a blob at a label-specific location, so a model can actually learn the
task. Used by CI and local smoke tests to exercise the whole pipeline without 42 GB of data.

    python -m cxr.data.synthetic --out data/synthetic --n 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cxr.data.splits import assign_splits
from cxr.data.store import write_dataset
from cxr.labels import LABELS


def _base_image(rng: np.random.Generator, size: int) -> np.ndarray:
    img = np.full((size, size), 40, np.uint8)
    c = size // 2
    for dx in (-size // 5, size // 5):
        cv2.ellipse(img, (c + dx, c), (size // 7, size // 3), 0, 0, 360, 110, -1)  # lungs
    cv2.ellipse(img, (c, int(c * 1.1)), (size // 9, size // 7), 0, 0, 360, 170, -1)  # heart
    noise = rng.normal(0, 12, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _finding_location(label_idx: int, size: int) -> tuple[int, int]:
    angle = 2 * np.pi * label_idx / len(LABELS)
    r = size * 0.3
    return int(size / 2 + r * np.cos(angle)), int(size / 2 + r * np.sin(angle))


def generate(n: int, size: int, seed: int) -> tuple[pd.DataFrame, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    prevalence = np.linspace(0.25, 0.05, len(LABELS))
    rows, images = [], []
    patient = 0
    while len(rows) < n:
        patient += 1
        sex, age = rng.choice(["M", "F"]), int(rng.integers(18, 90))
        for follow_up in range(int(rng.integers(1, 4))):
            if len(rows) >= n:
                break
            y = (rng.random(len(LABELS)) < prevalence).astype(np.uint8)
            img = _base_image(rng, size)
            for k in np.flatnonzero(y):
                cv2.circle(img, _finding_location(k, size), size // 14, 235, -1)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            images.append(img)
            findings = "|".join(LABELS[k] for k in np.flatnonzero(y)) or "No Finding"
            rows.append(
                {
                    "image": f"{patient:05d}_{follow_up:03d}.png",
                    "findings": findings,
                    "follow_up": follow_up,
                    "patient_id": patient,
                    "age": age,
                    "sex": sex,
                    "view": rng.choice(["PA", "AP"]),
                    **dict(zip(LABELS, y.tolist(), strict=True)),
                }
            )
    return pd.DataFrame(rows), images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, images = generate(args.n, args.size, args.seed)
    df = assign_splits(df, val_frac=0.15, test_frac=0.2, seed=args.seed)
    # assign_splits keeps row order, so `images` stays aligned with `df`
    write_dataset(Path(args.out), df, images, args.size, n_reference=500, seed=args.seed)


if __name__ == "__main__":
    main()
