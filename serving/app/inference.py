"""ONNX Runtime inference for the triage API (no torch dependency)."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from cxr.preprocess import image_stats, resize_for_storage, to_model_input


def ensure_model(model_dir: Path) -> None:
    """Download the bundle from the Hugging Face Hub on first boot if it is not baked in."""
    if (model_dir / "model.onnx").exists():
        return
    repo = os.environ.get("HF_MODEL_REPO")
    if not repo:
        raise RuntimeError(f"No model in {model_dir} and HF_MODEL_REPO is not set")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(model_dir),
                      revision=os.environ.get("HF_MODEL_REVISION"))  # fmt: skip


class TriageModel:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self.card = json.loads((self.dir / "model_card.json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        self.session = ort.InferenceSession(str(self.dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.labels: list[str] = self.card["labels"]
        self.img_size: int = int(self.card["img_size"])
        self.thresholds = np.array([self.card["thresholds"][label] for label in self.labels], dtype=np.float32)
        self.urgent = set(self.card["urgent_findings"])

    @property
    def version(self) -> str:
        return self.card["version"]

    def predict(self, img: np.ndarray) -> dict:
        stored = resize_for_storage(img)
        x = to_model_input(stored, self.img_size)[None].astype(np.float32)
        t0 = time.perf_counter()
        probs, cam = self.session.run(["probs", "cam"], {"image": x})
        inference_ms = (time.perf_counter() - t0) * 1000
        probs, cam = probs[0], np.maximum(cam[0], 0)

        findings = []
        for k, label in enumerate(self.labels):
            p, th = float(probs[k]), float(self.thresholds[k])
            findings.append({
                "label": label,
                "probability": round(p, 4),
                "threshold": round(th, 4),
                "positive": p >= th,
                "urgent": label in self.urgent,
                "score": round(p / max(th, 1e-6), 3),  # >1 means above the operating point
            })  # fmt: skip
        findings.sort(key=lambda f: -f["score"])
        flagged = [f for f in findings if f["positive"]]
        if any(f["urgent"] for f in flagged):
            triage = "urgent"
        elif flagged:
            triage = "review"
        else:
            triage = "routine"

        peak = cam.reshape(len(self.labels), -1).max(1).reshape(-1, 1, 1)
        heatmaps = {label: np.round(cam[k] / (peak[k] + 1e-8), 3).tolist() for k, label in enumerate(self.labels)}
        ok, png = cv2.imencode(".png", stored)
        return {
            "triage": triage,
            "findings": findings,
            "flagged": [f["label"] for f in flagged],
            "heatmaps": heatmaps,
            "image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else None,
            "image_stats": image_stats(stored),
            "inference_ms": round(inference_ms, 1),
            "model_version": self.version,
        }
