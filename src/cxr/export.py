"""Export a trained checkpoint to a self-contained ONNX "model bundle" for serving.

    python -m cxr.export --out models/bundle

Bundle layout (everything the API needs, nothing else):
    model.onnx                 inputs: image (B,3,S,S) -> outputs: probs (B,14), cam (B,14,h,w)
    model_card.json            labels, thresholds, temperature, metrics, provenance
    reference_images.csv       training-image statistics   } baselines for drift monitoring
    reference_predictions.csv  test-set probabilities      }

Temperature scaling and sigmoid are baked into the graph, and so is a class activation map
(CAM): the classifier weights projected onto the last feature map. One forward pass therefore
returns calibrated probabilities *and* a heatmap per class, with no torch at serving time.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from cxr.data.store import REFERENCE_FILE
from cxr.labels import LABELS, URGENT_FINDINGS


class ServingModel(nn.Module):
    def __init__(self, net: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.net = net
        self.register_buffer("inv_t", torch.tensor(1.0 / temperature))

    def forward(self, image: torch.Tensor):
        feats = self.net.forward_features(image)                  # (B, C, h, w)
        logits = self.net.forward_head(feats)                     # (B, K)
        weight = self.net.get_classifier().weight                 # (K, C)
        cam = torch.einsum("kc,bchw->bkhw", weight, feats)        # (B, K, h, w)
        return torch.sigmoid(logits * self.inv_t), cam


def export_onnx(net: nn.Module, img_size: int, temperature: float, path: Path) -> None:
    wrapper = ServingModel(net, temperature).eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    # torch >= 2.5 has a new dynamo-based exporter; the classic TorchScript one is more mature
    legacy = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["probs", "cam"],
        dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}, "cam": {0: "batch"}},
        opset_version=17,
        **legacy,
    )
    _check_parity(wrapper, path, img_size)


def _check_parity(wrapper: nn.Module, path: Path, img_size: int) -> None:
    import onnxruntime as ort

    x = torch.randn(2, 3, img_size, img_size)
    with torch.no_grad():
        ref = wrapper(x)[0].numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["probs"], {"image": x.numpy()})[0]
    err = float(np.abs(ref - got).max())
    if err > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {err:.2e}")
    print(f"ONNX parity check passed (max abs diff {err:.2e})")


def _json_safe(obj):
    """NaN (e.g. AUROC of a class with no test positives) is not valid JSON; store null."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def build_bundle(
    net: nn.Module,
    out: Path,
    img_size: int,
    backbone: str,
    calibration: dict | None = None,
    metrics: dict | None = None,
    reference_images: Path | None = None,
    reference_predictions: Path | None = None,
    version: str | None = None,
    run_id: str | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    net = net.cpu().float().eval()
    calibration = calibration or {}
    temperature = float(calibration.get("temperature", 1.0))
    export_onnx(net, img_size, temperature, out / "model.onnx")

    if reference_images and Path(reference_images).exists():
        shutil.copy(reference_images, out / "reference_images.csv")
    if reference_predictions and Path(reference_predictions).exists():
        preds = pd.read_csv(reference_predictions)
        preds.sample(min(len(preds), 5000), random_state=0).to_csv(out / "reference_predictions.csv", index=False)

    now = datetime.now(timezone.utc)
    card = {
        "name": "cxr-triage",
        "version": version or now.strftime("%Y.%m.%d-%H%M"),
        "backbone": backbone,
        "img_size": img_size,
        "labels": LABELS,
        "urgent_findings": list(URGENT_FINDINGS),
        "temperature": temperature,
        "target_sensitivity": calibration.get("target_sensitivity"),
        "thresholds": calibration.get("thresholds", {label: 0.5 for label in LABELS}),
        "metrics": {k: v for k, v in (metrics or {}).items() if k != "per_class"},
        "per_class_auroc": {k: v["auroc"] for k, v in (metrics or {}).get("per_class", {}).items()},
        "training_data": "NIH ChestX-ray14 (official patient-wise test split)",
        "intended_use": "Research demo of worklist prioritisation. Not a medical device; not for clinical decisions.",
        "mlflow_run_id": run_id,
        "created_at": now.isoformat(timespec="seconds"),
        "onnx_sha256": hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest(),
    }
    (out / "model_card.json").write_text(json.dumps(_json_safe(card), indent=2))
    print(f"Bundle written to {out} (version {card['version']})")
    return card


def main() -> None:
    from cxr.models.classifier import CXRClassifier

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-info", default="outputs/last_run.json")
    ap.add_argument("--ckpt")
    ap.add_argument("--data-dir")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="models/bundle")
    ap.add_argument("--version")
    args = ap.parse_args()

    info = json.loads(Path(args.run_info).read_text()) if Path(args.run_info).exists() else {}
    ckpt = args.ckpt or info["best_ckpt"]
    data_dir = Path(args.data_dir or info["data_dir"])
    reports = Path(args.reports)

    model = CXRClassifier.load_from_checkpoint(ckpt, map_location="cpu")
    read = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731
    build_bundle(
        model.net,
        Path(args.out),
        img_size=model.hparams.img_size,
        backbone=model.hparams.backbone,
        calibration=read(reports / "calibration.json"),
        metrics=read(reports / "metrics.json"),
        reference_images=data_dir / REFERENCE_FILE,
        reference_predictions=reports / "test_predictions.csv",
        version=args.version,
        run_id=info.get("run_id"),
    )


if __name__ == "__main__":
    main()
