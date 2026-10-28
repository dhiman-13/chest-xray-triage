"""Publish a model bundle to a public Hugging Face model repo.

    HF_TOKEN=... python deploy/publish_model.py --repo <user>/lunglens-model --bundle models/bundle

The serving container downloads the bundle from this repo on boot (HF_MODEL_REPO), so any
Docker host can run LungLens: Render, a Hugging Face Space, or your own machine.
Model repos are free; only Docker Spaces need a paid plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

FILES = ("model.onnx", "model_card.json", "reference_images.csv", "reference_predictions.csv")

README = """---
license: other
license_name: research-only
tags: [medical-imaging, chest-x-ray, onnx, densenet, explainability]
datasets: [nih-chest-xrays]
pipeline_tag: image-classification
---

# LungLens: chest X-ray triage model ({version})

DenseNet-121 fine-tuned on NIH ChestX-ray14 to score 14 thoracic findings, exported to ONNX with
temperature scaling and class activation maps built in. Served by the LungLens API
([source](https://github.com/dhiman-13/chest-xray-triage)).

| Metric | Value |
|---|---|
| Test macro AUROC | {auroc:.3f} (95% CI {lo:.3f} to {hi:.3f}) |
| Test images | {n_test:,} (official patient-wise split) |
| Calibration error (ECE) | {ece:.3f} |

**Research prototype, not a medical device.** Not for clinical decisions.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    args = ap.parse_args()

    card = json.loads((args.bundle / "model_card.json").read_text())
    sha = hashlib.sha256((args.bundle / "model.onnx").read_bytes()).hexdigest()
    if sha != card["onnx_sha256"]:
        raise SystemExit("model.onnx does not match the hash in model_card.json")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    api.update_repo_settings(args.repo, repo_type="model", private=False)

    m = card.get("metrics", {})
    lo, hi = m.get("auroc_macro_ci", [float("nan")] * 2)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for f in FILES:
            if (args.bundle / f).exists():
                shutil.copy(args.bundle / f, t / f)
        (t / "README.md").write_text(README.format(
            version=card["version"], auroc=m.get("auroc_macro", float("nan")), lo=lo, hi=hi,
            n_test=m.get("n_test", 0), ece=m.get("ece_calibrated", float("nan"))), encoding="utf-8")  # fmt: skip
        api.upload_folder(folder_path=str(t), repo_id=args.repo, repo_type="model",
                          commit_message=f"Publish model {card['version']}")  # fmt: skip
    print(f"Published https://huggingface.co/{args.repo} (version {card['version']})")


if __name__ == "__main__":
    main()
