"""Deploy the serving app to a Hugging Face Docker Space.

    HF_TOKEN=... python deploy/deploy_space.py --space <user>/lunglens --model-repo <user>/cxr-triage-model
    HF_TOKEN=... python deploy/deploy_space.py --space <user>/lunglens --bundle models/bundle

The Space gets only what the container needs (Dockerfile at its root, as Spaces require).
With --model-repo the model is not copied: the container downloads the bundle from the model
repo on boot, so shipping a new model = publishing a new bundle + restarting the Space.
With --bundle the bundle is baked into the image instead: one self-contained Space.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: LungLens - Chest X-ray Triage
emoji: 🫁
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

AI-assisted chest X-ray triage with explainable heatmaps. Research prototype, not a medical device.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-repo", help="HF model repo the container downloads the bundle from")
    source.add_argument("--bundle", type=Path, help="local bundle directory to bake into the image")
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))  # falls back to the cached `hf auth login` token
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    # A demo link must be shareable: accounts can default new repos to private, and
    # exist_ok keeps an existing private Space private, so set visibility explicitly.
    api.update_repo_settings(args.space, repo_type="space", private=False)
    if args.model_repo:
        api.add_space_variable(args.space, "HF_MODEL_REPO", args.model_repo)
    if os.environ.get("MONITOR_TOKEN"):
        api.add_space_secret(args.space, "MONITOR_TOKEN", os.environ["MONITOR_TOKEN"])

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        shutil.copy(ROOT / "serving/Dockerfile", t / "Dockerfile")
        shutil.copy(ROOT / "requirements-serve.txt", t / "requirements-serve.txt")
        (t / "src/cxr").mkdir(parents=True)
        for f in ("__init__.py", "labels.py", "preprocess.py"):
            shutil.copy(ROOT / "src/cxr" / f, t / "src/cxr" / f)
        shutil.copytree(ROOT / "serving", t / "serving",
                        ignore=shutil.ignore_patterns("__pycache__", "*.db", "*.onnx", "*.csv", "model_card.json", "Dockerfile"))  # fmt: skip
        if args.bundle:
            for f in ("model.onnx", "model_card.json", "reference_images.csv", "reference_predictions.csv"):
                if (args.bundle / f).exists():
                    shutil.copy(args.bundle / f, t / "serving/model_store" / f)
        (t / "README.md").write_text(SPACE_README, encoding="utf-8")
        api.upload_folder(folder_path=str(t), repo_id=args.space, repo_type="space",
                          commit_message="Deploy LungLens", delete_patterns=["*"])  # fmt: skip
    print(f"Deployed to https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
