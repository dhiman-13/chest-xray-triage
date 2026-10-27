"""Model registry operations: register, promote (quality gate), pull, publish.

    python -m cxr.registry register --bundle models/bundle
    python -m cxr.registry promote  --metric test/auroc_macro --min-delta 0.002
    python -m cxr.registry pull     --dst serving/model_store
    python -m cxr.registry publish  --bundle models/bundle --repo <hf-user>/cxr-triage-model

Promotion uses MLflow *aliases*: `champion` is what production serves, `previous` keeps the
last champion for one-command rollback. A new version only becomes champion if it beats the
current one on the held-out test metric by at least --min-delta.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "cxr-triage"


def _client(tracking_uri: str | None) -> MlflowClient:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def register(bundle: Path, run_id: str, name: str) -> str:
    import onnx

    with mlflow.start_run(run_id=run_id):
        info = mlflow.onnx.log_model(onnx.load(str(bundle / "model.onnx")), "model", registered_model_name=name)
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
    version = str(info.registered_model_version)
    MlflowClient().set_registered_model_alias(name, "challenger", version)
    print(f"Registered {name} v{version} (alias: challenger)")
    return version


def promote(client: MlflowClient, name: str, metric: str, min_delta: float) -> dict:
    challenger = client.get_model_version_by_alias(name, "challenger")
    new_score = client.get_run(challenger.run_id).data.metrics.get(metric)
    if new_score is None:
        raise SystemExit(f"Challenger run has no metric '{metric}'; run cxr.evaluate first")
    try:
        champion = client.get_model_version_by_alias(name, "champion")
        old_score = client.get_run(champion.run_id).data.metrics.get(metric, float("-inf"))
    except mlflow.exceptions.MlflowException:
        champion, old_score = None, float("-inf")

    promoted = new_score >= old_score + min_delta
    if promoted:
        if champion is not None and champion.version != challenger.version:
            client.set_registered_model_alias(name, "previous", champion.version)
        client.set_registered_model_alias(name, "champion", challenger.version)
    decision = {
        "challenger_version": challenger.version,
        "challenger_score": new_score,
        "champion_version": champion.version if champion else None,
        "champion_score": None if champion is None else old_score,
        "metric": metric,
        "promoted": promoted,
    }
    print(json.dumps(decision, indent=2))
    return decision


def pull(client: MlflowClient, name: str, alias: str, dst: Path) -> None:
    mv = client.get_model_version_by_alias(name, alias)
    dst.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="bundle", dst_path=str(dst.parent / "_dl"))
    for f in (dst.parent / "_dl" / "bundle").iterdir():
        f.replace(dst / f.name)
    print(f"Pulled {name}@{alias} (v{mv.version}) into {dst}")


def publish(bundle: Path, repo: str) -> None:
    """Push the bundle to a Hugging Face Hub model repo; the deployed API downloads it on boot."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    card = json.loads((bundle / "model_card.json").read_text())
    api.upload_folder(folder_path=str(bundle), repo_id=repo, repo_type="model",
                      commit_message=f"cxr-triage {card['version']}")  # fmt: skip
    print(f"Published bundle {card['version']} to https://huggingface.co/{repo}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["register", "promote", "pull", "publish"])
    ap.add_argument("--name", default=MODEL_NAME)
    ap.add_argument("--run-info", default="outputs/last_run.json")
    ap.add_argument("--run-id")
    ap.add_argument("--tracking-uri")
    ap.add_argument("--bundle", type=Path, default=Path("models/bundle"))
    ap.add_argument("--metric", default="test/auroc_macro")
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--dst", type=Path, default=Path("serving/model_store"))
    ap.add_argument("--repo", help="Hugging Face repo id, e.g. yourname/cxr-triage-model")
    args = ap.parse_args()

    info = json.loads(Path(args.run_info).read_text()) if Path(args.run_info).exists() else {}
    tracking_uri = args.tracking_uri or info.get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")

    if args.action == "publish":
        publish(args.bundle, args.repo or os.environ["HF_MODEL_REPO"])
        return
    client = _client(tracking_uri)
    if args.action == "register":
        register(args.bundle, args.run_id or info["run_id"], args.name)
    elif args.action == "promote":
        decision = promote(client, args.name, args.metric, args.min_delta)
        Path("reports").mkdir(exist_ok=True)
        Path("reports/promotion.json").write_text(json.dumps(decision, indent=2))
    elif args.action == "pull":
        pull(client, args.name, args.alias, args.dst)


if __name__ == "__main__":
    main()
