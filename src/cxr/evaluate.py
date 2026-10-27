"""Evaluate a checkpoint the way a medical-imaging paper would, and log it to MLflow.

    python -m cxr.evaluate                       # uses outputs/last_run.json from training
    python -m cxr.evaluate --ckpt path/best.ckpt --data-dir data/processed --run-id <mlflow id>

Produces in --out (default reports/):
    metrics.json            per-class + macro AUROC with 95% bootstrap CIs, AUPRC, sens@90%spec, ECE
    calibration.json        temperature + per-class operating thresholds (fit on validation only)
    subgroups.csv           macro AUROC by sex / age band / view position
    test_predictions.csv    calibrated test probabilities (baseline for prediction-drift monitoring)
    figures/*.png           ROC curves, reliability diagram, comparison with literature, Grad-CAM
    RESULTS.md              research-style summary table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from cxr.baselines import BASELINES
from cxr.data.dataset import CXRDataset
from cxr.data.store import load_metadata
from cxr.explain import save_gradcam_grid
from cxr.labels import LABELS, URGENT_FINDINGS, pretty
from cxr.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    fit_temperature,
    per_class_auprc,
    per_class_auroc,
    reliability_curve,
    sensitivity_at_specificity,
    sigmoid,
    thresholds_for_sensitivity,
)
from cxr.models.classifier import CXRClassifier


@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def subgroup_table(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    groups = {
        "sex": frame["sex"].astype(str),
        "age": pd.cut(frame["age"], [0, 40, 60, 80, 120], labels=["<40", "40-59", "60-79", "80+"]).astype(str),
        "view": frame["view"].astype(str),
    }
    rows = []
    for attr, values in groups.items():
        for value in sorted(values.unique()):
            m = (values == value).to_numpy()
            if value == "nan" or m.sum() < 50:
                continue
            rows.append(
                {"attribute": attr, "group": value, "n": int(m.sum()),
                 "auroc_macro": float(np.nanmean(per_class_auroc(y[m], p[m])))}
            )  # fmt: skip
    return pd.DataFrame(rows)


def plot_roc(y, p, auc, path):
    fig, axes = plt.subplots(3, 5, figsize=(16, 9.5))
    for k, ax in enumerate(axes.ravel()):
        if k >= len(LABELS):
            ax.axis("off")
            continue
        if np.isfinite(auc[k]):
            fpr, tpr, _ = roc_curve(y[:, k], p[:, k])
            ax.plot(fpr, tpr, lw=1.8, color="#2563eb")
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#9ca3af")
        ax.set_title(f"{pretty(LABELS[k])}  AUC={auc[k]:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.supxlabel("False positive rate")
    fig.supylabel("True positive rate")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(y, p_raw, p_cal, ece_raw, ece_cal, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="#9ca3af", label="perfect")
    for probs, name, ece, color in ((p_raw, "uncalibrated", ece_raw, "#f97316"), (p_cal, "temperature-scaled", ece_cal, "#2563eb")):
        conf, acc = reliability_curve(y, probs)
        ax.plot(conf, acc, marker="o", ms=3, color=color, label=f"{name} (ECE={ece:.3f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (all labels)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_vs_baselines(auc, lo, hi, path):
    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x, auc, yerr=[auc - lo, hi - auc], color="#2563eb", alpha=0.85, capsize=2, label="This work (95% CI)")
    markers = ["D", "o"]
    for (name, ref), mk in zip(BASELINES.items(), markers, strict=True):
        ax.scatter(x, [ref[label] for label in LABELS], marker=mk, color="#111827" if mk == "D" else "#f59e0b", zorder=3, s=22, label=name)
    ax.set_xticks(x, [pretty(label) for label in LABELS], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Test AUROC")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_results_md(path, model_name, metrics, subgroups: pd.DataFrame):
    lines = [
        f"# Results: {model_name}",
        "",
        f"Test set: {metrics['n_test']} images, patient-disjoint from training. "
        f"Macro AUROC **{metrics['auroc_macro']:.3f}** (95% CI {metrics['auroc_macro_ci'][0]:.3f}-{metrics['auroc_macro_ci'][1]:.3f}). "
        f"Calibration: ECE {metrics['ece_uncalibrated']:.3f} -> {metrics['ece_calibrated']:.3f} after temperature scaling (T={metrics['temperature']:.2f}).",
        "",
        "| Finding | AUROC (95% CI) | AUPRC | Sens @ 90% spec | Prevalence | Wang 2017 |",
        "|---|---|---|---|---|---|",
    ]
    wang = BASELINES["Wang et al. 2017"]
    for label in LABELS:
        m = metrics["per_class"][label]
        lines.append(
            f"| {pretty(label)} | {m['auroc']:.3f} ({m['auroc_ci'][0]:.3f}-{m['auroc_ci'][1]:.3f}) | "
            f"{m['auprc']:.3f} | {m['sens_at_90spec']:.3f} | {m['prevalence']:.3f} | {wang[label]:.3f} |"
        )
    lines += ["", "## Subgroup analysis (macro AUROC)", "", "| Attribute | Group | n | AUROC |", "|---|---|---|---|"]
    for r in subgroups.itertuples():
        lines.append(f"| {r.attribute} | {r.group} | {r.n} | {r.auroc_macro:.3f} |")
    lines += ["", "![ROC](figures/roc_curves.png)", "![Baselines](figures/auroc_vs_baselines.png)",
              "![Calibration](figures/reliability.png)", "![Grad-CAM](figures/gradcam.png)"]  # fmt: skip
    Path(path).write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-info", default="outputs/last_run.json")
    ap.add_argument("--ckpt")
    ap.add_argument("--data-dir")
    ap.add_argument("--run-id")
    ap.add_argument("--tracking-uri")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--target-sensitivity", type=float, default=0.9)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    info = json.loads(Path(args.run_info).read_text()) if Path(args.run_info).exists() else {}
    ckpt = args.ckpt or info["best_ckpt"]
    data_dir = args.data_dir or info["data_dir"]
    run_id = args.run_id or info.get("run_id")
    tracking_uri = args.tracking_uri or info.get("tracking_uri")

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CXRClassifier.load_from_checkpoint(ckpt, map_location="cpu").to(device)
    img_size = model.hparams.img_size

    meta = load_metadata(data_dir)
    frames = {s: meta[meta["split"] == s].reset_index(drop=True) for s in ("val", "test")}
    loaders = {
        s: DataLoader(CXRDataset(data_dir, f, img_size, train=False), batch_size=args.batch_size,
                      num_workers=args.num_workers)
        for s, f in frames.items()
    }  # fmt: skip
    logits = {s: predict_logits(model, dl, device) for s, dl in loaders.items()}
    y = {s: f[LABELS].to_numpy(np.float32) for s, f in frames.items()}

    # Everything that is *fit* (temperature, thresholds) uses validation data only.
    temperature = fit_temperature(logits["val"], y["val"])
    p_val = sigmoid(logits["val"] / temperature)
    thresholds = thresholds_for_sensitivity(y["val"], p_val, args.target_sensitivity)

    yt = y["test"]
    p_raw = sigmoid(logits["test"])
    p_test = sigmoid(logits["test"] / temperature)
    auc = per_class_auroc(yt, p_test)
    lo, hi = bootstrap_ci(yt, p_test, n_boot=args.n_boot)
    auprc = per_class_auprc(yt, p_test)
    sens90 = sensitivity_at_specificity(yt, p_test, 0.9)
    ece_raw, ece_cal = expected_calibration_error(yt, p_raw), expected_calibration_error(yt, p_test)
    pred = p_test >= thresholds
    achieved_sens = (pred & (yt == 1)).sum(0) / np.maximum(yt.sum(0), 1)
    achieved_spec = (~pred & (yt == 0)).sum(0) / np.maximum((yt == 0).sum(0), 1)

    metrics = {
        "model": model.hparams.backbone,
        "n_test": int(len(yt)),
        "auroc_macro": float(np.nanmean(auc)),
        "auroc_macro_ci": [float(lo[-1]), float(hi[-1])],
        "auprc_macro": float(np.nanmean(auprc)),
        "auroc_urgent_macro": float(np.nanmean([auc[LABELS.index(u)] for u in URGENT_FINDINGS])),
        "ece_uncalibrated": ece_raw,
        "ece_calibrated": ece_cal,
        "temperature": temperature,
        "per_class": {
            label: {
                "auroc": float(auc[k]), "auroc_ci": [float(lo[k]), float(hi[k])],
                "auprc": float(auprc[k]), "sens_at_90spec": float(sens90[k]),
                "threshold": float(thresholds[k]), "test_sensitivity": float(achieved_sens[k]),
                "test_specificity": float(achieved_spec[k]), "prevalence": float(yt[:, k].mean()),
            }
            for k, label in enumerate(LABELS)
        },  # fmt: skip
    }
    calibration = {
        "temperature": temperature,
        "target_sensitivity": args.target_sensitivity,
        "thresholds": {label: float(t) for label, t in zip(LABELS, thresholds, strict=True)},
    }
    subgroups = subgroup_table(frames["test"], yt, p_test)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2))
    subgroups.to_csv(out / "subgroups.csv", index=False)
    pd.DataFrame(p_test, columns=[f"prob_{label}" for label in LABELS]).to_csv(out / "test_predictions.csv", index=False)

    plot_roc(yt, p_test, auc, out / "figures/roc_curves.png")
    plot_reliability(yt, p_raw, p_test, ece_raw, ece_cal, out / "figures/reliability.png")
    plot_vs_baselines(auc, lo[:-1], hi[:-1], out / "figures/auroc_vs_baselines.png")

    # Grad-CAM on a few confidently-detected urgent findings
    ds = loaders["test"].dataset
    urgent_cols = [LABELS.index(u) for u in URGENT_FINDINGS]
    score = (p_test[:, urgent_cols] * yt[:, urgent_cols]).max(1)
    picks = np.argsort(-score)[:6]
    samples = [(ds[i][0][None], yt[i], p_test[i]) for i in picks]
    save_gradcam_grid(model.net, samples, out / "figures/gradcam.png")

    write_results_md(out / "RESULTS.md", model.hparams.backbone, metrics, subgroups)

    if run_id and tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                "test/auroc_macro": metrics["auroc_macro"],
                "test/auroc_macro_ci_low": metrics["auroc_macro_ci"][0],
                "test/auprc_macro": metrics["auprc_macro"],
                "test/auroc_urgent_macro": metrics["auroc_urgent_macro"],
                "test/ece_calibrated": ece_cal,
                "test/ece_uncalibrated": ece_raw,
                "calibration/temperature": temperature,
                **{f"test_auroc/{label}": float(a) for label, a in zip(LABELS, auc, strict=True) if np.isfinite(a)},
            })  # fmt: skip
            mlflow.log_artifacts(str(out), artifact_path="evaluation")

    print(f"Test macro AUROC {metrics['auroc_macro']:.4f} "
          f"(95% CI {metrics['auroc_macro_ci'][0]:.4f}-{metrics['auroc_macro_ci'][1]:.4f}), "
          f"ECE {ece_raw:.4f} -> {ece_cal:.4f}")  # fmt: skip


if __name__ == "__main__":
    main()
