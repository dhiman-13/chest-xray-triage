"""Data- and prediction-drift report: production traffic vs. the model's reference data.

    python monitoring/drift_report.py --bundle serving/model_store \
        --current https://<space>.hf.space/api/monitoring/export --token $MONITOR_TOKEN

Why both kinds of drift? Ground-truth labels arrive late (or never) in radiology, so we
cannot watch accuracy directly. We watch proxies instead:
  * data drift        - image statistics (exposure, contrast, edge density) vs. training images
  * prediction drift  - predicted probability per finding vs. the held-out test set
Each feature gets a Population Stability Index (PSI) and a two-sample KS test.
PSI < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift.

Exit code 2 with --fail-on-drift when the share of drifted features exceeds --max-drift-share,
which the scheduled GitHub Action turns into a "retraining recommended" issue.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.preprocess import STAT_COLUMNS  # noqa: E402


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def load_current(source: str, token: str | None) -> pd.DataFrame:
    if source.startswith("http"):
        req = urllib.request.Request(source, headers={"X-Monitor-Token": token} if token else {})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
        return pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame()
    return pd.read_csv(source)


def compare(ref: pd.DataFrame, cur: pd.DataFrame, columns: list[str], kind: str, psi_threshold: float) -> list[dict]:
    rows = []
    for col in columns:
        if col not in ref or col not in cur:
            continue
        r, c = ref[col].dropna().to_numpy(), cur[col].dropna().to_numpy()
        if len(r) < 10 or len(c) < 10:
            continue
        value = psi(r, c)
        rows.append({
            "feature": col, "kind": kind, "psi": round(value, 4),
            "ks_pvalue": float(ks_2samp(r, c).pvalue),
            "ref_mean": float(r.mean()), "cur_mean": float(c.mean()),
            "drift": value > psi_threshold,
        })  # fmt: skip
    return rows


def write_html(results: list[dict], summary: dict, path: Path) -> None:
    def row(r):
        status = "🔴 drift" if r["drift"] else ("🟡 watch" if r["psi"] > 0.1 else "🟢 stable")
        return (f"<tr><td>{r['feature']}</td><td>{r['kind']}</td><td>{r['psi']:.3f}</td>"
                f"<td>{r['ks_pvalue']:.3g}</td><td>{r['ref_mean']:.3f}</td><td>{r['cur_mean']:.3f}</td><td>{status}</td></tr>")  # fmt: skip

    html = f"""<!doctype html><meta charset="utf-8"><title>Drift report</title>
<style>body{{font-family:Inter,system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 16px;color:#0f1b2d}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:6px 10px;border-bottom:1px solid #e2e7ee;text-align:left}}
.box{{padding:14px 18px;border-radius:12px;background:{'#fee2e2' if summary['drift_detected'] else '#d1fae5'};margin:16px 0}}</style>
<h1>Drift report</h1><p>Generated {summary['generated_at']} · {summary['n_current']} production studies vs. reference</p>
<div class="box"><b>{'Drift detected - retraining recommended' if summary['drift_detected'] else 'No significant drift'}</b>
 · {summary['n_drifted']}/{summary['n_features']} features drifted (share {summary['drift_share']:.0%}, limit {summary['max_drift_share']:.0%})</div>
<table><tr><th>Feature</th><th>Type</th><th>PSI</th><th>KS p-value</th><th>Ref mean</th><th>Prod mean</th><th>Status</th></tr>
{''.join(row(r) for r in sorted(results, key=lambda r: -r['psi']))}</table>"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="serving/model_store", help="dir with reference_*.csv")
    ap.add_argument("--current", required=True, help="CSV path or API export URL")
    ap.add_argument("--token")
    ap.add_argument("--out", default="reports/drift")
    ap.add_argument("--psi-threshold", type=float, default=0.2)
    ap.add_argument("--max-drift-share", type=float, default=0.3)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--fail-on-drift", action="store_true")
    args = ap.parse_args()

    bundle, out = Path(args.bundle), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cur = load_current(args.current, args.token)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_current": int(len(cur))}
    if len(cur) < args.min_samples:
        summary.update({"status": "insufficient_data", "drift_detected": False})
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Only {len(cur)} production samples (< {args.min_samples}); skipping drift test.")
        return 0

    results = []
    if (bundle / "reference_images.csv").exists():
        results += compare(pd.read_csv(bundle / "reference_images.csv"), cur, STAT_COLUMNS, "data", args.psi_threshold)
    if (bundle / "reference_predictions.csv").exists():
        ref_p = pd.read_csv(bundle / "reference_predictions.csv")
        results += compare(ref_p, cur, [c for c in ref_p.columns if c.startswith("prob_")], "prediction", args.psi_threshold)

    n_drift = sum(r["drift"] for r in results)
    share = n_drift / max(len(results), 1)
    summary.update({
        "status": "ok", "n_features": len(results), "n_drifted": n_drift, "drift_share": share,
        "max_drift_share": args.max_drift_share, "drift_detected": share > args.max_drift_share,
        "top_features": [r["feature"] for r in sorted(results, key=lambda r: -r["psi"])[:5]],
    })  # fmt: skip
    pd.DataFrame(results).to_csv(out / "features.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_html(results, summary, out / "report.html")
    print(json.dumps(summary, indent=2))
    return 2 if (args.fail_on_drift and summary["drift_detected"]) else 0


if __name__ == "__main__":
    sys.exit(main())
