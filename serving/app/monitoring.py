"""Production telemetry: Prometheus metrics + a prediction/feedback log in SQLite.

Images are never stored (privacy by design): only probabilities, the triage decision and
label-free image statistics, which is all that drift monitoring needs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from prometheus_client import Counter, Histogram, Info

from cxr.labels import LABELS
from cxr.preprocess import STAT_COLUMNS

REQUESTS = Counter("cxr_http_requests_total", "HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram(
    "cxr_http_request_duration_seconds", "End-to-end request latency", ["path"],
    buckets=(0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2.5, 5),
)  # fmt: skip
INFERENCE_LATENCY = Histogram(
    "cxr_inference_seconds", "Model forward pass latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1, 2),
)  # fmt: skip
PREDICTIONS = Counter("cxr_predictions_total", "Predictions by triage level", ["triage"])
FLAGGED = Counter("cxr_findings_flagged_total", "Findings above their operating threshold", ["finding"])
FINDING_PROB = Histogram(
    "cxr_finding_probability", "Predicted probability per finding", ["finding"],
    buckets=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
)  # fmt: skip
INPUT_INTENSITY = Histogram(
    "cxr_input_mean_intensity", "Mean pixel intensity of inputs (drift signal)",
    buckets=(40, 60, 80, 100, 120, 140, 160, 180, 200),
)  # fmt: skip
FEEDBACK = Counter("cxr_feedback_total", "Clinician feedback", ["verdict"])
MODEL_INFO = Info("cxr_model", "Model currently being served")


def record_prediction_metrics(result: dict) -> None:
    PREDICTIONS.labels(result["triage"]).inc()
    INFERENCE_LATENCY.observe(result["inference_ms"] / 1000)
    INPUT_INTENSITY.observe(result["image_stats"]["mean_intensity"])
    for f in result["findings"]:
        FINDING_PROB.labels(f["label"]).observe(f["probability"])
        if f["positive"]:
            FLAGGED.labels(f["label"]).inc()


class PredictionStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock, self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY, ts REAL, model_version TEXT, triage TEXT,
                    top_finding TEXT, flagged TEXT, probs TEXT, stats TEXT, latency_ms REAL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    prediction_id TEXT, ts REAL, verdict TEXT, corrected TEXT, comment TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);
                """
            )

    def log_prediction(self, result: dict, latency_ms: float) -> str:
        pid = uuid.uuid4().hex[:12]
        probs = {f["label"]: f["probability"] for f in result["findings"]}
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)",
                (pid, time.time(), result["model_version"], result["triage"], result["findings"][0]["label"],
                 json.dumps(result["flagged"]), json.dumps(probs), json.dumps(result["image_stats"]), latency_ms),
            )  # fmt: skip
        return pid

    def add_feedback(self, prediction_id: str, verdict: str, corrected: list[str], comment: str) -> bool:
        with self.lock, self.conn:
            exists = self.conn.execute("SELECT 1 FROM predictions WHERE id=?", (prediction_id,)).fetchone()
            if not exists:
                return False
            self.conn.execute(
                "INSERT INTO feedback VALUES (?,?,?,?,?)",
                (prediction_id, time.time(), verdict, json.dumps(corrected), comment[:500]),
            )
        return True

    def recent(self, limit: int = 20) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                """SELECT p.id, p.ts, p.model_version, p.triage, p.top_finding, p.flagged, p.latency_ms,
                          f.verdict FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id
                   ORDER BY p.ts DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [{**dict(r), "flagged": json.loads(r["flagged"])} for r in rows]

    def summary(self, hours: float = 24) -> dict:
        since = time.time() - hours * 3600
        with self.lock:
            c = self.conn
            total = c.execute("SELECT COUNT(*) FROM predictions WHERE ts>=?", (since,)).fetchone()[0]
            by_triage = dict(c.execute("SELECT triage, COUNT(*) FROM predictions WHERE ts>=? GROUP BY triage", (since,)).fetchall())
            lat = c.execute("SELECT AVG(latency_ms) FROM predictions WHERE ts>=?", (since,)).fetchone()[0]
            fb = dict(c.execute("SELECT verdict, COUNT(*) FROM feedback WHERE ts>=? GROUP BY verdict", (since,)).fetchall())
            hourly = c.execute(
                "SELECT CAST((ts - ?) / 3600 AS INT) AS h, COUNT(*) FROM predictions WHERE ts>=? GROUP BY h",
                (since, since),
            ).fetchall()
            flagged_rows = c.execute("SELECT flagged FROM predictions WHERE ts>=?", (since,)).fetchall()
        counts = {label: 0 for label in LABELS}
        for (row,) in flagged_rows:
            for label in json.loads(row):
                counts[label] = counts.get(label, 0) + 1
        per_hour = [0] * int(hours)
        for h, n in hourly:
            if 0 <= h < len(per_hour):
                per_hour[h] = n
        n_fb = sum(fb.values())
        return {
            "window_hours": hours,
            "total": total,
            "by_triage": {k: by_triage.get(k, 0) for k in ("urgent", "review", "routine")},
            "avg_latency_ms": round(lat or 0.0, 1),
            "feedback": {"agree": fb.get("agree", 0), "disagree": fb.get("disagree", 0)},
            "agreement_rate": round(fb.get("agree", 0) / n_fb, 3) if n_fb else None,
            "flagged_counts": counts,
            "per_hour": per_hour,
        }

    def export_rows(self, limit: int = 5000) -> list[dict]:
        """Flat rows (image stats + probabilities) consumed by monitoring/drift_report.py."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT ts, model_version, probs, stats FROM predictions ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            stats, probs = json.loads(r["stats"]), json.loads(r["probs"])
            out.append({
                "ts": r["ts"], "model_version": r["model_version"],
                **{c: stats.get(c) for c in STAT_COLUMNS},
                **{f"prob_{label}": probs.get(label) for label in LABELS},
            })  # fmt: skip
        return out
