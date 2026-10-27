"""FastAPI service: triage API + web frontend + Prometheus metrics, in one container.

    uvicorn serving.app.main:app --port 7860
"""

from __future__ import annotations

import csv
import io
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from cxr.labels import LABELS
from cxr.preprocess import decode_image
from serving.app.inference import TriageModel, ensure_model
from serving.app.monitoring import (
    FEEDBACK,
    MODEL_INFO,
    REQUEST_LATENCY,
    REQUESTS,
    PredictionStore,
    record_prediction_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "model_store"))
DB_PATH = os.environ.get("DB_PATH", str(ROOT / "predictions.db"))
FRONTEND_DIR = ROOT / "frontend"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MONITOR_TOKEN = os.environ.get("MONITOR_TOKEN")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_model(MODEL_DIR)
    model = TriageModel(MODEL_DIR)
    state["model"] = model
    state["store"] = PredictionStore(DB_PATH)
    MODEL_INFO.info({"version": model.version, "backbone": model.card["backbone"]})
    yield
    state.clear()


app = FastAPI(
    title="CXR Triage API",
    description="Chest X-ray finding detection and worklist triage. Research demo, not a medical device.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", "static")  # route template keeps label cardinality bounded
    if path != "/metrics":
        REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(path).observe(time.perf_counter() - start)
    return response


class Feedback(BaseModel):
    prediction_id: str
    verdict: Literal["agree", "disagree"]
    corrected_labels: list[str] = Field(default_factory=list)
    comment: str = ""


@app.get("/health")
def health():
    model: TriageModel | None = state.get("model")
    return {"status": "ok" if model else "loading", "model_version": model.version if model else None}


@app.get("/api/model")
def model_card():
    return state["model"].card


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image larger than 15 MB")
    try:
        img = decode_image(data)
    except ValueError as e:
        raise HTTPException(415, str(e)) from e
    if min(img.shape[:2]) < 64:
        raise HTTPException(422, "Image too small: expected a chest radiograph of at least 64x64 px")

    result = state["model"].predict(img)
    latency_ms = (time.perf_counter() - start) * 1000
    result["prediction_id"] = state["store"].log_prediction(result, latency_ms)
    result["latency_ms"] = round(latency_ms, 1)
    record_prediction_metrics(result)
    return result


@app.post("/api/feedback")
def feedback(fb: Feedback):
    unknown = set(fb.corrected_labels) - set(LABELS)
    if unknown:
        raise HTTPException(422, f"Unknown labels: {sorted(unknown)}")
    if not state["store"].add_feedback(fb.prediction_id, fb.verdict, fb.corrected_labels, fb.comment):
        raise HTTPException(404, "Unknown prediction_id")
    FEEDBACK.labels(fb.verdict).inc()
    return {"status": "recorded"}


@app.get("/api/stats")
def stats(hours: float = 24):
    return state["store"].summary(min(max(hours, 1), 24 * 30))


@app.get("/api/predictions")
def predictions(limit: int = 20):
    return state["store"].recent(min(max(limit, 1), 200))


@app.get("/api/monitoring/export")
def export_for_drift(limit: int = 5000, x_monitor_token: str | None = Header(default=None)):
    """CSV of logged image statistics + probabilities, pulled by the scheduled drift job."""
    if MONITOR_TOKEN and not secrets.compare_digest(x_monitor_token or "", MONITOR_TOKEN):
        raise HTTPException(401, "Invalid monitor token")
    rows = state["store"].export_rows(min(limit, 50000))
    if not rows:
        return PlainTextResponse("", media_type="text/csv")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return PlainTextResponse(buf.getvalue(), media_type="text/csv")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Frontend last, so API routes take precedence.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
