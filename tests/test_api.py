import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tiny_bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tiny_bundle))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "pred.db"))
    monkeypatch.delenv("MONITOR_TOKEN", raising=False)
    import serving.app.main as main

    importlib.reload(main)  # pick up the env vars
    with TestClient(main.app) as c:
        yield c


def test_health_and_model_card(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "test"}
    card = client.get("/api/model").json()
    assert len(card["labels"]) == 14 and card["backbone"] == "resnet18"


def test_predict_returns_triage_findings_and_heatmaps(client, png_bytes):
    r = client.post("/api/predict", files={"file": ("x.png", png_bytes, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["triage"] in {"urgent", "review", "routine"}
    assert len(body["findings"]) == 14
    assert all(0 <= f["probability"] <= 1 for f in body["findings"])
    heat = body["heatmaps"]["Effusion"]
    assert len(heat) > 0 and max(max(row) for row in heat) <= 1.0
    assert body["image"].startswith("data:image/png;base64,")


def test_rejects_non_images(client):
    r = client.post("/api/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


def test_feedback_and_stats(client, png_bytes):
    pid = client.post("/api/predict", files={"file": ("x.png", png_bytes, "image/png")}).json()["prediction_id"]
    ok = client.post("/api/feedback", json={"prediction_id": pid, "verdict": "disagree", "corrected_labels": ["Mass"]})
    assert ok.status_code == 200
    assert client.post("/api/feedback", json={"prediction_id": "nope", "verdict": "agree"}).status_code == 404
    assert client.post("/api/feedback", json={"prediction_id": pid, "verdict": "agree", "corrected_labels": ["Cancer"]}).status_code == 422
    stats = client.get("/api/stats").json()
    assert stats["total"] == 1 and stats["feedback"]["disagree"] == 1
    assert client.get("/api/predictions").json()[0]["verdict"] == "disagree"


def test_metrics_and_drift_export(client, png_bytes):
    client.post("/api/predict", files={"file": ("x.png", png_bytes, "image/png")})
    metrics = client.get("/metrics").text
    assert "cxr_predictions_total" in metrics and "cxr_finding_probability_bucket" in metrics
    csv = client.get("/api/monitoring/export").text
    assert csv.splitlines()[0].startswith("ts,model_version,mean_intensity")


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "LungLens" in r.text
