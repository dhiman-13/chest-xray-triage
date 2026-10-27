"""Hit the API in-process with a random image: catches broken bundles before deployment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from serving.app.main import app

with TestClient(app) as client:
    ok, png = cv2.imencode(".png", (np.random.rand(300, 300) * 255).astype("uint8"))
    r = client.post("/api/predict", files={"file": ("x.png", png.tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["findings"]) == 14
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200, "frontend not served"
    print(f"API smoke test passed: triage={body['triage']} latency={body['latency_ms']} ms")
