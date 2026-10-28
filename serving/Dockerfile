# Inference image: FastAPI + ONNX Runtime + static frontend. No torch -> ~350 MB instead of ~2 GB.
# Build from the repo root:  docker build -f serving/Dockerfile -t cxr-triage .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_DIR=/app/serving/model_store \
    DB_PATH=/data/predictions.db \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only the torch-free part of the package is needed at serving time.
COPY src/cxr/__init__.py src/cxr/labels.py src/cxr/preprocess.py src/cxr/
COPY serving/ serving/

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 app && mkdir -p /data && chown -R app /data /app/serving/model_store
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

CMD ["sh", "-c", "uvicorn serving.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
