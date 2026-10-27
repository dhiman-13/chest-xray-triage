.PHONY: install smoke test lint serve docker monitor drift

install:
	pip install -r requirements-dev.txt && pip install -e .

# Full pipeline on synthetic data in ~2 minutes on a laptop CPU
smoke:
	python -m cxr.data.synthetic --out data/synthetic --n 400
	python -m cxr.train data=synthetic model=tiny_test train.epochs=2 train.batch_size=32 train.num_workers=0
	python -m cxr.evaluate --n-boot 50 --num-workers 0
	python -m cxr.export --out serving/model_store
	python scripts/smoke_api.py

test:
	pytest -q

lint:
	ruff check src serving monitoring tests deploy

serve:
	uvicorn serving.app.main:app --reload --port 7860

docker:
	docker build -f serving/Dockerfile -t cxr-triage .

monitor:
	cd monitoring && docker compose up --build

drift:
	python monitoring/drift_report.py --bundle serving/model_store --current http://localhost:7860/api/monitoring/export
