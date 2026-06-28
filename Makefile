.PHONY: install qdrant ingest run eval test

install:
	pip install -r requirements.txt

qdrant:
	docker compose up -d

ingest:
	python -m app.ingestion

run:
	uvicorn app.api:app --reload --port 8000

eval:
	python -m app.evaluate

test:
	pytest -q
