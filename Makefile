.PHONY: install run test lint format ingest eval docker-up docker-down clean

install:
	pip install -r requirements.txt

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

ingest:
	python scripts/ingest_sample_data.py

eval:
	python scripts/run_eval_suite.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
