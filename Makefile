.PHONY: setup data pipeline test lint api smoke

ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

setup:
	python -m venv .venv
	$(PY) -m pip install -r requirements.lock
	$(PY) -m pip install --no-deps -e .

data:
	$(PY) -c "import sys; from scorecard import ingest; from scorecard.config import load_config as c; r=c().paths.raw; m=[n for n in ingest.REQUIRED if not (r/(n+'.csv')).exists()]; print('Missing:', m, '- download from kaggle.com/c/home-credit-default-risk into data/raw') if m else print('All files present'); sys.exit(bool(m))"

pipeline:
	$(PY) -m scorecard.pipeline

smoke:
	$(PY) -m scorecard.pipeline --smoke

test:
	$(PY) -m pytest --cov=scorecard --cov-report=term --cov-fail-under=80

lint:
	$(PY) -m ruff check src tests
	$(PY) -m mypy src

api:
	$(PY) -m uvicorn scorecard.api.main:create_app --factory --port 8000
