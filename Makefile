VENV ?= .venv
PY   ?= $(VENV)/bin/python
PIP  ?= $(VENV)/bin/pip

.PHONY: venv dev lint test clean quickstart

venv:
	python -m venv $(VENV)

dev: venv
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/pylint  src tests

test:
	$(VENV)/bin/pytest -q

quickstart: dev
	$(PY) examples/quickstart.py

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache build dist **/*.egg-info **/**/**/__pycache__
