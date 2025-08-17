VENV ?= .venv
PY   ?= $(VENV)/bin/python
PIP  ?= $(VENV)/bin/pip

VERSION := $(shell $(PY) -c "import tomllib,sys; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['version'])" 2>/dev/null)

.PHONY: venv dev lint test clean quickstart build dist tag push-tag publish-testpypi publish-pypi publish

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
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache build dist
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

build: dev
	$(PIP) install -U build wheel twine
	$(PY) -m build
	$(PY) -m twine check dist/* || true

dist: build

tag:
	@test -n "$(VERSION)" || (echo "VERSION not detected; ensure venv active and pyproject has [project].version" && exit 1)
	git tag -a v$(VERSION) -m "v$(VERSION)"

push-tag: tag
	git push origin v$(VERSION)

publish-testpypi: build
	$(PY) -m twine upload --repository testpypi dist/*

publish-pypi: build
	$(PY) -m twine upload dist/*

INDEX ?= pypi
publish: build
	$(PY) -m twine upload $(if $(filter $(INDEX),testpypi),--repository testpypi,) dist/*
