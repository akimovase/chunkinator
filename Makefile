VENV ?= .venv
PYTHON ?= python3
BIN := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test cov check run clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

install: $(BIN)/python ## Create .venv, install dev dependencies and the pre-commit hook
	$(BIN)/pip install -e ".[dev]"
	$(BIN)/pre-commit install

lint: ## Check style with ruff (linter and formatter), as CI does
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Fix style: format code and apply safe lint fixes
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck: ## Type-check with mypy (strict mode from pyproject.toml)
	$(BIN)/mypy src tests main.py

test: ## Run tests
	$(BIN)/pytest

cov: ## Run tests with coverage; fails below 95%
	$(BIN)/pytest --cov --cov-report=term-missing

check: lint typecheck cov ## Run everything CI runs

run: ## Run the demo in main.py
	$(BIN)/python main.py

clean: ## Remove caches and coverage files
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage coverage.xml htmlcov build dist
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -exec rm -rf {} +
