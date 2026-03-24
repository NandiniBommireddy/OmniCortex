PYTHON   := python3
VENV     := .venv
BIN      := $(VENV)/bin
PIP      := $(BIN)/pip
REQ      := requirements.txt

.DEFAULT_GOAL := help

.PHONY: help venv install freeze run test lint format clean

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  venv      Create virtual environment using Python 3.14.3"
	@echo "  install   Install dependencies from $(REQ)"
	@echo "  freeze    Save installed packages to $(REQ)"
	@echo "  run       Run the project (edit the run target as needed)"
	@echo "  test      Run tests with pytest"
	@echo "  lint      Lint with flake8"
	@echo "  format    Format with black"
	@echo "  clean     Remove virtual environment and cache files"

# Create the virtual environment
venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

# Install all dependencies from requirements.txt
install: venv
	@if [ -f $(REQ) ]; then \
		$(PIP) install -r $(REQ); \
	else \
		echo "$(REQ) not found – skipping install"; \
	fi

# Freeze current environment into requirements.txt
freeze:
	$(PIP) freeze > $(REQ)
	@echo "Saved dependencies to $(REQ)"

# Run the project – adjust the entry point as needed
run: install
	$(BIN)/python main.py

# Run tests
test: install
	$(BIN)/pytest

# Lint
lint: install
	$(BIN)/flake8 .

# Format
format: install
	$(BIN)/black .

# Remove venv and cache artifacts
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned up"
