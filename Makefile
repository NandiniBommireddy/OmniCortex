PYTHON   := python3
VENV     := .venv
BIN      := $(VENV)/bin
PIP      := $(BIN)/pip
REQ      := requirements.txt
RADREQ   := requirements-radgraph.txt

PYTHON_RADGRAPH := python3.11
RADVENV         := .venv-radgraph
RADBIN          := $(RADVENV)/bin
RADPIP          := $(RADBIN)/pip
RADPY           := $(RADBIN)/python

.DEFAULT_GOAL := help

.PHONY: help venv install freeze run test lint format clean \
	venv-radgraph install-radgraph freeze-radgraph check-radgraph extract-train-radgraph clean-radgraph

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
	@echo ""
	@echo "RadGraph targets (isolated pinned env):"
	@echo "  venv-radgraph         Create .venv-radgraph with Python 3.11"
	@echo "  install-radgraph      Install pinned RadGraph dependencies"
	@echo "  freeze-radgraph       Save .venv-radgraph packages to $(RADREQ)"
	@echo "  check-radgraph        Print key package versions in .venv-radgraph"
	@echo "  extract-train-radgraph  Run triplet extraction on train split"
	@echo "  clean-radgraph        Remove .venv-radgraph"

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

# Create isolated RadGraph environment with pinned, compatible versions.
venv-radgraph:
	$(PYTHON_RADGRAPH) -m venv $(RADVENV)
	$(RADPIP) install --upgrade pip setuptools wheel

# Install a known-good dependency set for modern-radgraph-xl.
install-radgraph: venv-radgraph
	@if [ -f $(RADREQ) ]; then \
		$(RADPIP) install -r $(RADREQ); \
	else \
		$(RADPIP) install \
			"numpy<2" \
			"torch==2.4.1" \
			"transformers==4.49.0" \
			"tokenizers==0.21.0" \
			"huggingface_hub==0.26.5"; \
	fi
	# torchvision is unnecessary for RadGraph triplet extraction and often causes torch ABI conflicts.
	-$(RADPIP) uninstall -y torchvision
	$(RADPIP) install -e ./radgraph

# Freeze isolated RadGraph environment into its own requirements file.
freeze-radgraph: install-radgraph
	@$(RADPIP) freeze | \
		grep -E '^(numpy==|torch==|transformers==|tokenizers==|huggingface-hub==|huggingface_hub==)' | \
		sed 's/^huggingface_hub==/huggingface-hub==/' | \
		sort -f > $(RADREQ)
	@echo "Saved RadGraph core pinned dependencies to $(RADREQ)"

# Validate pinned versions quickly.
check-radgraph: install-radgraph
	$(RADPY) -c "import transformers, torch, numpy, huggingface_hub; print(transformers.__version__, torch.__version__, numpy.__version__, huggingface_hub.__version__)"

clean-radgraph:
	rm -rf $(RADVENV)
	@echo "Removed $(RADVENV)"
