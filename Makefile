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
	venv-radgraph install-radgraph freeze-radgraph check-radgraph extract-train-radgraph clean-radgraph \
	kg-prepare kg-neo4j-up kg-neo4j-down kg-neo4j-logs kg-load kg-verify kg-explore \
	kg-export-subgraph kg-all kg-clean

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
	@echo ""
	@echo "PrimeKG / Neo4j targets:"
	@echo "  kg-prepare            Download & preprocess PrimeKG for Neo4j"
	@echo "  kg-neo4j-up           Start Neo4j container (Docker Compose)"
	@echo "  kg-neo4j-down         Stop Neo4j container"
	@echo "  kg-neo4j-logs         Tail Neo4j container logs"
	@echo "  kg-load               Load PrimeKG CSVs into Neo4j"
	@echo "  kg-verify             Verify node/edge counts in Neo4j"
	@echo "  kg-explore            Run exploration queries (schema, diagnoses, multi-hop)"
	@echo "  kg-export-subgraph    Export radiology subgraph for pipeline use"
	@echo "  kg-all                Full pipeline: prepare -> neo4j-up -> load -> explore"
	@echo "  kg-clean              Remove PrimeKG data and stop Neo4j"

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

# ──────────────────────────────────────────────────────────────
#  PrimeKG / Neo4j
# ──────────────────────────────────────────────────────────────

KG_DATA     := kg/data
KG_NEO4J_URI := bolt://localhost:7687
KG_NEO4J_PW  := primekg123

# Download PrimeKG from Harvard Dataverse and prepare CSVs for Neo4j import.
kg-prepare:
	$(BIN)/python kg/prepare_primekg.py --data-dir $(KG_DATA)

# Start the Neo4j container via Docker Compose.
kg-neo4j-up:
	docker compose -f docker/docker-compose.yaml up -d
	@echo "Neo4j Browser: http://localhost:7474  (neo4j / $(KG_NEO4J_PW))"
	@echo "Waiting for Neo4j to be ready ..."
	@for i in $$(seq 1 30); do \
		docker compose -f docker/docker-compose.yaml exec -T neo4j cypher-shell -u neo4j -p $(KG_NEO4J_PW) "RETURN 1" > /dev/null 2>&1 && break; \
		sleep 2; \
	done
	@echo "Neo4j is ready."

# Stop the Neo4j container.
kg-neo4j-down:
	docker compose -f docker/docker-compose.yaml down

# Tail Neo4j container logs.
kg-neo4j-logs:
	docker compose -f docker/docker-compose.yaml logs -f neo4j

# Load prepared PrimeKG CSVs into Neo4j (requires neo4j to be running).
kg-load:
	$(BIN)/python kg/load_neo4j.py --uri $(KG_NEO4J_URI) --password $(KG_NEO4J_PW)

# Verify node/edge counts match the paper.
kg-verify:
	$(BIN)/python kg/load_neo4j.py --uri $(KG_NEO4J_URI) --password $(KG_NEO4J_PW) --verify-only

# Run exploration queries: schema, seed diagnoses, multi-hop chains.
kg-explore:
	$(BIN)/python kg/explore_primekg.py --uri $(KG_NEO4J_URI) --password $(KG_NEO4J_PW)

# Export the 2-hop radiology subgraph around seed diagnoses.
kg-export-subgraph:
	$(BIN)/python kg/explore_primekg.py --uri $(KG_NEO4J_URI) --password $(KG_NEO4J_PW) \
		--export-subgraph --output-dir $(KG_DATA)/subgraph

# Full pipeline: prepare data, start Neo4j, load, explore.
kg-all: kg-prepare kg-neo4j-up kg-load kg-explore
	@echo "PrimeKG fully loaded and explored!"

# Remove PrimeKG data and stop Neo4j container + volumes.
kg-clean: kg-neo4j-down
	docker compose -f docker/docker-compose.yaml down -v
	rm -rf $(KG_DATA)
	@echo "PrimeKG data and Neo4j volumes removed."
