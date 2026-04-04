# Installation & Run Guide

Codebase Structure

```
OmniCortex/
├── scripts/                 Pipeline scripts (RadGraph, datastore, LLaVA JSON, chains, eval)
├── kg/                      PrimeKG Neo4j setup (prepare, load, explore)
│   └── data/subgraph/       Exported radiology subgraph + diagnosis mapping
├── models/LLaVA/            LLaVA model (git submodule)
├── radgraph/                RadGraph model (git submodule)
├── MIMIC-NLE/               MIMIC-NLE dataset
├── docker/                  Docker Compose (Neo4j)
├── experiments/             Results and analysis
├── data/                    Entity maps and chain files
├── physionet.org/           Downloaded PhysioNet data (reports, metadata CSVs)
└── tmp/demo/                Pipeline outputs (datastore, LLaVA JSONs, eval results)
```


## 0. Prerequisites

- **PhysioNet** account with signed DUA for MIMIC-CXR-JPG v2.1.0 and MIMIC-NLE
- **Google Cloud Platform** project with a GCS bucket containing MIMIC-CXR images
- **Modal** account (for GPU training/eval)
- **Docker** (for Neo4j in step 4+)
- Python 3.11 via **pyenv**

## 1. Setup

```shell
git submodule add https://github.com/Stanford-AIMI/radgraph.git radgraph

pyenv install 3.11.12
pyenv local 3.11.12
```

### RadGraph environment (isolated, Python 3.11)

```shell
make install-radgraph
make check-radgraph
make freeze-radgraph
```

### Main environment

```shell
deactivate  # exit radgraph venv if active
make venv
make install

.venv/bin/pip install git+https://github.com/RyanWangZf/MedCLIP.git
.venv/bin/pip install faiss-cpu  # or faiss-gpu
.venv/bin/pip install google-cloud-storage
```

## 2. Data

Images are stored in GCS (Google Cloud Storage) bucket: `gs://mimic-cxr-jpg-2.1.0.physionet.org/files/`.
If you want to _actually_ install this project, please find all usages of `storage.Client(project="885253748539")` 
and replace with your Google Cloud Project ID.

```shell
gcloud auth application-default login
```

Download reports and metadata from PhysioNet:

```shell
# Reports (needed for RadGraph extraction in step 3a)
wget -r -N -c -np --user <physionet_user> --ask-password \
  https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip
unzip physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip \
  -d physionet.org/mimic-cxr/2.1.0/

# Metadata CSVs
wget -r -N -c -np --user <physionet_user> --ask-password \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/cxr-study-list.csv.gz
```

Expected paths:
- `physionet.org/mimic-cxr/2.1.0/files/p1*/p*/s*.txt` (reports)
- `physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz`
- `physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz`
- `physionet.org/mimic-cxr-jpg/2.1.0/cxr-study-list.csv.gz`

Modal GCS secret:

```shell
modal secret create gcs-mimic-cxr \
  GOOGLE_CREDENTIALS="$(cat ~/.config/gcloud/application_default_credentials.json)" \
  GOOGLE_PROJECT="$(gcloud config get-value project)"
```

To refresh the secret:

```shell
modal secret delete gcs-mimic-cxr
# then re-run the create command above
```

## 3. Base Pipeline (KG-LLaVA)

### 3a. Extract RadGraph triplets

Output: `tmp/demo/mimic-nle-{train,dev,test}-radgraph.json`

```shell
.venv-radgraph/bin/python scripts/extract_radgraph_triplets.py \
  --input-dir MIMIC-NLE/mimic-nle \
  --output-dir tmp/demo \
  --model-type modern-radgraph-xl \
  --batch-size 4 \
  --num-workers 4 \
  --reports-root physionet.org/mimic-cxr/2.1.0/files
```

### 3b. Build FAISS datastore (~2 hours on macOS CPU)

Output: `tmp/demo/datastore/retrieved_triplets.json`, `kg_nle_index`, `kg_nle_index_captions.json`

Run once for train, once for test:

```shell
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
.venv/bin/python scripts/build_demo_datastore.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --output-dir tmp/demo/datastore
```

### 3c. Build LLaVA JSON (baseline)

Output: `tmp/demo/mimic-nle-{train,test}-kg-llava.json`

```shell
# Train
.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --retrieved-triplets tmp/demo/datastore/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --output tmp/demo/mimic-nle-train-kg-llava.json

# Test
.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-test-radgraph.json \
  --retrieved-triplets tmp/demo/datastore_test/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --output tmp/demo/mimic-nle-test-kg-llava.json
```

## 4. PrimeKG / Neo4j Setup

Requires Docker.

```shell
make kg-prepare            # 1. Download & preprocess PrimeKG CSVs
make kg-neo4j-up           # 2. Start Neo4j container
make kg-load               # 3. Load nodes + edges into Neo4j
make kg-explore            # 4. Run exploration queries
make kg-export-subgraph    # 5. Export radiology subgraph
```

Or steps 1-4 at once: `make kg-all`

Use `make kg-verify` to confirm 129K nodes / 4M edges.

## 5. Multi-Hop Chain Pipeline

Requires Neo4j running with PrimeKG loaded (step 4) and subgraph exported.

### 5a. Install dependencies

```shell
.venv/bin/pip install scispacy rapidfuzz
.venv/bin/pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

### 5b. Build entity alignment map

Output: `data/entity_cui_map.json`

```shell
mkdir -p data
.venv/bin/python scripts/build_entity_cui_map.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --diagnosis-mapping kg/data/subgraph/diagnosis_node_mapping.json \
  --subgraph-nodes kg/data/subgraph/primekg_radiology_nodes.json \
  --spacy-model en_core_sci_lg \
  --output data/entity_cui_map.json
```

### 5c. Build chains + LLaVA JSON (train)

```shell
.venv/bin/python scripts/build_multihop_chains.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --entity-map data/entity_cui_map.json \
  --output data/radgraph-multihop.jsonl

# (no hop)
  .venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --retrieved-triplets tmp/demo/datastore/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --output tmp/demo/mimic-nle-train-kg-llava.json

# (or 1-hop)
.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --retrieved-triplets tmp/demo/datastore/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --chains-file data/radgraph-multihop.jsonl \
  --multihop \
  --output tmp/demo/mimic-nle-train-kg-llava-multihop.json
```

### 5d. Build chains + LLaVA JSON (test)

```shell
.venv/bin/python scripts/build_multihop_chains.py \
  --input tmp/demo/mimic-nle-test-radgraph.json \
  --entity-map data/entity_cui_map.json \
  --output data/radgraph-multihop-test.jsonl

# (no hop)
.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-test-radgraph.json \
  --retrieved-triplets tmp/demo/datastore_test/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --output tmp/demo/mimic-nle-test-kg-llava.json

# (or 1 hop)
.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-test-radgraph.json \
  --retrieved-triplets tmp/demo/datastore_test/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --chains-file data/radgraph-multihop-test.jsonl \
  --multihop \
  --output tmp/demo/mimic-nle-test-kg-llava-multihop.json
```

## 6. RadLex Enriched Triplets Pipeline (Alternative to PrimeKG)

Uses RadLex `May_Cause` edges to enrich RadGraph triplets inline — RadLex signs are
appended directly to each triplet string rather than injected as a separate chains block.

Before: `opacity suggestive of pneumonia`
After:  `opacity suggestive of pneumonia (also presents as: crazy-paving sign, ground glass pattern)`

No Docker or Neo4j required.

### 6a. Install dependencies

```shell
.venv/bin/pip install owlready2
```

### 6b. Download ontologies

Register for a free BioPortal API key at https://bioportal.bioontology.org/accounts/new

```shell
mkdir -p kg/data/radlex kg/data/gamuts
.venv/bin/python scripts/download_ontologies.py --api-key YOUR_BIOPORTAL_KEY
```

Output: `kg/data/radlex/radlex.owl`, `kg/data/gamuts/gamuts.owl`

Verify content (check that RadLex has useful May_Cause edges):

```shell
.venv/bin/python scripts/explore_radlex.py
```

### 6c. Build entity alignment map

Output: `data/entity_radlex_map.json`

```shell
mkdir -p data
.venv/bin/python scripts/build_entity_radlex_map.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --radlex-owl kg/data/radlex/radlex.owl \
  --output data/entity_radlex_map.json
```

### 6d. Build enriched LLaVA JSON (train)

```shell
.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --retrieved-triplets tmp/demo/datastore/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --entity-map data/entity_radlex_map.json \
  --radlex-owl kg/data/radlex/radlex.owl \
  --output tmp/demo/mimic-nle-train-kg-llava-radlex.json
```

### 6e. Build enriched LLaVA JSON (test)

```shell
.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-test-radgraph.json \
  --retrieved-triplets tmp/demo/datastore_test/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --entity-map data/entity_radlex_map.json \
  --radlex-owl kg/data/radlex/radlex.owl \
  --output tmp/demo/mimic-nle-test-kg-llava-radlex.json
```

## 7. Train & Eval on Modal

Toggle between conditions by editing the path constants at the top of each Modal script
(commented-out lines show baseline ↔ multihop ↔ radlex variants).

For reference, training 1 epoch on full MIMIC-CXR-JPG took 10h16m on Modal A10G.

### 7a. PrimeKG multihop (current default in scripts)

```shell
# scripts/modal_demo_train_llava.py already points to mimic-nle-train-kg-llava-multihop.json
modal run scripts/modal_demo_train_llava.py
modal run scripts/modal_demo_eval_llava.py
```

### 7b. RadLex condition

Edit path constants in both Modal scripts (swap commented lines):
- `LOCAL_DATA`  → `tmp/demo/mimic-nle-train-kg-llava-radlex.json`
- `REMOTE_DATA` → `{REMOTE_ROOT}/data/mimic-nle-train-kg-llava-radlex.json`
- `REMOTE_OUT`  → `{REMOTE_ROOT}/outputs-radlex`

(test script: same pattern with `-test` suffix and `REMOTE_TRAIN_OUT`)

```shell
modal run scripts/modal_demo_train_llava.py
modal run scripts/modal_demo_eval_llava.py
```

### 7c. Compute metrics (after eval)

**NLG metrics (BLEU, METEOR, ROUGE-L, CIDEr):**
```shell
.venv/bin/python scripts/metrics.py \
  --answers tmp/demo/llava_modal_eval_radlex/demo_answers.jsonl \
  --references tmp/demo/mimic-nle-test-kg-llava-radlex.json \
  --output tmp/demo/eval_results_radlex.json
```

**LLM-as-judge (Claude Haiku, 100 samples):**

`--answers` and `--references` must point to outputs from the **same test JSON** — `question_id` in the answers file is the row index in the references file. Mismatched files will silently misalign samples.

Each sample is scored by comparing the model's generated answer against the NLE ground truth written by a radiologist for that same image.

```shell
.venv/bin/python scripts/metrics_llm.py \
  --answers tmp/demo/llava_modal_eval_radlex/demo_answers.jsonl \
  --references tmp/demo/mimic-nle-test-kg-llava-radlex.json \
  --output tmp/demo/eval_results_llm_radlex.json \
  --max-samples 100 --seed 42
```

## 8. Cleanup

```shell
make kg-neo4j-down         # Stop Neo4j (PrimeKG pipeline only)
make kg-clean              # Remove PrimeKG data + Neo4j volumes
# RadLex pipeline has no Docker services to clean up
```
