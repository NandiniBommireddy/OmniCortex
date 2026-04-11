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

## 5. PrimeKG Chain Pipeline

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
.venv/bin/python scripts/build_primekg_chains.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --entity-map data/entity_cui_map.json \
  --output data/radgraph-primekg.jsonl

.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --retrieved-triplets tmp/demo/datastore/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --chains-file data/radgraph-primekg.jsonl \
  --multihop \
  --output tmp/demo/mimic-nle-train-kg-llava-primekg.json
```

### 5d. Build chains + LLaVA JSON (test)

```shell
.venv/bin/python scripts/build_primekg_chains.py \
  --input tmp/demo/mimic-nle-test-radgraph.json \
  --entity-map data/entity_cui_map.json \
  --output data/radgraph-primekg-test.jsonl

.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-test-radgraph.json \
  --retrieved-triplets tmp/demo/datastore_test/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --chains-file data/radgraph-primekg-test.jsonl \
  --multihop \
  --output tmp/demo/mimic-nle-test-kg-llava-primekg.json
```

## 6. RadLex Reasoning Chain Pipeline

Uses RadLex `May_Cause` edges to build 1-hop reasoning chains, injected as a separate
block in the prompt alongside the retrieved FAISS triplets. No Docker or Neo4j required.

### 6a. Install dependencies

```shell
.venv/bin/pip install owlready2 rapidfuzz
```

### 6b. Download ontologies

Register for a free BioPortal API key at https://bioportal.bioontology.org/accounts/new

```shell
mkdir -p kg/data/radlex kg/data/gamuts
.venv/bin/python scripts/download_ontologies.py --api-key YOUR_BIOPORTAL_KEY
```

Output: `kg/data/radlex/radlex.owl`, `kg/data/gamuts/gamuts.owl`

Verify content:

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

### 6d. Build reasoning chains

Output: `data/radgraph-multihop-radlex.jsonl`, `data/radgraph-multihop-radlex-test.jsonl`

```shell
# Train
.venv/bin/python scripts/build_radlex_chains.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --entity-map data/entity_radlex_map.json \
  --radlex-owl kg/data/radlex/radlex.owl \
  --output data/radgraph-multihop-radlex.jsonl

# Test
.venv/bin/python scripts/build_radlex_chains.py \
  --input tmp/demo/mimic-nle-test-radgraph.json \
  --entity-map data/entity_radlex_map.json \
  --radlex-owl kg/data/radlex/radlex.owl \
  --output data/radgraph-multihop-radlex-test.jsonl
```

### 6e. Build LLaVA JSON

Output: `tmp/demo/mimic-nle-{train,test}-kg-llava-radlex.json`

```shell
# Train
.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-train-radgraph.json \
  --retrieved-triplets tmp/demo/datastore/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --chains-file data/radgraph-multihop-radlex.jsonl \
  --multihop \
  --output tmp/demo/mimic-nle-train-kg-llava-radlex.json

# Test
.venv/bin/python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-test-radgraph.json \
  --retrieved-triplets tmp/demo/datastore_test/retrieved_triplets.json \
  --image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
  --metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
  --chains-file data/radgraph-multihop-radlex-test.jsonl \
  --multihop \
  --output tmp/demo/mimic-nle-test-kg-llava-radlex.json
```

## 7. Train & Eval on Modal

Both Modal scripts accept `--variant` and `--model` CLI flags. Outputs are written to
`tmp/demo/llava_modal_{train,eval}_{variant}_{model}/` locally and
`outputs-{variant}_{model}/` on the Modal volume.

For reference, training 1 epoch on full MIMIC-CXR-JPG took 10h16m on Modal A10G.

Both scripts accept `--version` (train) and `--conv-mode` (eval) for non-default model families:

| Model family                                 | `--version`        | `--conv-mode`      |
| -------------------------------------------- | ------------------ | ------------------ |
| LLaVA-1.5 / LLaVA-1.6 Vicuna (default)       | `v1`               | `llava_v1`         |
| Mistral-based (LLaVA-Med, LLaVA-1.6-Mistral) | `mistral_instruct` | `mistral_instruct` |

Example:

```shell
modal run --detach scripts/modal_demo_train_llava.py --variant radlex --model microsoft/llava-med-v1.5-mistral-7b --version mistral_instruct
modal run --detach scripts/modal_demo_eval_llava.py --variant radlex --model microsoft/llava-med-v1.5-mistral-7b --conv-mode mistral_instruct
```

Use `--detach` to run in the background. Before retraining, wipe the old output dir on the volume:

```shell
modal volume rm kg-llava-demo-train outputs-radlex_llava-v1.6-vicuna-7b
modal volume rm kg-llava-demo-train outputs-radlex_llava-v1.6-vicuna-13b
```

### 7a. Train — all 12 combinations (4 models × 3 variants)

For `microsoft/llava-med-v1.5-mistral-7b`, add `--version mistral_instruct`.

```shell
# liuhaotian/llava-v1.5-7b
modal run --detach scripts/modal_demo_train_llava.py --variant "" --model liuhaotian/llava-v1.5-7b
modal run --detach scripts/modal_demo_train_llava.py --variant radlex --model liuhaotian/llava-v1.5-7b
modal run --detach scripts/modal_demo_train_llava.py --variant primekg --model liuhaotian/llava-v1.5-7b

# liuhaotian/llava-v1.6-vicuna-7b
modal run --detach scripts/modal_demo_train_llava.py --variant "" --model liuhaotian/llava-v1.6-vicuna-7b
modal run --detach scripts/modal_demo_train_llava.py --variant radlex --model liuhaotian/llava-v1.6-vicuna-7b
modal run --detach scripts/modal_demo_train_llava.py --variant primekg --model liuhaotian/llava-v1.6-vicuna-7b

# liuhaotian/llava-v1.6-vicuna-13b
modal run --detach scripts/modal_demo_train_llava.py --variant "" --model liuhaotian/llava-v1.6-vicuna-13b
modal run --detach scripts/modal_demo_train_llava.py --variant radlex --model liuhaotian/llava-v1.6-vicuna-13b
modal run --detach scripts/modal_demo_train_llava.py --variant primekg --model liuhaotian/llava-v1.6-vicuna-13b

# microsoft/llava-med-v1.5-mistral-7b
modal run --detach scripts/modal_demo_train_llava.py --variant "" --model microsoft/llava-med-v1.5-mistral-7b --version mistral_instruct
modal run --detach scripts/modal_demo_train_llava.py --variant radlex --model microsoft/llava-med-v1.5-mistral-7b --version mistral_instruct
modal run --detach scripts/modal_demo_train_llava.py --variant primekg --model microsoft/llava-med-v1.5-mistral-7b --version mistral_instruct
```

### 7b. Eval — all 12 combinations

```shell
# liuhaotian/llava-v1.5-7b
modal run --detach scripts/modal_demo_eval_llava.py --variant "" --model liuhaotian/llava-v1.5-7b
modal run --detach scripts/modal_demo_eval_llava.py --variant radlex --model liuhaotian/llava-v1.5-7b
modal run --detach scripts/modal_demo_eval_llava.py --variant primekg --model liuhaotian/llava-v1.5-7b

# liuhaotian/llava-v1.6-vicuna-7b
modal run --detach scripts/modal_demo_eval_llava.py --variant "" --model liuhaotian/llava-v1.6-vicuna-7b
modal run --detach scripts/modal_demo_eval_llava.py --variant radlex --model liuhaotian/llava-v1.6-vicuna-7b
modal run --detach scripts/modal_demo_eval_llava.py --variant primekg --model liuhaotian/llava-v1.6-vicuna-7b

# liuhaotian/llava-v1.6-vicuna-13b
modal run --detach scripts/modal_demo_eval_llava.py --variant "" --model liuhaotian/llava-v1.6-vicuna-13b
modal run --detach scripts/modal_demo_eval_llava.py --variant radlex --model liuhaotian/llava-v1.6-vicuna-13b
modal run --detach scripts/modal_demo_eval_llava.py --variant primekg --model liuhaotian/llava-v1.6-vicuna-13b

# microsoft/llava-med-v1.5-mistral-7b
modal run --detach scripts/modal_demo_eval_llava.py --variant "" --model microsoft/llava-med-v1.5-mistral-7b
modal run --detach scripts/modal_demo_eval_llava.py --variant radlex --model microsoft/llava-med-v1.5-mistral-7b
modal run --detach scripts/modal_demo_eval_llava.py --variant primekg --model microsoft/llava-med-v1.5-mistral-7b
```

### 7c. Compute metrics (after eval)

`--answers` and `--references` must use outputs from the **same test JSON** — `question_id` is the row index in the references file.

For baseline (`variant=""`), the references file has no suffix: `mimic-nle-test-kg-llava.json`.

**NLG + RadGraph metrics:**

```shell
# llava-v1.5-7b
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval__llava-v1.5-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava.json --output tmp/demo/eval_results__llava-v1.5-7b.json
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval_radlex_llava-v1.5-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-radlex.json --output tmp/demo/eval_results_radlex_llava-v1.5-7b.json
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval_primekg_llava-v1.5-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-primekg.json --output tmp/demo/eval_results_primekg_llava-v1.5-7b.json

# llava-v1.6-vicuna-7b
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval__llava-v1.6-vicuna-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava.json --output tmp/demo/eval_results__llava-v1.6-vicuna-7b.json
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval_radlex_llava-v1.6-vicuna-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-radlex.json --output tmp/demo/eval_results_radlex_llava-v1.6-vicuna-7b.json
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval_primekg_llava-v1.6-vicuna-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-primekg.json --output tmp/demo/eval_results_primekg_llava-v1.6-vicuna-7b.json

# llava-v1.6-vicuna-13b
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval__llava-v1.6-vicuna-13b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava.json --output tmp/demo/eval_results__llava-v1.6-vicuna-13b.json
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval_radlex_llava-v1.6-vicuna-13b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-radlex.json --output tmp/demo/eval_results_radlex_llava-v1.6-vicuna-13b.json
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval_primekg_llava-v1.6-vicuna-13b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-primekg.json --output tmp/demo/eval_results_primekg_llava-v1.6-vicuna-13b.json

# llava-med-v1.5-mistral-7b
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval__llava-med-v1.5-mistral-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava.json --output tmp/demo/eval_results__llava-med-v1.5-mistral-7b.json
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval_radlex_llava-med-v1.5-mistral-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-radlex.json --output tmp/demo/eval_results_radlex_llava-med-v1.5-mistral-7b.json
.venv/bin/python scripts/metrics.py --answers tmp/demo/llava_modal_eval_primekg_llava-med-v1.5-mistral-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-primekg.json --output tmp/demo/eval_results_primekg_llava-med-v1.5-mistral-7b.json
```

**LLM-as-judge (100 samples, Claude Haiku):**

```shell
# llava-v1.5-7b
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval__llava-v1.5-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava.json --output tmp/demo/eval_results_llm__llava-v1.5-7b.json --max-samples 100 --seed 42
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval_radlex_llava-v1.5-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-radlex.json --output tmp/demo/eval_results_llm_radlex_llava-v1.5-7b.json --max-samples 100 --seed 42
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval_primekg_llava-v1.5-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-primekg.json --output tmp/demo/eval_results_llm_primekg_llava-v1.5-7b.json --max-samples 100 --seed 42

# llava-v1.6-vicuna-7b
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval__llava-v1.6-vicuna-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava.json --output tmp/demo/eval_results_llm__llava-v1.6-vicuna-7b.json --max-samples 100 --seed 42
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval_radlex_llava-v1.6-vicuna-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-radlex.json --output tmp/demo/eval_results_llm_radlex_llava-v1.6-vicuna-7b.json --max-samples 100 --seed 42
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval_primekg_llava-v1.6-vicuna-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-primekg.json --output tmp/demo/eval_results_llm_primekg_llava-v1.6-vicuna-7b.json --max-samples 100 --seed 42

# llava-v1.6-vicuna-13b
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval__llava-v1.6-vicuna-13b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava.json --output tmp/demo/eval_results_llm__llava-v1.6-vicuna-13b.json --max-samples 100 --seed 42
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval_radlex_llava-v1.6-vicuna-13b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-radlex.json --output tmp/demo/eval_results_llm_radlex_llava-v1.6-vicuna-13b.json --max-samples 100 --seed 42
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval_primekg_llava-v1.6-vicuna-13b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-primekg.json --output tmp/demo/eval_results_llm_primekg_llava-v1.6-vicuna-13b.json --max-samples 100 --seed 42

# llava-med-v1.5-mistral-7b
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval__llava-med-v1.5-mistral-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava.json --output tmp/demo/eval_results_llm__llava-med-v1.5-mistral-7b.json --max-samples 100 --seed 42
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval_radlex_llava-med-v1.5-mistral-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-radlex.json --output tmp/demo/eval_results_llm_radlex_llava-med-v1.5-mistral-7b.json --max-samples 100 --seed 42
.venv/bin/python scripts/metrics_llm.py --answers tmp/demo/llava_modal_eval_primekg_llava-med-v1.5-mistral-7b/demo_answers.jsonl --references tmp/demo/mimic-nle-test-kg-llava-primekg.json --output tmp/demo/eval_results_llm_primekg_llava-med-v1.5-mistral-7b.json --max-samples 100 --seed 42
```

**RadGraph F1 (requires .venv-radgraph):**

```shell
# Pattern: replace {variant} and {model} accordingly
.venv-radgraph/bin/python scripts/compute_radgraph_f1.py \
  --answers tmp/demo/llava_modal_eval_{variant}_{model}/demo_answers.jsonl \
  --references tmp/demo/mimic-nle-test-kg-llava-{variant}.json \
  --output tmp/demo/radgraph_f1_{variant}_{model}.json
```

## 8. Cleanup

```shell
make kg-neo4j-down         # Stop Neo4j (PrimeKG pipeline only)
make kg-clean              # Remove PrimeKG data + Neo4j volumes
# RadLex pipeline has no Docker services to clean up
```
