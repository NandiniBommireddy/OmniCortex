# OmniCortex — KG-LLaVA Reproducibility

## Project Overview

Reproduces **KG-LLaVA** from "LLaVA Needs More Knowledge: Retrieval Augmented Natural Language Generation with Knowledge Graph for Explaining Thoracic Pathologies" (AAAI 2025, Kyung Hee University).

**Architecture:**
- Vision encoder: MedCLIP ViT (512-dim embeddings)
- KG source: RadGraph `suggestive_of` triplets from MIMIC-CXR reports
- Retrieval: FAISS IndexFlatIP, k=7 neighbors (cross-modal: image query → text-embedded captions)
- LLM: LLaVA-1.5-7B with LoRA fine-tuning (lora_r=64, lora_alpha=128)
- Dataset: MIMIC-NLE (38,003 NLEs, 10 diagnoses, 3 certainty levels)
- **Target metrics:** AUC 83.0, BLEU-4 7.2, CIDEr 62.2, ROUGE-L 25.0

---

## Known Bugs (Fix Before Running)

1. **Image preprocessing mismatch** (`datastore_retrieval.py`): Uses broken `MedCLIPProcessor` for image encoding at query time; `build_demo_datastore.py` uses a custom `_MEDCLIP_IMG_TRANSFORM` torchvision pipeline. Embeddings are in different spaces → broken retrieval. Fix: copy the same `_MEDCLIP_IMG_TRANSFORM` into `datastore_retrieval.py`.

2. **Eval runs on training data** (`scripts/modal_demo_eval_llava.py` line 13): `LOCAL_DATA` points to `mimic-nle-train-kg-llava.json`. Should point to the dev/test split JSON.

3. **Triplets extracted from NLE sentences, not full reports** (`scripts/extract_radgraph_triplets.py`): Paper builds KG from full MIMIC-CXR reports; current code processes single NLE sentences → sparse triplets. ~50% of eval entries have empty triplet context.

4. **No metric computation**: Eval script only generates `demo_answers.jsonl`; does not compute BLEU, ROUGE, CIDEr, or AUC.

---

## Multi-Hop Reasoning Implementation Plan

Extends single-hop RadGraph triplets with multi-hop chains via **PrimeKG** (open-source precision medicine KG, ~4M edges, UMLS-derived nodes).

**Chain structure:** `finding → diagnosis → mechanism → clinical implication`
- Hop 1: RadGraph (`suggestive_of`) — already implemented
- Hop 2: PrimeKG `disease_phenotype`, `disease_anatomy` edges
- Hop 3: PrimeKG `phenotype_phenotype`, `anatomy_anatomy` edges

**Entity alignment:** RadGraph entity text → scispaCy UMLS linker → CUI → PrimeKG node name

### Artifact chain
```
PrimeKG subgraph CSV
    → entity_cui_map.json
    → radgraph-multihop.json  (triplets + chains)
    → kg_nle_index_multihop   (FAISS, uses chain text for caption embeddings)
    → mimic-nle-train-kg-llava-multihop.json  (2-section prompt)
    → Modal training (modified train script)
    → evaluation with chain-quality metrics
```

---

### Step 1 — Build PrimeKG subgraph (`scripts/build_primekg_subgraph.py`)

**Purpose:** Download PrimeKG and extract a radiology-relevant subgraph (diseases, phenotypes, anatomy nodes reachable within 2 hops from the 10 MIMIC-NLE diagnoses).

**Inputs:**
- PrimeKG `kg.csv` (download from Harvard Dataverse)

**Outputs:**
- `data/primekg_radiology_subgraph.csv` — filtered edges
- `data/primekg_node_index.json` — `{node_name: node_idx, ...}`

**Key logic:**
```python
SEED_DIAGNOSES = [
    "atelectasis", "consolidation", "edema", "cardiomegaly",
    "lung lesion", "lung opacity", "pleural effusion",
    "pleural other", "pneumonia", "pneumothorax",
]
ALLOWED_EDGE_TYPES = {
    "disease_phenotype_positive", "disease_phenotype_negative",
    "disease_anatomy", "phenotype_phenotype", "anatomy_anatomy",
}
# BFS up to depth=2 from seed disease nodes
```

---

### Step 2 — Build entity→CUI map (`scripts/build_entity_cui_map.py`)

**Purpose:** Map every unique RadGraph entity text to a PrimeKG node name via UMLS CUI.

**Inputs:**
- `data/radgraph-enriched.jsonl` (RadGraph extraction output)
- `data/primekg_node_index.json`

**Outputs:**
- `data/entity_cui_map.json` — `{entity_text: primekg_node_name | null}`

**Three-stage alignment:**
1. Exact string match (lowercase) against PrimeKG node names
2. scispaCy `en_core_sci_lg` + UMLS entity linker → top CUI → lookup in PrimeKG
3. `rapidfuzz.process.extractOne` (score_cutoff=85) against PrimeKG node names

**Dependencies:** `pip install scispacy en_core_sci_lg rapidfuzz`

---

### Step 3 — Build multi-hop chains (`scripts/build_multihop_chains.py`)

**Purpose:** For each image's RadGraph triplets, traverse PrimeKG to build 2-hop reasoning chains.

**Inputs:**
- `data/radgraph-enriched.jsonl`
- `data/primekg_radiology_subgraph.csv`
- `data/entity_cui_map.json`

**Outputs:**
- `data/radgraph-multihop.jsonl` — each row adds a `"chains"` field alongside existing `"triplets"`

**Chain format (string):**
```
"opacity --suggestive_of--> consolidation --disease_phenotype--> fever --phenotype_phenotype--> cough"
```

**Key logic:**
```python
def build_chains(triplets, entity_map, graph, max_hops=2):
    chains = []
    for subj, rel, obj in triplets:
        node1 = entity_map.get(obj)   # map diagnosis entity
        if node1 is None:
            continue
        for neighbor1, edge1 in graph.neighbors(node1):
            chain = f"{subj} --{rel}--> {obj} --{edge1}--> {neighbor1}"
            if max_hops >= 2:
                for neighbor2, edge2 in graph.neighbors(neighbor1):
                    chains.append(chain + f" --{edge2}--> {neighbor2}")
            else:
                chains.append(chain)
    return chains
```

---

### Step 4 — Modify datastore builder (`scripts/build_demo_datastore.py`)

**Changes:** Add `--multihop` flag. When set, caption for each image = `triplets + chains` (semicolon-joined). All other logic unchanged.

```python
parser.add_argument("--multihop", action="store_true")
# ...
if args.multihop:
    caption = "; ".join(r["triplets"] + r.get("chains", []))
else:
    caption = "; ".join(r["triplets"])
```

**Output:** `data/kg_nle_index_multihop` (FAISS), `data/retrieved_triplets_multihop.json`

---

### Step 5 — Modify LLaVA JSON builder (`scripts/build_demo_llava_json.py`)

**Changes:** Add `--chains-file` argument and `--multihop` flag. When set, use 2-section prompt.

**New prompt template:**
```
<image>
The image-specific triplets from the knowledge graph are: {kg_triplets}.
The multi-hop reasoning chains are: {kg_chains}.
And for the given image, {question}
```

**Key change in `main()`:**
```python
parser.add_argument("--chains-file", default=None)
parser.add_argument("--multihop", action="store_true")
# ...
chains_by_image = json.load(open(args.chains_file)) if args.chains_file else {}
# ...
kg_chains = "; ".join(chains_by_image.get(dicom_id, []))
if args.multihop and kg_chains:
    human_value = (
        "<image>\n"
        f"The image-specific triplets from the knowledge graph are: {kg_triplets}. "
        f"The multi-hop reasoning chains are: {kg_chains}. "
        f"And for the given image, {question}"
    )
```

**Output:** `tmp/demo/mimic-nle-train-kg-llava-multihop.json`

---

### Step 6 — Modify training script (`scripts/modal_demo_train_llava.py`)

**Changes (3 lines):**
```python
# Line 13 — change data path:
LOCAL_DATA = ROOT / "tmp" / "demo" / "mimic-nle-train-kg-llava-multihop.json"
# Line 18 — change remote data path:
REMOTE_DATA = f"{REMOTE_ROOT}/data/mimic-nle-train-kg-llava-multihop.json"
# Line 21 — change output model name:
REMOTE_TRAIN_OUT = f"{REMOTE_ROOT}/outputs-multihop"
```

---

### Step 7 — Fix eval + add metrics (`scripts/modal_demo_eval_llava.py` + `scripts/eval_multihop_quality.py`)

**Fix eval bug** in `modal_demo_eval_llava.py`:
```python
# Before (wrong — evaluates on training data):
LOCAL_DATA = ROOT / "tmp" / "demo" / "mimic-nle-train-kg-llava.json"
# After (correct — use dev split):
LOCAL_DATA = ROOT / "tmp" / "demo" / "mimic-nle-dev-kg-llava-multihop.json"
```

**New script `scripts/eval_multihop_quality.py`:**

Inputs:
- `demo_answers.jsonl` (model outputs)
- `mimic-nle-dev-kg-llava-multihop.json` (ground truth NLEs)

Metrics computed:
- **BLEU-1/2/4** via `nltk.translate.bleu_score`
- **ROUGE-L** via `rouge_score`
- **CIDEr** via `pycocoevalcap` or manual TF-IDF cosine
- **Chain coverage**: % of answers mentioning ≥1 entity from their multi-hop chain
- **Hop depth**: average number of hops referenced in the generated answer

---

## Running Order (Multi-hop)

```bash
# 1. Build PrimeKG subgraph
python scripts/build_primekg_subgraph.py \
    --primekg-csv data/kg.csv \
    --output-csv data/primekg_radiology_subgraph.csv \
    --output-index data/primekg_node_index.json

# 2. Build entity alignment map
python scripts/build_entity_cui_map.py \
    --input data/radgraph-enriched.jsonl \
    --node-index data/primekg_node_index.json \
    --output data/entity_cui_map.json

# 3. Build multi-hop chains
python scripts/build_multihop_chains.py \
    --input data/radgraph-enriched.jsonl \
    --subgraph data/primekg_radiology_subgraph.csv \
    --entity-map data/entity_cui_map.json \
    --output data/radgraph-multihop.jsonl

# 4. Build multihop datastore
python scripts/build_demo_datastore.py \
    --input data/radgraph-multihop.jsonl \
    --image-root physionet.org/mimic-cxr-jpg/2.1.0/files \
    --metadata-csv-gz data/mimic-cxr-2.0.0-metadata.csv.gz \
    --split-csv-gz data/mimic-cxr-2.0.0-split.csv.gz \
    --output-dir tmp/demo/datastore_multihop \
    --multihop

# 5. Build LLaVA JSON (train split)
python scripts/build_demo_llava_json.py \
    --input data/radgraph-multihop.jsonl \
    --retrieved-triplets tmp/demo/datastore_multihop/retrieved_triplets_multihop.json \
    --chains-file data/radgraph-multihop.jsonl \
    --image-root physionet.org/mimic-cxr-jpg/2.1.0/files \
    --metadata-csv-gz data/mimic-cxr-2.0.0-metadata.csv.gz \
    --split-csv-gz data/mimic-cxr-2.0.0-split.csv.gz \
    --output tmp/demo/mimic-nle-train-kg-llava-multihop.json \
    --multihop

# 6. Train on Modal
modal run scripts/modal_demo_train_llava.py

# 7. Eval on Modal
modal run scripts/modal_demo_eval_llava.py

# 8. Compute metrics
python scripts/eval_multihop_quality.py \
    --answers tmp/demo/llava_modal_eval/demo_answers.jsonl \
    --references tmp/demo/mimic-nle-dev-kg-llava-multihop.json \
    --output tmp/demo/eval_results.json
```

---

## Dependencies (additional for multi-hop)

```
scispacy
en_core_sci_lg  # python -m spacy download en_core_sci_lg
rapidfuzz
networkx
nltk
rouge_score
pycocoevalcap
```
