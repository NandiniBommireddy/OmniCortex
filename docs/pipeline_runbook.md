# KG-LLaVA Demo Pipeline Runbook

This document explains the exact script flow used in this repository for the
`demo50` end-to-end run:

1. extract MIMIC-NLE
2. run RadGraph triplet extraction
3. prepare a local image subset
4. build the MedCLIP + FAISS datastore on Modal
5. build the final LLaVA-format JSON
6. train on Modal
7. evaluate on Modal

This runbook exists because the root `README.md` still references
`dataset_preparation.py`, but that file does not exist in this repo. The actual
logic is split across the scripts listed below.

## What Each Script Does

### `scripts/extract_radgraph_triplets.py`

Purpose:
- reads extracted MIMIC-NLE JSONL
- runs RadGraph on the `nle` field
- keeps only `suggestive_of` relations
- writes back triplet-enriched JSONL

Inputs:
- extracted `mimic-nle-*.json`

Outputs:
- triplet-enriched JSONL
- optional sentence-to-triplets JSON map

### `scripts/build_demo_datastore.py`

Purpose:
- joins the RadGraph output to image metadata
- aggregates image-level triplets
- encodes captions and images with MedCLIP
- builds the FAISS datastore

Inputs:
- RadGraph-enriched JSON
- local image root
- `mimic-cxr-2.0.0-metadata.csv.gz`
- `mimic-cxr-2.0.0-split.csv.gz`

Outputs:
- `kg_nle_index`
- `kg_nle_index_captions.json`
- `retrieved_triplets.json`

Notes:
- this is the local implementation
- for the actual demo run, `modal_demo_datastore.py` was used instead

### `scripts/build_demo_llava_json.py`

Purpose:
- combines image paths, retrieved triplets, labels, and target NLE text
- writes the final LLaVA conversation-format dataset

Input:
- RadGraph-enriched JSON
- `retrieved_triplets.json`
- image metadata + split CSVs

Output:
- `mimic-nle-demo50-kg-llava.json`

### `scripts/modal_demo_datastore.py`

Purpose:
- uploads the local demo image subset, metadata, and build script to a Modal
  volume
- runs the datastore build remotely on Modal
- persists outputs in the Modal volume
- downloads the final artifacts back locally

Important:
- this script does **not** download the full raw PhysioNet dataset
- it works on the local subset you already prepared and uploaded
- it is a remote wrapper around `build_demo_datastore.py`

### `scripts/modal_demo_train_llava.py`

Purpose:
- uploads the local LLaVA code, final JSON, and local image subset to Modal
- runs a 1-epoch LoRA fine-tuning job remotely
- downloads training metadata back locally

Outputs:
- `trainer_state.json`
- `config.json`

### `scripts/modal_demo_eval_llava.py`

Purpose:
- loads the trained Modal output
- builds evaluation questions from the final demo JSON
- runs the LLaVA eval pipeline remotely
- downloads `demo_answers.jsonl` back locally

Output:
- `demo_answers.jsonl`

## Images: Local First, Then Uploaded

For the demo run, images were **not** downloaded with Modal.

Actual flow:
- the required JPG subset was downloaded locally from PhysioNet
- the Modal scripts uploaded that local subset into a Modal volume
- datastore build, training, and evaluation then ran remotely using that subset

This was intentional because downloading the full `MIMIC-CXR-JPG` dataset to a
laptop is not practical, and the demo only needed a small controlled subset.

## Environments

### MIMIC-NLE extraction
Use the `mimicnle` conda environment.

### RadGraph and local helper scripts
Use the `kgllava` conda environment.

### Modal runs
Use any shell where:
- `modal` is installed
- Modal authentication has already been completed

The Modal scripts define their own remote package dependencies.

## Exact Demo50 Command Order

### 1. Extract MIMIC-NLE

```bash
conda activate mimicnle
cd /Users/apple/Desktop/GenAI-Project/CS7180-OmniCortex/MIMIC-NLE
python extract_mimic_nle.py --reports_path "/Users/apple/Desktop/GenAI-Project/CS7180-OmniCortex/physionet.org/files 2"
```

Outputs:
- `MIMIC-NLE/mimic-nle/mimic-nle-train.json`
- `MIMIC-NLE/mimic-nle/mimic-nle-dev.json`
- `MIMIC-NLE/mimic-nle/mimic-nle-test.json`

### 2. Run RadGraph Triplet Extraction

```bash
conda activate kgllava
cd /Users/apple/Desktop/GenAI-Project/CS7180-OmniCortex
python scripts/extract_radgraph_triplets.py \
  --input MIMIC-NLE/mimic-nle/mimic-nle-train.json \
  --output tmp/demo/mimic-nle-train-demo50-radgraph.json \
  --triplets-json tmp/demo/mimic-nle-train-demo50-triplets.json \
  --model-type modern-radgraph-xl \
  --batch-size 4 \
  --limit 50
```

Outputs:
- `tmp/demo/mimic-nle-train-demo50-radgraph.json`
- `tmp/demo/mimic-nle-train-demo50-triplets.json`

### 3. Generate JPG URL List

```bash
python - <<'PY'
import csv, gzip, json
from pathlib import Path

demo = Path("tmp/demo/mimic-nle-train-demo50-radgraph.json")
meta = Path("physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz")
out = Path("tmp/demo/demo50_jpg_urls.txt")

wanted = set()
with open(demo) as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            wanted.add((row["patient_ID"][1:], row["report_ID"][1:]))

urls = []
with gzip.open(meta, "rt") as f:
    reader = csv.DictReader(f)
    for rec in reader:
        key = (rec["subject_id"], rec["study_id"])
        if key in wanted:
            subject = rec["subject_id"]
            study = rec["study_id"]
            dicom = rec["dicom_id"]
            urls.append(
                "https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
                f"p{subject[:2]}/p{subject}/s{study}/{dicom}.jpg"
            )

urls = sorted(set(urls))
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for url in urls:
        f.write(url + "\n")

print(out)
print("jpg urls:", len(urls))
PY
```

### 4. Download the Required JPGs Locally

```bash
wget -nc -c -x -nH --cut-dirs=2 --user <YOUR_PHYSIONET_USERNAME> --ask-password \
  -i tmp/demo/demo50_jpg_urls.txt \
  -P physionet.org
```

Important:
- this is local download
- Modal does not automatically fetch the full raw dataset for you

### 5. Build the Datastore on Modal

```bash
modal run scripts/modal_demo_datastore.py
```

Local outputs:
- `tmp/demo/datastore_modal/kg_nle_index`
- `tmp/demo/datastore_modal/kg_nle_index_captions.json`
- `tmp/demo/datastore_modal/retrieved_triplets.json`

### 6. Build the Final LLaVA JSON

```bash
python scripts/build_demo_llava_json.py \
  --input tmp/demo/mimic-nle-train-demo50-radgraph.json \
  --retrieved-triplets tmp/demo/datastore_modal/retrieved_triplets.json \
  --image-root physionet.org/2.0.0/files \
  --metadata-csv-gz physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --split-csv-gz physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz \
  --output tmp/demo/mimic-nle-demo50-kg-llava.json
```

Output:
- `tmp/demo/mimic-nle-demo50-kg-llava.json`

### 7. Train on Modal

```bash
modal run scripts/modal_demo_train_llava.py
```

Local outputs:
- `tmp/demo/llava_modal_train/trainer_state.json`
- `tmp/demo/llava_modal_train/config.json`

### 8. Evaluate on Modal

```bash
modal run scripts/modal_demo_eval_llava.py
```

Local output:
- `tmp/demo/llava_modal_eval/demo_answers.jsonl`

## Storage Guidance for Larger Runs

Do **not** use a laptop as the long-term storage location for the full
`MIMIC-CXR-JPG` dataset.

Recommended scale-up path:
- keep the full raw dataset in a lab server, external SSD, or cloud bucket
- use that as the source of truth
- use Modal for compute-heavy stages:
  - datastore build
  - training
  - evaluation

Short version:
- storage and compute should be separated
- Modal is helpful for compute, not as a replacement for a proper raw-data
  storage strategy
