git submodule add https://github.com/Stanford-AIMI/radgraph.git radgraph

clean all .venv folders

pyenv install 3.11.12
pyenv local 3.11.12

<!-- install radgraph first -->
python3.11 -m venv .venv-radgraph
source .venv-radgraph/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Pin HF stack to RadGraph-compatible versions
python -m pip install \
  "torch==2.2.2" \
  "transformers==4.44.2" \
  "tokenizers==0.19.1" \
  "huggingface_hub==0.24.6"

# Install RadGraph from your local submodule/folder
python -m pip install -e ./radgraph


<!-- install  -->
make venv
make install

Install MedCLIP
python -m pip install git+https://github.com/RyanWangZf/MedCLIP.git

python -m pip install faiss-gpu
or
python -m pip install faiss-cpu



<!-- Image -->
Download the following from mimic-cxr-jpg
- IMAGE_FILENAMES
- mimic-cxr-2.0.0-metadata.csv.gz
- mimic-cxr-2.0.0-split.csv.gz

To donwload images
```shell
cd physionet.org

head -n 507 mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES | wget -r -N -c -np -nH --cut-dirs=1 --user minh160302 --ask-password -i - --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```

<!-- Run -->
Extract RadGraph triplets. Expected output includes: MIMIC-NLE/mimic-nle/mimic-nle-dev.json
```shell
.venv/bin/python scripts/extract_radgraph_triplets.py \
--input MIMIC-NLE/mimic-nle/mimic-nle-train.json \
--output tmp/demo/mimic-nle-train-radgraph.json \
--triplets-json tmp/demo/dev-triplets-map.json \
--model-type modern-radgraph-xl \
--batch-size 4
```


Build datastore artifacts (local)
Expected outputs:
- tmp/demo/datastore/retrieved_triplets.json
- tmp/demo/datastore/kg_nle_index
- tmp/demo/datastore/kg_nle_index_captions.json
```shell
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
.venv/bin/python scripts/build_demo_datastore.py \
--input tmp/demo/mimic-nle-train-radgraph.json \
--image-root physionet.org/mimic-cxr-jpg/2.1.0/files \
--metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
--split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
--output-dir tmp/demo/datastore
```



Build LLaVA dataset JSON. Expected outputs: tmp/demo/mimic-nle-dev-kg-llava.json
```shell
.venv/bin/python scripts/build_demo_llava_json.py \
--input tmp/demo/mimic-nle-train-radgraph.json \
--retrieved-triplets tmp/demo/datastore/retrieved_triplets.json \
--image-root physionet.org/mimic-cxr-jpg/2.1.0/files \
--metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
--split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
--output tmp/demo/mimic-nle-train-kg-llava.json
```


To train with modal:

```
modal run scripts/modal_demo_train_llava.py
```

To eval
```
modal run scripts/modal_demo_eval_llava.py
```
