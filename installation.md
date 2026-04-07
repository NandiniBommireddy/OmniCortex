git submodule add https://github.com/Stanford-AIMI/radgraph.git radgraph

clean all .venv folders

pyenv install 3.11.12
pyenv local 3.11.12

<!-- install radgraph first -->
make install-radgraph
make check-radgraph
make freeze-radgraph


<!-- install  -->
deactivate
make venv
make install

Install MedCLIP
python -m pip install git+https://github.com/RyanWangZf/MedCLIP.git

python -m pip install faiss-gpu
or
python -m pip install faiss-cpu

python -m pip install google-cloud-storage


<!-- Image data — GCS bucket -->
Images are stored in GCS bucket: `gs://mimic-cxr-jpg-2.1.0.physionet.org/files/`

Authenticate with GCS:
```shell
gcloud auth application-default login
```

Download metadata CSVs (still needed locally):
- mimic-cxr-2.0.0-metadata.csv.gz
- mimic-cxr-2.0.0-split.csv.gz
- cxr-study-list.csv.gz

For Modal, create a secret from your gcloud credentials:
```shell
modal secret create gcs-mimic-cxr \
  GOOGLE_CREDENTIALS="$(cat ~/.config/gcloud/application_default_credentials.json)" \
  GOOGLE_PROJECT="$(gcloud config get-value project)"
```

To update the secret (e.g. after refreshing gcloud credentials), delete and recreate:
```shell
modal secret delete gcs-mimic-cxr
modal secret create gcs-mimic-cxr \
  GOOGLE_CREDENTIALS="$(cat ~/.config/gcloud/application_default_credentials.json)" \
  GOOGLE_PROJECT="$(gcloud config get-value project)"
```


<!-- Run -->
Extract RadGraph triplets. Expected output includes:
- tmp/demo/mimic-nle-train/dev/test-radgraph.json

```shell
# .venv-radgraph/bin/python scripts/extract_radgraph_triplets.py \
# --input MIMIC-NLE/mimic-nle/mimic-nle-train.json \
# --output tmp/demo/mimic-nle-train-radgraph.json \
# --triplets-json tmp/demo/train-triplets-map.json \
# --model-type modern-radgraph-xl \
# --batch-size 4

.venv-radgraph/bin/python scripts/extract_radgraph_triplets.py \
--input-dir MIMIC-NLE/mimic-nle \
--output-dir tmp/demo \
--model-type modern-radgraph-xl \
--batch-size 4 \
--num-workers 4 \
--reports-root physionet.org/mimic-cxr/2.1.0/files
```


Build datastore artifacts (local)
Expected outputs:
- tmp/demo/datastore/retrieved_triplets.json
- tmp/demo/datastore/kg_nle_index
- tmp/demo/datastore/kg_nle_index_captions.json

on local MacOS, this takes ~2 hours
```shell
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
.venv/bin/python scripts/build_demo_datastore.py \
--input tmp/demo/mimic-nle-(train|test)-radgraph.json \
--image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
--metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
--split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
--output-dir tmp/demo/datastore
```



Build LLaVA dataset JSON. Expected outputs: tmp/demo/mimic-nle-(dev|test|train)-kg-llava.json
```shell
.venv/bin/python scripts/build_demo_llava_json.py \
--input tmp/demo/mimic-nle-(train|test)-radgraph.json \
--retrieved-triplets tmp/demo/(datastore|datastore_test)/retrieved_triplets.json \
--image-root gs://mimic-cxr-jpg-2.1.0.physionet.org/files \
--metadata-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
--split-csv-gz physionet.org/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
--output tmp/demo/mimic-nle-(train|test)-kg-llava.json
```


To train with modal:

```
modal run scripts/modal_demo_train_llava.py
```

To eval
```
modal run scripts/modal_demo_eval_llava.py
```
