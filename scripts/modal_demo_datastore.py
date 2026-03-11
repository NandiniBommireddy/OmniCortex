from pathlib import Path
import os

import modal


APP_NAME = "kg-llava-demo-datastore"
VOLUME_NAME = "kg-llava-demo-datastore"

ROOT = Path("/Users/apple/Desktop/GenAI-Project/CS7180-OmniCortex")
LOCAL_SCRIPT = ROOT / "scripts" / "build_demo_datastore.py"
LOCAL_INPUT = ROOT / "tmp" / "demo" / "mimic-nle-train-demo50-radgraph.json"
LOCAL_METADATA = ROOT / "physionet.org" / "files" / "mimic-cxr-jpg" / "2.0.0" / "mimic-cxr-2.0.0-metadata.csv.gz"
LOCAL_SPLIT = ROOT / "physionet.org" / "files" / "mimic-cxr-jpg" / "2.0.0" / "mimic-cxr-2.0.0-split.csv.gz"
LOCAL_IMAGES = ROOT / "physionet.org" / "2.0.0" / "files"
LOCAL_OUTPUT_DIR = ROOT / "tmp" / "demo" / "datastore_modal"

REMOTE_ROOT = "/data"
REMOTE_SCRIPT = f"{REMOTE_ROOT}/scripts/build_demo_datastore.py"
REMOTE_INPUT = f"{REMOTE_ROOT}/inputs/mimic-nle-train-demo50-radgraph.json"
REMOTE_METADATA = f"{REMOTE_ROOT}/inputs/mimic-cxr-2.0.0-metadata.csv.gz"
REMOTE_SPLIT = f"{REMOTE_ROOT}/inputs/mimic-cxr-2.0.0-split.csv.gz"
REMOTE_IMAGES = f"{REMOTE_ROOT}/images"
REMOTE_OUTPUT_DIR = f"{REMOTE_ROOT}/outputs"


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "transformers==4.24.0",
        "numpy==1.26.4",
        "pillow==10.4.0",
        "pandas==2.2.2",
        "faiss-cpu==1.8.0.post1",
        "medclip @ git+https://github.com/RyanWangZf/MedCLIP.git",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _write_bytes(local_path: Path, data: bytes) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as handle:
        handle.write(data)


def _volume_relative(remote_path: str) -> str:
    prefix = f"{REMOTE_ROOT}/"
    if remote_path.startswith(prefix):
        return remote_path[len(prefix):]
    return remote_path


@app.function(volumes={REMOTE_ROOT: volume}, timeout=60 * 30, cpu=8)
def build_demo_datastore_remote() -> list[str]:
    import subprocess

    cmd = [
        "python",
        REMOTE_SCRIPT,
        "--input",
        REMOTE_INPUT,
        "--image-root",
        REMOTE_IMAGES,
        "--metadata-csv-gz",
        REMOTE_METADATA,
        "--split-csv-gz",
        REMOTE_SPLIT,
        "--output-dir",
        REMOTE_OUTPUT_DIR,
    ]
    subprocess.run(cmd, check=True)
    volume.commit()
    return [
        f"{REMOTE_OUTPUT_DIR}/kg_nle_index",
        f"{REMOTE_OUTPUT_DIR}/kg_nle_index_captions.json",
        f"{REMOTE_OUTPUT_DIR}/retrieved_triplets.json",
    ]


@app.function(volumes={REMOTE_ROOT: volume}, timeout=60 * 5, cpu=1)
def fetch_output_bytes(remote_paths: list[str]) -> dict[str, bytes]:
    out = {}
    for remote_path in remote_paths:
        name = os.path.basename(remote_path)
        out[name] = b"".join(volume.read_file(_volume_relative(remote_path)))
    return out


@app.local_entrypoint()
def main():
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(LOCAL_SCRIPT), "scripts/build_demo_datastore.py")
        batch.put_file(str(LOCAL_INPUT), "inputs/mimic-nle-train-demo50-radgraph.json")
        batch.put_file(str(LOCAL_METADATA), "inputs/mimic-cxr-2.0.0-metadata.csv.gz")
        batch.put_file(str(LOCAL_SPLIT), "inputs/mimic-cxr-2.0.0-split.csv.gz")
        batch.put_directory(str(LOCAL_IMAGES), "images")

    remote_paths = build_demo_datastore_remote.remote()
    outputs = {
        "kg_nle_index": LOCAL_OUTPUT_DIR / "kg_nle_index",
        "kg_nle_index_captions.json": LOCAL_OUTPUT_DIR / "kg_nle_index_captions.json",
        "retrieved_triplets.json": LOCAL_OUTPUT_DIR / "retrieved_triplets.json",
    }

    for name, data in fetch_output_bytes.remote(remote_paths).items():
        target = outputs[name]
        _write_bytes(target, data)
        print(f"downloaded {name} -> {target}")
