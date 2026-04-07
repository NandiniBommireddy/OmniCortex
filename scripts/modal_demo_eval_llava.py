from pathlib import Path
import json
import os

import modal


APP_NAME = "kg-llava-demo-eval"
VOLUME_NAME = "kg-llava-demo-train"

ROOT = Path(__file__).resolve().parent.parent
LOCAL_LLAVA = ROOT / "models" / "LLaVA"

GCS_BUCKET = "mimic-cxr-jpg-2.1.0.physionet.org"
GCS_PREFIX = "files/"
GCS_SECRET_NAME = "gcs-mimic-cxr"

REMOTE_ROOT = "/workspace"
REMOTE_LLAVA = f"{REMOTE_ROOT}/LLaVA"
REMOTE_IMAGES = f"{REMOTE_ROOT}/images"

REMOTE_LORA_MODEL = f"{REMOTE_ROOT}/llava-lora-demo"
REMOTE_QFILE = f"{REMOTE_ROOT}/eval/demo_questions.jsonl"
REMOTE_ANS = f"{REMOTE_ROOT}/eval/demo_answers.jsonl"


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.37.2",
        "tokenizers==0.15.1",
        "sentencepiece==0.1.99",
        "protobuf==4.25.3",
        "accelerate==0.21.0",
        "peft==0.7.1",
        "bitsandbytes",
        "pydantic",
        "markdown2[all]",
        "numpy",
        "scikit-learn==1.2.2",
        "gradio==4.16.0",
        "gradio_client==0.8.1",
        "requests",
        "httpx==0.24.0",
        "uvicorn",
        "fastapi",
        "einops==0.6.1",
        "einops-exts==0.0.4",
        "timm==0.6.13",
        "shortuuid",
        "google-cloud-storage",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _setup_gcs_credentials():
    """Write GCS credentials from GOOGLE_CREDENTIALS env var to a file."""
    creds = os.environ.get("GOOGLE_CREDENTIALS")
    if not creds:
        return
    creds_path = "/tmp/gcs_credentials.json"
    with open(creds_path, "w") as f:
        f.write(creds)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path


def _download_images_from_gcs(data_path, dest_dir, bucket_name, prefix):
    """Download images referenced in the data JSON from GCS (parallel)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from google.cloud import storage

    data = json.load(open(data_path))
    rel_paths = sorted(set(row["image"] for row in data))

    # Pre-create all destination directories
    dirs = set(os.path.join(dest_dir, os.path.dirname(rp)) for rp in rel_paths)
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    to_download = [rp for rp in rel_paths if not os.path.exists(os.path.join(dest_dir, rp))]
    print(f"[GCS] {len(to_download)} to download ({len(rel_paths) - len(to_download)} cached)")
    if not to_download:
        return

    client = storage.Client(project="885253748539")
    bucket = client.bucket(bucket_name)
    lock = threading.Lock()
    done = [0]

    def download_one(rel_path):
        bucket.blob(prefix + rel_path).download_to_filename(os.path.join(dest_dir, rel_path))
        with lock:
            done[0] += 1
            if done[0] % 200 == 0:
                print(f"[GCS]   {done[0]}/{len(to_download)}")

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(download_one, rp) for rp in to_download]
        for f in as_completed(futures):
            f.result()

    print(f"[GCS] done — {len(to_download)} downloaded")


def _write_bytes(local_path: Path, data: bytes) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as handle:
        handle.write(data)


def _volume_relative(remote_path: str) -> str:
    prefix = f"{REMOTE_ROOT}/"
    if remote_path.startswith(prefix):
        return remote_path[len(prefix):]
    return remote_path


@app.function(
    volumes={REMOTE_ROOT: volume},
    secrets=[modal.Secret.from_name(GCS_SECRET_NAME)],
    timeout=60 * 60,
    gpu="A100",
)
def run_demo_eval(remote_data: str, remote_train_out: str, model: str = "liuhaotian/llava-v1.5-7b", conv_mode: str = "llava_v1") -> str:
    import subprocess
    import shutil

    _setup_gcs_credentials()

    _download_images_from_gcs(remote_data, REMOTE_IMAGES, GCS_BUCKET, GCS_PREFIX)

    os.makedirs(os.path.dirname(REMOTE_QFILE), exist_ok=True)
    if os.path.exists(REMOTE_LORA_MODEL):
        shutil.rmtree(REMOTE_LORA_MODEL)
    shutil.copytree(remote_train_out, REMOTE_LORA_MODEL)

    # Use latest checkpoint subdir if model weights are nested there
    import glob
    checkpoints = sorted(glob.glob(f"{REMOTE_LORA_MODEL}/checkpoint-*"))
    if checkpoints:
        model_path = checkpoints[-1]
        import torch as _torch
        # Checkpoint dirs don't have config.json — copy it from the base model
        ckpt_config = os.path.join(model_path, "config.json")
        if not os.path.exists(ckpt_config):
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=model, filename="config.json")
            shutil.copy2(downloaded, ckpt_config)
        # Checkpoint dirs don't have non_lora_trainables.bin — create empty stub
        # so builder.py doesn't try to fetch it from HF Hub using a local path
        non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
        if not os.path.exists(non_lora_path):
            _torch.save({}, non_lora_path)
    else:
        model_path = REMOTE_LORA_MODEL

    data = json.load(open(remote_data))
    with open(REMOTE_QFILE, "w") as handle:
        for idx, row in enumerate(data):
            prompt = row["conversations"][0]["value"]
            if prompt.startswith("<image>\n"):
                prompt = prompt[len("<image>\n"):]
            q = {
                "question_id": idx,
                "image": row["image"],
                "text": prompt,
            }
            handle.write(json.dumps(q) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_LLAVA
    env["HF_HOME"] = f"{REMOTE_ROOT}/hf_cache"

    cmd = [
        "python",
        "-m",
        "llava.eval.model_vqa_loader",
        "--model-path", model_path,
        "--model-base", model,
        "--question-file", REMOTE_QFILE,
        "--image-folder", REMOTE_IMAGES,
        "--answers-file", REMOTE_ANS,
        "--temperature", "0",
        "--conv-mode", conv_mode,
    ]

    subprocess.run(cmd, check=True, env=env, cwd=REMOTE_LLAVA)
    volume.commit()
    return REMOTE_ANS


@app.function(volumes={REMOTE_ROOT: volume}, timeout=60 * 5)
def fetch_eval_output(remote_path: str) -> bytes:
    return b"".join(volume.read_file(_volume_relative(remote_path)))


@app.local_entrypoint()
def main(variant: str = "radlex", model: str = "liuhaotian/llava-v1.5-7b", conv_mode: str = "llava_v1"):
    """
    Run evaluation for a given data variant and base model.

    Examples:
        modal run scripts/modal_demo_eval_llava.py
        modal run scripts/modal_demo_eval_llava.py --variant multihop
        modal run scripts/modal_demo_eval_llava.py --model liuhaotian/llava-v1.5-13b
        modal run scripts/modal_demo_eval_llava.py --variant ""   # base (no suffix)
    """
    model_tag = model.split("/")[-1]
    suffix = f"-{variant}" if variant else ""
    local_data = ROOT / "tmp" / "demo" / f"mimic-nle-test-kg-llava{suffix}.json"
    local_output_dir = ROOT / "tmp" / "demo" / f"llava_modal_eval{suffix.replace('-', '_')}_{model_tag}"
    remote_data = f"{REMOTE_ROOT}/data/mimic-nle-test-kg-llava{suffix}.json"
    remote_train_out = f"{REMOTE_ROOT}/outputs{suffix}_{model_tag}"

    local_output_dir.mkdir(parents=True, exist_ok=True)

    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_LLAVA), "LLaVA")
        batch.put_file(str(local_data), _volume_relative(remote_data))

    remote_path = run_demo_eval.remote(remote_data, remote_train_out, model=model, conv_mode=conv_mode)
    data = fetch_eval_output.remote(remote_path)
    target = local_output_dir / "demo_answers.jsonl"
    _write_bytes(target, data)
    print(f"downloaded demo_answers.jsonl -> {target}")
