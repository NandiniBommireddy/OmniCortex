from pathlib import Path
import os

import modal


APP_NAME = "kg-llava-demo-train"
VOLUME_NAME = "kg-llava-demo-train"

ROOT = Path(__file__).resolve().parent.parent
LOCAL_LLAVA = ROOT / "models" / "LLaVA"

GCS_BUCKET = "mimic-cxr-jpg-2.1.0.physionet.org"
GCS_PREFIX = "files/"
GCS_SECRET_NAME = "gcs-mimic-cxr"

REMOTE_ROOT = "/workspace"
REMOTE_LLAVA = f"{REMOTE_ROOT}/LLaVA"
REMOTE_IMAGES = f"{REMOTE_ROOT}/images"


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
        "deepspeed==0.12.6",
        "ninja",
        "google-cloud-storage",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


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
    import json
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
    timeout=60 * 60 * 12,
    gpu="A100-40GB"
)
def run_demo_train(remote_data: str, remote_out: str, model: str = "liuhaotian/llava-v1.5-7b", version: str = "v1") -> list[str]:
    import subprocess

    _setup_gcs_credentials()

    _download_images_from_gcs(remote_data, REMOTE_IMAGES, GCS_BUCKET, GCS_PREFIX)

    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_LLAVA
    env["HF_HOME"] = f"{REMOTE_ROOT}/hf_cache"

    cmd = [
        "python",
        f"{REMOTE_LLAVA}/llava/train/train.py",
        "--lora_enable", "True",
        "--lora_r", "64",
        "--lora_alpha", "128",
        "--mm_projector_lr", "2e-5",
        "--model_name_or_path", model,
        "--version", version,
        "--data_path", remote_data,
        "--image_folder", REMOTE_IMAGES,
        "--vision_tower", "openai/clip-vit-large-patch14-336",
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--group_by_modality_length", "True",
        "--bf16", "True",
        "--output_dir", remote_out,
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "100",
        "--save_total_limit", "1",
        "--learning_rate", "2e-4",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "100",
        "--tf32", "True",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "2",
        "--lazy_preprocess", "True",
        "--report_to", "none",
    ]
    subprocess.run(cmd, check=True, env=env)
    volume.commit()
    return [remote_out]


@app.function(volumes={REMOTE_ROOT: volume}, timeout=60 * 5)
def fetch_train_outputs(remote_paths: list[str]) -> dict[str, bytes]:
    out = {}
    for remote_path in remote_paths:
        rel = _volume_relative(remote_path)
        for name in ["trainer_state.json", "config.json"]:
            candidate = f"{rel}/{name}"
            try:
                out[name] = b"".join(volume.read_file(candidate))
            except FileNotFoundError:
                pass
    return out


@app.local_entrypoint()
def main(variant: str = "radlex", model: str = "liuhaotian/llava-v1.5-7b", version: str = "v1"):
    """
    Run training for a given data variant and base model.

    Examples:
        modal run scripts/modal_demo_train_llava.py
        modal run scripts/modal_demo_train_llava.py --variant multihop
        modal run scripts/modal_demo_train_llava.py --model liuhaotian/llava-v1.6-vicuna-13b
        modal run scripts/modal_demo_train_llava.py --variant ""   # base (no suffix)
    """
    model_tag = model.split("/")[-1]
    suffix = f"-{variant}" if variant else ""
    local_data = ROOT / "tmp" / "demo" / f"mimic-nle-train-kg-llava{suffix}.json"
    local_output_dir = ROOT / "tmp" / "demo" / f"llava_modal_train{suffix.replace('-', '_')}_{model_tag}"
    remote_data = f"{REMOTE_ROOT}/data/mimic-nle-train-kg-llava{suffix}.json"
    remote_out = f"{REMOTE_ROOT}/outputs{suffix}_{model_tag}"

    local_output_dir.mkdir(parents=True, exist_ok=True)

    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_LLAVA), "LLaVA")
        batch.put_file(str(local_data), _volume_relative(remote_data))

    remote_paths = run_demo_train.remote(remote_data, remote_out, model=model, version=version)
    fetched = fetch_train_outputs.remote(remote_paths)
    for name, data in fetched.items():
        target = local_output_dir / name
        _write_bytes(target, data)
        print(f"downloaded {name} -> {target}")
