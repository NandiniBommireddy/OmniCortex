from pathlib import Path
import os

import modal


APP_NAME = "kg-llava-demo-train"
VOLUME_NAME = "kg-llava-demo-train"

ROOT = Path(__file__).resolve().parent.parent
LOCAL_LLAVA = ROOT / "models" / "LLaVA"
LOCAL_DATA = ROOT / "tmp" / "demo" / "mimic-nle-train-kg-llava.json"
LOCAL_IMAGES = ROOT / "physionet.org" / "mimic-cxr-jpg" / "2.1.0" / "files"
LOCAL_OUTPUT_DIR = ROOT / "tmp" / "demo" / "llava_modal_train"

REMOTE_ROOT = "/workspace"
REMOTE_LLAVA = f"{REMOTE_ROOT}/LLaVA"
REMOTE_DATA = f"{REMOTE_ROOT}/data/mimic-nle-train-kg-llava.json"
REMOTE_IMAGES = f"{REMOTE_ROOT}/images"
REMOTE_OUT = f"{REMOTE_ROOT}/outputs"


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


@app.function(volumes={REMOTE_ROOT: volume}, timeout=60 * 60 * 4, gpu="A10G")
def run_demo_train() -> list[str]:
    import subprocess

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
        "--model_name_or_path", "liuhaotian/llava-v1.5-7b",
        "--version", "v1",
        "--data_path", REMOTE_DATA,
        "--image_folder", REMOTE_IMAGES,
        "--vision_tower", "openai/clip-vit-large-patch14-336",
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--group_by_modality_length", "True",
        "--bf16", "True",
        "--output_dir", REMOTE_OUT,
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
        "--logging_steps", "1",
        "--tf32", "True",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "2",
        "--lazy_preprocess", "True",
        "--report_to", "none",
    ]
    subprocess.run(cmd, check=True, env=env)
    volume.commit()
    return [REMOTE_OUT]


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
def main():
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_LLAVA), "LLaVA")
        batch.put_file(str(LOCAL_DATA), "data/mimic-nle-train-kg-llava.json")
        batch.put_directory(str(LOCAL_IMAGES), "images")

    remote_paths = run_demo_train.remote()
    fetched = fetch_train_outputs.remote(remote_paths)
    for name, data in fetched.items():
        target = LOCAL_OUTPUT_DIR / name
        _write_bytes(target, data)
        print(f"downloaded {name} -> {target}")
