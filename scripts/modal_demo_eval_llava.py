from pathlib import Path
import json
import os

import modal


APP_NAME = "kg-llava-demo-eval"
VOLUME_NAME = "kg-llava-demo-train"

ROOT = Path("/Users/apple/Desktop/GenAI-Project/CS7180-OmniCortex")
LOCAL_LLAVA = ROOT / "models" / "LLaVA"
LOCAL_DATA = ROOT / "tmp" / "demo" / "mimic-nle-demo50-kg-llava.json"
LOCAL_IMAGES = ROOT / "physionet.org" / "2.0.0" / "files"
LOCAL_OUTPUT_DIR = ROOT / "tmp" / "demo" / "llava_modal_eval"

REMOTE_ROOT = "/workspace"
REMOTE_LLAVA = f"{REMOTE_ROOT}/LLaVA"
REMOTE_DATA = f"{REMOTE_ROOT}/data/mimic-nle-demo50-kg-llava.json"
REMOTE_IMAGES = f"{REMOTE_ROOT}/images"
REMOTE_TRAIN_OUT = f"{REMOTE_ROOT}/outputs"
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
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _write_bytes(local_path: Path, data: bytes) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as handle:
        handle.write(data)


def _volume_relative(remote_path: str) -> str:
    prefix = f"{REMOTE_ROOT}/"
    if remote_path.startswith(prefix):
        return remote_path[len(prefix):]
    return remote_path


@app.function(volumes={REMOTE_ROOT: volume}, timeout=60 * 60, gpu="A10G")
def run_demo_eval() -> str:
    import subprocess
    import shutil

    os.makedirs(os.path.dirname(REMOTE_QFILE), exist_ok=True)
    if os.path.exists(REMOTE_LORA_MODEL):
        shutil.rmtree(REMOTE_LORA_MODEL)
    shutil.copytree(REMOTE_TRAIN_OUT, REMOTE_LORA_MODEL)

    data = json.load(open(REMOTE_DATA))
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
        "--model-path", REMOTE_LORA_MODEL,
        "--model-base", "liuhaotian/llava-v1.5-7b",
        "--question-file", REMOTE_QFILE,
        "--image-folder", REMOTE_IMAGES,
        "--answers-file", REMOTE_ANS,
        "--temperature", "0",
        "--conv-mode", "llava_v1",
    ]
    subprocess.run(cmd, check=True, env=env, cwd=REMOTE_LLAVA)
    volume.commit()
    return REMOTE_ANS


@app.function(volumes={REMOTE_ROOT: volume}, timeout=60 * 5)
def fetch_eval_output(remote_path: str) -> bytes:
    return b"".join(volume.read_file(_volume_relative(remote_path)))


@app.local_entrypoint()
def main():
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_LLAVA), "LLaVA")
        batch.put_file(str(LOCAL_DATA), "data/mimic-nle-demo50-kg-llava.json")
        batch.put_directory(str(LOCAL_IMAGES), "images")

    remote_path = run_demo_eval.remote()
    data = fetch_eval_output.remote(remote_path)
    target = LOCAL_OUTPUT_DIR / "demo_answers.jsonl"
    _write_bytes(target, data)
    print(f"downloaded demo_answers.jsonl -> {target}")
