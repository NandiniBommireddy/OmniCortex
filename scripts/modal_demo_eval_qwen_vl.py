"""Evaluate Qwen2.5-VL-7B-Instruct fine-tuned with LoRA on MIMIC-NLE test set.

Outputs answers in the same JSONL format as modal_demo_eval_llava.py so
metrics.py, metrics_llm.py, and compute_radgraph_f1.py work unchanged.

Usage:
    modal run scripts/modal_demo_eval_qwen_vl.py --variant radlex
    modal run scripts/modal_demo_eval_qwen_vl.py --variant ""
    modal run scripts/modal_demo_eval_qwen_vl.py --variant primekg
"""

from pathlib import Path
import os
import json

import modal


APP_NAME = "kg-qwen-vl-eval"
VOLUME_NAME = "kg-llava-demo-train"

ROOT = Path(__file__).resolve().parent.parent

GCS_BUCKET = "mimic-cxr-jpg-2.1.0.physionet.org"
GCS_PREFIX = "files/"
GCS_SECRET_NAME = "gcs-mimic-cxr"

REMOTE_ROOT = "/workspace"
REMOTE_IMAGES = f"{REMOTE_ROOT}/images"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "transformers==4.49.0",
        "accelerate>=0.30.0",
        "peft>=0.11.0",
        "Pillow",
        "numpy",
        "google-cloud-storage",
        "qwen-vl-utils",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _setup_gcs_credentials():
    creds = os.environ.get("GOOGLE_CREDENTIALS")
    if not creds:
        return
    creds_path = "/tmp/gcs_credentials.json"
    with open(creds_path, "w") as f:
        f.write(creds)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path


def _download_images_from_gcs(data_path, dest_dir, bucket_name, prefix):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from google.cloud import storage

    data = json.load(open(data_path))
    rel_paths = sorted(set(row["image"] for row in data))
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
    return remote_path[len(prefix):] if remote_path.startswith(prefix) else remote_path


@app.function(
    volumes={REMOTE_ROOT: volume},
    secrets=[modal.Secret.from_name(GCS_SECRET_NAME)],
    timeout=60 * 60 * 3,
    gpu="A100",
)
def run_qwen_vl_eval(remote_data: str, remote_train_out: str, remote_ans: str) -> str:
    import torch
    torch.set_float32_matmul_precision('high')
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

    _setup_gcs_credentials()
    _download_images_from_gcs(remote_data, REMOTE_IMAGES, GCS_BUCKET, GCS_PREFIX)

    os.environ["HF_HOME"] = f"{REMOTE_ROOT}/hf_cache"

    print(f"Loading processor from adapter: {remote_train_out}")
    processor = AutoProcessor.from_pretrained(
        remote_train_out,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    print(f"Loading base model: {MODEL_ID}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, remote_train_out)
    model.eval()
    torch._dynamo.config.disable = True

    data = json.load(open(remote_data))
    os.makedirs(os.path.dirname(remote_ans), exist_ok=True)

    print(f"Running inference on {len(data)} samples...")
    with open(remote_ans, "w") as out_f:
        for idx, row in enumerate(data):
            image = Image.open(os.path.join(REMOTE_IMAGES, row["image"])).convert("RGB")

            prompt = row["conversations"][0]["value"]
            if prompt.startswith("<image>\n"):
                prompt = prompt[len("<image>\n"):]

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[0][input_len:]
            answer = processor.decode(generated, skip_special_tokens=True).strip()

            out_f.write(json.dumps({
                "question_id": idx,
                "prompt": prompt,
                "text": answer,
                "model_id": MODEL_ID.split("/")[-1],
                "metadata": {},
            }) + "\n")
            out_f.flush()

            if (idx + 1) % 10 == 0:
                print(f"  {idx + 1}/{len(data)}")

    volume.commit()
    return remote_ans


@app.function(volumes={REMOTE_ROOT: volume}, timeout=60 * 5)
def fetch_eval_output(remote_path: str) -> bytes:
    return b"".join(volume.read_file(_volume_relative(remote_path)))


@app.local_entrypoint()
def main(variant: str = "radlex"):
    """
    Evaluate Qwen2.5-VL-7B-Instruct fine-tuned on MIMIC-NLE.

    Examples:
        modal run scripts/modal_demo_eval_qwen_vl.py --variant radlex
        modal run scripts/modal_demo_eval_qwen_vl.py --variant primekg
        modal run scripts/modal_demo_eval_qwen_vl.py --variant ""
    """
    model_tag = MODEL_ID.split("/")[-1]
    suffix = f"-{variant}" if variant else ""
    local_data = ROOT / "tmp" / "demo" / f"mimic-nle-test-kg-llava{suffix}.json"
    local_output_dir = ROOT / "tmp" / "demo" / f"hf_modal_eval{suffix.replace('-', '_')}_{model_tag}"
    remote_data = f"{REMOTE_ROOT}/data/mimic-nle-test-kg-llava{suffix}.json"
    remote_train_out = f"{REMOTE_ROOT}/outputs-hf{suffix}_{model_tag}"
    remote_ans = f"{REMOTE_ROOT}/eval/qwen_vl_answers{suffix}.jsonl"

    local_output_dir.mkdir(parents=True, exist_ok=True)

    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_data), _volume_relative(remote_data))

    remote_path = run_qwen_vl_eval.remote(remote_data, remote_train_out, remote_ans)
    data = fetch_eval_output.remote(remote_path)
    target = local_output_dir / "demo_answers.jsonl"
    _write_bytes(target, data)
    print(f"downloaded demo_answers.jsonl -> {target}")
