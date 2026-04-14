"""Train Llama-3.2-11B-Vision with LoRA on MIMIC-NLE.

Requires HuggingFace token (gated model): modal secret create huggingface HF_TOKEN=hf_xxx
Same volume as LLaVA/PaliGemma runs.

Usage:
    modal run scripts/modal_demo_train_llama_vision.py --variant radlex
    modal run scripts/modal_demo_train_llama_vision.py --variant ""
    modal run scripts/modal_demo_train_llama_vision.py --variant primekg
"""

from pathlib import Path
import os
import json

import modal


APP_NAME = "kg-llama-vision-train"
VOLUME_NAME = "kg-llava-demo-train"

ROOT = Path(__file__).resolve().parent.parent

GCS_BUCKET = "mimic-cxr-jpg-2.1.0.physionet.org"
GCS_PREFIX = "files/"
GCS_SECRET_NAME = "gcs-mimic-cxr"
HF_SECRET_NAME = "huggingface"

REMOTE_ROOT = "/workspace"
REMOTE_IMAGES = f"{REMOTE_ROOT}/images"
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

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
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


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
    secrets=[
        modal.Secret.from_name(GCS_SECRET_NAME),
        modal.Secret.from_name(HF_SECRET_NAME),
    ],
    timeout=60 * 60 * 14,
    gpu="A100-80GB",
)
def run_llama_vision_train(remote_data: str, remote_out: str) -> list[str]:
    import torch
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from transformers import AutoProcessor, MllamaForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import TrainingArguments, Trainer

    _setup_gcs_credentials()
    _download_images_from_gcs(remote_data, REMOTE_IMAGES, GCS_BUCKET, GCS_PREFIX)

    os.environ["HF_HOME"] = f"{REMOTE_ROOT}/hf_cache"
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading processor and model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    base_model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    # Apply LoRA — target both language model and cross-attention layers
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "cross_attn_q_proj", "cross_attn_k_proj",
            "cross_attn_v_proj", "cross_attn_o_proj",
        ],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model_peft = get_peft_model(base_model, lora_config)
    model_peft.print_trainable_parameters()

    class NLEDataset(Dataset):
        def __init__(self, data_path, image_dir, processor):
            self.data = json.load(open(data_path))
            self.image_dir = image_dir
            self.processor = processor

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data[idx]
            image = Image.open(os.path.join(self.image_dir, row["image"])).convert("RGB")

            prompt = row["conversations"][0]["value"]
            if prompt.startswith("<image>\n"):
                prompt = prompt[len("<image>\n"):]
            answer = row["conversations"][1]["value"]

            # Build prompt-only and full sequences
            messages_prompt = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            messages_full = messages_prompt + [{"role": "assistant", "content": answer}]

            prompt_text = processor.apply_chat_template(
                messages_prompt, add_generation_prompt=True
            )
            full_text = processor.apply_chat_template(
                messages_full, add_generation_prompt=False
            )

            prompt_inputs = processor(
                images=image, text=prompt_text, return_tensors="pt",
                padding=False, truncation=True, max_length=1024
            )
            full_inputs = processor(
                images=image, text=full_text, return_tensors="pt",
                padding=False, truncation=True, max_length=1024
            )

            input_ids = full_inputs["input_ids"].squeeze(0)
            labels = input_ids.clone()

            # Mask prompt tokens — only compute loss on answer tokens
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[:prompt_len] = -100

            result = {k: v.squeeze(0) for k, v in full_inputs.items()}
            result["labels"] = labels
            return result

    dataset = NLEDataset(remote_data, REMOTE_IMAGES, processor)

    training_args = TrainingArguments(
        output_dir=remote_out,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        logging_steps=100,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model_peft,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=True)
    model_peft.save_pretrained(remote_out)
    processor.save_pretrained(remote_out)
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
def main(variant: str = "radlex"):
    """
    Train Llama-3.2-11B-Vision on MIMIC-NLE with KG augmentation.

    Examples:
        modal run scripts/modal_demo_train_llama_vision.py --variant radlex
        modal run scripts/modal_demo_train_llama_vision.py --variant primekg
        modal run scripts/modal_demo_train_llama_vision.py --variant ""
    """
    model_tag = MODEL_ID.split("/")[-1]
    suffix = f"-{variant}" if variant else ""
    local_data = ROOT / "tmp" / "demo" / f"mimic-nle-train-kg-llava{suffix}.json"
    local_output_dir = ROOT / "tmp" / "demo" / f"hf_modal_train{suffix.replace('-', '_')}_{model_tag}"
    remote_data = f"{REMOTE_ROOT}/data/mimic-nle-train-kg-llava{suffix}.json"
    remote_out = f"{REMOTE_ROOT}/outputs-hf{suffix}_{model_tag}"

    local_output_dir.mkdir(parents=True, exist_ok=True)

    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_data), _volume_relative(remote_data))

    remote_paths = run_llama_vision_train.remote(remote_data, remote_out)
    fetched = fetch_train_outputs.remote(remote_paths)
    for name, data in fetched.items():
        target = local_output_dir / name
        _write_bytes(target, data)
        print(f"downloaded {name} -> {target}")
