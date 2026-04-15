#!/usr/bin/env python3
"""LLM-as-judge evaluation for multihop KG-LLaVA outputs via OpenRouter.

Metrics: clinical accuracy, completeness, reasoning quality, overall score (1-5 each).

Usage:
    python scripts/metrics_llm.py \
        --answers experiments/demo_answers_one_hop.jsonl \
        --references tmp/demo/mimic-nle-test-kg-llava-multihop.json \
        --output tmp/demo/eval_results_llm.json

    # With X-ray images from GCS (requires GOOGLE_APPLICATION_CREDENTIALS):
    python scripts/metrics_llm.py \
        --answers experiments/demo_answers_one_hop.jsonl \
        --references tmp/demo/mimic-nle-test-kg-llava-multihop.json \
        --output tmp/demo/eval_results_llm.json \
        --with-image

Requires: OPENROUTER_API_KEY environment variable.
"""

import argparse
import base64
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

DEFAULT_MODEL = "anthropic/claude-haiku-4-5"

GCS_BUCKET = "mimic-cxr-jpg-2.1.0.physionet.org"
GCS_PREFIX = "files/"

JUDGE_PROMPT = """\
You are evaluating a medical AI model's chest X-ray explanation against a reference explanation written by a radiologist.

**Reference explanation:**
{reference}

**Model explanation:**
{hypothesis}

Rate the model explanation on these 4 dimensions (1-5 scale each):

1. **Clinical Accuracy** — Are the medical findings factually correct? Does it avoid hallucinating conditions not present?
2. **Completeness** — Does it cover the key findings mentioned in the reference?
3. **Reasoning Quality** — Is the clinical reasoning sound? Are causal links (e.g., opacity → pneumonia) appropriate?
4. **Language Quality** — Is it clear, concise, and uses appropriate medical terminology?

Respond with ONLY a JSON object, no other text:
{{"clinical_accuracy": <int>, "completeness": <int>, "reasoning_quality": <int>, "language_quality": <int>}}"""

JUDGE_PROMPT_IMAGE = """\
You are evaluating a medical AI model's chest X-ray explanation. The actual X-ray image is provided above.

**Reference explanation (by a radiologist):**
{reference}

**Model explanation:**
{hypothesis}

Rate the model explanation on these 4 dimensions (1-5 scale each):

1. **Clinical Accuracy** — Are the findings consistent with what is visible in the X-ray? Does it avoid hallucinating conditions not present in the image?
2. **Completeness** — Does it cover the key findings visible in the image and mentioned in the reference?
3. **Reasoning Quality** — Is the clinical reasoning sound? Are causal links (e.g., opacity → pneumonia) appropriate given the image?
4. **Language Quality** — Is it clear, concise, and uses appropriate medical terminology?

Respond with ONLY a JSON object, no other text:
{{"clinical_accuracy": <int>, "completeness": <int>, "reasoning_quality": <int>, "language_quality": <int>}}"""


def load_answers(path: Path) -> dict[int, str]:
    answers = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                answers[row["question_id"]] = row["text"]
    return answers


def load_references(path: Path) -> tuple[list[str], list[str]]:
    data = json.load(open(path))
    texts = [row["conversations"][1]["value"] for row in data]
    image_paths = [row["image"] for row in data]
    return texts, image_paths


def fetch_image_b64(bucket, rel_path: str, images_dir: Path) -> str:
    """Return base64-encoded image, downloading from GCS to disk cache if needed."""
    local = images_dir / rel_path
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(GCS_PREFIX + rel_path).download_to_filename(str(local))
    return base64.b64encode(local.read_bytes()).decode("utf-8")


def extract_json(text: str) -> dict:
    """Extract JSON object from text that may contain markdown or extra content."""
    import re
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try finding first { ... }
    m = re.search(r"\{[^{}]+\}", text)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("No JSON found", text, 0)


REQUIRED_DIMS = ["clinical_accuracy", "completeness", "reasoning_quality", "language_quality"]


def judge_single(client: OpenAI, model: str, hypothesis: str, reference: str, image_b64: str | None = None, retries: int = 2) -> dict:
    if image_b64:
        prompt = JUDGE_PROMPT_IMAGE.format(reference=reference, hypothesis=hypothesis)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]
    else:
        prompt = JUDGE_PROMPT.format(reference=reference, hypothesis=hypothesis)
        content = prompt
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": content}],
        )
        text = resp.choices[0].message.content.strip()
        scores = extract_json(text)
        for d in REQUIRED_DIMS:
            if d not in scores or not isinstance(scores[d], (int, float)):
                if attempt < retries:
                    break
                raise KeyError(d)
        else:
            return scores
    return scores


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation")
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("tmp/demo/eval_results_llm.json"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenRouter judge model name")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--with-image", action="store_true", help="Include X-ray image in judge prompt (downloads from GCS, cached locally)")
    parser.add_argument("--images-dir", type=Path, default=Path("tmp/demo/images"), help="Local cache directory for GCS images")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    gcs_bucket = None
    if args.with_image:
        from google.cloud import storage
        gcs_bucket = storage.Client().bucket(GCS_BUCKET)
        args.images_dir.mkdir(parents=True, exist_ok=True)
        print(f"GCS image cache enabled (bucket: {GCS_BUCKET}, local: {args.images_dir})")

    references, image_paths = load_references(args.references)
    answers = load_answers(args.answers)
    hypotheses = [answers.get(i, "") for i in range(len(references))]

    if args.max_samples and args.max_samples < len(hypotheses):
        random.seed(args.seed)
        indices = sorted(random.sample(range(len(hypotheses)), args.max_samples))
        hypotheses = [hypotheses[i] for i in indices]
        references = [references[i] for i in indices]
        image_paths = [image_paths[i] for i in indices]
        print(f"Sampled {len(indices)} indices (seed={args.seed}): {indices[:10]}{'...' if len(indices) > 10 else ''}")

    num_samples = len(hypotheses)
    print(f"Evaluating {num_samples} samples with {args.model} (images={'yes' if args.with_image else 'no'})...")

    dims = ["clinical_accuracy", "completeness", "reasoning_quality", "language_quality"]
    totals = {d: 0.0 for d in dims}
    scored = 0
    per_sample = []

    for i, (hyp, ref, img_path) in enumerate(zip(hypotheses, references, image_paths)):
        if not hyp:
            per_sample.append(None)
            continue
        try:
            image_b64 = fetch_image_b64(gcs_bucket, img_path, args.images_dir) if gcs_bucket else None
            scores = judge_single(client, args.model, hyp, ref, image_b64=image_b64)
            per_sample.append(scores)
            for d in dims:
                totals[d] += scores[d]
            scored += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{num_samples} done")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Sample {i}: parse error ({e}), skipping")
            per_sample.append(None)
        time.sleep(0.1)  # gentle rate limit

    averages = {d: round(totals[d] / scored, 2) if scored else 0.0 for d in dims}
    averages["overall"] = round(sum(averages[d] for d in dims) / len(dims), 2)

    results = {
        "averages": averages,
        "num_scored": scored,
        "num_samples": num_samples,
        "model": args.model,
        "with_image": args.with_image,
        "per_sample": per_sample,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults ({scored}/{num_samples} scored):")
    for k, v in averages.items():
        print(f"  {k}: {v}/5")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
