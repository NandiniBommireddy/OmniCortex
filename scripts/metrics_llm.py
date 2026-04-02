#!/usr/bin/env python3
"""LLM-as-judge evaluation for multihop KG-LLaVA outputs using Claude Haiku.

Metrics: clinical accuracy, completeness, reasoning quality, overall score (1-5 each).

Usage:
    python scripts/metrics_llm.py \
        --answers experiments/demo_answers_one_hop.jsonl \
        --references tmp/demo/mimic-nle-test-kg-llava-multihop.json \
        --output tmp/demo/eval_results_llm.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import anthropic

MODEL = "claude-haiku-4-5-20251001"

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


def load_answers(path: Path) -> dict[int, str]:
    answers = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                answers[row["question_id"]] = row["text"]
    return answers


def load_references(path: Path) -> list[str]:
    data = json.load(open(path))
    return [row["conversations"][1]["value"] for row in data]


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


def judge_single(client: anthropic.Anthropic, hypothesis: str, reference: str, retries: int = 2) -> dict:
    prompt = JUDGE_PROMPT.format(reference=reference, hypothesis=hypothesis)
    for attempt in range(retries + 1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
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
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    references = load_references(args.references)
    answers = load_answers(args.answers)
    hypotheses = [answers.get(i, "") for i in range(len(references))]

    if args.max_samples and args.max_samples < len(hypotheses):
        random.seed(args.seed)
        indices = sorted(random.sample(range(len(hypotheses)), args.max_samples))
        hypotheses = [hypotheses[i] for i in indices]
        references = [references[i] for i in indices]
        print(f"Sampled {len(indices)} indices (seed={args.seed}): {indices[:10]}{'...' if len(indices) > 10 else ''}")

    num_samples = len(hypotheses)
    print(f"Evaluating {num_samples} samples with {MODEL}...")

    dims = ["clinical_accuracy", "completeness", "reasoning_quality", "language_quality"]
    totals = {d: 0.0 for d in dims}
    scored = 0
    per_sample = []

    for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
        if not hyp:
            per_sample.append(None)
            continue
        try:
            scores = judge_single(client, hyp, ref)
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
        "model": MODEL,
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
