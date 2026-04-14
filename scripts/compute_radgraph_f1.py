"""Compute RadGraph F1 between model answers and reference NLEs.

Runs RadGraph on both hypothesis and reference texts, extracts clinical
entities, then computes precision / recall / F1 at the entity level.

Must be run with .venv-radgraph/bin/python (Python 3.11).

Usage:
    .venv-radgraph/bin/python scripts/compute_radgraph_f1.py \
        --answers tmp/demo/llava_modal_eval_radlex_llava-v1.6-vicuna-7b/demo_answers.jsonl \
        --references tmp/demo/mimic-nle-test-kg-llava-radlex.json \
        --output tmp/demo/radgraph_f1_radlex_llava-v1.6-vicuna-7b.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RADGRAPH_ROOT = ROOT / "radgraph"
if str(RADGRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(RADGRAPH_ROOT))

from radgraph import RadGraph, get_radgraph_processed_annotations  # noqa: E402


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


def extract_entities(triplets: list[str]) -> set[str]:
    """Extract entity tokens from triplet strings like 'opacity suggestive of pneumonia'."""
    entities = set()
    for t in triplets:
        t = t.strip().lower()
        if " suggestive of " in t:
            parts = t.split(" suggestive of ", 1)
            entities.add(parts[0].strip())
            entities.add(parts[1].strip())
        else:
            entities.add(t)
    return entities


def run_radgraph(texts: list[str], batch_size: int, model_cache_dir: str, tokenizer_cache_dir: str) -> list[list[str]]:
    """Run RadGraph on a list of texts, return list of triplet lists."""
    model = RadGraph(
        model_type="modern-radgraph-xl",
        batch_size=batch_size,
        model_cache_dir=model_cache_dir,
        tokenizer_cache_dir=tokenizer_cache_dir,
    )

    all_triplets = []
    batch = []
    indices = []

    def flush(batch, indices):
        annotations = model(batch)
        for local_idx, _ in enumerate(batch):
            processed = get_radgraph_processed_annotations(
                {"0": annotations[str(local_idx)]}
            )
            triplets = []
            for annotation in processed["processed_annotations"]:
                triplets.extend(annotation.get("suggestive_of") or [])
            all_triplets.append(triplets)

    for i, text in enumerate(texts):
        batch.append(text or "")
        indices.append(i)
        if len(batch) == batch_size:
            flush(batch, indices)
            batch = []
            indices = []
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(texts)}")
    if batch:
        flush(batch, indices)

    return all_triplets


def compute_f1(hyp_entities: set[str], ref_entities: set[str]) -> tuple[float, float, float]:
    if not hyp_entities and not ref_entities:
        return 1.0, 1.0, 1.0
    if not hyp_entities or not ref_entities:
        return 0.0, 0.0, 0.0
    tp = len(hyp_entities & ref_entities)
    precision = tp / len(hyp_entities)
    recall = tp / len(ref_entities)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-cache-dir", default=str(ROOT / "tmp" / "radgraph_cache"))
    parser.add_argument("--tokenizer-cache-dir", default=str(ROOT / "tmp" / "hf_cache"))
    args = parser.parse_args()

    print("Loading data...")
    answers = load_answers(args.answers)
    references = load_references(args.references)
    hypotheses = [answers.get(i, "") for i in range(len(references))]
    print(f"  {len(hypotheses)} samples, {sum(1 for h in hypotheses if h)} with answers")

    Path(args.model_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tokenizer_cache_dir).mkdir(parents=True, exist_ok=True)

    print("Running RadGraph on hypotheses...")
    hyp_triplets = run_radgraph(hypotheses, args.batch_size, args.model_cache_dir, args.tokenizer_cache_dir)

    print("Running RadGraph on references...")
    ref_triplets = run_radgraph(references, args.batch_size, args.model_cache_dir, args.tokenizer_cache_dir)

    print("Computing F1...")
    precisions, recalls, f1s = [], [], []
    for hyp_t, ref_t in zip(hyp_triplets, ref_triplets):
        hyp_e = extract_entities(hyp_t)
        ref_e = extract_entities(ref_t)
        p, r, f = compute_f1(hyp_e, ref_e)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    n = len(f1s)
    results = {
        "radgraph_precision": round(sum(precisions) / n * 100, 2),
        "radgraph_recall": round(sum(recalls) / n * 100, 2),
        "radgraph_f1": round(sum(f1s) / n * 100, 2),
        "num_samples": n,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
