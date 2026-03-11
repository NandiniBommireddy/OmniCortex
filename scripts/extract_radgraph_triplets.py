import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RADGRAPH_ROOT = ROOT / "radgraph"
if str(RADGRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(RADGRAPH_ROOT))

from radgraph import RadGraph, get_radgraph_processed_annotations  # noqa: E402


def read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


def unique_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def extract_triplets(processed):
    triplets = []
    for annotation in processed["processed_annotations"]:
        suggestive = annotation.get("suggestive_of") or []
        triplets.extend(suggestive)
    return unique_preserve_order(triplets)


def batched(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input extracted MIMIC-NLE JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file with triplets field")
    parser.add_argument(
        "--triplets-json",
        help="Optional output JSON mapping sentence_ID to suggestive_of triplets",
    )
    parser.add_argument(
        "--model-type",
        default="modern-radgraph-xl",
        choices=["radgraph", "radgraph-xl", "modern-radgraph-xl"],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--model-cache-dir",
        default=str(ROOT / "tmp" / "radgraph_cache"),
        help="Writable cache directory for RadGraph model files",
    )
    parser.add_argument(
        "--tokenizer-cache-dir",
        default=str(ROOT / "tmp" / "hf_cache"),
        help="Writable cache directory for tokenizer files",
    )
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    texts = [row.get("nle", "") for row in rows]

    Path(args.model_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tokenizer_cache_dir).mkdir(parents=True, exist_ok=True)

    model = RadGraph(
        model_type=args.model_type,
        batch_size=args.batch_size,
        model_cache_dir=args.model_cache_dir,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
    )

    output_rows = []
    triplet_map = {}

    processed_count = 0
    for row_batch, text_batch in zip(batched(rows, args.batch_size), batched(texts, args.batch_size)):
        annotations = model(text_batch)
        for idx, row in enumerate(row_batch):
            processed = get_radgraph_processed_annotations({"0": annotations[str(idx)]})
            triplets = extract_triplets(processed)
            updated = dict(row)
            updated["triplets"] = triplets
            output_rows.append(updated)
            triplet_map[row["sentence_ID"]] = triplets
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"processed {processed_count}/{len(rows)}")

    write_jsonl(args.output, output_rows)
    if args.triplets_json:
        with open(args.triplets_json, "w") as handle:
            json.dump(triplet_map, handle, indent=2)

    print(f"wrote {len(output_rows)} rows to {args.output}")
    if args.triplets_json:
        print(f"wrote triplet map to {args.triplets_json}")


if __name__ == "__main__":
    main()
