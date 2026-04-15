import argparse
import json
from pathlib import Path


def strip_image_prefix(text: str) -> str:
    prefix = "<image>\n"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def extract_question(prompt: str) -> str:
    marker = "And for the given image, "
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def load_data_index(data_path: Path):
    rows = json.loads(data_path.read_text())
    index = {}
    for idx, row in enumerate(rows):
        human = strip_image_prefix(row["conversations"][0]["value"])
        reference = row["conversations"][1]["value"]
        key = {
            "image": row.get("image", ""),
            "question": extract_question(human),
            "reference": reference,
        }
        index[idx] = {
            "key": key,
            "id": row.get("id", ""),
            "prompt": human,
        }
    return index


def load_answers_by_qid(path: Path):
    answers = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            answers[row["question_id"]] = row
    return answers


def main():
    parser = argparse.ArgumentParser(
        description="Align base/radlex/primekg eval outputs by stable sample keys (no retraining needed)."
    )
    parser.add_argument("--base-data", required=True)
    parser.add_argument("--radlex-data", required=True)
    parser.add_argument("--primekg-data", required=True)
    parser.add_argument("--base-answers", required=True)
    parser.add_argument("--radlex-answers", required=True)
    parser.add_argument("--primekg-answers", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = {
        "base": load_data_index(Path(args.base_data)),
        "radlex": load_data_index(Path(args.radlex_data)),
        "primekg": load_data_index(Path(args.primekg_data)),
    }
    answers = {
        "base": load_answers_by_qid(Path(args.base_answers)),
        "radlex": load_answers_by_qid(Path(args.radlex_answers)),
        "primekg": load_answers_by_qid(Path(args.primekg_answers)),
    }

    by_key = {}
    for variant in ("base", "radlex", "primekg"):
        for qid, sample in data[variant].items():
            key = (
                sample["key"]["image"],
                sample["key"]["question"],
                sample["key"]["reference"],
            )
            entry = by_key.setdefault(
                key,
                {
                    "image": sample["key"]["image"],
                    "question": sample["key"]["question"],
                    "reference": sample["key"]["reference"],
                    "variants": {},
                },
            )
            ans = answers[variant].get(qid)
            entry["variants"][variant] = {
                "question_id": qid,
                "id": sample["id"],
                "prompt": ans.get("prompt") if ans else None,
                "text": ans.get("text") if ans else None,
                "answer_id": ans.get("answer_id") if ans else None,
            }

    aligned = []
    for _, row in by_key.items():
        variants = row["variants"]
        row["is_complete"] = all(v in variants for v in ("base", "radlex", "primekg"))
        row["texts_match"] = (
            row["is_complete"]
            and variants["base"]["text"] == variants["radlex"]["text"] == variants["primekg"]["text"]
        )
        aligned.append(row)

    aligned.sort(key=lambda x: (x["image"], x["question"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in aligned:
            f.write(json.dumps(row) + "\n")

    total = len(aligned)
    complete = sum(1 for x in aligned if x["is_complete"])
    text_match = sum(1 for x in aligned if x["texts_match"])
    print(f"wrote {total} aligned rows to {output_path}")
    print(f"complete rows (all 3 variants): {complete}")
    print(f"complete rows with identical generated text: {text_match}")


if __name__ == "__main__":
    main()
