import argparse
import csv
import gzip
import json
from pathlib import Path


DIAGNOSIS_LIST = [
    "Atelectasis",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]

CERTAINTY_LIST = ["negative", "uncertain", "positive"]

QUESTION_TEMPLATE = "Which signs show that the patient has {pathologies}?"


def read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_metadata(metadata_csv_gz, split_csv_gz):
    image_by_study = {}
    with gzip.open(metadata_csv_gz, "rt") as handle:
        for row in csv.DictReader(handle):
            key = (row["subject_id"], row["study_id"])
            image_by_study.setdefault(key, []).append(row["dicom_id"])

    split_by_image = {}
    with gzip.open(split_csv_gz, "rt") as handle:
        for row in csv.DictReader(handle):
            split_by_image[(row["subject_id"], row["study_id"], row["dicom_id"])] = row["split"]

    return image_by_study, split_by_image


def get_pathologies(img_labels):
    labels = []
    for idx, diagnosis in enumerate(img_labels):
        if diagnosis[1]:
            labels.append(f"{CERTAINTY_LIST[1]} {DIAGNOSIS_LIST[idx]}")
        if diagnosis[2]:
            labels.append(f"{CERTAINTY_LIST[2]} {DIAGNOSIS_LIST[idx]}")
    return ", ".join(labels)


def choose_image(subject_id, study_id, image_root, image_by_study):
    for dicom_id in sorted(image_by_study.get((subject_id, study_id), [])):
        rel_path = f"p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        abs_path = image_root / rel_path
        if abs_path.exists():
            return dicom_id, rel_path
    return None, None


def unique_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def to_triplet_blocks(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        blocks = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                blocks.append(text)
        return blocks
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="RadGraph-enriched demo JSONL")
    parser.add_argument("--retrieved-triplets", required=True, help="Image-level retrieved triplets JSON")
    parser.add_argument("--image-root", required=True, help="Root containing JPG files")
    parser.add_argument("--metadata-csv-gz", required=True)
    parser.add_argument("--split-csv-gz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image_root = Path(args.image_root)
    image_by_study, split_by_image = load_metadata(args.metadata_csv_gz, args.split_csv_gz)
    retrieved_triplets = json.load(open(args.retrieved_triplets))

    output_rows = []
    seen = set()
    stats = {
        "rows_with_retrieved_triplets": 0,
        "rows_with_row_triplets_fallback": 0,
        "rows_without_triplets": 0,
    }

    for row in read_jsonl(args.input):
        subject_id = row["patient_ID"][1:]
        study_id = row["report_ID"][1:]
        dicom_id, rel_path = choose_image(subject_id, study_id, image_root, image_by_study)
        if dicom_id is None:
            continue

        key = (dicom_id, row["sentence_ID"])
        if key in seen:
            continue
        seen.add(key)

        split = split_by_image.get((subject_id, study_id, dicom_id), "train")
        pathologies = get_pathologies(row["img_labels"])
        question = QUESTION_TEMPLATE.format(pathologies=pathologies)
        retrieved_blocks = to_triplet_blocks(retrieved_triplets.get(dicom_id))
        row_blocks = to_triplet_blocks(row.get("triplets"))
        if retrieved_blocks:
            triplet_texts = retrieved_blocks
            stats["rows_with_retrieved_triplets"] += 1
        elif row_blocks:
            triplet_texts = row_blocks
            stats["rows_with_row_triplets_fallback"] += 1
        else:
            triplet_texts = []
            stats["rows_without_triplets"] += 1
        all_triplets = []
        for block in triplet_texts:
            for t in block.split(";"):
                t = t.strip()
                if t:
                    all_triplets.append(t)
        kg_triplets = "; ".join(unique_preserve_order(all_triplets))

        human_value = (
            "<image>\n"
            f"The image-specific triplets from the knowledge graph are: {kg_triplets}. "
            f"And for the given image, {question}"
        )

        output_row = {
            "id": dicom_id,
            "split": split,
            "image": rel_path,
            "conversations": [
                {
                    "from": "human",
                    "value": human_value,
                },
                {
                    "from": "gpt",
                    "value": row["nle"],
                },
            ],
        }
        output_rows.append(output_row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(output_rows, handle, indent=2)

    print(f"wrote {len(output_rows)} records to {output_path}")
    print(
        "triplet source stats: "
        f"retrieved_rows={stats['rows_with_retrieved_triplets']}, "
        f"row_fallback_rows={stats['rows_with_row_triplets_fallback']}, "
        f"empty_rows={stats['rows_without_triplets']}"
    )


if __name__ == "__main__":
    main()
