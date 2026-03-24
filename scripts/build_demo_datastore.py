import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
import torchvision.transforms as T
from medclip import MedCLIPModel, MedCLIPProcessor, MedCLIPVisionModelViT
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_metadata(metadata_csv_gz, split_csv_gz):
    by_study = defaultdict(list)
    with gzip.open(metadata_csv_gz, "rt") as handle:
        for row in csv.DictReader(handle):
            by_study[(row["subject_id"], row["study_id"])].append(row["dicom_id"])

    split_map = {}
    with gzip.open(split_csv_gz, "rt") as handle:
        for row in csv.DictReader(handle):
            split_map[(row["subject_id"], row["study_id"], row["dicom_id"])] = row["split"]

    return by_study, split_map


def choose_dicom(subject_id, study_id, image_root, by_study):
    dicoms = sorted(by_study.get((subject_id, study_id), []))
    for dicom_id in dicoms:
        image_path = image_root / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"
        if image_path.exists():
            return dicom_id, image_path
    return None, None


def join_rows(rows, image_root, by_study, split_map):
    joined = []
    for row in rows:
        subject_id = row["patient_ID"][1:]
        study_id = row["report_ID"][1:]
        dicom_id, image_path = choose_dicom(subject_id, study_id, image_root, by_study)
        if image_path is None:
            continue

        split = split_map.get((subject_id, study_id, dicom_id), "train")
        image_rel_path = str(image_path.relative_to(image_root))
        merged = dict(row)
        merged["img_id"] = dicom_id
        merged["img_path"] = str(image_path)
        merged["image_rel_path"] = image_rel_path
        merged["split"] = split
        joined.append(merged)
    return joined


def aggregate_triplets_by_image(joined_rows):
    aggregated = {}
    for row in joined_rows:
        key = row["img_id"]
        record = aggregated.setdefault(
            key,
            {
                "img_id": row["img_id"],
                "img_path": row["img_path"],
                "image_rel_path": row["image_rel_path"],
                "split": row["split"],
                "triplets": [],
            },
        )
        for triplet in row.get("triplets", []):
            if triplet not in record["triplets"]:
                record["triplets"].append(triplet)
    return list(aggregated.values())


def load_clip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MedCLIPProcessor()
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    original_torch_load = torch.load
    original_load_state_dict = model.load_state_dict

    def cpu_safe_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        return original_torch_load(*args, **kwargs)

    def compat_load_state_dict(state_dict, strict=True):
        state_dict.pop("text_model.model.embeddings.position_ids", None)
        return original_load_state_dict(state_dict, strict=False)

    torch.load = cpu_safe_torch_load
    model.load_state_dict = compat_load_state_dict
    try:
        model.from_pretrained()
    finally:
        torch.load = original_torch_load
        model.load_state_dict = original_load_state_dict
    model = model.to(device)
    model.eval()
    return model, processor, device


def encode_text_cpu(model, input_ids, attention_mask=None):
    input_ids = input_ids.to(next(model.parameters()).device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(next(model.parameters()).device)
    text_embeds = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds


def encode_captions(captions, model, device, processor):
    encoded = []
    for caption in captions:
        inputs = processor(text=[caption], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            emb = encode_text_cpu(model, inputs["input_ids"], inputs.get("attention_mask")).cpu().numpy()
        encoded.append(emb[0])
    return np.array(encoded)


# Replicate MedCLIPFeatureExtractor preprocessing directly with torchvision,
# bypassing the broken MedCLIPProcessor image pipeline (incompatible with
# transformers >= 4.26 which changed CLIPImageProcessor.resize() to expect
# dict-style size and numpy arrays instead of PIL Images).
_MEDCLIP_IMG_TRANSFORM = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=0.5862785803043838, std=0.27950088968644304),
])


def encode_images(image_paths, model, processor, device):
    encoded = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        inputs = _MEDCLIP_IMG_TRANSFORM(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(pixel_values=inputs).cpu().numpy()
        encoded.append(emb[0])
    return np.array(encoded)


def build_retrieval_outputs(records, model, processor, device):
    captions = ["; ".join(r["triplets"]) for r in records]
    image_paths = [r["img_path"] for r in records]
    image_ids = [r["img_id"] for r in records]

    caption_emb = encode_captions(captions, model, device, processor).astype(np.float32)
    image_emb = encode_images(image_paths, model, processor, device).astype(np.float32)

    faiss.normalize_L2(caption_emb)
    index = faiss.IndexFlatIP(caption_emb.shape[1])
    index.add(caption_emb)

    faiss.normalize_L2(image_emb)
    _, neighbors = index.search(image_emb, min(8, len(records)))

    retrieved = {}
    for row_idx, img_id in enumerate(image_ids):
        triplet_lists = []
        for nn_idx in neighbors[row_idx]:
            if image_ids[nn_idx] == img_id:
                continue
            triplet_lists.append(captions[nn_idx])
            if len(triplet_lists) == 7:
                break
        retrieved[img_id] = triplet_lists

    return index, captions, retrieved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="RadGraph-enriched demo JSONL")
    parser.add_argument("--image-root", required=True, help="Root directory containing JPG study folders")
    parser.add_argument("--metadata-csv-gz", required=True)
    parser.add_argument("--split-csv-gz", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(args.input))
    by_study, split_map = load_metadata(args.metadata_csv_gz, args.split_csv_gz)
    joined_rows = join_rows(rows, Path(args.image_root), by_study, split_map)
    aggregated = aggregate_triplets_by_image(joined_rows)
    aggregated = [row for row in aggregated if row["triplets"]]

    joined_path = output_dir / "demo_annotations_joined.json"
    with open(joined_path, "w") as handle:
        json.dump(joined_rows, handle, indent=2)

    aggregated_path = output_dir / "demo_annotations_image_level.json"
    with open(aggregated_path, "w") as handle:
        json.dump(aggregated, handle, indent=2)

    model, processor, device = load_clip_model()
    index, captions, retrieved = build_retrieval_outputs(aggregated, model, processor, device)

    faiss.write_index(index, str(output_dir / "kg_nle_index"))
    with open(output_dir / "kg_nle_index_captions.json", "w") as handle:
        json.dump(captions, handle, indent=2)
    with open(output_dir / "retrieved_triplets.json", "w") as handle:
        json.dump(retrieved, handle, indent=2)

    print(f"joined rows: {len(joined_rows)}")
    print(f"image-level rows with triplets: {len(aggregated)}")
    print(f"wrote datastore artifacts to {output_dir}")


if __name__ == "__main__":
    main()
