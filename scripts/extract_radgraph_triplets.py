import argparse
import json
import re
import sys
from collections import OrderedDict
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


def group_rows_by_report(rows):
    groups = OrderedDict()
    for row in rows:
        report_id = row.get("report_ID")
        if not report_id:
            report_id = row.get("sentence_ID")
        groups.setdefault(report_id, []).append(row)
    return groups


def build_report_text_from_nles(rows_for_report):
    sentences = []
    for row in rows_for_report:
        nle = (row.get("nle") or "").strip()
        if nle:
            sentences.append(nle)
    return " ".join(sentences)


def build_report_txt_path(reports_root, patient_id, report_id):
    patient = str(patient_id or "")
    report = str(report_id or "")

    if not patient:
        return None
    if not report:
        return None

    if not patient.startswith("p"):
        patient = f"p{patient}"
    if not report.startswith("s"):
        report = f"s{report}"

    if len(patient) < 3:
        return None

    patient_prefix = f"p{patient[1:3]}"
    return reports_root / patient_prefix / patient / f"{report}.txt"


_SECTION_HEADER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z /_-]{1,40}):\s*(.*)$")


def extract_findings_impression(report_text):
    sections = OrderedDict()
    current_section = None

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _SECTION_HEADER_RE.match(line)
        if match:
            section_name = match.group(1).strip().lower()
            current_section = section_name
            sections.setdefault(current_section, [])
            remainder = match.group(2).strip()
            if remainder:
                sections[current_section].append(remainder)
            continue

        if current_section is not None:
            sections[current_section].append(line)

    picked = []
    for wanted in ("findings", "impression"):
        for section_name, lines in sections.items():
            if section_name.startswith(wanted):
                text = " ".join(lines).strip()
                if text:
                    picked.append(text)

    if picked:
        return " ".join(picked)

    return " ".join(report_text.split())


def load_report_text(row, reports_root):
    report_path = build_report_txt_path(
        reports_root=reports_root,
        patient_id=row.get("patient_ID"),
        report_id=row.get("report_ID"),
    )
    if report_path is None:
        return None
    if not report_path.exists():
        return None

    try:
        with open(report_path) as handle:
            raw_text = handle.read()
    except OSError:
        return None

    if not raw_text.strip():
        return None

    return extract_findings_impression(raw_text)


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
    parser.add_argument(
        "--reports-root",
        default=None,
        help=(
            "Optional root to raw MIMIC-CXR report files (e.g. "
            "physionet.org/mimic-cxr/2.1.0/files). If set, triplets are extracted "
            "from report findings/impression text by report_ID."
        ),
    )
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    report_groups = group_rows_by_report(rows)
    report_ids = list(report_groups.keys())

    reports_root = Path(args.reports_root) if args.reports_root else None
    report_texts = []
    missing_report_files = 0
    used_nle_fallback = 0

    for report_id in report_ids:
        rows_for_report = report_groups[report_id]
        report_text = None
        if reports_root is not None:
            report_text = load_report_text(rows_for_report[0], reports_root)
            if report_text is None:
                missing_report_files += 1

        if report_text is None:
            report_text = build_report_text_from_nles(rows_for_report)
            used_nle_fallback += 1

        report_texts.append(report_text)

    Path(args.model_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tokenizer_cache_dir).mkdir(parents=True, exist_ok=True)

    model = RadGraph(
        model_type=args.model_type,
        batch_size=args.batch_size,
        model_cache_dir=args.model_cache_dir,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
    )

    print(f"total rows: {len(rows)}, unique reports: {len(report_ids)}")
    output_rows = []
    triplet_map = {}
    report_triplets = {}

    processed_count = 0
    for report_id_batch, text_batch in zip(
        batched(report_ids, args.batch_size),
        batched(report_texts, args.batch_size),
    ):
        annotations = model(text_batch)
        for idx, report_id in enumerate(report_id_batch):
            processed = get_radgraph_processed_annotations({"0": annotations[str(idx)]})
            triplets = extract_triplets(processed)
            report_triplets[report_id] = triplets
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"processed {processed_count}/{len(report_ids)} reports")

    for row in rows:
        report_id = row.get("report_ID") or row["sentence_ID"]
        triplets = report_triplets.get(report_id, [])
        updated = dict(row)
        updated["triplets"] = triplets
        output_rows.append(updated)
        triplet_map[row["sentence_ID"]] = triplets

    write_jsonl(args.output, output_rows)
    if args.triplets_json:
        with open(args.triplets_json, "w") as handle:
            json.dump(triplet_map, handle, indent=2)

    print(f"wrote {len(output_rows)} rows to {args.output}")
    empty_reports = sum(1 for t in report_triplets.values() if not t)
    print(
        "report-level triplet stats: "
        f"reports={len(report_triplets)}, "
        f"empty_reports={empty_reports}, "
        f"empty_report_pct={100 * empty_reports / max(len(report_triplets), 1):.1f}%"
    )
    if reports_root is not None:
        print(
            "report text source stats: "
            f"missing_report_files={missing_report_files}, "
            f"used_nle_fallback={used_nle_fallback}"
        )
    if args.triplets_json:
        print(f"wrote triplet map to {args.triplets_json}")


if __name__ == "__main__":
    main()
