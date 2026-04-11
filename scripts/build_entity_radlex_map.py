#!/usr/bin/env python3
"""Map RadGraph entity text to RadLex concept names via owlready2.

Two-stage alignment against RadLex OWL (loaded in-memory, no Neo4j needed):
  1. Exact match: lowercased entity vs all RadLex English labels + synonyms
  2. Fuzzy match (rapidfuzz, score_cutoff=88) against English labels only
     (RadLex is 100% radiology — no cross-domain noise like PrimeKG)

Only English labels are used for matching and output, since chain strings
must be readable in the generated NLE prompts.

Usage:
    python scripts/build_entity_radlex_map.py \
        --input tmp/demo/mimic-nle-train-radgraph.json \
        --radlex-owl kg/data/radlex/radlex.owl \
        --output data/entity_radlex_map.json
"""

import argparse
import json
import sys
from pathlib import Path

from owlready2 import get_ontology

# ---------------------------------------------------------------------------
# Manual map for important RadGraph entities missing from RadLex exact/fuzzy
# Keys are lowercased RadGraph entity texts; values are RadLex canonical labels.
# These are verified to exist in RadLex and have May_Cause edges.
# ---------------------------------------------------------------------------

MANUAL_MAP: dict[str, str] = {
    # Radiological signs
    "blunting":               "blunting of costophrenic angle",
    "scarring":               "scar",
    "scaring":                "scar",
    "interstitial markings":  "interstitial markings",
    "interstitial pattern":   "interstitial pattern",
    "haziness":               "hazy opacity",
    "ground glass":           "ground-glass opacity",
    "ground-glass":           "ground-glass opacity",
    "ground glass opacity":   "ground-glass opacity",
    "interstitial markings":  "interstitial markings",
    "pulmonary markings":     "pulmonary vascularity",
    "vascular markings":      "pulmonary vascularity",
    "markings":               "pulmonary vascularity",
    # Diseases / findings
    "cardiomegaly":           "cardiomegaly",
    "pulmonary edema":        "pulmonary edema",
    "pleural thickening":     "pleural thickening",
    "vascular congestion":    "pulmonary venous congestion",
    "pulmonary congestion":   "pulmonary venous congestion",
    # Explicitly block bad fuzzy matches
    "aeration":               None,   # would fuzzy-match to "laceration" — wrong
    # Common typos / variants in RadGraph output
    "edama":                  "edema",
    "edena":                  "edema",
    "edmea":                  "edema",
    "pnuemonia":              "pneumonia",
    "pneumothoraces":         "pneumothorax",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_entities(input_path: str) -> set[str]:
    """Extract unique entity texts from RadGraph triplets (JSONL)."""
    entities: set[str] = set()
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for triplet in entry.get("triplets", []):
                parts = triplet.split(" suggestive of ")
                if len(parts) == 2:
                    entities.add(parts[0].strip().lower())
                    entities.add(parts[1].strip().lower())
    return entities


def get_english_labels(cls) -> list[str]:
    """Return all English labels for an owlready2 class, lowercased."""
    labels = []
    for label in cls.label:
        # owlready2 locstr has a 'lang' attribute
        lang = getattr(label, "lang", None)
        if lang is None or lang == "en":
            labels.append(str(label).lower())
    return labels


def build_label_index(ontology) -> dict[str, tuple[object, str]]:
    """Build {lowered_english_label: (cls, canonical_english_label)} index.

    Includes the class name (IRI local part) as a fallback key.
    Skips non-English labels entirely.
    """
    index: dict[str, tuple[object, str]] = {}
    for cls in ontology.classes():
        en_labels = []
        for label in cls.label:
            lang = getattr(label, "lang", None)
            if lang is None or lang == "en":
                en_labels.append(str(label))

        if not en_labels:
            # No English label — use the IRI local name as a last resort
            name = cls.name
            if name:
                en_labels = [name.replace("_", " ")]

        canonical = en_labels[0] if en_labels else cls.name
        for lbl in en_labels:
            key = lbl.lower().strip()
            if key:
                index.setdefault(key, (cls, canonical))

    return index


# ---------------------------------------------------------------------------
# Stage 1: Exact match
# ---------------------------------------------------------------------------

def exact_match(entities: list[str], label_index: dict) -> dict[str, str]:
    """Return {entity_text: canonical_english_label} for exact matches."""
    matched: dict[str, str] = {}
    for ent in entities:
        hit = label_index.get(ent)
        if hit:
            _, canonical = hit
            matched[ent] = canonical
    print(f"  [exact] {len(matched)}/{len(entities)} matched")
    return matched


# ---------------------------------------------------------------------------
# Stage 2: Fuzzy match
# ---------------------------------------------------------------------------

def fuzzy_match(
    entities: list[str],
    label_index: dict,
    score_cutoff: int = 88,
) -> dict[str, str]:
    """Fuzzy match remaining entities against RadLex English labels.

    RadLex is 100% radiology-domain, so it is safe to fuzzy match here
    without cross-domain noise like PrimeKG.

    Guards:
    - Skips very short terms (<= 3 chars)
    - Rejects matches where the RadLex term is < 50% the length of the query
      (prevents short RadLex terms like "tin", "normal", "duct" matching
      longer RadGraph phrases like "blunting", "abnormalities", "abduction")
    """
    try:
        from rapidfuzz import process
    except ImportError:
        print("  [fuzzy] rapidfuzz not installed — skipping (pip install rapidfuzz)")
        return {}

    all_keys = list(label_index.keys())
    matched: dict[str, str] = {}
    skipped = 0
    length_rejected = 0

    for idx, ent in enumerate(entities):
        if idx > 0 and idx % 200 == 0:
            print(f"  [fuzzy] processed {idx}/{len(entities)}")
        if len(ent) <= 3:
            skipped += 1
            continue
        result = process.extractOne(ent, all_keys, score_cutoff=score_cutoff)
        if result is not None:
            best_key, _score, _ = result
            # Reject if matched RadLex term is much shorter than the query
            if len(best_key) < 0.7 * len(ent):
                length_rejected += 1
                continue
            _, canonical = label_index[best_key]
            matched[ent] = canonical

    print(f"  [fuzzy] processed {len(entities)}/{len(entities)} "
          f"(skipped {skipped} short terms, {length_rejected} length-ratio rejected), "
          f"{len(matched)} matched")
    return matched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Map RadGraph entity text to RadLex concept names"
    )
    parser.add_argument(
        "--input",
        default="tmp/demo/mimic-nle-train-radgraph.json",
        help="Path to RadGraph JSONL file",
    )
    parser.add_argument(
        "--radlex-owl",
        default="kg/data/radlex/radlex.owl",
        help="Path to RadLex OWL file",
    )
    parser.add_argument(
        "--output",
        default="data/entity_radlex_map.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--fuzzy-cutoff",
        type=int,
        default=88,
        help="Fuzzy match score cutoff 0-100 (default: 88)",
    )
    parser.add_argument(
        "--no-fuzzy",
        action="store_true",
        help="Disable fuzzy matching (exact only)",
    )
    args = parser.parse_args()

    owl_path = Path(args.radlex_owl)
    if not owl_path.exists():
        print(f"ERROR: RadLex OWL not found at {owl_path}")
        print("Run: python scripts/download_ontologies.py --api-key YOUR_KEY")
        sys.exit(1)

    # --- Load entities ---
    print("Loading entities from RadGraph triplets...")
    entities = load_entities(args.input)
    print(f"  Found {len(entities)} unique entity texts")

    # --- Load RadLex ---
    print(f"Loading RadLex from {owl_path} ...")
    radlex = get_ontology(f"file://{owl_path.resolve()}").load()
    num_classes = len(list(radlex.classes()))
    print(f"  {num_classes} classes loaded")

    # --- Build label index ---
    print("Building English label index...")
    label_index = build_label_index(radlex)
    print(f"  {len(label_index)} unique English label keys")

    # --- Align ---
    result: dict[str, str | None] = {}
    entity_list = sorted(entities)

    # Stage 0: manual map (highest priority — verified correct)
    manual_hits = 0
    for ent in entity_list:
        if ent in MANUAL_MAP:
            result[ent] = MANUAL_MAP[ent]
            manual_hits += 1
    remaining = [e for e in entity_list if e not in result]
    print(f"\nStage 0: Manual map -> {manual_hits} matched, {len(remaining)} remaining")

    # Stage 1: exact
    print("\nStage 1: Exact match...")
    exact = exact_match(remaining, label_index)
    for ent, name in exact.items():
        result[ent] = name
    remaining = [e for e in remaining if e not in exact]
    print(f"  -> {len(exact)} matched, {len(remaining)} remaining\n")

    # Stage 2: fuzzy
    if not args.no_fuzzy:
        print("Stage 2: Fuzzy match (RadLex English labels only)...")
        fuzzy = fuzzy_match(remaining, label_index, args.fuzzy_cutoff)
        for ent, name in fuzzy.items():
            result[ent] = name
        remaining = [e for e in remaining if e not in fuzzy]
        print(f"  -> {len(fuzzy)} matched, {len(remaining)} remaining\n")
    else:
        print("Stage 2: Skipped (--no-fuzzy)\n")
        fuzzy = {}

    # Fill unmatched with null
    for ent in entities:
        if ent not in result:
            result[ent] = None

    # --- Stats ---
    total = len(result)
    matched_count = sum(1 for v in result.values() if v is not None)
    null_count = total - matched_count
    print("=" * 50)
    print(f"Total entities:     {total}")
    print(f"  Stage 0 (manual): {manual_hits}")
    print(f"  Stage 1 (exact):  {len(exact)}")
    print(f"  Stage 2 (fuzzy):  {len(fuzzy)}")
    print(f"  Matched total:    {matched_count} ({100*matched_count/total:.1f}%)")
    print(f"  Unmatched (null): {null_count}")
    print("=" * 50)

    # Show a sample of matched entities for sanity check
    print("\nSample matches:")
    sample = [(k, v) for k, v in sorted(result.items()) if v is not None][:20]
    for ent, name in sample:
        print(f"  {ent!r:40s} -> {name!r}")

    print("\nUnmatched entities (sample):")
    unmatched = [k for k, v in sorted(result.items()) if v is None][:20]
    for ent in unmatched:
        print(f"  {ent!r}")

    # --- Write output ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
