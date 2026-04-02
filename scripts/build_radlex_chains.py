#!/usr/bin/env python3
"""Build 1-hop reasoning chains from RadGraph triplets via RadLex OWL.

For each image's RadGraph triplets, maps entity objects to RadLex concepts
using the entity radlex map, then traverses May_Cause / May_Be_Caused_By
edges from RadLex to build chains.

Chain format (same as PrimeKG pipeline, compatible with build_demo_llava_json.py):
    "opacity --suggestive_of--> pneumonia --May_Cause--> crazy-paving sign"
    "blunting --suggestive_of--> pleural effusion --May_Be_Caused_By-- fissure sign"

Usage:
    python scripts/build_radlex_chains.py \
        --input tmp/demo/mimic-nle-train-radgraph.json \
        --entity-map data/entity_radlex_map.json \
        --radlex-owl kg/data/radlex/radlex.owl \
        --output data/radgraph-multihop-radlex.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from owlready2 import get_ontology

MAX_NEIGHBORS = 10   # max chains per entity node (same cap as PrimeKG pipeline)


def parse_triplet(triplet_str: str):
    """Parse 'subject suggestive_of object' into (subj, rel, obj)."""
    sep = " suggestive of "
    if sep in triplet_str:
        idx = triplet_str.index(sep)
        subj = triplet_str[:idx].strip()
        obj = triplet_str[idx + len(sep):].strip()
        return subj, "suggestive_of", obj
    return triplet_str, None, None


def build_label_index(ontology) -> dict[str, object]:
    """Build {lowered_english_label: cls} for fast lookup."""
    index: dict[str, object] = {}
    for cls in ontology.classes():
        for label in cls.label:
            lang = getattr(label, "lang", None)
            if lang is None or lang == "en":
                index.setdefault(str(label).lower(), cls)
        name = cls.name
        if name:
            index.setdefault(name.lower().replace("_", " "), cls)
    return index


def get_english_label(cls) -> str:
    """Return the first English label for a class, or the class name."""
    for label in cls.label:
        lang = getattr(label, "lang", None)
        if lang is None or lang == "en":
            return str(label)
    return cls.name or str(cls)


def get_neighbors(cls_name: str, label_index: dict, cache: dict) -> list[dict]:
    """Return May_Cause and May_Be_Caused_By neighbors for a RadLex class.

    Returns list of {"edge": str, "neighbor": str} dicts.
    Results are cached by cls_name.
    """
    if cls_name in cache:
        return cache[cls_name]

    cls = label_index.get(cls_name.lower())
    if cls is None:
        cache[cls_name] = []
        return []

    results = []

    # Traverse May_Cause (outgoing): disease/finding -> imaging sign
    for prop in cls.namespace.ontology.object_properties():
        if prop.name in ("May_Cause", "may_cause"):
            for subj, obj in prop.get_relations():
                if subj == cls:
                    neighbor_label = get_english_label(obj)
                    results.append({"edge": "May_Cause", "neighbor": neighbor_label})
                    if len(results) >= MAX_NEIGHBORS:
                        break
            if len(results) >= MAX_NEIGHBORS:
                break

    # Traverse May_Be_Caused_By (outgoing from cls): sign <- disease
    for prop in cls.namespace.ontology.object_properties():
        if prop.name in ("May_Be_Caused_By", "may_be_caused_by"):
            for subj, obj in prop.get_relations():
                if subj == cls:
                    neighbor_label = get_english_label(obj)
                    results.append({"edge": "May_Be_Caused_By", "neighbor": neighbor_label})
                    if len(results) >= MAX_NEIGHBORS:
                        break
            if len(results) >= MAX_NEIGHBORS:
                break

    cache[cls_name] = results[:MAX_NEIGHBORS]
    return cache[cls_name]


def build_chain_string(subj: str, rel: str, obj: str, row: dict) -> str:
    """Build chain string from RadGraph triplet + RadLex neighbor."""
    return f"{subj} --{rel}--> {obj} --{row['edge']}--> {row['neighbor']}"


def main():
    parser = argparse.ArgumentParser(
        description="Build 1-hop reasoning chains via RadLex May_Cause edges"
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("tmp/demo/mimic-nle-train-radgraph.json"),
        help="RadGraph triplets file (JSONL)",
    )
    parser.add_argument(
        "--entity-map", type=Path,
        default=Path("data/entity_radlex_map.json"),
        help="Entity-to-RadLex label mapping (JSON)",
    )
    parser.add_argument(
        "--radlex-owl", type=Path,
        default=Path("kg/data/radlex/radlex.owl"),
        help="RadLex OWL file",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/radgraph-multihop-radlex.jsonl"),
        help="Output JSONL file",
    )
    args = parser.parse_args()

    # --- Load entity map ---
    print(f"Loading entity map from {args.entity_map} ...")
    with open(args.entity_map) as f:
        entity_map: dict[str, str | None] = json.load(f)
    mapped = sum(1 for v in entity_map.values() if v is not None)
    print(f"  {mapped}/{len(entity_map)} entities have RadLex mappings")

    # --- Load RadLex ---
    owl_path = args.radlex_owl
    if not owl_path.exists():
        print(f"ERROR: RadLex OWL not found at {owl_path}")
        sys.exit(1)
    print(f"Loading RadLex from {owl_path} ...")
    radlex = get_ontology(f"file://{owl_path.resolve()}").load()
    num_classes = len(list(radlex.classes()))
    print(f"  {num_classes} classes loaded")

    # --- Build label index ---
    print("Building label index ...")
    label_index = build_label_index(radlex)
    print(f"  {len(label_index)} English label keys")

    # --- Load input rows ---
    print(f"Loading RadGraph data from {args.input} ...")
    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"  {len(rows)} rows loaded")

    # --- Build chains ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    neighbor_cache: dict[str, list] = {}
    total_chains = 0
    rows_with_chains = 0

    with open(args.output, "w") as out:
        for i, row in enumerate(rows):
            triplets = row.get("triplets", [])
            img_id = row.get("sentence_ID", row.get("report_ID", f"row_{i}"))
            chains = []
            seen_edges: set = set()

            for triplet_str in triplets:
                subj, rel, obj = parse_triplet(triplet_str)
                if rel is None:
                    continue

                # Look up both subject and object in the entity map
                for entity_text in (obj, subj):
                    node_name = entity_map.get(entity_text) or entity_map.get(entity_text.lower())
                    if not node_name:
                        continue

                    neighbors = get_neighbors(node_name, label_index, neighbor_cache)
                    for nbr in neighbors:
                        dedup_key = (node_name, nbr["edge"], nbr["neighbor"])
                        if dedup_key in seen_edges:
                            continue
                        seen_edges.add(dedup_key)
                        chain_str = build_chain_string(subj, rel, obj, nbr)
                        chains.append(chain_str)

            record = {"img_id": img_id, "triplets": triplets, "chains": chains}
            out.write(json.dumps(record) + "\n")

            total_chains += len(chains)
            if chains:
                rows_with_chains += 1

            if (i + 1) % 500 == 0:
                print(
                    f"  [{i+1}/{len(rows)}] chains: {total_chains:,}, "
                    f"cache: {len(neighbor_cache)}"
                )

    print(f"\n{'='*50}")
    print(f"Total rows:        {len(rows):,}")
    print(f"Rows with chains:  {rows_with_chains:,} ({100*rows_with_chains/len(rows):.1f}%)")
    print(f"Total chains:      {total_chains:,}")
    print(f"Cache entries:     {len(neighbor_cache):,}")
    print(f"Written to:        {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
