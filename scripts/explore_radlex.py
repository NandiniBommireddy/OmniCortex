#!/usr/bin/env python3
"""Explore RadLex and Radiology Gamuts ontologies to verify coverage.

This is the MVE (minimum viable experiment) gate: run this BEFORE building
the full pipeline to verify that the Gamuts ontology has useful causal
edges for the 10 MIMIC-NLE diagnoses.

Usage:
    python scripts/explore_radlex.py
    python scripts/explore_radlex.py --radlex-owl kg/data/radlex/radlex.owl --gamuts-owl kg/data/gamuts/gamuts.owl
"""

import argparse
from pathlib import Path

from owlready2 import get_ontology


# The 10 MIMIC-NLE seed diagnoses + common RadGraph entity texts
SEED_DIAGNOSES = [
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

# Common RadGraph entity texts to check alignment
SAMPLE_ENTITIES = [
    "opacity",
    "effusion",
    "consolidation",
    "atelectasis",
    "edema",
    "pneumonia",
    "cardiomegaly",
    "scarring",
    "blunting",
    "markings",
    "infiltrate",
    "pneumothorax",
]


def find_class(ontology, term: str):
    """Search for a class by label (case-insensitive)."""
    # Try exact label match first
    result = ontology.search_one(label=term)
    if result:
        return result
    # Try case-insensitive
    for cls in ontology.classes():
        for label in cls.label:
            if label.lower() == term.lower():
                return cls
    return None


def get_all_labels(ontology) -> dict[str, object]:
    """Build {lowered_label: class} map for quick lookup."""
    label_map = {}
    for cls in ontology.classes():
        for label in cls.label:
            label_map.setdefault(label.lower(), cls)
        # Also index the class name (IRI local part)
        name = cls.name
        if name:
            label_map.setdefault(name.lower().replace("_", " "), cls)
    return label_map


def explore_object_properties(ontology, name: str):
    """Print all object properties and their relation counts."""
    print(f"\n{'='*60}")
    print(f"Object properties in {name}")
    print(f"{'='*60}")
    props = list(ontology.object_properties())
    if not props:
        print("  (none found)")
        return props
    for prop in sorted(props, key=lambda p: p.name):
        rels = list(prop.get_relations())
        print(f"  {prop.name}: {len(rels)} relations")
        # Show a few example relations
        for subj, obj in rels[:3]:
            subj_label = subj.label[0] if subj.label else subj.name
            obj_label = obj.label[0] if obj.label else obj.name
            print(f"    {subj_label} --{prop.name}--> {obj_label}")
        if len(rels) > 3:
            print(f"    ... and {len(rels) - 3} more")
    return props


def explore_data_properties(ontology, name: str):
    """Print all data/annotation properties."""
    print(f"\n{'='*60}")
    print(f"Data/annotation properties in {name}")
    print(f"{'='*60}")
    for prop in sorted(ontology.data_properties(), key=lambda p: p.name):
        print(f"  {prop.name}")
    for prop in sorted(ontology.annotation_properties(), key=lambda p: p.name):
        print(f"  {prop.name} (annotation)")


def explore_class_neighbors(cls, ontology, obj_props):
    """Print all object property relations for a given class."""
    neighbors = []
    for prop in obj_props:
        for subj, obj in prop.get_relations():
            if subj == cls:
                obj_label = obj.label[0] if obj.label else obj.name
                neighbors.append((prop.name, obj_label, "outgoing"))
            elif obj == cls:
                subj_label = subj.label[0] if subj.label else subj.name
                neighbors.append((prop.name, subj_label, "incoming"))
    return neighbors


def main():
    parser = argparse.ArgumentParser(description="Explore RadLex + Gamuts ontologies")
    parser.add_argument("--radlex-owl", type=Path, default=Path("kg/data/radlex/radlex.owl"))
    parser.add_argument("--gamuts-owl", type=Path, default=Path("kg/data/gamuts/gamuts.owl"))
    args = parser.parse_args()

    # --- Load ontologies ---
    print("Loading ontologies (this may take a minute)...")

    gamuts = None
    if args.gamuts_owl.exists():
        print(f"  Loading Gamuts from {args.gamuts_owl}...")
        gamuts = get_ontology(f"file://{args.gamuts_owl.resolve()}").load()
        num_classes = len(list(gamuts.classes()))
        print(f"  Gamuts loaded: {num_classes} classes")
    else:
        print(f"  WARNING: {args.gamuts_owl} not found — skipping Gamuts")

    radlex = None
    if args.radlex_owl.exists():
        print(f"  Loading RadLex from {args.radlex_owl}...")
        radlex = get_ontology(f"file://{args.radlex_owl.resolve()}").load()
        num_classes = len(list(radlex.classes()))
        print(f"  RadLex loaded: {num_classes} classes")
    else:
        print(f"  WARNING: {args.radlex_owl} not found — skipping RadLex")

    if not gamuts and not radlex:
        print("\nERROR: No ontologies loaded. Run scripts/download_ontologies.py first.")
        return

    # --- Explore object properties ---
    gamuts_props = []
    radlex_props = []
    if gamuts:
        gamuts_props = explore_object_properties(gamuts, "Gamuts")
    if radlex:
        radlex_props = explore_object_properties(radlex, "RadLex")

    # --- Check diagnosis coverage ---
    print(f"\n{'='*60}")
    print("Diagnosis coverage check (10 MIMIC-NLE diagnoses)")
    print(f"{'='*60}")

    for ontology, name, props in [
        (gamuts, "Gamuts", gamuts_props),
        (radlex, "RadLex", radlex_props),
    ]:
        if not ontology:
            continue
        print(f"\n--- {name} ---")
        label_map = get_all_labels(ontology)
        matched = 0
        for diag in SEED_DIAGNOSES:
            cls = label_map.get(diag.lower())
            if cls:
                matched += 1
                labels = cls.label if cls.label else [cls.name]
                parents = [p.label[0] if hasattr(p, 'label') and p.label else str(p)
                           for p in cls.is_a[:3]]
                print(f"  ✓ {diag}")
                print(f"    Labels: {labels}")
                print(f"    Parents: {parents}")

                # Show object property neighbors
                if props:
                    neighbors = explore_class_neighbors(cls, ontology, props)
                    if neighbors:
                        print(f"    Relations ({len(neighbors)}):")
                        for edge, target, direction in neighbors[:8]:
                            arrow = "-->" if direction == "outgoing" else "<--"
                            print(f"      {arrow} {edge}: {target}")
                        if len(neighbors) > 8:
                            print(f"      ... and {len(neighbors) - 8} more")
                    else:
                        print(f"    Relations: (none)")
            else:
                print(f"  ✗ {diag} — NOT FOUND")
        print(f"\n  Coverage: {matched}/{len(SEED_DIAGNOSES)}")

    # --- Check sample RadGraph entities ---
    print(f"\n{'='*60}")
    print("RadGraph entity coverage check")
    print(f"{'='*60}")

    for ontology, name in [(gamuts, "Gamuts"), (radlex, "RadLex")]:
        if not ontology:
            continue
        print(f"\n--- {name} ---")
        label_map = get_all_labels(ontology)
        matched = 0
        for ent in SAMPLE_ENTITIES:
            cls = label_map.get(ent.lower())
            if cls:
                matched += 1
                print(f"  ✓ {ent} → {cls.label[0] if cls.label else cls.name}")
            else:
                print(f"  ✗ {ent}")
        print(f"\n  Coverage: {matched}/{len(SAMPLE_ENTITIES)}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY — Is this worth pursuing?")
    print(f"{'='*60}")
    print("""
Check the output above for:
1. Does Gamuts have 'may_cause' or similar causal object properties?
2. Do the 10 diagnoses match in Gamuts? (need >= 8/10)
3. Are the causal neighbors clinically relevant to chest X-rays?
   Good: "atelectasis --may_cause--> volume loss"
   Bad:  "atelectasis --is_a--> Clinical finding" (too generic)
4. Do common RadGraph entities (opacity, effusion, etc.) have matches?

If YES to all: proceed to Step 1 (entity alignment script).
If NO:  the Gamuts ontology doesn't have useful chest X-ray coverage.
""")


if __name__ == "__main__":
    main()
