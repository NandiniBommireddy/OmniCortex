#!/usr/bin/env python
"""Map RadGraph entity text to PrimeKG node names via Neo4j.

Three-stage alignment:
  1. Exact match (lowercased) in Neo4j
  2. scispaCy UMLS CUI linking -> Neo4j lookup by CUI or name
  3. Fuzzy match (rapidfuzz, score_cutoff=85) against all PrimeKG node names

Pre-built diagnosis mappings are loaded from a JSON file and applied first.
"""

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase


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


def load_diagnosis_mapping(path: str) -> dict[str, str | None]:
    """Load pre-built diagnosis-to-PrimeKG mapping.

    Returns a dict {lowercase_diagnosis: first_matched_node_name or None}.
    """
    with open(path) as f:
        data = json.load(f)
    mapping: dict[str, str | None] = {}
    for diag, matches in data.items():
        key = diag.strip().lower()
        if matches:
            mapping[key] = matches[0]["node_name"]
        else:
            mapping[key] = None
    return mapping


# ---------------------------------------------------------------------------
# Stage 1: Exact match via Neo4j
# ---------------------------------------------------------------------------

def exact_match_batch(driver, entities: list[str]) -> dict[str, str]:
    """Return {entity_text: primekg_node_name} for exact (lowered) matches."""
    matched: dict[str, str] = {}
    batch_size = 500
    for i in range(0, len(entities), batch_size):
        batch = entities[i : i + batch_size]
        with driver.session() as session:
            result = session.run(
                "UNWIND $names AS name "
                "MATCH (n:Node) WHERE toLower(n.node_name) = name "
                "RETURN name, n.node_name AS node_name",
                names=batch,
            )
            for record in result:
                matched[record["name"]] = record["node_name"]
        if (i + batch_size) % 500 == 0 or i + batch_size >= len(entities):
            print(f"  [exact] processed {min(i + batch_size, len(entities))}/{len(entities)}")
    return matched


# ---------------------------------------------------------------------------
# Stage 2: scispaCy CUI linking -> Neo4j
# ---------------------------------------------------------------------------

def cui_match(driver, entities: list[str], spacy_model: str) -> dict[str, str]:
    """Use scispaCy UMLS linker to get CUIs, then look them up in Neo4j."""
    try:
        import spacy
        import scispacy  # noqa: F401
        from scispacy.linking import EntityLinker  # noqa: F401  # registers the factory
    except ImportError:
        print("  [cui] scispacy not available, skipping CUI stage")
        return {}

    print(f"  [cui] loading {spacy_model} + UMLS linker (this may take a minute)...")
    nlp = spacy.load(spacy_model)
    # Try both factory names across scispacy versions
    for factory_name in ("scispacy_linker", "entity_linker"):
        try:
            nlp.add_pipe(
                factory_name,
                config={"resolve_abbreviations": True, "linker_name": "umls"},
            )
            linker = nlp.get_pipe(factory_name)
            break
        except ValueError:
            continue
    else:
        print("  [cui] could not register scispacy linker, skipping CUI stage")
        return {}

    matched: dict[str, str] = {}
    for idx, entity_text in enumerate(entities):
        doc = nlp(entity_text)
        if idx > 0 and idx % 500 == 0:
            print(f"  [cui] processed {idx}/{len(entities)}")

        for ent in doc.ents:
            if not ent._.kb_ents:
                continue
            top_cui, _score = ent._.kb_ents[0]
            cui_info = linker.kb.cui_to_entity.get(top_cui)
            if cui_info is None:
                continue

            # Try canonical name lookup in Neo4j
            canonical = cui_info.canonical_name.lower()
            with driver.session() as session:
                result = session.run(
                    "MATCH (n:Node) WHERE toLower(n.node_name) = $name "
                    "RETURN n.node_name LIMIT 1",
                    name=canonical,
                )
                rec = result.single()
                if rec:
                    matched[entity_text] = rec["n.node_name"]
                    break  # first entity match wins

            # Try aliases
            if entity_text not in matched and cui_info.aliases:
                for alias in list(cui_info.aliases)[:5]:
                    with driver.session() as session:
                        result = session.run(
                            "MATCH (n:Node) WHERE toLower(n.node_name) = $name "
                            "RETURN n.node_name LIMIT 1",
                            name=alias.lower(),
                        )
                        rec = result.single()
                        if rec:
                            matched[entity_text] = rec["n.node_name"]
                            break

    print(f"  [cui] processed {len(entities)}/{len(entities)}")
    return matched


# ---------------------------------------------------------------------------
# Stage 3: Fuzzy match via rapidfuzz
# ---------------------------------------------------------------------------

def fuzzy_match(
    driver, entities: list[str], subgraph_nodes_path: str | None = None,
    score_cutoff: int = 85,
) -> dict[str, str]:
    """Fuzzy match entities against PrimeKG node names.

    If subgraph_nodes_path is provided, only matches against nodes in the
    radiology subgraph (2-hop neighborhood of seed diagnoses). This
    naturally filters out irrelevant nodes (ear infections, adipose tissue,
    etc.) without hardcoded blocklists.

    Skips very short entities (<=3 chars) which produce noisy matches.
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        print("  [fuzzy] rapidfuzz not available, skipping fuzzy stage")
        return {}

    if subgraph_nodes_path:
        print(f"  [fuzzy] loading radiology subgraph nodes from {subgraph_nodes_path}...")
        with open(subgraph_nodes_path) as f:
            subgraph = json.load(f)
        all_names = [node["node_name"] for node in subgraph]
        print(f"  [fuzzy] loaded {len(all_names)} subgraph node names (radiology-relevant only)")
    else:
        print("  [fuzzy] fetching all PrimeKG node names from Neo4j...")
        with driver.session() as session:
            result = session.run("MATCH (n:Node) RETURN n.node_name")
            all_names = [record["n.node_name"] for record in result]
        print(f"  [fuzzy] loaded {len(all_names)} node names")

    # Build lowered lookup for returning original name
    name_lower_map = {n.lower(): n for n in all_names}
    all_names_lower = list(name_lower_map.keys())

    matched: dict[str, str] = {}
    skipped = 0
    for idx, entity_text in enumerate(entities):
        if idx > 0 and idx % 500 == 0:
            print(f"  [fuzzy] processed {idx}/{len(entities)}")
        # Skip very short terms — they produce noisy matches
        if len(entity_text) <= 3:
            skipped += 1
            continue
        result = process.extractOne(
            entity_text, all_names_lower, score_cutoff=score_cutoff
        )
        if result is not None:
            best_match_lower, score, _ = result
            matched[entity_text] = name_lower_map[best_match_lower]

    print(f"  [fuzzy] processed {len(entities)}/{len(entities)} (skipped {skipped} short terms)")
    return matched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Map RadGraph entity text to PrimeKG node names"
    )
    parser.add_argument(
        "--input",
        default="tmp/demo/mimic-nle-train-radgraph.json",
        help="Path to RadGraph JSONL file",
    )
    parser.add_argument(
        "--diagnosis-mapping",
        default="kg/data/subgraph/diagnosis_node_mapping.json",
        help="Pre-built diagnosis-to-PrimeKG mapping",
    )
    parser.add_argument(
        "--subgraph-nodes",
        default="kg/data/subgraph/primekg_radiology_nodes.json",
        help="Radiology subgraph nodes JSON (restricts fuzzy matching to relevant nodes)",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_sci_lg",
        help="spaCy/ scispaCy model to use for CUI linking (e.g., en_core_sci_lg, en_core_sci_scibert)",
    )
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-password", default="primekg123")
    parser.add_argument(
        "--output",
        default="data/entity_cui_map.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    # --- Load entities ---
    print("Loading entities from RadGraph triplets...")
    entities = load_entities(args.input)
    print(f"  Found {len(entities)} unique entity texts")

    # --- Load pre-built diagnosis mapping ---
    print("Loading pre-built diagnosis mapping...")
    diag_map = load_diagnosis_mapping(args.diagnosis_mapping)
    print(f"  {sum(1 for v in diag_map.values() if v is not None)} diagnoses with PrimeKG matches")

    # --- Connect to Neo4j ---
    print(f"Connecting to Neo4j at {args.neo4j_uri}...")
    driver = GraphDatabase.driver(
        args.neo4j_uri, auth=("neo4j", args.neo4j_password)
    )
    driver.verify_connectivity()
    print("  Connected.")

    # --- Build result map ---
    result: dict[str, str | None] = {}

    # Pre-populate from diagnosis mapping
    for ent in entities:
        if ent in diag_map:
            result[ent] = diag_map[ent]

    # Remaining entities to resolve
    remaining = sorted([e for e in entities if e not in result])
    print(f"\n{len(result)} entities resolved from diagnosis mapping, {len(remaining)} remaining.\n")

    # Stage 1: Exact match
    print("Stage 1: Exact match in Neo4j...")
    exact = exact_match_batch(driver, remaining)
    stage1_count = len(exact)
    for ent, name in exact.items():
        result[ent] = name
    remaining = [e for e in remaining if e not in exact]
    print(f"  -> {stage1_count} matched, {len(remaining)} remaining\n")

    # Stage 2: CUI linking
    print("Stage 2: scispaCy CUI linking...")
    cui = cui_match(driver, remaining, args.spacy_model)
    stage2_count = len(cui)
    for ent, name in cui.items():
        result[ent] = name
    remaining = [e for e in remaining if e not in cui]
    print(f"  -> {stage2_count} matched, {len(remaining)} remaining\n")

    # Stage 3: Fuzzy match (disabled — too noisy for radiology terms)
    stage3_count = 0
    print("Stage 3: Skipped (exact + CUI only for higher precision)\n")

    # Fill unmatched with null
    for ent in entities:
        if ent not in result:
            result[ent] = None

    driver.close()

    # --- Stats ---
    total = len(result)
    matched_count = sum(1 for v in result.values() if v is not None)
    null_count = total - matched_count
    print("=" * 50)
    print(f"Total entities:     {total}")
    print(f"  Diagnosis map:    {sum(1 for e in entities if e in diag_map and diag_map[e] is not None)}")
    print(f"  Stage 1 (exact):  {stage1_count}")
    print(f"  Stage 2 (CUI):    {stage2_count}")
    print(f"  Stage 3 (fuzzy):  {stage3_count}")
    print(f"  Matched total:    {matched_count}")
    print(f"  Unmatched (null):  {null_count}")
    print("=" * 50)

    # --- Write output ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
