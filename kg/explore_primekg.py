#!/usr/bin/env python3
"""Explore PrimeKG in Neo4j with Cypher queries for multi-hop analysis.

Runs a suite of queries useful for the KG-LLaVA multi-hop extension:
  - Schema overview (labels, relationship types)
  - Seed diagnosis lookup (10 MIMIC-NLE diagnoses)
  - Multi-hop traversals (1-hop, 2-hop chains)
  - Radiology subgraph export

Usage:
    python kg/explore_primekg.py [--uri bolt://localhost:7687] [--export-subgraph]
"""

import argparse
import csv
import json
from pathlib import Path

from neo4j import GraphDatabase

# 10 MIMIC-NLE seed diagnoses mapped to PrimeKG search terms.
# PrimeKG uses formal disease names; MIMIC uses radiology finding labels.
# Each diagnosis maps to a list of alternative terms to broaden coverage.
DIAGNOSIS_SEARCH_TERMS = {
    "Atelectasis": ["atelectasis", "lung collapse", "collapsed lung"],
    "Consolidation": ["consolidation", "lung consolidation", "airspace disease"],
    "Edema": ["pulmonary edema", "lung edema"],
    "Enlarged Cardiomediastinum": ["cardiomegaly", "cardiac enlargement", "mediastinal"],
    "Lung Lesion": ["lung neoplasm", "pulmonary nodule", "lung mass", "lung tumor"],
    "Lung Opacity": ["pulmonary opacity", "ground glass", "lung infiltrate"],
    "Pleural Effusion": ["pleural effusion", "hydrothorax", "pleural fluid"],
    "Pleural Other": ["pleural disease", "pleurisy", "pleural thickening", "pleural"],
    "Pneumonia": ["pneumonia"],
    "Pneumothorax": ["pneumothorax"],
}

# Edge types relevant to multi-hop reasoning in radiology
MULTIHOP_EDGE_TYPES = [
    "disease_phenotype_positive",
    "disease_phenotype_negative",
    "disease_protein",
    "disease_disease",
    "phenotype_phenotype",
    "drug_protein",
    "indication",
    "contraindication",
]


def _find_nodes(session, search_terms: list[str], limit_per_term: int = 20):
    """Search all PrimeKG nodes using multiple alternative terms.

    Returns deduplicated list of (node_index, node_name, matched_term) tuples.
    Searches across all node types (disease, effect__phenotype, anatomy, etc.)
    since radiology findings may exist under any label.
    """
    seen = set()
    results = []
    for term in search_terms:
        records = session.run(
            "MATCH (d:Node) "
            "WHERE toLower(d.node_name) CONTAINS $name "
            "RETURN d.node_index AS idx, d.node_name AS name "
            "ORDER BY d.node_name LIMIT $lim",
            name=term.lower(),
            lim=limit_per_term,
        )
        for r in records:
            if r["idx"] not in seen:
                seen.add(r["idx"])
                results.append((r["idx"], r["name"], term))
    return results


def _build_where_clause(session) -> str:
    """Build a Cypher WHERE clause matching all disease nodes across all search terms."""
    all_ids = []
    for terms in DIAGNOSIS_SEARCH_TERMS.values():
        for idx, _, _ in _find_nodes(session, terms):
            all_ids.append(idx)
    all_ids = list(set(all_ids))
    if not all_ids:
        return "FALSE"
    return "d.node_index IN [" + ", ".join(f"'{i}'" for i in all_ids) + "]"


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def schema_overview(session):
    """Show all node labels and relationship types."""
    print_header("Schema Overview")

    print("\nNode Labels:")
    records = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
    for r in records:
        count = session.run(
            f"MATCH (n:`{r['label']}`) RETURN count(n) AS c"
        ).single()["c"]
        print(f"  {r['label']:30s} {count:>10,}")

    print("\nRelationship Types:")
    records = session.run(
        "CALL db.relationshipTypes() YIELD relationshipType "
        "RETURN relationshipType ORDER BY relationshipType"
    )
    for r in records:
        rtype = r["relationshipType"]
        count = session.run(
            f"MATCH ()-[r:`{rtype}`]-() RETURN count(r) AS c"
        ).single()["c"]
        print(f"  {rtype:40s} {count:>10,}")


def find_seed_diagnoses(session):
    """Search for the 10 MIMIC-NLE seed diagnoses in PrimeKG using alternative terms."""
    print_header("Seed Diagnosis Lookup")

    total_matched = 0
    for diagnosis, terms in DIAGNOSIS_SEARCH_TERMS.items():
        matches = _find_nodes(session, terms)
        total_matched += len(matches)
        status = f"({len(matches)} match{'es' if len(matches) != 1 else ''})"
        print(f"\n  '{diagnosis}' {status}  [terms: {', '.join(terms)}]")
        for idx, name, term in matches[:10]:
            print(f"    [{idx}] {name}  (via \"{term}\")")
        if not matches:
            print(f"    NO MATCH — will need scispaCy/CUI mapping in build_entity_cui_map.py")
    print(f"\n  Total: {total_matched} unique disease nodes matched across all diagnoses")


def multihop_exploration(session):
    """Explore multi-hop chains from seed diagnoses with best-matched nodes."""
    print_header("Multi-Hop Exploration")

    # Pick up to 3 diagnoses that have matches for demonstration
    sample = []
    for diagnosis, terms in DIAGNOSIS_SEARCH_TERMS.items():
        matches = _find_nodes(session, terms, limit_per_term=5)
        if matches:
            # Pick the most specific match (shortest name = most direct)
            best = min(matches, key=lambda m: len(m[1]))
            sample.append((diagnosis, best[0], best[1]))
        if len(sample) >= 3:
            break

    for diagnosis, node_idx, node_name in sample:
        print(f"\n--- {diagnosis.upper()} (matched: {node_name} [{node_idx}]) ---")

        # 1-hop: disease -> phenotype
        print(f"\n  1-hop (disease -> phenotype):")
        records = session.run(
            "MATCH (d:disease)-[:disease_phenotype_positive]-(p:effect__phenotype) "
            "WHERE d.node_index = $idx "
            "RETURN d.node_name AS disease, p.node_name AS phenotype "
            "LIMIT 10",
            idx=node_idx,
        )
        for r in records:
            print(f"    {r['disease']}  -->  {r['phenotype']}")

        # 2-hop: disease -> phenotype -> phenotype
        print(f"\n  2-hop (disease -> phenotype -> phenotype):")
        records = session.run(
            "MATCH (d:disease)-[:disease_phenotype_positive]-(p1:effect__phenotype)"
            "-[:phenotype_phenotype]-(p2:effect__phenotype) "
            "WHERE d.node_index = $idx "
            "RETURN d.node_name AS disease, p1.node_name AS hop1, "
            "       p2.node_name AS hop2 "
            "LIMIT 10",
            idx=node_idx,
        )
        for r in records:
            print(f"    {r['disease']}  -->  {r['hop1']}  -->  {r['hop2']}")

        # 1-hop: disease -> protein
        print(f"\n  1-hop (disease -> protein):")
        records = session.run(
            "MATCH (d:disease)-[:disease_protein]-(g:gene__protein) "
            "WHERE d.node_index = $idx "
            "RETURN d.node_name AS disease, g.node_name AS protein "
            "LIMIT 10",
            idx=node_idx,
        )
        for r in records:
            print(f"    {r['disease']}  -->  {r['protein']}")

        # 1-hop: disease -> disease
        print(f"\n  1-hop (disease -> disease):")
        records = session.run(
            "MATCH (d1:disease)-[:disease_disease]-(d2:disease) "
            "WHERE d1.node_index = $idx "
            "RETURN d1.node_name AS disease1, d2.node_name AS disease2 "
            "LIMIT 10",
            idx=node_idx,
        )
        for r in records:
            print(f"    {r['disease1']}  -->  {r['disease2']}")

        # Reachability summary
        record = session.run(
            "MATCH (d:disease) WHERE d.node_index = $idx "
            "OPTIONAL MATCH (d)-[*1..2]-(target) "
            "RETURN d.node_name AS disease, count(DISTINCT target) AS reachable",
            idx=node_idx,
        ).single()
        if record:
            print(f"\n  Reachable within 2 hops: {record['reachable']:,} nodes")


def exposure_analysis(session):
    """Replicate the article's exposure-disease query for validation."""
    print_header("Exposure-Disease Analysis (article validation)")

    print("\n  Diabetes ↔ Exposure:")
    records = session.run(
        "MATCH (e:exposure)-[:exposure_disease]-(d:disease) "
        "WHERE d.node_name CONTAINS 'diabetes' "
        "RETURN d.node_name AS disease, e.node_name AS exposure "
        "ORDER BY d.node_name, e.node_name LIMIT 20"
    )
    for r in records:
        print(f"    {r['disease']}  <-->  {r['exposure']}")


def export_radiology_subgraph(session, output_dir: Path):
    """Export the radiology-relevant subgraph for offline pipeline use.

    Uses DIAGNOSIS_SEARCH_TERMS (alternative terms) to find all matching
    disease nodes, then exports their 2-hop neighborhoods.
    """
    print_header("Exporting Radiology Subgraph")

    output_dir.mkdir(parents=True, exist_ok=True)

    edges_path = output_dir / "primekg_radiology_subgraph.csv"
    nodes_path = output_dir / "primekg_radiology_nodes.json"
    mapping_path = output_dir / "diagnosis_node_mapping.json"

    # Step 1: Find all matching disease node IDs via alternative terms
    diagnosis_mapping = {}
    all_disease_ids = set()
    for diagnosis, terms in DIAGNOSIS_SEARCH_TERMS.items():
        matches = _find_nodes(session, terms)
        diagnosis_mapping[diagnosis] = [
            {"node_index": idx, "node_name": name, "matched_term": term}
            for idx, name, term in matches
        ]
        for idx, _, _ in matches:
            all_disease_ids.add(idx)

    matched_count = sum(len(v) for v in diagnosis_mapping.values())
    print(f"  Found {matched_count} disease nodes across {len(DIAGNOSIS_SEARCH_TERMS)} diagnoses")
    for dx, matches in diagnosis_mapping.items():
        print(f"    {dx}: {len(matches)} nodes")

    # Save the diagnosis-to-node mapping for downstream pipeline use
    with open(mapping_path, "w") as f:
        json.dump(diagnosis_mapping, f, indent=2)
    print(f"  [done] {mapping_path}")

    if not all_disease_ids:
        print("  WARNING: No disease nodes matched — subgraph will be empty.")
        return

    # Step 2: Export 2-hop neighborhood of all matched disease nodes
    id_list = list(all_disease_ids)
    print(f"  Querying 2-hop neighborhood of {len(id_list)} disease nodes ...")

    # Use [labels(n) | l WHERE l <> 'Node'] to filter out the shared Node label
    records = session.run(
        "MATCH (d:disease) WHERE d.node_index IN $ids "
        "MATCH path = (d)-[*1..2]-(target) "
        "UNWIND relationships(path) AS rel "
        "WITH DISTINCT startNode(rel) AS src, rel, endNode(rel) AS dst "
        "RETURN src.node_index AS src_index, src.node_name AS src_name, "
        "       [l IN labels(src) WHERE l <> 'Node'][0] AS src_label, "
        "       type(rel) AS rel_type, "
        "       dst.node_index AS dst_index, dst.node_name AS dst_name, "
        "       [l IN labels(dst) WHERE l <> 'Node'][0] AS dst_label",
        ids=id_list,
    )

    rows = list(records)
    print(f"  Found {len(rows):,} edges in subgraph")

    # Write edges CSV
    with open(edges_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "src_index", "src_name", "src_label",
                "rel_type",
                "dst_index", "dst_name", "dst_label",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))
    print(f"  [done] {edges_path}")

    # Collect unique nodes
    node_set = {}
    for r in rows:
        for prefix in ("src", "dst"):
            idx = r[f"{prefix}_index"]
            if idx not in node_set:
                node_set[idx] = {
                    "node_index": idx,
                    "node_name": r[f"{prefix}_name"],
                    "label": r[f"{prefix}_label"],
                }

    with open(nodes_path, "w") as f:
        json.dump(list(node_set.values()), f, indent=2)
    print(f"  [done] {nodes_path} ({len(node_set):,} unique nodes)")


def main():
    parser = argparse.ArgumentParser(description="Explore PrimeKG in Neo4j")
    parser.add_argument(
        "--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI"
    )
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="primekg123", help="Neo4j password")
    parser.add_argument(
        "--export-subgraph",
        action="store_true",
        help="Export the radiology subgraph to kg/data/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kg/data/subgraph"),
        help="Output directory for subgraph export",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only show schema overview",
    )
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    with driver.session() as session:
        schema_overview(session)

        if not args.schema_only:
            find_seed_diagnoses(session)
            multihop_exploration(session)
            exposure_analysis(session)

        if args.export_subgraph:
            export_radiology_subgraph(session, args.output_dir)

    driver.close()
    print("\nExploration complete!")


if __name__ == "__main__":
    main()
