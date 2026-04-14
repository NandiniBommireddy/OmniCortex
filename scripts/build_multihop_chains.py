#!/usr/bin/env python3
"""Build 2-hop reasoning chains from RadGraph triplets via Neo4j.

For each image's RadGraph triplets, maps entity objects to PrimeKG nodes
using the entity CUI map, then queries Neo4j for 2-hop chains.

Traversal strategy:
  - Disease start nodes: disease → disease → disease/phenotype (stays in
    common disease space, avoids rare-disease phenotype explosion)
  - Phenotype start nodes: phenotype → phenotype only (avoids going
    phenotype → rare disease → unrelated phenotypes)
  - Other node types (gene, drug, etc.): skipped

Usage:
    python scripts/build_multihop_chains.py \
        --input tmp/demo/mimic-nle-train-radgraph.json \
        --entity-map data/entity_cui_map.json \
        --output data/radgraph-multihop.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

ALLOWED_EDGES = [
    "disease_disease",
    "phenotype_phenotype",
]

# Filter out high-degree hub nodes (e.g. "Mendelian disease" with 1524 edges)
# that add no clinical value.
MAX_NEIGHBOR_DEGREE = 200

CYPHER_1HOP = """
MATCH (n:Node {node_name: $node_name})-[r]-(neighbor)
WHERE type(r) IN $allowed_edges
  AND COUNT { (neighbor)-[]-() } < $max_degree
RETURN n.node_name AS start, type(r) AS edge, neighbor.node_name AS neighbor
LIMIT 10
"""


def parse_triplet(triplet_str: str):
    """Parse 'subject suggestive_of object' into (subj, rel, obj)."""
    sep = " suggestive of "
    if sep in triplet_str:
        idx = triplet_str.index(sep)
        subj = triplet_str[:idx].strip()
        obj = triplet_str[idx + len(sep) :].strip()
        return subj, "suggestive_of", obj
    return triplet_str, None, None


def query_neighbors(session, node_name: str, cache: dict) -> list[dict]:
    """Query Neo4j for direct neighbors of a node, with caching."""
    if node_name in cache:
        return cache[node_name]

    records = session.run(
        CYPHER_1HOP, node_name=node_name,
        allowed_edges=ALLOWED_EDGES, max_degree=MAX_NEIGHBOR_DEGREE,
    )
    results = [dict(r) for r in records]
    cache[node_name] = results
    return results


def build_chain_string(subj, rel, obj, row):
    """Build a chain string from the RadGraph triplet + Neo4j neighbor."""
    return (
        f"{subj} --{rel}--> {obj} "
        f"--{row['edge']}--> {row['neighbor']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build 2-hop reasoning chains from RadGraph triplets via Neo4j"
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("tmp/demo/mimic-nle-train-radgraph.json"),
        help="RadGraph triplets file (JSONL)",
    )
    parser.add_argument(
        "--entity-map", type=Path,
        default=Path("data/entity_cui_map.json"),
        help="Entity-to-PrimeKG node name mapping (JSON)",
    )
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-password", default="primekg123")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/radgraph-multihop.jsonl"),
        help="Output JSONL file",
    )
    args = parser.parse_args()

    # Load entity map
    print(f"Loading entity map from {args.entity_map} ...")
    with open(args.entity_map) as f:
        entity_map = json.load(f)
    mapped = sum(1 for v in entity_map.values() if v is not None)
    print(f"  {mapped}/{len(entity_map)} entities have PrimeKG mappings")

    # Load input rows
    print(f"Loading RadGraph data from {args.input} ...")
    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"  {len(rows)} rows loaded")

    # Connect to Neo4j
    print(f"Connecting to Neo4j at {args.neo4j_uri} ...")
    driver = GraphDatabase.driver(
        args.neo4j_uri, auth=("neo4j", args.neo4j_password)
    )
    driver.verify_connectivity()
    print("  Connected.")

    # Process each row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chain_cache = {}
    total_chains = 0
    rows_with_chains = 0

    with driver.session() as session, open(args.output, "w") as out:
        for i, row in enumerate(rows):
            triplets = row.get("triplets", [])
            img_id = row.get("sentence_ID", row.get("report_ID", f"row_{i}"))
            chains = []
            seen_edges = set()  # deduplicate by (entity, edge, neighbor)

            for triplet_str in triplets:
                subj, rel, obj = parse_triplet(triplet_str)
                if rel is None:
                    continue

                node_name = entity_map.get(obj) or entity_map.get(obj.lower())
                if node_name is None:
                    continue

                neighbors = query_neighbors(session, node_name, chain_cache)
                for nbr in neighbors:
                    dedup_key = (node_name, nbr["edge"], nbr["neighbor"])
                    if dedup_key in seen_edges:
                        continue
                    seen_edges.add(dedup_key)
                    chain_str = build_chain_string(subj, rel, obj, nbr)
                    chains.append(chain_str)

            record = {
                "img_id": img_id,
                "triplets": triplets,
                "chains": chains,
            }
            out.write(json.dumps(record) + "\n")

            total_chains += len(chains)
            if chains:
                rows_with_chains += 1

            if (i + 1) % 500 == 0:
                print(
                    f"  [{i+1}/{len(rows)}] "
                    f"chains so far: {total_chains:,}, "
                    f"cache size: {len(chain_cache)}"
                )

    driver.close()

    print(f"\n{'='*50}")
    print(f"Total rows:        {len(rows):,}")
    print(f"Rows with chains:  {rows_with_chains:,} ({100*rows_with_chains/len(rows):.1f}%)")
    print(f"Total chains:      {total_chains:,}")
    print(f"Cache entries:     {len(chain_cache):,}")
    print(f"Written to:        {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
