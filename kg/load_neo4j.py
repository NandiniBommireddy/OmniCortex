#!/usr/bin/env python3
"""Load prepared PrimeKG CSVs into Neo4j via the Python driver.

Expects Neo4j to be running (e.g. via docker compose) with CSV files
mounted at /import/ inside the container.

Usage:
    python kg/load_neo4j.py [--uri bolt://localhost:7687] [--password primekg123]
"""

import argparse
import time

from neo4j import GraphDatabase

# Expected counts from the PrimeKG paper (Chandak et al., 2023)
EXPECTED_NODES = 129_375
EXPECTED_EDGES = 4_050_249

ALL_LABELS = [
    "disease", "gene__protein", "drug", "anatomy", "effect__phenotype",
    "biological_process", "molecular_function", "cellular_component",
    "exposure", "pathway",
]


def wait_for_connection(driver, max_retries=30, delay=2):
    """Wait for Neo4j to become available."""
    for i in range(max_retries):
        try:
            driver.verify_connectivity()
            print("Connected to Neo4j.")
            return
        except Exception:
            if i < max_retries - 1:
                print(f"  Waiting for Neo4j... ({i+1}/{max_retries})")
                time.sleep(delay)
    raise RuntimeError("Could not connect to Neo4j after retries.")


def load_nodes(session):
    """Load nodes from the mounted CSV.

    CSV columns: node_index, node_id, label, node_name, node_source
    """
    print("Loading nodes ...")
    # Every node gets both its specific label (e.g. disease) AND a shared
    # "Node" label so we can create a single index for fast edge lookups.
    result = session.run("""
        LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
        CALL {
            WITH row
            CALL apoc.create.node(['Node', row.label], {
                node_index: row.node_index,
                node_id: row.node_id,
                node_name: row.node_name,
                node_source: row.node_source
            }) YIELD node
            RETURN count(*) AS cnt
        } IN TRANSACTIONS OF 5000 ROWS
        RETURN sum(cnt) AS total_nodes
    """)
    record = result.single()
    total = record["total_nodes"] if record else 0
    print(f"  Loaded {total:,} nodes (expected ~{EXPECTED_NODES:,})")
    return total


def create_node_indexes(session):
    """Create index on the shared Node label — needed for fast edge loading."""
    print("Creating index on :Node(node_index) ...")
    session.run(
        "CREATE INDEX node_index_idx IF NOT EXISTS FOR (n:Node) ON (n.node_index)"
    )
    # Also create per-type indexes for query-time convenience
    for label in ALL_LABELS:
        session.run(
            f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.node_index)"
        )
    # Wait for all indexes to come online before loading edges
    session.run("CALL db.awaitIndexes(300)")
    print("  Indexes online.")


def load_edges(session):
    """Load edges from the mounted CSV.

    CSV columns: rel_type, display_relation, start_id, end_id
    """
    print("Loading edges (this will take several minutes) ...")
    result = session.run("""
        LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
        CALL {
            WITH row
            MATCH (a:Node {node_index: row.start_id})
            MATCH (b:Node {node_index: row.end_id})
            CALL apoc.create.relationship(a, row.rel_type, {
                display_relation: row.display_relation
            }, b) YIELD rel
            RETURN count(*) AS cnt
        } IN TRANSACTIONS OF 5000 ROWS
        RETURN sum(cnt) AS total_edges
    """)
    record = result.single()
    total = record["total_edges"] if record else 0
    print(f"  Loaded {total:,} edges (expected ~{EXPECTED_EDGES:,})")
    return total


def verify(session):
    """Verify node and edge counts match the paper."""
    print("\n=== Verification ===")
    node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    edge_count = session.run(
        "MATCH ()-[r]-() RETURN count(DISTINCT r) AS c"
    ).single()["c"]

    print(f"  Nodes: {node_count:,} (expected {EXPECTED_NODES:,})")
    print(f"  Edges: {edge_count:,} (expected {EXPECTED_EDGES:,})")

    if node_count == EXPECTED_NODES and edge_count == EXPECTED_EDGES:
        print("  All counts match the paper!")
    else:
        print("  WARNING: counts differ from paper — check data.")

    # Show label distribution
    print("\n=== Node Labels ===")
    records = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
    for r in records:
        count = session.run(
            f"MATCH (n:`{r['label']}`) RETURN count(n) AS c"
        ).single()["c"]
        print(f"  {r['label']}: {count:,}")

    # Show relationship types
    print("\n=== Relationship Types ===")
    records = session.run(
        "CALL db.relationshipTypes() YIELD relationshipType "
        "RETURN relationshipType ORDER BY relationshipType"
    )
    for r in records:
        print(f"  {r['relationshipType']}")


def main():
    parser = argparse.ArgumentParser(description="Load PrimeKG into Neo4j")
    parser.add_argument(
        "--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI"
    )
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="primekg123", help="Neo4j password")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification queries, skip loading",
    )
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    wait_for_connection(driver)

    with driver.session() as session:
        if not args.verify_only:
            load_nodes(session)
            create_node_indexes(session)
            load_edges(session)

        verify(session)

    driver.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
