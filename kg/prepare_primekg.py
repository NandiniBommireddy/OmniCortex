#!/usr/bin/env python3
"""Download and preprocess PrimeKG data for Neo4j import.

Steps:
  1. Download nodes.csv and edges.csv from Harvard Dataverse
  2. Reformat CSV headers to Neo4j import format
  3. Clean node labels (replace '/' with '__')
  4. Deduplicate directed edges into undirected pairs

Usage:
    python kg/prepare_primekg.py [--data-dir kg/data] [--skip-download]
"""

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd

# Harvard Dataverse file IDs for PrimeKG v2.1
# IDs from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
DATAVERSE_BASE = "https://dataverse.harvard.edu/api/access/datafile/"
FILES = {
    "nodes.tab": "6180617",
    "edges.csv": "6180616",
    "kg.csv": "6180620",
}


def download_file(url: str, dest: Path, desc: str = "") -> None:
    if dest.exists():
        print(f"  [skip] {dest} already exists")
        return
    print(f"  Downloading {desc or dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fSL", "-o", str(dest), url],
        check=True,
    )
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"  [done] {dest} ({size_mb:.1f} MB)")


def download_primekg(data_dir: Path) -> None:
    """Download raw PrimeKG files from Harvard Dataverse."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, file_id in FILES.items():
        url = f"{DATAVERSE_BASE}{file_id}"
        download_file(url, raw_dir / filename, desc=filename)


def reformat_nodes(data_dir: Path) -> Path:
    """Reformat nodes.csv headers for Neo4j and clean labels."""
    raw_path = data_dir / "raw" / "nodes.tab"
    out_path = data_dir / "neo4j" / "nodes.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Processing nodes ...")
    # .tab files from Dataverse are tab-delimited
    sep = "\t" if raw_path.suffix == ".tab" else ","
    df = pd.read_csv(raw_path, sep=sep, dtype=str, keep_default_na=False)

    # Rename columns to clean names (no colons — avoids Cypher escaping issues)
    df.columns = ["node_index", "node_id", "label", "node_name", "node_source"]

    # Neo4j label restriction: replace '/' with '__'
    df["label"] = df["label"].str.replace("/", "__", regex=False)

    # Sanitize fields: remove embedded quotes that break Neo4j's CSV parser
    for col in df.columns:
        df[col] = df[col].str.replace('"', "", regex=False)

    df.to_csv(out_path, index=False, quoting=1)  # QUOTE_ALL for safety
    print(f"  [done] {out_path} ({len(df)} nodes)")
    return out_path


def reformat_and_dedup_edges(data_dir: Path) -> Path:
    """Reformat edges.csv headers and deduplicate bidirectional pairs."""
    raw_path = data_dir / "raw" / "edges.csv"
    out_path = data_dir / "neo4j" / "edges.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Processing edges.csv (this may take a minute) ...")
    df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)

    # Rename columns to clean names (no colons — avoids Cypher escaping issues)
    df.columns = ["rel_type", "display_relation", "start_id", "end_id"]

    # Deduplicate: the raw file has both (A->B) and (B->A) for each edge.
    # Group by frozenset of {start, end, type, relation} and keep first.
    print("  Deduplicating directed edges into undirected pairs ...")
    original_count = len(df)
    group_key = df[["start_id", "end_id", "rel_type", "display_relation"]].agg(
        frozenset, axis=1
    )
    df = df.groupby(group_key).first().reset_index(drop=True)
    print(f"  Reduced {original_count:,} -> {len(df):,} edges")

    df.to_csv(out_path, index=False)
    print(f"  [done] {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare PrimeKG data for Neo4j")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("kg/data"),
        help="Directory for PrimeKG data (default: kg/data)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, assume raw files already exist",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    print(f"Data directory: {data_dir}\n")

    # Step 1: Download
    if not args.skip_download:
        print("=== Step 1: Download PrimeKG from Harvard Dataverse ===")
        download_primekg(data_dir)
    else:
        print("=== Step 1: Download skipped ===")

    # Step 2: Reformat nodes
    print("\n=== Step 2: Reformat nodes for Neo4j ===")
    reformat_nodes(data_dir)

    # Step 3: Reformat & deduplicate edges
    print("\n=== Step 3: Reformat & deduplicate edges for Neo4j ===")
    reformat_and_dedup_edges(data_dir)

    print("\n=== Done! Files ready for Neo4j import in: ===")
    print(f"  {data_dir / 'neo4j' / 'nodes.csv'}")
    print(f"  {data_dir / 'neo4j' / 'edges.csv'}")


if __name__ == "__main__":
    main()
