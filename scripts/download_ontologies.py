#!/usr/bin/env python3
"""Download RadLex and Radiology Gamuts ontologies from BioPortal.

Requires a free BioPortal API key (register at https://bioportal.bioontology.org/accounts/new).

Usage:
    python scripts/download_ontologies.py --api-key YOUR_KEY
    python scripts/download_ontologies.py --api-key-env BIOPORTAL_API_KEY
"""

import argparse
import os
import sys
from pathlib import Path

import requests

ONTOLOGIES = {
    "radlex": {
        "id": "RADLEX",
        "filename": "radlex.owl",
        "subdir": "radlex",
        "description": "RadLex Radiology Lexicon (~34K concepts)",
    },
    "gamuts": {
        "id": "GAMUTS",
        "filename": "gamuts.owl",
        "subdir": "gamuts",
        "description": "Radiology Gamuts Ontology (~17K classes, 55K causal relations)",
    },
}

BIOPORTAL_DOWNLOAD_URL = (
    "https://data.bioontology.org/ontologies/{ont_id}/download"
    "?apikey={api_key}&download_format=csv"
)


def download_ontology(ont_id: str, api_key: str, dest: Path) -> bool:
    """Download an ontology from BioPortal. Returns True on success."""
    # First try the OWL download endpoint
    url = f"https://data.bioontology.org/ontologies/{ont_id}/download?apikey={api_key}"
    print(f"  Requesting {url[:80]}...")

    resp = requests.get(url, stream=True, timeout=120)
    if resp.status_code != 200:
        print(f"  ERROR: HTTP {resp.status_code} — {resp.text[:200]}")
        return False

    content_type = resp.headers.get("Content-Type", "")
    content_length = resp.headers.get("Content-Length", "unknown")
    print(f"  Content-Type: {content_type}, Size: {content_length}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)

    size_mb = total / (1024 * 1024)
    print(f"  Saved to {dest} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download RadLex and Gamuts from BioPortal")
    parser.add_argument("--api-key", type=str, default=None,
                        help="BioPortal API key")
    parser.add_argument("--api-key-env", type=str, default="BIOPORTAL_API_KEY",
                        help="Environment variable holding the API key (default: BIOPORTAL_API_KEY)")
    parser.add_argument("--output-dir", type=Path, default=Path("kg/data"),
                        help="Base output directory (default: kg/data)")
    parser.add_argument("--only", type=str, choices=["radlex", "gamuts"], default=None,
                        help="Download only one ontology")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        print(
            "ERROR: No API key provided.\n"
            "  Register at https://bioportal.bioontology.org/accounts/new\n"
            "  Then: python scripts/download_ontologies.py --api-key YOUR_KEY\n"
            "  Or:   export BIOPORTAL_API_KEY=YOUR_KEY"
        )
        sys.exit(1)

    targets = [args.only] if args.only else list(ONTOLOGIES.keys())

    for name in targets:
        ont = ONTOLOGIES[name]
        dest = args.output_dir / ont["subdir"] / ont["filename"]
        if dest.exists():
            print(f"[{name}] Already exists: {dest} — skipping (delete to re-download)")
            continue
        print(f"[{name}] Downloading {ont['description']}...")
        ok = download_ontology(ont["id"], api_key, dest)
        if not ok:
            print(f"[{name}] FAILED")
            sys.exit(1)
        print(f"[{name}] Done.\n")

    print("All ontologies downloaded.")


if __name__ == "__main__":
    main()
