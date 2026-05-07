"""
scripts/build_graph.py
Build the legal knowledge graph and save it to data/graph.json.

Run once after uploading data to Qdrant, or whenever data is updated.

Usage
-----
    python scripts/build_graph.py
    # or via Makefile:
    make build-graph
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY, GRAPH_FILE
from retrieval.graph.builder import build_graph, report


def main() -> None:
    print("\n" + "=" * 50)
    print("  Building legal knowledge graph from Qdrant")
    print("=" * 50)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

    graph = build_graph(client)

    GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved: {GRAPH_FILE}")
    report(graph)
    print("\n  Restart the server to load the new graph.\n")


if __name__ == "__main__":
    main()
