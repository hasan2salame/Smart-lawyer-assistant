"""
retrieval/graph/builder.py
Builds the legal knowledge graph and writes graph.json.

Run once (or after data updates):
    python scripts/build_graph.py

Three edge types
----------------
[1] Article refs   — article explicitly cites another in its body text
[2] Topic mapping  — fonon templates linked to related statutory articles
[3] Adjacency      — numeric neighbours (article_num ± 1)

Design decisions
----------------
- bab/fasl metadata is corrupted (all articles collapse to one group)
  → replaced by numeric adjacency
- "lawsuit" (دعوى) removed from TOPIC_OSOUL — appears in 26/28 templates,
  creates noisy cross-links with no semantic value
- Cross-law refs are gated behind an explicit law name mention in the text
  → prevents false positives between the two corpora
"""

import json
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from config import (
    COL_LAWS, COL_OSOUL, COL_FONON,
    GRAPH_FILE, DATA_PROCESSED,
    QDRANT_URL, QDRANT_API_KEY,
)

load_dotenv()

# ── Topic → article-number mappings ──────────────────────────────────────

TOPIC_LAWS: dict[str, list[int]] = {
    "زواج":   [5, 6, 7, 8, 9, 10, 40, 41, 42],
    "مهر":    [52, 53, 54, 55, 56, 57, 58],
    "طلاق":   [85, 86, 87, 88, 89, 90, 91],
    "خلع":    [95, 96, 97, 98, 99, 100],
    "تفريق":  [105, 106, 107, 108, 109, 110],
    "نسب":    [120, 121, 122, 123, 124, 125],
    "حضانة":  [137, 138, 139, 140, 141, 146, 154],
    "رؤية":   [154, 155, 156],
    "نفقة":   [72, 73, 74, 75, 76, 77, 78],
    "وصاية":  [160, 161, 162, 163],
    "ولاية":  [163, 164, 165, 166, 167, 168, 169, 170],
    "وصية":   [200, 201, 202, 203, 204, 205],
    "ميراث":  [265, 270, 275, 280, 285, 290, 295, 300],
}

# "lawsuit" excluded — appears in 26/28 templates, generates noise
TOPIC_OSOUL: dict[str, list[int]] = {
    "تبليغ":   [10, 11, 12, 13, 14, 15],
    "إثبات":   [65, 66, 67, 68, 69, 70],
    "استئناف": [150, 151, 152, 153],
    "تنفيذ":   [280, 281, 282, 283],
    "اختصاص":  [30, 31, 32, 33, 34],
    "طعن":     [170, 171, 172, 173],
    "خبرة":    [72, 73, 74, 75],
    "حجز":     [290, 291, 292, 293],
}

# Explicit law names that indicate a cross-corpus citation
_OSOUL_NAMES = ["أصول المحاكمات", "قانون الأصول", "الإجراءات المدنية"]
_LAWS_NAMES  = ["الأحوال الشخصية", "قانون الأحوال"]

# Article number range for reference extraction
_ARTICLE_MIN, _ARTICLE_MAX = 1, 600

# ── Helpers ───────────────────────────────────────────────────────────────

def make_key(payload: dict) -> str:
    """Build a natural string key from a Qdrant point payload."""
    source = payload.get("source", "")
    if source == "fonon":
        return f"fonon_{payload.get('template_id', '')}"
    return f"{source}_{payload.get('article_num', '')}"


def _fetch_all(client: QdrantClient, collection: str) -> list:
    """Paginate through an entire Qdrant collection and return all points."""
    points, offset = [], None
    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(results)
        if next_offset is None:
            break
        offset = next_offset
    print(f"  fetched {collection}: {len(points)} points")
    return points


def _extract_refs(text: str) -> list[int]:
    """
    Extract explicitly cited article numbers from body text.
    Requires "المادة / المواد / م." prefix to avoid false positives.
    """
    nums: set[int] = set()
    for pattern in [r"المادة\s+(\d+)", r"المواد\s+(\d+)"]:
        for m in re.finditer(pattern, str(text)):
            n = int(m.group(1))
            if _ARTICLE_MIN <= n <= _ARTICLE_MAX:
                nums.add(n)
    for m in re.finditer(r"\bم\s*[./]\s*(\d+)\b", str(text)):
        n = int(m.group(1))
        if _ARTICLE_MIN <= n <= _ARTICLE_MAX:
            nums.add(n)
    return list(nums)


def _add_edge(graph: dict, src: str, dst: str) -> None:
    """Add a directed edge, ignoring self-loops and duplicates."""
    if src and dst and src != dst:
        if dst not in graph.get(src, []):
            graph.setdefault(src, []).append(dst)


def _build_num_index(points: list) -> dict[int, str]:
    """Map article_num → key for a list of points."""
    return {
        int(p.payload["article_num"]): make_key(p.payload)
        for p in points
        if p.payload.get("article_num") is not None
    }


# ── Graph construction ────────────────────────────────────────────────────

def _link_article_refs(
    graph: dict,
    points: list,
    self_idx: dict[int, str],
    other_idx: dict[int, str],
    cross_law_markers: list[str],
) -> None:
    """
    Edge type [1] — article explicitly cites another article.
    Cross-law refs are only added when the text contains an explicit
    law name mention (prevents false cross-corpus links).
    """
    for p in points:
        src_key = make_key(p.payload)
        text    = p.payload.get("original_text", "")
        cur_num = p.payload.get("article_num")
        is_cross = any(name in text for name in cross_law_markers)

        for ref in _extract_refs(text):
            if cur_num and ref == int(cur_num):
                continue
            if ref in self_idx:
                _add_edge(graph, src_key, self_idx[ref])
                _add_edge(graph, self_idx[ref], src_key)
            elif is_cross and ref in other_idx:
                _add_edge(graph, src_key, other_idx[ref])
                _add_edge(graph, other_idx[ref], src_key)


def _link_topic_mapping(
    graph: dict,
    fonon_points: list,
    laws_idx: dict[int, str],
    osoul_idx: dict[int, str],
) -> None:
    """
    Edge type [2] — fonon templates linked to related statutory articles
    via TOPIC_LAWS and TOPIC_OSOUL keyword mappings.
    """
    for p in fonon_points:
        fkey = make_key(p.payload)
        text = " ".join(filter(None, [
            p.payload.get("title", ""),
            p.payload.get("category", ""),
            p.payload.get("formal_text", ""),
            p.payload.get("intro_notes", ""),
        ]))
        for topic, nums in TOPIC_LAWS.items():
            if topic in text:
                for num in nums:
                    if num in laws_idx:
                        _add_edge(graph, fkey, laws_idx[num])
                        _add_edge(graph, laws_idx[num], fkey)
        for topic, nums in TOPIC_OSOUL.items():
            if topic in text:
                for num in nums:
                    if num in osoul_idx:
                        _add_edge(graph, fkey, osoul_idx[num])
                        _add_edge(graph, osoul_idx[num], fkey)


def _link_adjacency(
    graph: dict,
    *num_indices: dict[int, str],
) -> None:
    """Edge type [3] — numeric neighbours (article_num ± 1)."""
    for idx in num_indices:
        for num, key in idx.items():
            for nb in (num - 1, num + 1):
                if nb in idx:
                    _add_edge(graph, key, idx[nb])
                    _add_edge(graph, idx[nb], key)


def build_graph(client: QdrantClient) -> dict:
    """
    Fetch all collections from Qdrant and build the full knowledge graph.

    Returns
    -------
    dict  — {key: [neighbour_key, ...]} adjacency list
    """
    print("\n[1] Fetching data from Qdrant...")
    laws_pts  = _fetch_all(client, COL_LAWS)
    osoul_pts = _fetch_all(client, COL_OSOUL)
    fonon_pts = _fetch_all(client, COL_FONON)

    # Initialise all nodes (ensures isolated nodes appear in the graph)
    graph: dict = {}
    for p in laws_pts + osoul_pts + fonon_pts:
        key = make_key(p.payload)
        if key:
            graph.setdefault(key, [])

    laws_idx  = _build_num_index(laws_pts)
    osoul_idx = _build_num_index(osoul_pts)
    print(f"  laws: {len(laws_idx)} articles, osoul: {len(osoul_idx)} articles")

    print("\n[2] Building article reference edges...")
    _link_article_refs(graph, laws_pts,  laws_idx,  osoul_idx, _OSOUL_NAMES)
    _link_article_refs(graph, osoul_pts, osoul_idx, laws_idx,  _LAWS_NAMES)

    print("\n[3] Building topic mapping edges (fonon → statutory)...")
    _link_topic_mapping(graph, fonon_pts, laws_idx, osoul_idx)

    print("\n[4] Building numeric adjacency edges (± 1)...")
    _link_adjacency(graph, laws_idx, osoul_idx)

    return graph


def report(graph: dict) -> None:
    """Print graph statistics to stdout."""
    total_nodes = len(graph)
    total_edges = sum(len(v) for v in graph.values())
    isolated    = sum(1 for v in graph.values() if not v)
    top5 = sorted(graph.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    print(f"\n{'=' * 50}")
    print(f"  Nodes      : {total_nodes}")
    print(f"  Edges      : {total_edges}")
    print(f"  Isolated   : {isolated}")
    print(f"  Avg degree : {total_edges / max(total_nodes, 1):.1f}")
    print("\n  Most connected articles:")
    for key, neighbours in top5:
        print(f"    {key:20} — {len(neighbours)} edges")
    print("=" * 50)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    _client = QdrantClient(
        url=os.getenv("QDRANT_URL", ""),
        api_key=os.getenv("QDRANT_API_KEY", ""),
        timeout=60,
    )

    graph = build_graph(_client)
    report(graph)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"\nSaved → {GRAPH_FILE}")
