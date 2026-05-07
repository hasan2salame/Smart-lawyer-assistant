#!/usr/bin/env python3
"""
scripts/upload_osoul.py
Upload Syrian Civil Procedure Law articles to Qdrant.

Usage
-----
    python scripts/upload_osoul.py
    # or via Makefile:
    make upload-osoul
"""

import json
import time
import uuid
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY

INPUT_FILE      = "data/processed/osoul.json"
COLLECTION_NAME = "legal_osoul"
BATCH_SIZE      = 30
SLEEP_SECS      = 35

Settings.embed_model = CohereEmbedding(
    api_key=COHERE_API_KEY,
    model_name="embed-multilingual-v3.0",
    input_type="search_document",
)
Settings.llm           = None
Settings.chunk_size    = 8192   # keeps each article as a single vector node
Settings.chunk_overlap = 0


def main() -> None:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
        print(f"Dropped: {COLLECTION_NAME}")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print(f"Created: {COLLECTION_NAME}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} articles")

    docs: list[Document] = []
    for item in data:
        meta       = item["metadata"]
        topics_str = " | ".join(meta.get("topics", []))

        embedding_text = (
            f"{item['article_number_str']} — {meta.get('law_name', '')}\n"
            f"{item['text']}"
        )

        docs.append(Document(
            doc_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"osoul_v2_{item['id']}")),
            text=embedding_text,
            metadata={
                "source":        "osoul",
                "article_str":   item["article_number_str"],
                "article_num":   item["article_number"],
                "law_name":      meta.get("law_name", ""),
                "law_year":      meta.get("law_year", 0),
                "bab":           meta.get("bab", ""),
                "fasl":          meta.get("fasl", ""),
                "fara":          meta.get("fara", ""),
                "topics":        topics_str,
                "original_text": item["text"],
            },
            excluded_embed_metadata_keys=[
                "source", "article_num", "law_year", "bab", "fasl", "fara",
                "topics", "original_text",
            ],
            excluded_llm_metadata_keys=["source", "article_num", "law_year", "topics"],
        ))

    vector_store    = QdrantVectorStore(client=qdrant, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    total         = len(docs)
    uploaded      = 0
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, total, BATCH_SIZE):
        batch     = docs[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        for attempt in range(3):
            try:
                VectorStoreIndex.from_documents(
                    batch, storage_context=storage_context, show_progress=False,
                )
                uploaded += len(batch)
                info = qdrant.get_collection(COLLECTION_NAME)
                print(f"  batch {batch_num}/{total_batches} "
                      f"— {uploaded}/{total} uploaded "
                      f"({info.points_count} in collection)")
                break
            except Exception as exc:
                wait = 15 * (attempt + 1)
                print(f"  attempt {attempt + 1}: {str(exc)[:100]} — retrying in {wait}s")
                time.sleep(wait)
        else:
            print(f"  skipping batch {batch_num} after 3 failed attempts")

        if i + BATCH_SIZE < total:
            time.sleep(SLEEP_SECS)

    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"\nlegal_osoul — {info.points_count} points")
    if info.points_count == total:
        print("  OK — one point per article")
    elif info.points_count < total:
        print(f"  WARNING: {total - info.points_count} articles missing — re-run script")


if __name__ == "__main__":
    main()
