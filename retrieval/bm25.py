"""
retrieval/bm25.py
BM25Okapi sparse retriever built without external libraries.

The index is built once at startup from the in-memory payload_index.
COL_FONON is intentionally excluded — template lookup uses Rerank-all,
so a BM25 index over 28 short titles would add noise without value.
"""

import math
import re
from collections import Counter
from typing import Literal

from config import COL_LAWS, COL_OSOUL, TOP_K_BM25

# Collection type accepted by search()
Collection = Literal["legal_laws", "legal_osoul"]

# ── Arabic stop-words ─────────────────────────────────────────────────────
_STOP_WORDS: frozenset[str] = frozenset({
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "التي", "الذي",
    "ذلك", "تلك", "هو", "هي", "هم", "هن", "نحن", "انتم", "كان", "كانت",
    "يكون", "تكون", "قد", "لقد", "لا", "لم", "لن", "إن", "أن", "بأن",
    "أو", "و", "ف", "ثم", "حتى", "إذا", "إذ", "عند", "بين", "بعد", "قبل",
    "أي", "كل", "بما", "مما", "وفق", "وفقا", "حيث", "كما",
})


def _tokenize(text: str) -> list[str]:
    """
    Normalise and tokenise Arabic text for BM25.

    Pipeline
    --------
    1. Remove tashkeel and tatweel.
    2. Normalise hamza, ta marbuta, alef maqsura.
    3. Keep only Arabic characters and spaces.
    4. Split on whitespace and remove stop-words and short tokens.
    """
    # Remove tashkeel (U+064B – U+065F) and tatweel (U+0640)
    text = re.sub(r"[\u064B-\u065F\u0640]", "", text)

    # Normalise hamza variants → bare alef
    text = re.sub(r"[أإآ]", "ا", text)

    # Normalise ta marbuta → ha, alef maqsura → ya
    text = text.replace("ة", "ه").replace("ى", "ي")

    # Keep Arabic letters and spaces only
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)

    tokens = text.split()
    return [t for t in tokens if len(t) > 2 and t not in _STOP_WORDS]


class BM25Searcher:
    """
    BM25Okapi index built from the full legal corpus at startup.

    Only COL_LAWS and COL_OSOUL are indexed.  COL_FONON is skipped because
    template retrieval uses a Rerank-all strategy (see HybridRetriever).

    Parameters
    ----------
    payload_index : dict  — {key: payload} mapping from HybridRetriever
    corpus_keys   : dict  — {collection_name: [key, ...]} from HybridRetriever
    k1            : float — BM25 term-frequency saturation (default 1.5)
    b             : float — BM25 document-length normalisation (default 0.75)
    """

    def __init__(
        self,
        payload_index: dict,
        corpus_keys: dict,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b  = b
        self._payload_index = payload_index

        # Build one index per legal collection (fonon excluded)
        self._indices: dict[str, dict] = {}
        for col in (COL_LAWS, COL_OSOUL):
            keys = corpus_keys.get(col, [])
            self._indices[col] = self._build(keys)

        total = sum(len(v["keys"]) for v in self._indices.values())
        print(f"[BM25Searcher] Indexed {total} documents "
              f"({COL_LAWS}: {len(self._indices[COL_LAWS]['keys'])}, "
              f"{COL_OSOUL}: {len(self._indices[COL_OSOUL]['keys'])})")

    # ── Public API ────────────────────────────────────────────────────────

    def search(self, query: str, collection: Collection) -> list[dict]:
        """
        Return up to TOP_K_BM25 results from the specified collection,
        sorted by BM25 score descending.

        Returns an empty list if the collection is not indexed or the
        query produces no tokens.
        """
        index = self._indices.get(collection)
        if not index:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._score(tokens, index)
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_BM25]

        return [
            {
                "key":     key,
                "payload": self._payload_index[key],
                "score":   score,
                "source":  "bm25",
            }
            for key, score in top
            if score > 0
        ]

    # ── Internal ──────────────────────────────────────────────────────────

    def _build(self, keys: list[str]) -> dict:
        """Build BM25 index data for a list of document keys."""
        tokenized: list[list[str]] = []
        for key in keys:
            payload = self._payload_index.get(key, {})
            text    = payload.get("original_text") or payload.get("formal_text", "")
            tokenized.append(_tokenize(text))

        doc_count = len(tokenized)
        avg_dl    = sum(len(t) for t in tokenized) / max(doc_count, 1)

        # Document-frequency per term
        df: Counter = Counter()
        for tokens in tokenized:
            df.update(set(tokens))

        # IDF per term (BM25 variant: log((N - df + 0.5) / (df + 0.5)))
        idf: dict[str, float] = {
            term: math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

        return {
            "keys":      keys,
            "tokenized": tokenized,
            "idf":       idf,
            "avg_dl":    avg_dl,
        }

    def _score(self, tokens: list[str], index: dict) -> dict[str, float]:
        """Compute BM25Okapi scores for all documents in the index."""
        k1, b   = self._k1, self._b
        avg_dl  = index["avg_dl"]
        idf     = index["idf"]
        scores: dict[str, float] = {}

        for key, doc_tokens in zip(index["keys"], index["tokenized"]):
            tf  = Counter(doc_tokens)
            dl  = len(doc_tokens)
            val = 0.0
            for term in tokens:
                if term not in idf:
                    continue
                freq = tf.get(term, 0)
                val += idf[term] * (
                    freq * (k1 + 1)
                    / (freq + k1 * (1 - b + b * dl / max(avg_dl, 1)))
                )
            scores[key] = val

        return scores
