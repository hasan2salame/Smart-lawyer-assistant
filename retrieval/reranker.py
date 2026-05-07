"""
retrieval/reranker.py
Cohere cross-encoder reranker (rerank-multilingual-v3.0).

Builds a rich document string for each candidate so the cross-encoder has
enough context: article identifier + law name + first 400 chars of body text.
"""

import cohere

from config import COHERE_RERANK_MODEL


class Reranker:
    """
    Thin wrapper around the Cohere Rerank API.

    Parameters
    ----------
    api_key : str — Cohere API key
    """

    def __init__(self, api_key: str) -> None:
        self._client = cohere.Client(api_key)

    def rerank(
        self,
        query: str,
        items: list[dict],
        top_n: int,
    ) -> list[dict]:
        """
        Re-score *items* against *query* using the Cohere cross-encoder.

        Parameters
        ----------
        query  : cleaned query string
        items  : list of dicts with at least {"key", "payload", "score"}
        top_n  : maximum number of results to return

        Returns
        -------
        list[dict] — subset of *items* re-sorted by relevance score,
                     each dict updated with {"score": float, "source": "rerank"}.
        Returns the original *items* unchanged if the API call fails.
        """
        if not items or not query:
            return items

        docs = [self._build_doc(item["payload"]) for item in items]

        try:
            response = self._client.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=docs,
                top_n=min(top_n, len(items)),
            )
        except Exception as exc:
            print(f"[Reranker] API error — falling back to original order: {exc}")
            return items[:top_n]

        return [
            {
                **items[result.index],
                "score":  result.relevance_score,
                "source": "rerank",
            }
            for result in response.results
        ]

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_doc(payload: dict) -> str:
        """
        Construct the document string fed to the cross-encoder.

        Format: "<article_str> - <law_name>\n<body_text[:400]>"
        """
        article_str = payload.get("article_str") or payload.get("title", "")
        law_name    = payload.get("law_name", "")
        body        = (
            payload.get("original_text") or payload.get("formal_text", "")
        )[:400]

        header = " - ".join(filter(None, [article_str, law_name]))
        return f"{header}\n{body}".strip()
