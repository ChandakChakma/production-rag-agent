"""
Hybrid Retriever — combines BM25 (lexical) + dense ANN (semantic)
using Reciprocal Rank Fusion (RRF).

Why hybrid?
  - Dense retrieval misses exact keyword matches ("GPT-4", "RFC 2616")
  - BM25 misses paraphrases and conceptual similarity
  - RRF fusion provably outperforms either alone (SIGIR 2009, Cormack et al.)

RRF formula: score(d) = Σ_r 1 / (k + rank_r(d))
  k=60 is the standard smoothing constant.

Corpus is rebuilt from ChromaDB on each retriever instantiation (dev) or
kept in memory as a class attribute (prod, see InMemoryBM25 pattern).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import BaseVectorStore, SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

_RRF_K = 60


@dataclass
class HybridResult:
    text: str
    metadata: dict[str, Any]
    rrf_score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    chunk_id: str = ""


class HybridRetriever:
    """
    Two-stage retrieval:
      Stage 1 — Hybrid BM25 + dense ANN → top_k=50 candidates via RRF
      Stage 2 — Cross-encoder reranking → top_k=5 (done in reranker.py)

    alpha controls the blend:
      0.0 → pure BM25
      1.0 → pure dense
      0.5 → balanced (recommended default)
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: Embedder,
        alpha: float = 0.5,
    ) -> None:
        self._vs = vector_store
        self._embedder = embedder
        self.alpha = alpha

        # BM25 corpus is rebuilt lazily when collection changes
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict[str, Any]] = []  # [{text, metadata, chunk_id}]
        self._bm25_collection: str | None = None

    # ── Public ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        collection: str = "default",
        filters: dict[str, Any] | None = None,
    ) -> list[HybridResult]:
        """
        Hybrid retrieve with RRF fusion.
        Returns up to top_k results sorted by descending RRF score.
        """
        query_embedding = self._embedder.embed_text(query)

        # ── Dense retrieval ──────────────────────────────────────────
        dense_results = self._vs.search(
            query_embedding=query_embedding,
            top_k=top_k,
            collection=collection,
            filters=filters,
        )
        logger.debug("dense_retrieved", count=len(dense_results))

        # ── Sparse (BM25) retrieval ──────────────────────────────────
        self._ensure_bm25(collection)
        bm25_results = self._bm25_search(query, top_k=top_k)
        logger.debug("bm25_retrieved", count=len(bm25_results))

        # ── RRF fusion ───────────────────────────────────────────────
        fused = self._reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_k)
        logger.info(
            "hybrid_retrieve_complete",
            query_preview=query[:60],
            dense=len(dense_results),
            bm25=len(bm25_results),
            fused=len(fused),
        )
        return fused

    def add_to_bm25_corpus(self, texts: list[str], metadatas: list[dict[str, Any]], chunk_ids: list[str]) -> None:
        """Called by ingestion pipeline to keep BM25 index in sync."""
        for text, meta, cid in zip(texts, metadatas, chunk_ids):
            self._bm25_docs.append({"text": text, "metadata": meta, "chunk_id": cid})
        self._bm25 = None  # Invalidate; rebuilt on next retrieve

    def invalidate_bm25(self) -> None:
        self._bm25 = None
        self._bm25_docs = []

    # ── BM25 helpers ──────────────────────────────────────────────────

    def _ensure_bm25(self, collection: str) -> None:
        """Lazily build or rebuild BM25 index."""
        if self._bm25 is not None and self._bm25_collection == collection:
            return
        if not self._bm25_docs:
            logger.debug("bm25_corpus_empty_skipping")
            return

        tokenized = [self._tokenize(d["text"]) for d in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_collection = collection
        logger.debug("bm25_index_built", docs=len(self._bm25_docs))

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._bm25 or not self._bm25_docs:
            return []
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(range(len(self._bm25_docs)), scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for idx, score in ranked:
            doc = self._bm25_docs[idx]
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "chunk_id": doc["chunk_id"],
                "score": float(score),
            })
        return results

    # ── RRF fusion ────────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense: list[SearchResult],
        bm25: list[dict[str, Any]],
        top_k: int,
    ) -> list[HybridResult]:
        """
        Combine dense and BM25 rankings using RRF.
        Key: text content used as dedup key (chunk_id may differ across indices).
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        # Dense rankings (weighted by alpha)
        for rank, result in enumerate(dense):
            key = result.chunk_id or result.text[:50]
            contribution = self.alpha / (_RRF_K + rank + 1)
            rrf_scores[key] += contribution
            dense_scores[key] = result.score
            doc_map[key] = {"text": result.text, "metadata": result.metadata, "chunk_id": result.chunk_id}

        # BM25 rankings (weighted by 1-alpha)
        for rank, result in enumerate(bm25):
            key = result.get("chunk_id") or result["text"][:50]
            contribution = (1.0 - self.alpha) / (_RRF_K + rank + 1)
            rrf_scores[key] += contribution
            bm25_scores[key] = result["score"]
            if key not in doc_map:
                doc_map[key] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "chunk_id": result.get("chunk_id", ""),
                }

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        return [
            HybridResult(
                text=doc_map[k]["text"],
                metadata=doc_map[k]["metadata"],
                rrf_score=rrf_scores[k],
                dense_score=dense_scores.get(k, 0.0),
                bm25_score=bm25_scores.get(k, 0.0),
                chunk_id=doc_map[k]["chunk_id"],
            )
            for k in sorted_keys
        ]
