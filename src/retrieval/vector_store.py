"""
Vector store abstraction layer.
Default: ChromaDB (local, zero-infra).
Production alternative: Pinecone (swap by setting VECTOR_STORE_TYPE=pinecone).

Design pattern: Abstract base class + factory function.
Both implementations expose the same interface, making retrieval code agnostic.
"""
from __future__ import annotations

import abc
import os
from typing import Any

from src.config.settings import get_settings
from src.ingestion.chunker import Chunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────


class SearchResult:
    __slots__ = ("text", "metadata", "score", "chunk_id")

    def __init__(
        self,
        text: str,
        metadata: dict[str, Any],
        score: float,
        chunk_id: str = "",
    ) -> None:
        self.text = text
        self.metadata = metadata
        self.score = score
        self.chunk_id = chunk_id

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, text={self.text[:60]!r})"


# ── Abstract base ─────────────────────────────────────────────────────────────


class BaseVectorStore(abc.ABC):
    @abc.abstractmethod
    def upsert(self, chunks: list[Chunk], collection: str = "default") -> int:
        """Insert or update chunks. Returns count upserted."""

    @abc.abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        collection: str = "default",
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """ANN search. Returns top_k results sorted by descending score."""

    @abc.abstractmethod
    def delete_collection(self, collection: str) -> None:
        """Drop all vectors in a collection."""

    @abc.abstractmethod
    def count(self, collection: str = "default") -> int:
        """Return number of vectors in collection."""


# ── ChromaDB implementation ───────────────────────────────────────────────────


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_dir: str) -> None:
        import chromadb
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collections: dict[str, Any] = {}
        logger.info("chroma_initialized", persist_dir=persist_dir)

    def _get_or_create(self, collection: str) -> Any:
        if collection not in self._collections:
            self._collections[collection] = self._client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[collection]

    def upsert(self, chunks: list[Chunk], collection: str = "default") -> int:
        if not chunks:
            return 0
        col = self._get_or_create(collection)

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        embeddings = [c.metadata.pop("embedding") for c in chunks]
        metadatas = [
            {k: str(v) for k, v in c.metadata.items() if not isinstance(v, (list, dict))}
            for c in chunks
        ]

        col.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("chroma_upsert", collection=collection, count=len(chunks))
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        collection: str = "default",
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        col = self._get_or_create(collection)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, col.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        res = col.query(**kwargs)
        results: list[SearchResult] = []
        for doc, meta, dist, cid in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            res["ids"][0],
        ):
            # Chroma returns cosine distance; convert to similarity
            score = 1.0 - dist
            results.append(SearchResult(text=doc, metadata=meta, score=score, chunk_id=cid))
        return results

    def delete_collection(self, collection: str) -> None:
        self._client.delete_collection(collection)
        self._collections.pop(collection, None)
        logger.info("collection_deleted", collection=collection)

    def count(self, collection: str = "default") -> int:
        col = self._get_or_create(collection)
        return col.count()


# ── Factory ───────────────────────────────────────────────────────────────────


def get_vector_store() -> BaseVectorStore:
    settings = get_settings()
    if settings.vector_store_type == "chroma":
        return ChromaVectorStore(persist_dir=settings.chroma_persist_dir)
    if settings.vector_store_type == "pinecone":
        raise NotImplementedError(
            "Pinecone support: set VECTOR_STORE_TYPE=chroma for local dev, "
            "or implement PineconeVectorStore following ChromaVectorStore interface."
        )
    raise ValueError(f"Unknown vector_store_type: {settings.vector_store_type}")
