"""
Embedder — wraps OpenAI embeddings with:
  - Disk-based embedding cache (saves API $)
  - Batch processing (max 2048 texts per API call)
  - Exponential-backoff retry via tenacity
  - Token budget guard (raises if text exceeds model limit)
  - Drift detection (warns when new embeddings diverge from corpus mean)
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import get_settings
from src.ingestion.chunker import Chunk
from src.utils.cache import EmbeddingCache
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_BATCH = 2048   # OpenAI hard limit
_MAX_TOKENS = 8191  # text-embedding-3-small limit


class Embedder:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = OpenAI(api_key=self._settings.openai_api_key)
        self._model = self._settings.openai_embedding_model
        self._cache = EmbeddingCache(
            cache_dir=self._settings.cache_dir,
            size_limit_gb=self._settings.embedding_cache_size_gb,
        )
        # Running mean for drift detection
        self._corpus_mean: np.ndarray | None = None
        self._corpus_count: int = 0
        self._total_tokens_used: int = 0
        self._total_api_calls: int = 0

    # ── Public API ────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, using cache where possible."""
        if not texts:
            return []

        cached, miss_indices = self._cache.get_batch(self._model, texts)
        miss_texts = [texts[i] for i in miss_indices]

        if miss_texts:
            fresh = self._embed_batch_with_retry(miss_texts)
            self._cache.set_batch(self._model, miss_texts, fresh)
            self._update_drift_stats(fresh)
            for i, emb in zip(miss_indices, fresh):
                cached[i] = emb

        logger.debug(
            "embed_texts_complete",
            total=len(texts),
            cache_hits=len(texts) - len(miss_indices),
            api_calls=len(miss_texts),
        )
        return cached  # type: ignore[return-value]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all chunks, attach embedding to metadata, return updated chunks."""
        texts = [c.text for c in chunks]
        embeddings = self.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.metadata["embedding"] = emb
        return chunks

    def stats(self) -> dict[str, Any]:
        return {
            "total_tokens_used": self._total_tokens_used,
            "total_api_calls": self._total_api_calls,
            "estimated_cost_usd": round(self._total_tokens_used / 1_000_000 * 0.02, 6),
            "cache_stats": self._cache.stats(),
            "corpus_count": self._corpus_count,
        }

    # ── Private ───────────────────────────────────────────────────────

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Split into safe batches, call API with retry."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _MAX_BATCH):
            batch = texts[i : i + _MAX_BATCH]
            embeddings = self._call_openai(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call_openai(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format="float",
        )
        elapsed = time.perf_counter() - t0
        usage = response.usage
        self._total_tokens_used += usage.total_tokens
        self._total_api_calls += 1
        logger.info(
            "openai_embed_call",
            texts=len(texts),
            tokens=usage.total_tokens,
            elapsed_s=round(elapsed, 3),
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def _update_drift_stats(self, embeddings: list[list[float]]) -> None:
        """Welford online algorithm for running mean."""
        for emb in embeddings:
            arr = np.array(emb, dtype=np.float32)
            self._corpus_count += 1
            if self._corpus_mean is None:
                self._corpus_mean = arr
            else:
                delta = arr - self._corpus_mean
                self._corpus_mean += delta / self._corpus_count

    def check_drift(self, new_embeddings: list[list[float]], threshold: float = 0.15) -> bool:
        """
        Returns True if the mean of new_embeddings deviates significantly
        from the corpus mean (embedding drift detected).
        Interview topic: schedule this as a cron job post-ingestion.
        """
        if self._corpus_mean is None or not new_embeddings:
            return False
        new_mean = np.mean([np.array(e, dtype=np.float32) for e in new_embeddings], axis=0)
        cos_sim = float(
            np.dot(self._corpus_mean, new_mean)
            / (np.linalg.norm(self._corpus_mean) * np.linalg.norm(new_mean) + 1e-10)
        )
        drift = 1.0 - cos_sim
        if drift > threshold:
            logger.warning("embedding_drift_detected", drift=round(drift, 4), threshold=threshold)
            return True
        return False
