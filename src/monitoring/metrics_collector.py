"""
Prometheus metrics for the RAG system.
Exposes /metrics endpoint consumed by Prometheus scraper.

Key metrics:
  - request latency histogram (p50/p95/p99)
  - retrieval stage latencies
  - LLM token usage and cost
  - cache hit/miss counters
  - error rates by component
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ── Metric definitions ────────────────────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "End-to-end request latency",
    ["endpoint", "status"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval stage latency",
    ["stage"],  # dense | bm25 | rerank | total
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

LLM_TOKENS = Counter(
    "rag_llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "type"],  # type: prompt | completion
)

LLM_COST_USD = Counter(
    "rag_llm_cost_usd_total",
    "Estimated LLM cost in USD",
    ["model"],
)

EMBEDDING_TOKENS = Counter(
    "rag_embedding_tokens_total",
    "Total embedding tokens consumed",
    ["model"],
)

CACHE_OPS = Counter(
    "rag_cache_operations_total",
    "Cache hit/miss counts",
    ["cache_type", "result"],  # result: hit | miss
)

AGENT_ITERATIONS = Histogram(
    "rag_agent_iterations",
    "Number of ReAct iterations per query",
    buckets=[1, 2, 3, 4, 5, 6, 8, 10],
)

INGESTION_CHUNKS = Counter(
    "rag_ingestion_chunks_total",
    "Total chunks ingested into vector store",
    ["collection"],
)

ERRORS = Counter(
    "rag_errors_total",
    "Error counts by component",
    ["component", "error_type"],
)

ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "Currently in-flight requests",
)

VECTOR_STORE_SIZE = Gauge(
    "rag_vector_store_size",
    "Number of vectors in store",
    ["collection"],
)

SYSTEM_INFO = Info("rag_system", "RAG system metadata")
SYSTEM_INFO.info({"version": "1.0.0", "framework": "custom-react"})


# ── Context managers ──────────────────────────────────────────────────────────


@contextmanager
def track_request(endpoint: str) -> Generator[None, None, None]:
    ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.perf_counter() - t0
        REQUEST_LATENCY.labels(endpoint=endpoint, status=status).observe(elapsed)
        ACTIVE_REQUESTS.dec()


@contextmanager
def track_retrieval(stage: str) -> Generator[None, None, None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        RETRIEVAL_LATENCY.labels(stage=stage).observe(time.perf_counter() - t0)


# ── Helper functions ──────────────────────────────────────────────────────────


def record_llm_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    LLM_TOKENS.labels(model=model, type="prompt").inc(prompt_tokens)
    LLM_TOKENS.labels(model=model, type="completion").inc(completion_tokens)
    # Rough cost estimate (GPT-4o-mini pricing)
    cost = (prompt_tokens * 0.00015 + completion_tokens * 0.0006) / 1000
    LLM_COST_USD.labels(model=model).inc(cost)


def record_cache_hit(cache_type: str) -> None:
    CACHE_OPS.labels(cache_type=cache_type, result="hit").inc()


def record_cache_miss(cache_type: str) -> None:
    CACHE_OPS.labels(cache_type=cache_type, result="miss").inc()


def record_error(component: str, error_type: str) -> None:
    ERRORS.labels(component=component, error_type=error_type).inc()


def get_metrics_output() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
