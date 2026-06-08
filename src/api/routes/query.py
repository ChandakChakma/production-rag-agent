"""
Query routes — streaming, semantic cache, stats endpoint.

POST /api/v1/query         → standard JSON
POST /api/v1/query/stream  → Server-Sent Events (tokens stream live)
GET  /api/v1/stats         → vector count, cache hit rate, model info
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings
from src.monitoring.metrics_collector import AGENT_ITERATIONS, track_request
from src.monitoring.tracer import traced_span
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    use_agent: bool = Field(True)
    top_k: int = Field(5, ge=1, le=20)
    collection: str = Field("default")


class SourceDoc(BaseModel):
    text: str
    metadata: dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    latency_seconds: float
    total_tokens: int
    iterations: int | None = None
    cache_hit: bool = False
    request_id: str | None = None


# ── Standard batch query ──────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request, settings: Settings = Depends(get_settings)):
    components = request.app.state.components
    with traced_span("query", {"query": req.query[:100], "use_agent": req.use_agent}):
        with track_request("/api/v1/query"):
            if req.use_agent:
                return await _agent_query(req, components, request)
            return await _direct_rag_query(req, components, request)


# ── Streaming query ───────────────────────────────────────────────────────────

@router.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    """
    SSE streaming. Event types in order:
      status  → pipeline stage message
      sources → retrieved docs array
      token   → one LLM output token
      stats   → latency / token count / cache_hit
      done    → stream complete
      error   → failure message
    """
    return StreamingResponse(
        _stream_generator(req, request.app.state.components),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_generator(req: QueryRequest, components: dict) -> AsyncGenerator[str, None]:
    settings = components["settings"]
    embedder  = components["embedder"]
    retriever = components["retriever"]
    reranker  = components["reranker"]
    t_start   = time.perf_counter()

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    try:
        # 1. Semantic cache check
        yield sse({"type": "status", "message": "Checking semantic cache..."})
        query_emb = await asyncio.to_thread(embedder.embed_text, req.query)

        sem_cache = components.get("semantic_cache")
        if sem_cache and settings.enable_semantic_cache:
            cached = sem_cache.lookup(query_emb)
            if cached:
                yield sse({"type": "status", "message": "⚡ Cache hit — returning instantly"})
                for word in cached.answer.split(" "):
                    yield sse({"type": "token", "content": word + " "})
                    await asyncio.sleep(0.008)
                yield sse({"type": "stats", "latency": round(time.perf_counter() - t_start, 3),
                           "tokens": 0, "cache_hit": True})
                yield sse({"type": "done"})
                return

        # 2. Retrieve
        yield sse({"type": "status", "message": "Running hybrid retrieval (BM25 + dense + RRF)..."})
        candidates = await asyncio.to_thread(
            retriever.retrieve, req.query, settings.retrieval_top_k, req.collection
        )

        # 3. Rerank
        yield sse({"type": "status", "message": f"Retrieved {len(candidates)} candidates — reranking..."})
        ranked = await asyncio.to_thread(reranker.rerank, req.query, candidates, req.top_k)

        sources_payload = [
            {"text": r.text, "metadata": r.metadata, "score": round(r.rerank_score, 4)}
            for r in ranked
        ]
        yield sse({"type": "sources", "sources": sources_payload})
        yield sse({"type": "status", "message": f"Top {len(ranked)} sources — generating answer..."})

        # 4. Stream LLM
        client  = OpenAI(api_key=settings.openai_api_key)
        context = "\n\n".join(f"[{i+1}] {r.text}" for i, r in enumerate(ranked))
        prompt  = (
            "Answer using ONLY the provided context. "
            "Cite sources as [1], [2], etc. If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {req.query}\n\nAnswer:"
        )

        full_answer = ""
        stream = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800,
                stream=True,
            )
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_answer += token
                yield sse({"type": "token", "content": token})

        # 5. Store in semantic cache
        if sem_cache and settings.enable_semantic_cache and full_answer:
            sem_cache.store(query_embedding=query_emb, answer=full_answer, sources=sources_payload)

        yield sse({"type": "stats",
                   "latency": round(time.perf_counter() - t_start, 3),
                   "tokens": 0, "cache_hit": False})
        yield sse({"type": "done"})

    except asyncio.CancelledError:
        return
    except Exception as exc:
        logger.error("stream_error", error=str(exc))
        yield sse({"type": "error", "message": str(exc)})
        yield sse({"type": "done"})


# ── Stats endpoint (feeds UI header) ─────────────────────────────────────────

@router.get("/stats")
async def stats(request: Request, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    components = request.app.state.components
    try:
        vector_count = components["vector_store"].count(collection=settings.default_collection)
    except Exception:
        vector_count = 0

    embed_stats = components["embedder"].stats()
    cache = components.get("semantic_cache")

    return {
        "vector_count": vector_count,
        "model": settings.openai_model,
        "embedding_model": settings.openai_embedding_model,
        "embedding_cache_hit_rate": embed_stats["cache_stats"]["hit_rate"],
        "embedding_tokens_used": embed_stats["total_tokens_used"],
        "estimated_cost_usd": embed_stats["estimated_cost_usd"],
        "semantic_cache": cache.stats() if cache else None,
        "collection": settings.default_collection,
        "retrieval_top_k": settings.retrieval_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "hybrid_alpha": settings.hybrid_alpha,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _agent_query(req: QueryRequest, components: dict, request: Request) -> QueryResponse:
    agent = components["agent"]
    try:
        response = agent.run(req.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    AGENT_ITERATIONS.observe(response.iterations)
    sources = [SourceDoc(text="(see agent sources)", metadata=s, score=0.0) for s in response.sources]
    return QueryResponse(
        answer=response.answer, sources=sources,
        latency_seconds=response.latency_seconds, total_tokens=response.total_tokens,
        iterations=response.iterations, request_id=request.headers.get("X-Request-ID"),
    )


async def _direct_rag_query(req: QueryRequest, components: dict, request: Request) -> QueryResponse:
    settings  = components["settings"]
    embedder  = components["embedder"]
    retriever = components["retriever"]
    reranker  = components["reranker"]
    client    = OpenAI(api_key=settings.openai_api_key)
    t0        = time.perf_counter()

    query_emb = embedder.embed_text(req.query)
    sem_cache = components.get("semantic_cache")
    if sem_cache and settings.enable_semantic_cache:
        cached = sem_cache.lookup(query_emb)
        if cached:
            return QueryResponse(
                answer=cached.answer,
                sources=[SourceDoc(text=s["text"], metadata=s["metadata"], score=s["score"])
                         for s in cached.sources],
                latency_seconds=round(time.perf_counter() - t0, 3),
                total_tokens=0, cache_hit=True,
                request_id=request.headers.get("X-Request-ID"),
            )

    candidates = retriever.retrieve(query=req.query, top_k=settings.retrieval_top_k, collection=req.collection)
    ranked     = reranker.rerank(query=req.query, candidates=candidates, top_k=req.top_k)
    context    = "\n\n".join(f"[{i+1}] {r.text}" for i, r in enumerate(ranked))

    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": f"Answer using context only:\n{context}\n\nQuestion: {req.query}\n\nAnswer:"}],
        temperature=0.0, max_tokens=800,
    )
    answer       = resp.choices[0].message.content or ""
    total_tokens = resp.usage.total_tokens if resp.usage else 0
    src_payload  = [{"text": r.text, "metadata": r.metadata, "score": round(r.rerank_score, 4)} for r in ranked]

    if sem_cache and settings.enable_semantic_cache:
        sem_cache.store(query_embedding=query_emb, answer=answer, sources=src_payload)

    return QueryResponse(
        answer=answer, sources=[SourceDoc(**s) for s in src_payload],
        latency_seconds=round(time.perf_counter() - t0, 3), total_tokens=total_tokens,
        request_id=request.headers.get("X-Request-ID"),
    )
