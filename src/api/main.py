"""FastAPI app — wires semantic cache into app state for query route."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.api.middleware import RequestIDMiddleware, RateLimitMiddleware
from src.api.routes import eval as eval_router
from src.api.routes import ingest as ingest_router
from src.api.routes import query as query_router
from src.config.settings import get_settings
from src.monitoring.metrics_collector import get_metrics_output
from src.monitoring.tracer import setup_tracing
from src.utils.logger import configure_logging, get_logger

logger   = get_logger(__name__)
_state: dict[str, Any] = {}
UI_DIR   = Path(__file__).parent.parent.parent / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, log_format=settings.log_format)
    logger.info("starting_up", model=settings.openai_model)

    setup_tracing(
        service_name="rag-agent-api",
        otlp_endpoint=settings.otlp_endpoint if settings.enable_tracing else None,
        enabled=settings.enable_tracing,
    )

    from src.ingestion.embedder import Embedder
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.vector_store import get_vector_store
    from src.agents.rag_agent import RAGAgent
    from src.agents.tools import ToolRegistry, RAGRetrieveTool, CalculatorTool, CurrentDateTool
    from src.utils.cache import SemanticCache

    vector_store   = get_vector_store()
    embedder       = Embedder()
    retriever      = HybridRetriever(vector_store=vector_store, embedder=embedder, alpha=settings.hybrid_alpha)
    reranker       = CrossEncoderReranker(model_name=settings.reranker_model)
    semantic_cache = SemanticCache(threshold=settings.semantic_cache_threshold, cache_dir=settings.cache_dir) \
                     if settings.enable_semantic_cache else None

    tool_registry = ToolRegistry()
    tool_registry.register(RAGRetrieveTool(retriever=retriever, reranker=reranker))
    tool_registry.register(CalculatorTool())
    tool_registry.register(CurrentDateTool())

    agent = RAGAgent(tool_registry=tool_registry)

    _state.update({
        "vector_store": vector_store,
        "embedder": embedder,
        "retriever": retriever,
        "reranker": reranker,
        "semantic_cache": semantic_cache,
        "tool_registry": tool_registry,
        "agent": agent,
        "settings": settings,
    })
    app.state.components = _state
    logger.info("startup_complete")
    yield
    logger.info("shutting_down")
    embedder._cache.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Production RAG + Agent System", version="1.0.0",
                  docs_url="/docs", redoc_url="/redoc", lifespan=lifespan)

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.api_rate_limit)

    app.include_router(query_router.router,  prefix="/api/v1", tags=["Query"])
    app.include_router(ingest_router.router, prefix="/api/v1", tags=["Ingest"])
    app.include_router(eval_router.router,   prefix="/api/v1", tags=["Evaluate"])

    @app.get("/", include_in_schema=False)
    async def serve_ui() -> FileResponse:
        return FileResponse(str(UI_DIR / "index.html"))

    @app.get("/health", tags=["System"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/metrics", tags=["System"], include_in_schema=False)
    async def metrics() -> Response:
        data, ct = get_metrics_output()
        return Response(content=data, media_type=ct)

    @app.exception_handler(Exception)
    async def err(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled", path=str(request.url), error=str(exc))
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    return app


app = create_app()

