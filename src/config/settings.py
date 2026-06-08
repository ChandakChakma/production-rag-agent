"""
Central configuration using Pydantic Settings v2.
All env vars are validated at startup — fail fast on misconfiguration.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o-mini", description="Chat completion model")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model"
    )
    embedding_dimension: int = Field(1536, description="Embedding vector dimension")

    # ── Vector Store ─────────────────────────────────────────────────
    vector_store_type: Literal["chroma", "pinecone"] = "chroma"
    chroma_persist_dir: str = "./data/chroma_db"
    pinecone_api_key: str | None = None
    pinecone_index_name: str | None = None
    default_collection: str = "default"

    # ── Retrieval ────────────────────────────────────────────────────
    retrieval_top_k: int = Field(50, ge=1, le=500)
    rerank_top_k: int = Field(5, ge=1, le=50)
    hybrid_alpha: float = Field(
        0.5, ge=0.0, le=1.0, description="0=BM25 only, 1=dense only"
    )
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Chunking ─────────────────────────────────────────────────────
    chunk_size: int = Field(512, ge=64, le=2048)
    chunk_overlap: int = Field(64, ge=0, le=512)

    # ── Agent ────────────────────────────────────────────────────────
    agent_max_iterations: int = Field(8, ge=1, le=20)
    agent_temperature: float = Field(0.0, ge=0.0, le=2.0)
    memory_max_tokens: int = Field(2000, ge=100)
    memory_window_size: int = Field(10, ge=1, description="Message pairs to retain")

    # ── API ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = Field(8000, ge=1, le=65535)
    api_key: str = Field("dev-secret-key", description="Bearer token for API auth")
    api_rate_limit: int = Field(100, description="Requests per minute per client")

    # ── Observability ────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    otlp_endpoint: str = "http://localhost:4317"
    prometheus_port: int = 8001
    enable_tracing: bool = True

    # ── Caching ──────────────────────────────────────────────────────
    cache_dir: str = "./data/cache"
    embedding_cache_size_gb: float = Field(1.0, ge=0.1, le=100.0)
    semantic_cache_threshold: float = Field(0.95, ge=0.0, le=1.0)
    enable_semantic_cache: bool = True

    # ── Evaluation ───────────────────────────────────────────────────
    eval_llm_model: str = "gpt-4o"
    eval_batch_size: int = Field(5, ge=1, le=50)

    # ── Circuit Breaker ──────────────────────────────────────────────
    cb_failure_threshold: int = Field(5, description="Failures before opening circuit")
    cb_recovery_timeout: int = Field(60, description="Seconds before half-open retry")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({chunk_size})")
        return v

    @field_validator("pinecone_api_key")
    @classmethod
    def pinecone_key_required_if_pinecone(cls, v: str | None, info) -> str | None:
        if info.data.get("vector_store_type") == "pinecone" and not v:
            raise ValueError("pinecone_api_key is required when vector_store_type=pinecone")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — call once at startup."""
    return Settings()
