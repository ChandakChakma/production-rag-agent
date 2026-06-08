"""Shared pytest fixtures."""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")
    monkeypatch.setenv("CACHE_DIR", "/tmp/test_cache")
    monkeypatch.setenv("LOG_FORMAT", "console")


@pytest.fixture
def sample_chunks():
    from src.ingestion.chunker import Chunk
    return [
        Chunk(
            text="RAG combines retrieval with generation to ground LLM outputs.",
            metadata={"source": "test.txt"},
            chunk_id="chunk001",
            doc_id="doc1",
            chunk_index=0,
            token_count=15,
        ),
        Chunk(
            text="Hybrid retrieval fuses BM25 and dense vectors using RRF.",
            metadata={"source": "test.txt"},
            chunk_id="chunk002",
            doc_id="doc1",
            chunk_index=1,
            token_count=14,
        ),
        Chunk(
            text="Cross-encoder reranking improves precision by scoring query-document pairs jointly.",
            metadata={"source": "test.txt"},
            chunk_id="chunk003",
            doc_id="doc1",
            chunk_index=2,
            token_count=16,
        ),
    ]


@pytest.fixture
def mock_embedder():
    import numpy as np
    embedder = MagicMock()
    embedder.embed_text.return_value = np.random.rand(1536).tolist()
    embedder.embed_texts.side_effect = lambda texts: [np.random.rand(1536).tolist() for _ in texts]
    return embedder


@pytest.fixture
def mock_openai_response():
    response = MagicMock()
    response.choices[0].message.content = "This is a test answer about RAG systems."
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.tool_calls = None
    response.usage.total_tokens = 100
    response.usage.prompt_tokens = 80
    response.usage.completion_tokens = 20
    return response
