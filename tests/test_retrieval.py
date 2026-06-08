"""Tests for hybrid retrieval and reranking pipeline."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.chunker import RecursiveChunker
from src.ingestion.document_loader import Document, DocumentLoader
from src.retrieval.hybrid_retriever import HybridRetriever, HybridResult


class TestDocumentLoader:
    def test_load_text(self):
        loader = DocumentLoader()
        doc = loader.load_text("Hello world", metadata={"source": "test"})
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "test"
        assert doc.doc_id != ""

    def test_load_dicts(self):
        loader = DocumentLoader()
        docs = loader.load_dicts([
            {"content": "First document", "metadata": {"id": 1}},
            {"content": "Second document", "metadata": {"id": 2}},
            {"content": "", "metadata": {}},  # empty — should be skipped
        ])
        assert len(docs) == 2


class TestChunker:
    def test_chunk_short_doc(self):
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=0)
        doc = Document(content="This is a short document.", metadata={})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        assert chunks[0].text == "This is a short document."

    def test_chunk_long_doc(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        long_text = " ".join([f"Sentence number {i} is here." for i in range(50)])
        doc = Document(content=long_text, metadata={})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 80  # some slack for overlap

    def test_chunk_inherits_metadata(self):
        chunker = RecursiveChunker(chunk_size=512)
        doc = Document(content="Test content", metadata={"source": "test.pdf"})
        chunks = chunker.chunk_document(doc)
        assert chunks[0].metadata["source"] == "test.pdf"
        assert chunks[0].doc_id == doc.doc_id


class TestHybridRetriever:
    def _make_retriever(self, mock_vs, mock_embedder):
        return HybridRetriever(
            vector_store=mock_vs,
            embedder=mock_embedder,
            alpha=0.5,
        )

    def test_rrf_fusion_combines_results(self, mock_embedder):
        mock_vs = MagicMock()
        from src.retrieval.vector_store import SearchResult
        mock_vs.search.return_value = [
            SearchResult(text="dense doc 1", metadata={}, score=0.9, chunk_id="d1"),
            SearchResult(text="dense doc 2", metadata={}, score=0.8, chunk_id="d2"),
        ]

        retriever = self._make_retriever(mock_vs, mock_embedder)
        retriever._bm25_docs = [
            {"text": "bm25 doc 1", "metadata": {}, "chunk_id": "b1"},
            {"text": "dense doc 1", "metadata": {}, "chunk_id": "d1"},
        ]

        results = retriever.retrieve("test query", top_k=10)
        assert isinstance(results, list)
        assert all(isinstance(r, HybridResult) for r in results)
        # dense doc 1 appears in both → should have higher RRF score
        texts = [r.text for r in results]
        assert "dense doc 1" in texts

    def test_empty_corpus_returns_dense_only(self, mock_embedder):
        mock_vs = MagicMock()
        from src.retrieval.vector_store import SearchResult
        mock_vs.search.return_value = [
            SearchResult(text="only dense", metadata={}, score=0.85, chunk_id="x1"),
        ]

        retriever = self._make_retriever(mock_vs, mock_embedder)
        results = retriever.retrieve("query", top_k=5)
        assert len(results) >= 1

    def test_rrf_score_always_positive(self, mock_embedder):
        mock_vs = MagicMock()
        from src.retrieval.vector_store import SearchResult
        mock_vs.search.return_value = [
            SearchResult(text=f"doc {i}", metadata={}, score=0.9 - i * 0.1, chunk_id=f"c{i}")
            for i in range(5)
        ]
        retriever = self._make_retriever(mock_vs, mock_embedder)
        results = retriever.retrieve("query")
        for r in results:
            assert r.rrf_score > 0


class TestCrossEncoderReranker:
    def test_rerank_reduces_to_top_k(self, sample_chunks):
        from src.retrieval.reranker import CrossEncoderReranker
        from src.retrieval.hybrid_retriever import HybridResult

        candidates = [
            HybridResult(text=c.text, metadata=c.metadata, rrf_score=0.5, chunk_id=c.chunk_id)
            for c in sample_chunks
        ]

        reranker = CrossEncoderReranker()
        # Mock model to avoid downloading in tests
        reranker._model = MagicMock()
        import numpy as np
        reranker._model.predict.return_value = np.array([0.9, 0.5, 0.7])

        ranked = reranker.rerank("test query", candidates, top_k=2)
        assert len(ranked) == 2
        assert ranked[0].rerank_score >= ranked[1].rerank_score
