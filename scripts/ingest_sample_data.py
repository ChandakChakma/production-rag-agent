#!/usr/bin/env python
"""
Ingest sample documents into the vector store.
Run: python scripts/ingest_sample_data.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.progress import track

console = Console()


def main():
    console.print("\n[bold cyan]🚀 Production RAG System — Sample Data Ingestion[/bold cyan]\n")

    from src.config.settings import get_settings
    from src.ingestion.chunker import RecursiveChunker
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.embedder import Embedder
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.vector_store import get_vector_store
    from src.utils.logger import configure_logging

    settings = get_settings()
    configure_logging(log_level="WARNING", log_format="console")

    console.print(f"[dim]LLM model:    {settings.openai_model}[/dim]")
    console.print(f"[dim]Vector store: {settings.vector_store_type}[/dim]")
    console.print(f"[dim]Collection:   {settings.default_collection}[/dim]\n")

    # ── Load sample documents ──────────────────────────────────────────
    loader = DocumentLoader()
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_docs")
    docs = loader.load_directory(sample_dir)
    console.print(f"[green]✓[/green] Loaded {len(docs)} documents from {sample_dir}")

    # ── Chunk ──────────────────────────────────────────────────────────
    chunker = RecursiveChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    console.print(f"[green]✓[/green] Created {len(chunks)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    # ── Embed ──────────────────────────────────────────────────────────
    console.print("[yellow]⟳[/yellow] Embedding chunks (may call OpenAI API)...")
    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)
    stats = embedder.stats()
    console.print(f"[green]✓[/green] Embedded {len(chunks)} chunks | cache hit rate: {stats['cache_stats']['hit_rate']:.1%} | API calls: {stats['total_api_calls']}")

    # ── Store ──────────────────────────────────────────────────────────
    vector_store = get_vector_store()
    count = vector_store.upsert(chunks, collection=settings.default_collection)
    console.print(f"[green]✓[/green] Upserted {count} vectors into '{settings.default_collection}'")

    # ── Sync BM25 ─────────────────────────────────────────────────────
    retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
    retriever.add_to_bm25_corpus(
        texts=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        chunk_ids=[c.chunk_id for c in chunks],
    )

    # ── Smoke test retrieval ───────────────────────────────────────────
    console.print("\n[bold]Running smoke test retrieval...[/bold]")
    from src.retrieval.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()

    test_query = "What is retrieval-augmented generation?"
    candidates = retriever.retrieve(test_query, top_k=20)
    ranked = reranker.rerank(test_query, candidates, top_k=3)

    console.print(f"\n[bold]Query:[/bold] {test_query}")
    console.print(f"[bold]Top {len(ranked)} results:[/bold]")
    for i, r in enumerate(ranked, 1):
        console.print(f"  [{i}] (score={r.rerank_score:.3f}) {r.text[:100]}...")

    console.print("\n[bold green]✅ Ingestion complete! Run `make run` to start the API.[/bold green]\n")


if __name__ == "__main__":
    main()
