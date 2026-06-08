"""Ingest route — accepts documents and runs the full ingestion pipeline."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.monitoring.metrics_collector import INGESTION_CHUNKS
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class IngestDocument(BaseModel):
    content: str = Field(..., min_length=10)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(..., min_length=1)
    collection: str = Field("default")
    chunk_size: int = Field(512, ge=64, le=2048)
    chunk_overlap: int = Field(64, ge=0)


class IngestResponse(BaseModel):
    ingested_documents: int
    total_chunks: int
    collection: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request) -> IngestResponse:
    components = request.app.state.components
    vector_store = components["vector_store"]
    embedder = components["embedder"]
    retriever = components["retriever"]

    from src.ingestion.chunker import RecursiveChunker
    from src.ingestion.document_loader import Document

    chunker = RecursiveChunker(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
    docs = [Document(content=d.content, metadata=d.metadata) for d in req.documents]

    try:
        chunks = chunker.chunk_documents(docs)
        chunks = embedder.embed_chunks(chunks)
        count = vector_store.upsert(chunks, collection=req.collection)

        # Keep BM25 index in sync
        retriever.add_to_bm25_corpus(
            texts=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            chunk_ids=[c.chunk_id for c in chunks],
        )

        INGESTION_CHUNKS.labels(collection=req.collection).inc(count)
        logger.info("ingest_complete", docs=len(docs), chunks=count, collection=req.collection)

        return IngestResponse(
            ingested_documents=len(docs),
            total_chunks=count,
            collection=req.collection,
        )
    except Exception as exc:
        logger.error("ingest_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
