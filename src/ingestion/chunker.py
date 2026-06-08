"""
Semantic-aware document chunker.

Strategy:
  1. Split on double-newlines (paragraph boundaries) first.
  2. If paragraphs exceed chunk_size, split recursively on sentences → words.
  3. Apply token-count-aware overlap (not naive character overlap).
  4. Each chunk inherits parent document metadata plus chunk position.

Why this matters: naive character splitting breaks mid-sentence, degrading
retrieval quality. Paragraph-first splitting preserves semantic units.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from src.ingestion.document_loader import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.chunk_id:
            import hashlib
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:12]


class RecursiveChunker:
    """
    Recursively splits documents into token-bounded chunks with controlled overlap.
    Respects paragraph → sentence → word boundaries in that priority order.
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    # ── Public ────────────────────────────────────────────────────────

    def chunk_document(self, doc: Document) -> list[Chunk]:
        raw_chunks = self._split(doc.content, self._SEPARATORS)
        chunks_with_overlap = self._apply_overlap(raw_chunks)
        result: list[Chunk] = []
        for i, text in enumerate(chunks_with_overlap):
            tok = _count_tokens(text, self._enc)
            meta = {**doc.metadata, "chunk_index": i, "total_chunks": len(chunks_with_overlap)}
            result.append(
                Chunk(
                    text=text,
                    metadata=meta,
                    doc_id=doc.doc_id,
                    chunk_index=i,
                    token_count=tok,
                )
            )
        logger.debug("chunked_document", doc_id=doc.doc_id, num_chunks=len(result))
        return result

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(
            "chunked_all_documents",
            num_docs=len(docs),
            total_chunks=len(all_chunks),
        )
        return all_chunks

    # ── Private ───────────────────────────────────────────────────────

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text trying separators in order."""
        if not separators:
            return [text]

        sep = separators[0]
        parts = text.split(sep) if sep else list(text)
        good: list[str] = []
        bad: list[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if _count_tokens(part, self._enc) <= self.chunk_size:
                good.append(part)
            else:
                bad.append(part)

        result: list[str] = []
        # Merge good parts that are too small
        current = ""
        for part in good:
            candidate = (current + sep + part).strip() if current else part
            if _count_tokens(candidate, self._enc) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                current = part
        if current:
            result.append(current)

        # Recursively split parts that were too large
        for part in bad:
            result.extend(self._split(part, separators[1:]))

        return [r for r in result if r.strip()]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend tail of previous chunk to provide context continuity."""
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = self._enc.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-self.chunk_overlap :]
            overlap_text = self._enc.decode(overlap_tokens).strip()
            merged = (overlap_text + " " + chunks[i]).strip()
            result.append(merged)
        return result
