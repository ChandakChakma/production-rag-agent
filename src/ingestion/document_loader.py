"""
Document loader — ingests raw text, PDF, markdown, and URLs into
a unified Document schema consumed by the chunker.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """Universal document container passed through the entire pipeline."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            import hashlib
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class DocumentLoader:
    """
    Loads documents from multiple sources into Document objects.
    Supports: .txt, .md, .pdf, raw strings, dicts.
    """

    def load_text(self, text: str, metadata: dict[str, Any] | None = None) -> Document:
        return Document(content=text.strip(), metadata=metadata or {})

    def load_file(self, file_path: str | Path) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            content = path.read_text(encoding="utf-8")
            metadata = {"source": str(path), "type": suffix[1:]}
            logger.info("loaded_text_file", path=str(path), chars=len(content))
            return Document(content=content.strip(), metadata=metadata)

        if suffix == ".pdf":
            return self._load_pdf(path)

        raise ValueError(f"Unsupported file type: {suffix}")

    def _load_pdf(self, path: Path) -> Document:
        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            content = "\n\n".join(p for p in pages if p.strip())
            metadata = {
                "source": str(path),
                "type": "pdf",
                "pages": len(reader.pages),
            }
            logger.info("loaded_pdf", path=str(path), pages=len(reader.pages))
            return Document(content=content.strip(), metadata=metadata)
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")

    def load_directory(self, dir_path: str | Path) -> list[Document]:
        """Recursively load all supported files from a directory."""
        docs: list[Document] = []
        supported = {".txt", ".md", ".pdf"}
        for root, _, files in os.walk(dir_path):
            for fname in sorted(files):
                fpath = Path(root) / fname
                if fpath.suffix.lower() in supported:
                    try:
                        docs.append(self.load_file(fpath))
                    except Exception as exc:
                        logger.warning("skipped_file", path=str(fpath), error=str(exc))
        logger.info("loaded_directory", dir=str(dir_path), total_docs=len(docs))
        return docs

    def load_dicts(self, records: list[dict[str, Any]]) -> list[Document]:
        """Bulk-load from API payload: [{"content": "...", "metadata": {...}}]"""
        docs = []
        for r in records:
            content = r.get("content", "")
            if not content.strip():
                continue
            docs.append(Document(content=content.strip(), metadata=r.get("metadata", {})))
        return docs
