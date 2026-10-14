"""Deterministic, word-based chunking."""

from hashlib import sha256

from groundtruth_rag.ingestion.loaders import LoadedPage
from groundtruth_rag.schemas import DocumentChunk


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def chunk_pages(pages: list[LoadedPage], chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= chunk_overlap < chunk_size")
    chunks: list[DocumentChunk] = []
    step = chunk_size - chunk_overlap
    for page in pages:
        words = normalize_text(page.text).split()
        for chunk_index, start in enumerate(range(0, len(words), step)):
            end = min(start + chunk_size, len(words))
            text = " ".join(words[start:end])
            content_hash = sha256(text.encode("utf-8")).hexdigest()
            identity = f"{page.document_id}\0{page.page}\0{chunk_index}\0{content_hash}"
            chunks.append(DocumentChunk(
                chunk_id=sha256(identity.encode("utf-8")).hexdigest(),
                document_id=page.document_id, source=page.source, page=page.page, text=text,
                metadata={"chunk_index": chunk_index, "word_start": start,
                          "word_end": end, "content_hash": content_hash},
            ))
            if end == len(words):
                break
    return chunks
