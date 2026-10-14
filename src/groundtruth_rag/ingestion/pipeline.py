"""Document-to-corpus ingestion workflow."""

from dataclasses import dataclass
from pathlib import Path

from groundtruth_rag.ingestion.chunking import chunk_pages
from groundtruth_rag.ingestion.corpus import write_corpus
from groundtruth_rag.ingestion.loaders import discover_documents, load_document


@dataclass(frozen=True)
class IngestSummary:
    documents: int
    pages: int
    chunks: int
    ignored: list[Path]


def ingest(input_path: Path, output_path: Path, chunk_size: int, chunk_overlap: int) -> IngestSummary:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= chunk_overlap < chunk_size")
    files, ignored = discover_documents(input_path)
    if any(path.resolve() == output_path.resolve() for path, _ in files):
        raise ValueError("output must not overwrite an input document")
    pages = []
    for path, source in files:
        pages.extend(load_document(path, source))
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("input documents contain no extractable text")
    write_corpus(output_path, chunks)
    return IngestSummary(len(files), len(pages), len(chunks), ignored)
