"""Broad, deduplicated candidate discovery for human review."""

from dataclasses import dataclass, field

from groundtruth_rag.retrieval.base import Retriever
from groundtruth_rag.retrieval.hybrid import rrf_fuse
from groundtruth_rag.schemas import DocumentChunk, RetrievalHit


@dataclass
class Candidate:
    chunk: DocumentChunk
    matches: list[RetrievalHit] = field(default_factory=list)


def discover_candidates(query: str, chunks: list[DocumentChunk], dense: Retriever,
                        bm25: Retriever, depth: int) -> list[Candidate]:
    if depth <= 0:
        raise ValueError("candidate depth must be positive")
    dense_hits = dense.retrieve(query, depth)
    bm25_hits = bm25.retrieve(query, depth)
    hybrid_hits = rrf_fuse(dense_hits, bm25_hits)[:depth]
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    found: dict[str, Candidate] = {}
    for hits in (dense_hits, bm25_hits, hybrid_hits):
        for hit in hits:
            if hit.chunk_id not in by_id:
                raise ValueError(f"retriever returned unknown chunk ID: {hit.chunk_id}")
            found.setdefault(hit.chunk_id, Candidate(by_id[hit.chunk_id])).matches.append(hit)
    return list(found.values())


def search_corpus(chunks: list[DocumentChunk], phrase: str) -> list[Candidate]:
    if not phrase.strip():
        raise ValueError("search phrase must not be empty")
    return [Candidate(chunk) for chunk in chunks if phrase.casefold() in chunk.text.casefold()]
