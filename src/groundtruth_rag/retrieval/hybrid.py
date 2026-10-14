"""Dense/BM25 rank fusion without mixing raw score scales."""

from collections.abc import Sequence

from groundtruth_rag.retrieval.base import Retriever
from groundtruth_rag.retrieval.common import validate_chunks, validate_k
from groundtruth_rag.schemas import DocumentChunk, RetrievalHit


def rrf_fuse(dense: Sequence[RetrievalHit], bm25: Sequence[RetrievalHit],
             rrf_k: int = 60) -> list[RetrievalHit]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    details: dict[str, dict[str, float | int]] = {}
    for label, hits in (("dense", dense), ("bm25", bm25)):
        if len({hit.chunk_id for hit in hits}) != len(hits):
            raise ValueError(f"{label} returned duplicate chunk IDs")
        for hit in hits:
            info = details.setdefault(hit.chunk_id, {})
            info[f"{label}_rank"] = hit.rank
            info[f"{label}_score"] = hit.score
    scored = []
    for chunk_id, info in details.items():
        score = sum(1 / (rrf_k + int(info[f"{label}_rank"]))
                    for label in ("dense", "bm25") if f"{label}_rank" in info)
        scored.append((chunk_id, score, info))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [RetrievalHit(chunk_id=chunk_id, score=score, rank=rank, retriever="hybrid_rrf",
                         metadata={**info, "rrf_score": score})
            for rank, (chunk_id, score, info) in enumerate(scored, 1)]


class HybridRetriever(Retriever):
    def __init__(self, dense: Retriever, bm25: Retriever, rrf_k: int = 60,
                 dense_candidates: int = 20, bm25_candidates: int = 20) -> None:
        if rrf_k <= 0 or dense_candidates <= 0 or bm25_candidates <= 0:
            raise ValueError("RRF and candidate values must be positive")
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.dense_candidates = dense_candidates
        self.bm25_candidates = bm25_candidates

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        indexed = validate_chunks(chunks)
        self.dense.index(indexed)
        self.bm25.index(indexed)

    def retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        validate_k(k)
        dense_hits = self.dense.retrieve(query, self.dense_candidates)
        bm25_hits = self.bm25.retrieve(query, self.bm25_candidates)
        return rrf_fuse(dense_hits, bm25_hits, self.rrf_k)[:k]
