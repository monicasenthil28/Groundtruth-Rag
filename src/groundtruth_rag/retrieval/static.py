"""Fixed ranked hits for offline tests and demonstrations."""

from collections.abc import Mapping, Sequence

from groundtruth_rag.retrieval.base import Retriever
from groundtruth_rag.schemas import DocumentChunk, RetrievalHit


class StaticRetriever(Retriever):
    def __init__(self, results: Mapping[str, Sequence[str]]) -> None:
        self._results = {query: tuple(ids) for query, ids in results.items()}
        for ids in self._results.values():
            if len(ids) != len(set(ids)):
                raise ValueError("static results must have unique chunk IDs")

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """No index is needed; results are supplied at construction."""

    def retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        if k <= 0:
            raise ValueError("k must be positive")
        return [
            RetrievalHit(chunk_id=chunk_id, score=1.0 / rank, rank=rank, retriever="static")
            for rank, chunk_id in enumerate(self._results.get(query, ())[:k], 1)
        ]
