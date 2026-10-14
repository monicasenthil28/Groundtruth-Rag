"""Simple lexical retrieval using rank-bm25."""

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from groundtruth_rag.retrieval.base import Retriever
from groundtruth_rag.retrieval.common import ranked_indices, validate_chunks, validate_k
from groundtruth_rag.schemas import DocumentChunk, RetrievalHit

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25Retriever(Retriever):
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._index: BM25Okapi | None = None

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        self._index = None
        self._chunks = validate_chunks(chunks)
        tokenized = [tokenize(chunk.text) for chunk in self._chunks]
        if not any(tokenized):
            raise ValueError("BM25 corpus contains no word tokens")
        self._index = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        validate_k(k)
        if self._index is None:
            raise ValueError("BM25 retriever must be indexed before retrieval")
        scores = self._index.get_scores(tokenize(query))
        indices = ranked_indices(scores, self._chunks, k)
        return [RetrievalHit(chunk_id=self._chunks[i].chunk_id, score=float(scores[i]),
                             rank=rank, retriever="bm25") for rank, i in enumerate(indices, 1)]
