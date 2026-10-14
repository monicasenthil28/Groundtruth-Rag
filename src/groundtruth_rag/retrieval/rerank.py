"""CrossEncoder reranking over a bounded hybrid candidate pool."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from groundtruth_rag.retrieval.base import Retriever
from groundtruth_rag.retrieval.common import validate_chunks, validate_k
from groundtruth_rag.schemas import DocumentChunk, RetrievalHit

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankedHybridRetriever(Retriever):
    def __init__(self, hybrid: Retriever, candidate_pool: int = 20,
                 model: str = DEFAULT_RERANKER_MODEL, reranker: Any = None) -> None:
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        self.hybrid = hybrid
        self.candidate_pool = candidate_pool
        self.model_name = model
        self._reranker = reranker
        self._chunks: dict[str, DocumentChunk] = {}

    def _model(self) -> Any:
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("Reranking requires: pip install -e '.[models]'") from exc
            self._reranker = CrossEncoder(self.model_name)
        return self._reranker

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        self._chunks = {}
        indexed = validate_chunks(chunks)
        self.hybrid.index(indexed)
        self._model()
        self._chunks = {chunk.chunk_id: chunk for chunk in indexed}

    def retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        validate_k(k)
        if not self._chunks:
            raise ValueError("reranked retriever must be indexed before retrieval")
        candidates = self.hybrid.retrieve(query, self.candidate_pool)
        if not candidates:
            return []
        pairs = [(query, self._chunks[hit.chunk_id].text) for hit in candidates]
        scores = np.asarray(self._model().predict(pairs), dtype=float).reshape(-1)
        if len(scores) != len(candidates) or any(not math.isfinite(score) for score in scores):
            raise ValueError("CrossEncoder returned invalid scores")
        ordered = sorted(zip(candidates, scores), key=lambda item: (-item[1], item[0].chunk_id))[:k]
        return [RetrievalHit(chunk_id=hit.chunk_id, score=float(score), rank=rank,
                             retriever="hybrid_crossencoder",
                             metadata={**hit.metadata, "crossencoder_score": float(score)})
                for rank, (hit, score) in enumerate(ordered, 1)]
