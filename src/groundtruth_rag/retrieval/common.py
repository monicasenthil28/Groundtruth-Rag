"""Shared retrieval input checks and deterministic ordering."""

import math
from collections.abc import Sequence

from groundtruth_rag.schemas import DocumentChunk


def validate_chunks(chunks: Sequence[DocumentChunk]) -> list[DocumentChunk]:
    indexed = list(chunks)
    if not indexed:
        raise ValueError("cannot index an empty corpus")
    if len({chunk.chunk_id for chunk in indexed}) != len(indexed):
        raise ValueError("corpus contains duplicate chunk IDs")
    return indexed


def validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def ranked_indices(scores: Sequence[float], chunks: Sequence[DocumentChunk], k: int) -> list[int]:
    if len(scores) != len(chunks):
        raise ValueError("score count does not match corpus")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("retrieval scores must be finite")
    return sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i].chunk_id))[:k]
