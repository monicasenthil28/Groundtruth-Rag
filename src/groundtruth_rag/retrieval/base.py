"""Interchangeable retriever interface."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from groundtruth_rag.schemas import DocumentChunk, RetrievalHit


class Retriever(ABC):
    @abstractmethod
    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Prepare an index from domain chunks."""

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        """Return up to k hits, ordered with one-based consecutive ranks."""
