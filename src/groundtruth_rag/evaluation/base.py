"""Separate retrieval and answer evaluation contracts."""

from abc import ABC, abstractmethod

from groundtruth_rag.schemas import EvalQuery, RetrievalRunResult


class RetrievalMetric(ABC):
    @abstractmethod
    def evaluate(self, query: EvalQuery, result: RetrievalRunResult) -> float:
        """Score a retrieval result against relevance judgments."""


class GenerationMetric(ABC):
    @abstractmethod
    def evaluate(self, query: EvalQuery, answer: str, context: list[str]) -> float:
        """Score a generated answer using the query and supplied context."""
