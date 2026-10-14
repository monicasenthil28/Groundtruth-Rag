"""Domain contracts shared by retrievers and evaluators."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class DocumentChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalQuery(BaseModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_chunk_ids: set[str] = Field(min_length=1)
    reference_answer: str | None = None
    tags: set[str] = Field(default_factory=set)


class RetrievalHit(BaseModel):
    chunk_id: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1)
    retriever: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalRunResult(BaseModel):
    query_id: str = Field(min_length=1)
    hits: list[RetrievalHit] = Field(default_factory=list)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    configuration: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ranks(self) -> "RetrievalRunResult":
        ranks = [hit.rank for hit in self.hits]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("hit ranks must be consecutive and ordered from 1")
        return self
