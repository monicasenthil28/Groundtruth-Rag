"""Retriever-independent evaluation runner and versioned result models."""

from collections.abc import Sequence
from datetime import datetime, timezone
from statistics import mean, median
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from groundtruth_rag.evaluation.metrics import mean_reciprocal_rank, query_metrics
from groundtruth_rag.retrieval.base import Retriever
from groundtruth_rag.schemas import EvalQuery, RetrievalHit, RetrievalRunResult

DEFAULT_K_VALUES = (1, 3, 5, 10)
SCHEMA_VERSION = "1.0"


class QueryResult(BaseModel):
    query_id: str
    query: str
    relevant_chunk_ids: list[str]
    hits: list[RetrievalHit]
    metrics: dict[str, float]
    latency_ms: float = Field(ge=0)


class LatencyStats(BaseModel):
    mean_ms: float = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)


class BenchmarkRun(BaseModel):
    schema_version: str
    run_id: str
    timestamp: datetime
    configuration_name: str
    configuration: dict[str, Any]
    dataset: str
    query_count: int = Field(ge=1)
    k_values: list[int]
    aggregate_metrics: dict[str, float]
    latency: LatencyStats
    per_query_results: list[QueryResult]

    @model_validator(mode="after")
    def validate_run(self) -> "BenchmarkRun":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.query_count != len(self.per_query_results):
            raise ValueError("query_count does not match per_query_results")
        if not self.dataset.strip() or not self.configuration_name.strip():
            raise ValueError("dataset and configuration_name must not be empty")
        return self


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower])


def evaluate_retriever(
    retriever: Retriever,
    queries: Sequence[EvalQuery],
    *,
    dataset: str,
    configuration_name: str,
    configuration: dict[str, Any] | None = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> BenchmarkRun:
    if not queries:
        raise ValueError("evaluation requires at least one query")
    if not k_values or any(k <= 0 for k in k_values) or len(k_values) != len(set(k_values)):
        raise ValueError("k_values must be unique positive integers")
    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError("query IDs must be unique")
    if not dataset.strip() or not configuration_name.strip():
        raise ValueError("dataset and configuration_name must not be empty")

    results: list[QueryResult] = []
    run_config = configuration or {}
    for query in queries:
        start = perf_counter()
        hits = retriever.retrieve(query.query, max(k_values))
        elapsed_ms = (perf_counter() - start) * 1000
        if len(hits) > max(k_values):
            raise ValueError(f"retriever returned more than requested k for {query.query_id}")
        checked = RetrievalRunResult(query_id=query.query_id, hits=hits, latency_ms=elapsed_ms, configuration=run_config)
        ids = [hit.chunk_id for hit in checked.hits]
        if len(ids) != len(set(ids)):
            raise ValueError(f"retriever returned duplicate chunk IDs for {query.query_id}")
        results.append(QueryResult(
            query_id=query.query_id,
            query=query.query,
            relevant_chunk_ids=sorted(query.relevant_chunk_ids),
            hits=checked.hits,
            metrics=query_metrics(query.relevant_chunk_ids, ids, k_values),
            latency_ms=elapsed_ms,
        ))

    metric_names = [name for name in results[0].metrics if name != "reciprocal_rank"]
    aggregates = {name: mean(result.metrics[name] for result in results) for name in metric_names}
    aggregates["mrr"] = mean_reciprocal_rank([result.metrics["reciprocal_rank"] for result in results])
    latencies = [result.latency_ms for result in results]
    return BenchmarkRun(
        schema_version=SCHEMA_VERSION,
        run_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc),
        configuration_name=configuration_name,
        configuration=run_config,
        dataset=dataset,
        query_count=len(results),
        k_values=list(k_values),
        aggregate_metrics=aggregates,
        latency=LatencyStats(mean_ms=mean(latencies), p50_ms=median(latencies), p95_ms=_p95(latencies)),
        per_query_results=results,
    )
