"""Metric-by-metric comparison without a composite verdict."""

from dataclasses import dataclass

from groundtruth_rag.evaluation.runner import BenchmarkRun


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline: float
    candidate: float
    absolute_delta: float
    relative_delta_pct: float | None
    direction: str


def compare_runs(baseline: BenchmarkRun, candidate: BenchmarkRun) -> list[MetricDelta]:
    if baseline.dataset != candidate.dataset:
        raise ValueError("runs use different datasets")
    baseline_queries = {row.query_id: (row.query, row.relevant_chunk_ids) for row in baseline.per_query_results}
    candidate_queries = {row.query_id: (row.query, row.relevant_chunk_ids) for row in candidate.per_query_results}
    if baseline_queries != candidate_queries:
        raise ValueError("runs contain different golden queries or relevance judgments")
    if baseline.k_values != candidate.k_values:
        raise ValueError("runs use different k values")
    baseline_metrics = {**baseline.aggregate_metrics, "latency_mean_ms": baseline.latency.mean_ms,
                        "latency_p50_ms": baseline.latency.p50_ms, "latency_p95_ms": baseline.latency.p95_ms}
    candidate_metrics = {**candidate.aggregate_metrics, "latency_mean_ms": candidate.latency.mean_ms,
                         "latency_p50_ms": candidate.latency.p50_ms, "latency_p95_ms": candidate.latency.p95_ms}
    if baseline_metrics.keys() != candidate_metrics.keys():
        raise ValueError("runs have different metric sets")
    rows: list[MetricDelta] = []
    for name, old in baseline_metrics.items():
        new = candidate_metrics[name]
        delta = new - old
        direction = "lower" if name.startswith("latency_") else "higher"
        rows.append(MetricDelta(name, old, new, delta, delta / abs(old) * 100 if old else None, direction))
    return rows
