"""Text-free query ranking baselines and aligned retrieval diagnostics."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundtruth_rag.evaluation.metrics import query_metrics
from groundtruth_rag.evaluation.runner import BenchmarkRun
from groundtruth_rag.labeling.models import atomic_write
from groundtruth_rag.reporting.regression import (
    Baseline, BenchmarkSummary, PromotionProvenance, _read_json, load_summary,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryRank(StrictModel):
    query_id: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    relevant_positions: list[int]

    @model_validator(mode="after")
    def valid_rank(self) -> "QueryRank":
        if (len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids)
                or not all(self.relevant_chunk_ids)):
            raise ValueError("duplicate or empty relevant chunk IDs")
        if (self.relevant_chunk_ids != sorted(self.relevant_chunk_ids)
                or self.relevant_positions != sorted(set(self.relevant_positions))
                or any(rank < 1 for rank in self.relevant_positions)
                or len(self.relevant_positions) > len(self.relevant_chunk_ids)):
            raise ValueError("invalid relevant positions or IDs")
        return self


class RetrieverRanks(StrictModel):
    run_id: str = Field(min_length=1)
    configuration_name: str = Field(min_length=1)
    retriever_type: str = Field(min_length=1)
    queries: list[QueryRank]

    @model_validator(mode="after")
    def unique_queries(self) -> "RetrieverRanks":
        if len({query.query_id for query in self.queries}) != len(self.queries):
            raise ValueError("duplicate query IDs")
        return self


class QueryBaseline(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_name: str = Field(min_length=1)
    corpus_fingerprint: str = Field(min_length=1)
    dataset_sha256: str = Field(min_length=1)
    query_count: int = Field(gt=0)
    k_values: list[int]
    retrievers: dict[str, RetrieverRanks]
    promotion_provenance: PromotionProvenance | None = None

    @model_validator(mode="after")
    def valid_baseline(self) -> "QueryBaseline":
        if not self.retrievers or not self.k_values or len(set(self.k_values)) != len(self.k_values):
            raise ValueError("query baseline needs retrievers and unique K values")
        ids = None
        labels = None
        for run in self.retrievers.values():
            if len(run.queries) != self.query_count:
                raise ValueError("query count differs")
            current = {row.query_id: row.relevant_chunk_ids for row in run.queries}
            if ids is not None and set(current) != ids:
                raise ValueError("INCOMPATIBLE QUERY SET")
            if labels is not None and current != labels:
                raise ValueError("INCOMPATIBLE GOLDEN LABELS")
            ids, labels = set(current), current
        return self


def _load_run(summary_path: Path, summary: BenchmarkSummary, name: str) -> BenchmarkRun:
    if name not in summary.runs:
        raise ValueError(f"missing retriever {name}")
    path = summary_path.parent / f"{name}.json"
    run = BenchmarkRun.model_validate(_read_json(path))
    aggregate = summary.runs[name]
    if (run.run_id != aggregate.run_id or run.dataset != summary.dataset
            or run.query_count != summary.query_count or run.k_values != summary.k_values
            or run.configuration.get("corpus_fingerprint") != summary.corpus_fingerprint
            or run.configuration.get("dataset_sha256") != summary.dataset_sha256
            or run.configuration.get("retriever", {}).get("type") != name
            or run.aggregate_metrics != aggregate.metrics
            or run.latency.model_dump() != aggregate.latency):
        raise ValueError(f"{name}: run JSON contradicts summary")
    ids = [row.query_id for row in run.per_query_results]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query IDs")
    for row in run.per_query_results:
        hit_ids = [hit.chunk_id for hit in row.hits]
        if (not row.relevant_chunk_ids or len(set(row.relevant_chunk_ids)) != len(row.relevant_chunk_ids)
                or len(set(hit_ids)) != len(hit_ids)
                or len(hit_ids) > max(run.k_values)
                or [hit.rank for hit in row.hits] != list(range(1, len(row.hits) + 1))):
            raise ValueError(f"{name}.{row.query_id}: invalid ranking or labels")
        expected = query_metrics(row.relevant_chunk_ids, hit_ids, run.k_values)
        if any(abs(row.metrics.get(key, -1) - value) > 1e-9 for key, value in expected.items()):
            raise ValueError(f"{name}.{row.query_id}: metrics contradict ranking")
    return run


def _positions(relevant: list[str], hits: list[str]) -> list[int]:
    wanted = set(relevant)
    return [rank for rank, chunk_id in enumerate(hits, 1) if chunk_id in wanted]


def create_query_baseline(summary_path: Path, aggregate: Baseline, output: Path,
                          *, force: bool = False) -> QueryBaseline:
    summary = load_summary(summary_path)
    if (aggregate.corpus_fingerprint != summary.corpus_fingerprint
            or aggregate.dataset_sha256 is None
            or aggregate.dataset_sha256 != summary.dataset_sha256
            or aggregate.query_count != summary.query_count
            or aggregate.k_values != summary.k_values
            or set(aggregate.runs) != set(summary.runs)):
        raise ValueError("INCOMPATIBLE BENCHMARKS: aggregate baseline differs from source runs")
    runs = {}
    for name in summary.runs:
        run = _load_run(summary_path, summary, name)
        if run.run_id != aggregate.runs[name].run_id:
            raise ValueError(f"{name}: run ID differs from aggregate baseline")
        runs[name] = RetrieverRanks(
            run_id=run.run_id, configuration_name=run.configuration_name,
            retriever_type=name,
            queries=[QueryRank(query_id=row.query_id,
                               relevant_chunk_ids=sorted(row.relevant_chunk_ids),
                               relevant_positions=_positions(row.relevant_chunk_ids,
                                                             [hit.chunk_id for hit in row.hits]))
                     for row in run.per_query_results],
        )
    baseline = QueryBaseline(
        benchmark_name=aggregate.name, corpus_fingerprint=summary.corpus_fingerprint,
        dataset_sha256=summary.dataset_sha256, query_count=summary.query_count,
        k_values=summary.k_values, retrievers=runs,
        promotion_provenance=aggregate.promotion_provenance,
    )
    content = baseline.model_dump_json(indent=2) + "\n"
    if force:
        atomic_write(output, content)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    return baseline


def load_query_baseline(path: Path) -> QueryBaseline:
    return QueryBaseline.model_validate(_read_json(path))


def compare_queries(query_base: QueryBaseline, aggregate: Baseline,
                    candidate_path: Path, summary: BenchmarkSummary,
                    retrievers: list[str], *, critical_k: list[int],
                    warn_rank_drop: int | None) -> dict:
    if (query_base.benchmark_name != aggregate.name
            or query_base.corpus_fingerprint != aggregate.corpus_fingerprint
            or query_base.corpus_fingerprint != summary.corpus_fingerprint
            or query_base.dataset_sha256 != aggregate.dataset_sha256
            or query_base.dataset_sha256 != summary.dataset_sha256
            or query_base.query_count != aggregate.query_count
            or query_base.query_count != summary.query_count
            or query_base.k_values != aggregate.k_values
            or query_base.k_values != summary.k_values):
        raise ValueError("INCOMPATIBLE BENCHMARKS: query baseline provenance differs")
    if set(critical_k) - set(query_base.k_values):
        raise ValueError("INCOMPATIBLE BENCHMARKS: critical K missing")
    analyzed = {}
    for name in retrievers:
        old = query_base.retrievers.get(name)
        if old is None or name not in aggregate.runs or old.run_id != aggregate.runs[name].run_id:
            raise ValueError(f"INCOMPATIBLE BENCHMARKS: query baseline missing {name}")
        run = _load_run(candidate_path, summary, name)
        if old.retriever_type != name:
            raise ValueError(f"INCOMPATIBLE BENCHMARKS: retriever identity differs for {name}")
        old_rows = {row.query_id: row for row in old.queries}
        new_rows = {row.query_id: row for row in run.per_query_results}
        if set(old_rows) != set(new_rows):
            raise ValueError("INCOMPATIBLE QUERY SET")
        rows = []
        for query_id in sorted(old_rows):
            before, after = old_rows[query_id], new_rows[query_id]
            if set(before.relevant_chunk_ids) != set(after.relevant_chunk_ids):
                raise ValueError("INCOMPATIBLE GOLDEN LABELS")
            old_pos = before.relevant_positions
            new_pos = _positions(after.relevant_chunk_ids, [hit.chunk_id for hit in after.hits])
            first_old = old_pos[0] if old_pos else None
            first_new = new_pos[0] if new_pos else None
            flags = []
            if first_old is None and first_new is not None:
                flags.append("RANK_IMPROVED")
            elif first_new is None and first_old is not None:
                flags.append("RANK_REGRESSED")
            elif first_old is not None and first_new is not None:
                if first_new < first_old:
                    flags.append("RANK_IMPROVED")
                elif first_new > first_old:
                    flags.append("RANK_REGRESSED")
            for k in query_base.k_values:
                old_hit = any(rank <= k for rank in old_pos)
                new_hit = any(rank <= k for rank in new_pos)
                if old_hit and not new_hit:
                    flags.append(f"NEW_FAILURE_AT_{k}")
                elif new_hit and not old_hit:
                    flags.append(f"RECOVERED_AT_{k}")
            if not new_pos:
                flags.append("STILL_FAILED_AT_K")
            elif critical_k and all(rank > k for rank in new_pos for k in critical_k):
                flags.append("STILL_FAILED_AT_K")
            old_recall = {str(k): len([rank for rank in old_pos if rank <= k]) / len(before.relevant_chunk_ids)
                          for k in query_base.k_values}
            new_recall = {str(k): query_metrics(after.relevant_chunk_ids,
                          [hit.chunk_id for hit in after.hits], [k])[f"recall@{k}"]
                          for k in query_base.k_values}
            if any(new_recall[str(k)] > old_recall[str(k)] for k in query_base.k_values):
                flags.append("RECALL_IMPROVED")
            if any(new_recall[str(k)] < old_recall[str(k)] for k in query_base.k_values):
                flags.append("RECALL_REGRESSED")
            if old_pos == new_pos:
                flags.insert(0, "UNCHANGED")
            new_critical = [k for k in critical_k if f"NEW_FAILURE_AT_{k}" in flags]
            drop = (first_new - first_old) if first_old is not None and first_new is not None else None
            severity = ("FAIL" if new_critical else "WARN" if drop is not None
                        and warn_rank_drop is not None and drop >= warn_rank_drop
                        else "INFO")
            rows.append({"query_id": query_id, "baseline_first_relevant_rank": first_old,
                         "candidate_first_relevant_rank": first_new, "rank_delta": drop,
                         "baseline_recall_at_k": old_recall, "candidate_recall_at_k": new_recall,
                         "flags": flags, "severity": severity})
        analyzed[name] = {"new_failures": {str(k): [r["query_id"] for r in rows
                          if f"NEW_FAILURE_AT_{k}" in r["flags"]] for k in query_base.k_values},
                          "recoveries": {str(k): [r["query_id"] for r in rows
                          if f"RECOVERED_AT_{k}" in r["flags"]] for k in query_base.k_values},
                          "rank_improvements": [r["query_id"] for r in rows if "RANK_IMPROVED" in r["flags"]],
                          "rank_regressions": [r["query_id"] for r in rows if "RANK_REGRESSED" in r["flags"]],
                          "queries": rows}
    return {"enabled": True, "schema_version": "1.0", "retrievers": analyzed}
