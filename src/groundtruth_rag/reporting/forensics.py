"""Evidence-only tracing of relevant chunks across retrieval stages."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from groundtruth_rag.evaluation.metrics import recall_at_k
from groundtruth_rag.evaluation.runner import BenchmarkRun
from groundtruth_rag.labeling.models import atomic_write
from groundtruth_rag.reporting.regression import _read_json

STAGES = ("dense", "bm25", "hybrid", "rerank")
FLAGS = (
    "DENSE_MISS", "BM25_MISS", "BOTH_BASE_RETRIEVERS_MISS", "DENSE_ONLY",
    "BM25_ONLY", "BOTH_FOUND", "HYBRID_RECOVERY", "HYBRID_REGRESSION",
    "CANDIDATE_POOL_MISS", "RERANK_RECOVERY", "RERANK_REGRESSION",
    "PERSISTENT_FAILURE", "LATE_RELEVANT_RESULT", "MULTI_RELEVANT_PARTIAL",
    "MULTI_RELEVANT_COMPLETE",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StageEvidence(StrictModel):
    found: bool | None
    rank: int | None = Field(default=None, ge=1)
    score: float | None = Field(default=None, allow_inf_nan=False)
    observed_depth: int = Field(ge=0)


class RelevantChunkTrace(StrictModel):
    relevant_chunk_id: str
    dense: StageEvidence
    bm25: StageEvidence
    hybrid: StageEvidence
    candidate_present: bool | None
    candidate_rank: int | None = Field(default=None, ge=1)
    final: StageEvidence


class QueryForensics(StrictModel):
    query_id: str
    relevant_chunk_count: int = Field(gt=0)
    first_relevant_rank: dict[str, int | None]
    relevant_found_at_k: int = Field(ge=0)
    recall_at_k: float = Field(ge=0, le=1)
    flags: list[str]
    relevant_chunks: list[RelevantChunkTrace]


class ForensicsReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    corpus_fingerprint: str
    dataset_sha256: str
    query_count: int = Field(gt=0)
    target_k: int = Field(gt=0)
    observed_depth: int = Field(gt=0)
    summary: dict[str, int]
    queries: list[QueryForensics]


def load_run(path: Path) -> BenchmarkRun:
    return BenchmarkRun.model_validate(_read_json(path))


def _validated(runs: dict[str, BenchmarkRun], target_k: int) -> tuple[list[str], int]:
    if set(runs) != set(STAGES) or target_k <= 0:
        raise ValueError("four stage runs and a positive target K are required")
    first = runs["dense"]
    depth = max(first.k_values)
    if target_k not in first.k_values:
        raise ValueError("target K must be a recorded K value")
    fingerprint = first.configuration.get("corpus_fingerprint")
    dataset_hash = first.configuration.get("dataset_sha256")
    source_hash = first.configuration.get("source_sha256")
    if not fingerprint or not dataset_hash:
        raise ValueError("corpus fingerprint and dataset hash are required")
    ids = [row.query_id for row in first.per_query_results]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query IDs")
    judgments = {row.query_id: (row.query, set(row.relevant_chunk_ids))
                 for row in first.per_query_results}
    for name, run in runs.items():
        config = run.configuration
        if (run.schema_version != "1.0" or config.get("retriever", {}).get("type") !=
                ("hybrid_rerank" if name == "rerank" else name)
                or run.dataset != first.dataset or run.query_count != first.query_count
                or run.k_values != first.k_values
                or config.get("corpus_fingerprint") != fingerprint
                or config.get("dataset_sha256") != dataset_hash
                or config.get("source_sha256") != source_hash):
            raise ValueError(f"INCOMPATIBLE RUNS: {name} provenance or schema differs")
        run_ids = [row.query_id for row in run.per_query_results]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError(f"{name}: duplicate query IDs")
        if set(run_ids) != set(ids):
            raise ValueError("INCOMPATIBLE QUERY SET")
        for row in run.per_query_results:
            if not row.relevant_chunk_ids or len(set(row.relevant_chunk_ids)) != len(row.relevant_chunk_ids):
                raise ValueError(f"{name}.{row.query_id}: invalid relevance IDs")
            if (row.query, set(row.relevant_chunk_ids)) != judgments[row.query_id]:
                raise ValueError("INCOMPATIBLE GOLDEN LABELS")
            hits = row.hits
            if (len(hits) > depth or len({hit.chunk_id for hit in hits}) != len(hits)
                    or [hit.rank for hit in hits] != list(range(1, len(hits) + 1))):
                raise ValueError(f"{name}.{row.query_id}: invalid ranking")
    hybrid = runs["hybrid"].configuration["retriever"]
    rerank = runs["rerank"].configuration["retriever"]
    dense = runs["dense"].configuration["retriever"]
    for key in ("model", "batch_size"):
        if dense.get(key) != hybrid.get(key) or hybrid.get(key) != rerank.get(key):
            raise ValueError("INCOMPATIBLE RUNS: dense configuration differs")
    for key in ("rrf_k", "dense_candidates", "bm25_candidates"):
        if hybrid.get(key) != rerank.get(key):
            raise ValueError("INCOMPATIBLE RUNS: hybrid candidate settings differ")
    if any(not isinstance(rerank.get(key), int) or rerank[key] <= 0
           for key in ("candidate_pool", "dense_candidates", "bm25_candidates")):
        raise ValueError("candidate depths must be positive integers")
    return sorted(ids), depth


def _hit_map(row) -> dict[str, Any]:
    return {hit.chunk_id: hit for hit in row.hits}


def _component(chunk_id: str, name: str, base: dict, fused: dict,
               final: dict, candidate_depth: int, observed_depth: int) -> StageEvidence:
    hit = base.get(chunk_id)
    if hit is not None:
        return StageEvidence(found=True, rank=hit.rank, score=hit.score,
                             observed_depth=observed_depth)
    carrier = fused.get(chunk_id) or final.get(chunk_id)
    if carrier is not None:
        rank = carrier.metadata.get(f"{name}_rank")
        score = carrier.metadata.get(f"{name}_score")
        return StageEvidence(found=rank is not None, rank=rank, score=score,
                             observed_depth=candidate_depth)
    return StageEvidence(found=False if candidate_depth <= observed_depth else None,
                         observed_depth=observed_depth)


def analyze_forensics(dense_run: BenchmarkRun, bm25_run: BenchmarkRun,
                      hybrid_run: BenchmarkRun, rerank_run: BenchmarkRun,
                      *, target_k: int = 3) -> ForensicsReport:
    runs = {"dense": dense_run, "bm25": bm25_run,
            "hybrid": hybrid_run, "rerank": rerank_run}
    ids, depth = _validated(runs, target_k)
    rows = {name: {row.query_id: row for row in run.per_query_results}
            for name, run in runs.items()}
    settings = rerank_run.configuration["retriever"]
    pool = settings["candidate_pool"]
    queries = []
    for query_id in ids:
        relevant = sorted(rows["dense"][query_id].relevant_chunk_ids)
        maps = {name: _hit_map(data[query_id]) for name, data in rows.items()}
        traces = []
        flags = set()
        for chunk_id in relevant:
            dense = _component(chunk_id, "dense", maps["dense"], maps["hybrid"],
                               maps["rerank"], settings["dense_candidates"], depth)
            bm25 = _component(chunk_id, "bm25", maps["bm25"], maps["hybrid"],
                              maps["rerank"], settings["bm25_candidates"], depth)
            fused = maps["hybrid"].get(chunk_id)
            final = maps["rerank"].get(chunk_id)
            hybrid_rank = fused.rank if fused else None
            if final and hybrid_rank is not None and hybrid_rank > pool:
                raise ValueError(f"INCOMPATIBLE RUNS: {query_id} final hit lies outside candidate pool")
            if final and not fused and pool <= depth:
                raise ValueError(f"INCOMPATIBLE RUNS: {query_id} final hit absent from saved candidate pool")
            if fused:
                hybrid = StageEvidence(found=True, rank=fused.rank, score=fused.score,
                                       observed_depth=depth)
            elif final:
                hybrid = StageEvidence(found=True, score=final.metadata.get("rrf_score"),
                                       observed_depth=pool)
            else:
                hybrid = StageEvidence(found=False if pool <= depth else None,
                                       observed_depth=depth)
            present = True if final or (hybrid_rank is not None and hybrid_rank <= pool) else (
                False if pool <= depth else None)
            end = StageEvidence(found=final is not None, rank=final.rank if final else None,
                                score=final.metadata.get("crossencoder_score") if final else None,
                                observed_depth=depth)
            traces.append(RelevantChunkTrace(
                relevant_chunk_id=chunk_id, dense=dense, bm25=bm25, hybrid=hybrid,
                candidate_present=present, candidate_rank=hybrid_rank if present else None,
                final=end,
            ))
            if dense.found is False:
                flags.add("DENSE_MISS")
            if bm25.found is False:
                flags.add("BM25_MISS")
            if dense.found is False and bm25.found is False:
                flags.add("BOTH_BASE_RETRIEVERS_MISS")
            elif dense.found and bm25.found is False:
                flags.add("DENSE_ONLY")
            elif bm25.found and dense.found is False:
                flags.add("BM25_ONLY")
            elif dense.found and bm25.found:
                flags.add("BOTH_FOUND")
            component_ranks = [rank for rank in (dense.rank, bm25.rank) if rank is not None]
            if hybrid_rank is not None and component_ranks:
                if hybrid_rank < min(component_ranks):
                    flags.add("HYBRID_RECOVERY")
                elif hybrid_rank > min(component_ranks):
                    flags.add("HYBRID_REGRESSION")
            if present is False and (dense.found or bm25.found):
                flags.add("CANDIDATE_POOL_MISS")
            if present and hybrid_rank is not None and final:
                if final.rank < hybrid_rank:
                    flags.add("RERANK_RECOVERY")
                elif final.rank > hybrid_rank:
                    flags.add("RERANK_REGRESSION")
            elif present and hybrid_rank is not None and final is None and hybrid_rank <= depth:
                flags.add("RERANK_REGRESSION")
            if final and final.rank > target_k:
                flags.add("LATE_RELEVANT_RESULT")
        first = {stage: min((getattr(trace, stage).rank for trace in traces
                             if getattr(trace, stage).rank is not None), default=None)
                 for stage in STAGES if stage != "rerank"}
        first["rerank"] = min((trace.final.rank for trace in traces if trace.final.rank is not None),
                              default=None)
        found = sum(trace.final.rank is not None and trace.final.rank <= target_k for trace in traces)
        recall = recall_at_k(relevant, [hit.chunk_id for hit in rows["rerank"][query_id].hits],
                             target_k)
        if found == 0:
            flags.add("PERSISTENT_FAILURE")
        if len(relevant) > 1:
            if found == len(relevant):
                flags.add("MULTI_RELEVANT_COMPLETE")
            elif found > 0:
                flags.add("MULTI_RELEVANT_PARTIAL")
        queries.append(QueryForensics(
            query_id=query_id, relevant_chunk_count=len(relevant), first_relevant_rank=first,
            relevant_found_at_k=found, recall_at_k=recall,
            flags=[flag for flag in FLAGS if flag in flags], relevant_chunks=traces,
        ))
    return ForensicsReport(
        corpus_fingerprint=dense_run.configuration["corpus_fingerprint"],
        dataset_sha256=dense_run.configuration["dataset_sha256"], query_count=len(queries),
        target_k=target_k, observed_depth=depth,
        summary={**{flag.lower(): sum(flag in row.flags for row in queries) for flag in FLAGS},
                 "unresolved_candidate_pool": sum(
                     any(trace.candidate_present is None for trace in row.relevant_chunks)
                     for row in queries)},
        queries=queries,
    )


def render_forensics(report: ForensicsReport) -> str:
    lines = ["# Retrieval Failure Forensics", "", "## Benchmark Identity", "",
             f"- Corpus fingerprint: {report.corpus_fingerprint}",
             f"- Dataset SHA-256: {report.dataset_sha256}",
             f"- Queries: {report.query_count}", f"- Target K: {report.target_k}",
             f"- Saved ranking depth: {report.observed_depth}", "",
             "## Summary", "", "| Failure / Behavior | Queries |", "| --- | ---: |"]
    lines.extend(f"| {flag} | {report.summary[flag.lower()]} |" for flag in FLAGS)
    lines.append(f"| UNRESOLVED_CANDIDATE_POOL | {report.summary['unresolved_candidate_pool']} |")
    for heading, selected in (
        ("Base Retriever Complementarity", ("DENSE_ONLY", "BM25_ONLY", "BOTH_FOUND", "BOTH_BASE_RETRIEVERS_MISS")),
        ("Hybrid Analysis", ("HYBRID_RECOVERY", "HYBRID_REGRESSION", "CANDIDATE_POOL_MISS")),
        ("CrossEncoder Analysis", ("RERANK_RECOVERY", "RERANK_REGRESSION", "PERSISTENT_FAILURE")),
    ):
        lines.extend(["", f"## {heading}", ""])
        for flag in selected:
            query_ids = [row.query_id for row in report.queries if flag in row.flags]
            lines.append(f"- {flag}: {', '.join(query_ids) or 'none'}")
    lines.extend(["", "## Query Details", ""])
    for row in report.queries:
        ranks = row.first_relevant_rank
        lines.extend([f"### {row.query_id}", "",
                      f"- Relevant chunks: {row.relevant_chunk_count}",
                      f"- Found at K: {row.relevant_found_at_k}; recall@{report.target_k}: {row.recall_at_k:g}",
                      f"- First relevant ranks: Dense {ranks['dense'] or '-'}, BM25 {ranks['bm25'] or '-'}, "
                      f"Hybrid {ranks['hybrid'] or '-'}, CrossEncoder {ranks['rerank'] or '-'}",
                      f"- Flags: {', '.join(row.flags) or 'none'}", ""])
    lines.extend(["Unknown ranks or pool membership mean the saved run depth did not establish them.", ""])
    return "\n".join(lines)


def write_forensics(report: ForensicsReport, output: Path) -> tuple[Path, Path]:
    json_path = output / "forensics.json"
    markdown_path = output / "FORENSICS.md"
    atomic_write(json_path, report.model_dump_json(indent=2) + "\n")
    atomic_write(markdown_path, render_forensics(report))
    return json_path, markdown_path


def load_forensics(path: Path) -> ForensicsReport:
    return ForensicsReport.model_validate(_read_json(path))
