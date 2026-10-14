"""Summary and ranking-only error analysis from completed run files."""

from typing import Any

from groundtruth_rag.evaluation.runner import BenchmarkRun

CONFIG_NAMES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def summarize_runs(runs: dict[str, BenchmarkRun]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if set(runs) != set(CONFIG_NAMES):
        raise ValueError("benchmark requires dense, bm25, hybrid, and hybrid_rerank runs")
    first = runs["dense"]
    for name, run in runs.items():
        if run.dataset != first.dataset or run.k_values != first.k_values:
            raise ValueError(f"{name}: incompatible dataset or K values")
        if run.configuration.get("corpus_fingerprint") != first.configuration.get("corpus_fingerprint"):
            raise ValueError(f"{name}: incompatible corpus fingerprint")
        if {item.query_id for item in run.per_query_results} != {
            item.query_id for item in first.per_query_results
        }:
            raise ValueError(f"{name}: incompatible queries")
        judgments = {item.query_id: (item.query, item.relevant_chunk_ids)
                     for item in run.per_query_results}
        baseline_judgments = {item.query_id: (item.query, item.relevant_chunk_ids)
                              for item in first.per_query_results}
        if judgments != baseline_judgments:
            raise ValueError(f"{name}: incompatible relevance judgments")
    summary = {
        "schema_version": "1.0",
        "corpus_fingerprint": first.configuration.get("corpus_fingerprint"),
        "dataset": first.dataset, "query_count": first.query_count,
        "k_values": first.k_values,
        "runs": {name: {"run_id": run.run_id, "metrics": run.aggregate_metrics,
                        "latency": run.latency.model_dump()} for name, run in runs.items()},
    }
    lines = ["# OSHA Retrieval Benchmark", "", "Human-verified labels; one fixed corpus and chunk configuration.", "",
             "| Retriever | Recall@3 | MRR | nDCG@3 | p95 latency (ms) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in CONFIG_NAMES:
        run = runs[name]
        metrics = run.aggregate_metrics
        lines.append(f"| {name} | {metrics['recall@3']:.4f} | {metrics['mrr']:.4f} | "
                     f"{metrics['ndcg@3']:.4f} | {run.latency.p95_ms:.2f} |")
    by_run = {name: {item.query_id: item for item in run.per_query_results}
              for name, run in runs.items()}
    misses = {name: sorted(qid for qid, item in records.items()
                           if item.metrics["recall@3"] == 0)
              for name, records in by_run.items()}
    late = {name: sorted(qid for qid, item in records.items()
                         if item.metrics["recall@3"] == 0 and item.metrics.get("recall@10", 0) > 0)
            for name, records in by_run.items()}
    def improvement(before: str, after: str) -> list[str]:
        return sorted(qid for qid in by_run[before]
                      if by_run[after][qid].metrics["recall@3"] > by_run[before][qid].metrics["recall@3"])

    def moved_up() -> list[str]:
        result = []
        for qid, hybrid in by_run["hybrid"].items():
            reranked = by_run["hybrid_rerank"][qid]
            relevant = set(hybrid.relevant_chunk_ids)
            old_rank = min((hit.rank for hit in hybrid.hits if hit.chunk_id in relevant), default=float("inf"))
            new_rank = min((hit.rank for hit in reranked.hits if hit.chunk_id in relevant), default=float("inf"))
            if new_rank < old_rank:
                result.append(qid)
        return sorted(result)

    analysis = {
        "recall_at_3_zero": misses,
        "relevant_only_after_rank_3": late,
        "bm25_succeeds_dense_fails": sorted(set(misses["dense"]) - set(misses["bm25"])),
        "dense_succeeds_bm25_fails": sorted(set(misses["bm25"]) - set(misses["dense"])),
        "hybrid_improves_over_dense": improvement("dense", "hybrid"),
        "hybrid_improves_over_bm25": improvement("bm25", "hybrid"),
        "reranking_moves_first_relevant_up": moved_up(),
    }
    return summary, "\n".join(lines) + "\n", analysis
