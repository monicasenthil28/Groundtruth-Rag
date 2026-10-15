"""Synthetic stage traces, with no corpus, network, or model dependencies."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from groundtruth_rag.cli import app
from groundtruth_rag.evaluation.runner import evaluate_retriever
from groundtruth_rag.reporting.forensics import (
    ForensicsReport, analyze_forensics, render_forensics, write_forensics,
)
from groundtruth_rag.retrieval.static import StaticRetriever
from groundtruth_rag.schemas import EvalQuery


OTHER = [f"x{i}" for i in range(20)]
KS = [1, 3, 5, 10]


def _runs(dense, bm25, hybrid, rerank, *, relevant=("a",), pool=10):
    query = EvalQuery(query_id="synthetic-1", query="Synthetic question text",
                      relevant_chunk_ids=set(relevant))
    configs = {
        "dense": {"type": "dense", "model": "fixed", "batch_size": 2},
        "bm25": {"type": "bm25"},
        "hybrid": {"type": "hybrid", "model": "fixed", "batch_size": 2,
                   "rrf_k": 60, "dense_candidates": 10, "bm25_candidates": 10},
        "rerank": {"type": "hybrid_rerank", "model": "fixed", "batch_size": 2,
                   "rrf_k": 60, "dense_candidates": 10, "bm25_candidates": 10,
                   "candidate_pool": pool},
    }
    rankings = {"dense": dense, "bm25": bm25, "hybrid": hybrid, "rerank": rerank}
    result = {}
    for name in configs:
        result[name] = evaluate_retriever(
            StaticRetriever({query.query: rankings[name]}), [query], dataset="synthetic.jsonl",
            configuration_name=name, configuration={
                "retriever": configs[name], "corpus_fingerprint": "fixed-corpus",
                "dataset_sha256": "fixed-golden", "source_sha256": "fixed-source",
            }, k_values=KS,
        )
    for name in ("hybrid", "rerank"):
        for hit in result[name].per_query_results[0].hits:
            metadata = {}
            for component in ("dense", "bm25"):
                try:
                    position = rankings[component].index(hit.chunk_id) + 1
                except ValueError:
                    continue
                metadata[f"{component}_rank"] = position
                metadata[f"{component}_score"] = 1.0 / position
            metadata["rrf_score"] = 0.1
            if name == "rerank":
                metadata["crossencoder_score"] = hit.score
            hit.metadata = metadata
    return result


def _analyze(runs, k=3):
    return analyze_forensics(runs["dense"], runs["bm25"], runs["hybrid"],
                             runs["rerank"], target_k=k).queries[0]


@pytest.mark.parametrize("dense,bm25,expected", [
    (["a", *OTHER[:9]], OTHER[:10], {"DENSE_ONLY", "BM25_MISS"}),
    (OTHER[:10], ["a", *OTHER[:9]], {"BM25_ONLY", "DENSE_MISS"}),
    (["a", *OTHER[:9]], ["a", *OTHER[:9]], {"BOTH_FOUND"}),
    (OTHER[:10], OTHER[:10], {"BOTH_BASE_RETRIEVERS_MISS"}),
])
def test_base_complementarity(dense, bm25, expected):
    row = _analyze(_runs(dense, bm25, ["a", *OTHER[:9]], ["a", *OTHER[:9]]))
    assert expected <= set(row.flags)


def test_hybrid_and_rerank_rank_changes():
    component = [*OTHER[:4], "a", *OTHER[4:9]]
    fused = ["a", *OTHER[:9]]
    final = [*OTHER[:3], "a", *OTHER[3:9]]
    row = _analyze(_runs(component, component, fused, final))
    assert "HYBRID_RECOVERY" in row.flags
    assert "RERANK_REGRESSION" in row.flags
    row = _analyze(_runs(["a", *OTHER[:9]], ["a", *OTHER[:9]], final, fused))
    assert "HYBRID_REGRESSION" in row.flags
    assert "RERANK_RECOVERY" in row.flags


def test_candidate_pool_miss_only_when_proven():
    base = ["a", *OTHER[:9]]
    no_relevant = OTHER[:10]
    row = _analyze(_runs(base, no_relevant, no_relevant, no_relevant, pool=3))
    assert "CANDIDATE_POOL_MISS" in row.flags
    assert row.relevant_chunks[0].candidate_present is False
    row = _analyze(_runs(base, no_relevant, no_relevant, no_relevant, pool=20))
    assert "CANDIDATE_POOL_MISS" not in row.flags
    assert row.relevant_chunks[0].candidate_present is None


def test_rerank_truncation_is_observed_regression():
    base = ["a", *OTHER[:9]]
    row = _analyze(_runs(base, base, base, OTHER[:10]))
    assert "RERANK_REGRESSION" in row.flags
    assert row.relevant_chunks[0].candidate_present is True
    assert row.relevant_chunks[0].final.rank is None


def test_contradictory_candidate_pool_rejected():
    base = ["a", *OTHER[:9]]
    runs = _runs(base, base, OTHER[:10], base, pool=3)
    with pytest.raises(ValueError, match="absent from saved candidate pool"):
        _analyze(runs)


def test_deeper_component_rank_from_final_metadata():
    missing = OTHER[:10]
    final = [*OTHER[:9], "a"]
    runs = _runs(missing, missing, missing, final, pool=20)
    for name in ("hybrid", "rerank"):
        runs[name].configuration["retriever"]["dense_candidates"] = 20
        runs[name].configuration["retriever"]["bm25_candidates"] = 20
    hit = runs["rerank"].per_query_results[0].hits[-1]
    hit.metadata["dense_rank"] = 12
    hit.metadata["dense_score"] = 0.5
    row = _analyze(runs)
    trace = row.relevant_chunks[0]
    assert trace.dense.rank == 12 and trace.dense.found is True
    assert trace.bm25.found is False
    assert trace.candidate_present is True and trace.candidate_rank is None
    assert "DENSE_ONLY" in row.flags
    assert "RERANK_REGRESSION" not in row.flags


def test_duplicate_query_ids_rejected():
    runs = _runs(["a", *OTHER[:9]], ["a", *OTHER[:9]],
                 ["a", *OTHER[:9]], ["a", *OTHER[:9]])
    for run in runs.values():
        run.per_query_results.append(run.per_query_results[0].model_copy(deep=True))
        run.query_count = 2
    with pytest.raises(ValueError, match="duplicate query IDs"):
        _analyze(runs)


def test_final_failure_late_and_multi_relevance():
    late = [*OTHER[:4], "a", *OTHER[4:9]]
    row = _analyze(_runs(late, late, late, late))
    assert "LATE_RELEVANT_RESULT" in row.flags
    assert "PERSISTENT_FAILURE" in row.flags
    assert row.first_relevant_rank["rerank"] == 5
    row = _analyze(_runs(OTHER[:10], OTHER[:10], OTHER[:10], OTHER[:10]))
    assert "PERSISTENT_FAILURE" in row.flags
    partial = _analyze(_runs(["a", "b", *OTHER[:8]], ["a", "b", *OTHER[:8]],
                             ["a", "b", *OTHER[:8]], ["a", *OTHER[:9]], relevant=("a", "b")))
    assert "MULTI_RELEVANT_PARTIAL" in partial.flags
    assert partial.relevant_found_at_k == 1 and partial.recall_at_k == 0.5
    complete = _analyze(_runs(["a", "b", *OTHER[:8]], ["a", "b", *OTHER[:8]],
                              ["a", "b", *OTHER[:8]], ["a", "b", *OTHER[:8]],
                              relevant=("a", "b")))
    assert "MULTI_RELEVANT_COMPLETE" in complete.flags
    assert complete.recall_at_k == 1.0


def test_incompatible_runs_and_duplicate_hits():
    runs = _runs(["a", *OTHER[:9]], ["a", *OTHER[:9]],
                 ["a", *OTHER[:9]], ["a", *OTHER[:9]])
    runs["bm25"].configuration["corpus_fingerprint"] = "different"
    with pytest.raises(ValueError, match="INCOMPATIBLE RUNS"):
        _analyze(runs)
    runs["bm25"].configuration["corpus_fingerprint"] = "fixed-corpus"
    runs["bm25"].configuration["dataset_sha256"] = "different"
    with pytest.raises(ValueError, match="INCOMPATIBLE RUNS"):
        _analyze(runs)
    runs["bm25"].configuration["dataset_sha256"] = "fixed-golden"
    runs["bm25"].per_query_results[0].query_id = "wrong"
    with pytest.raises(ValueError, match="INCOMPATIBLE QUERY SET"):
        _analyze(runs)
    runs["bm25"].per_query_results[0].query_id = "synthetic-1"
    runs["bm25"].per_query_results[0].relevant_chunk_ids = ["different"]
    with pytest.raises(ValueError, match="INCOMPATIBLE GOLDEN LABELS"):
        _analyze(runs)
    runs["bm25"].per_query_results[0].relevant_chunk_ids = ["a"]
    hits = runs["bm25"].per_query_results[0].hits
    hits[1].chunk_id = hits[0].chunk_id
    with pytest.raises(ValueError, match="invalid ranking"):
        _analyze(runs)


def test_json_markdown_cli_and_inspection(tmp_path: Path):
    runs = _runs(["a", *OTHER[:9]], OTHER[:10],
                 ["a", *OTHER[:9]], [*OTHER[:3], "a", *OTHER[3:9]])
    paths = {}
    for name, run in runs.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(run.model_dump_json(), encoding="utf-8")
    output = tmp_path / "out"
    args = ["analyze-failures", "--dense", str(paths["dense"]),
            "--bm25", str(paths["bm25"]), "--hybrid", str(paths["hybrid"]),
            "--rerank", str(paths["rerank"]), "--output", str(output), "--target-k", "5"]
    cli = CliRunner()
    analyzed = cli.invoke(app, args)
    assert analyzed.exit_code == 0, analyzed.output
    data = json.loads((output / "forensics.json").read_text())
    assert data["schema_version"] == "1.0" and data["target_k"] == 5
    assert "Synthetic question text" not in (output / "forensics.json").read_text()
    assert "Synthetic question text" not in (output / "FORENSICS.md").read_text()
    assert "Base Retriever Complementarity" in render_forensics(ForensicsReport.model_validate(data))
    assert cli.invoke(app, ["failure-inspect", str(output / "forensics.json"),
                            "--query", "synthetic-1"]).exit_code == 0
    missing = cli.invoke(app, ["failure-inspect", str(output / "forensics.json"),
                               "--query", "missing"])
    assert missing.exit_code == 2 and "query not found" in missing.output
    assert write_forensics(ForensicsReport.model_validate(data), output)[0].exists()
