import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from groundtruth_rag.cli import app
from groundtruth_rag.datasets import DatasetValidationError, load_dataset
from groundtruth_rag.evaluation.metrics import (
    hit_rate_at_k, mean_reciprocal_rank, ndcg_at_k, precision_at_k,
    recall_at_k, reciprocal_rank,
)
from groundtruth_rag.evaluation.runner import BenchmarkRun, evaluate_retriever
from groundtruth_rag.reporting.compare import compare_runs
from groundtruth_rag.retrieval.static import StaticRetriever
from groundtruth_rag.schemas import EvalQuery


@pytest.mark.parametrize(("retrieved", "expected"), [(["A"], 1), (["B", "A"], .5), (["B"], 0)])
def test_reciprocal_rank(retrieved, expected):
    assert reciprocal_rank({"A"}, retrieved) == expected


def test_recall_precision_hit_rate_and_ndcg():
    assert [recall_at_k({"A", "B"}, ["A", "C", "B"], k) for k in (1, 2, 3)] == [.5, .5, 1]
    assert precision_at_k({"A"}, ["A", "B", "C"], 1) == 1
    assert precision_at_k({"A"}, ["A", "B", "C"], 3) == pytest.approx(1 / 3)
    assert precision_at_k({"A"}, ["A"], 10) == 1
    assert hit_rate_at_k({"A"}, ["B", "A"], 1) == 0
    assert hit_rate_at_k({"A"}, ["B", "A"], 2) == 1
    assert ndcg_at_k({"A"}, ["A", "B"], 2) > ndcg_at_k({"A"}, ["B", "A"], 2)
    assert mean_reciprocal_rank([1, .5, 0]) == .5


def test_empty_and_invalid_metric_inputs():
    assert recall_at_k({"A"}, [], 10) == 0
    assert precision_at_k({"A"}, [], 10) == 0
    assert ndcg_at_k({"A"}, [], 10) == 0
    assert reciprocal_rank({"A"}, ["B", "A"]) == .5
    with pytest.raises(ValueError):
        recall_at_k({"A"}, ["A"], 0)
    with pytest.raises(ValueError):
        recall_at_k(set(), ["A"], 1)
    with pytest.raises(ValueError):
        recall_at_k({"A"}, ["A", "A"], 1)
    with pytest.raises(ValueError):
        recall_at_k(["A", "A"], ["A"], 1)


@pytest.mark.parametrize("content, message", [
    ("{bad}", "invalid JSON"),
    ('{"query_id":"q","query":"x","relevant_chunk_ids":["a"]}\n' * 2, "duplicate query_id"),
    ('{"query_id":"q","query":"x","relevant_chunk_ids":[]}', "non-empty list"),
    ('{"query_id":"q","query":"x","relevant_chunk_ids":["a","a"]}', "duplicate relevant"),
    ('{"query_id":"q","query":" ","relevant_chunk_ids":["a"]}', "query must"),
    ('{"query":"x","relevant_chunk_ids":["a"]}', "query_id must"),
    ('{"query_id":"q","query":"x"}', "relevant_chunk_ids must"),
    ('{"query_id":"q","query":"x","relevant_chunk_ids":["a"]}\n\n', "blank JSONL"),
])
def test_bad_dataset(tmp_path: Path, content: str, message: str):
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=message):
        load_dataset(path)
    result = CliRunner().invoke(app, ["validate-dataset", str(path)])
    assert result.exit_code != 0
    assert message in result.output


def test_demo_validation_and_runner(tmp_path: Path):
    demo = Path(__file__).resolve().parents[1] / "datasets" / "demo"
    queries = load_dataset(demo / "queries.jsonl")
    rankings = json.loads((demo / "rankings.json").read_text(encoding="utf-8"))
    run = evaluate_retriever(StaticRetriever(rankings), queries, dataset="demo/queries.jsonl",
                             configuration_name="static-demo", k_values=(1, 3))
    assert run.query_count == 3
    assert run.aggregate_metrics["recall@1"] == pytest.approx(1 / 2)
    assert run.aggregate_metrics["mrr"] == pytest.approx((1 + .5 + 1) / 3)
    assert run.aggregate_metrics["hit_rate@3"] == 1
    assert run.latency.p95_ms >= run.latency.p50_ms >= 0
    assert BenchmarkRun.model_validate_json(run.model_dump_json()).aggregate_metrics == run.aggregate_metrics
    assert CliRunner().invoke(app, ["validate-dataset", str(demo / "queries.jsonl")]).exit_code == 0
    output = tmp_path / "run.json"
    assert CliRunner().invoke(app, ["evaluate-demo", str(output)]).exit_code == 0
    assert BenchmarkRun.model_validate_json(output.read_text(encoding="utf-8")).query_count == 3
    comparison = CliRunner().invoke(app, ["compare", str(output), str(output)])
    assert comparison.exit_code == 0
    assert "recall@1" in comparison.output
    assert "latency_mean_ms" in comparison.output


def test_compare_deltas_and_dataset_rejection():
    queries = [EvalQuery(query_id="q", query="question", relevant_chunk_ids={"A"})]
    old = evaluate_retriever(StaticRetriever({"question": ["B", "A"]}), queries,
                             dataset="same", configuration_name="old", k_values=(1, 2))
    new = evaluate_retriever(StaticRetriever({"question": ["A", "B"]}), queries,
                             dataset="same", configuration_name="new", k_values=(1, 2))
    rows = {row.name: row for row in compare_runs(old, new)}
    assert rows["recall@1"].absolute_delta == 1
    assert rows["mrr"].absolute_delta == .5
    assert rows["mrr"].relative_delta_pct == 100
    assert rows["latency_mean_ms"].direction == "lower"
    new.dataset = "different"
    with pytest.raises(ValueError, match="different datasets"):
        compare_runs(old, new)
    new.dataset = "same"
    new.per_query_results[0].relevant_chunk_ids = ["B"]
    with pytest.raises(ValueError, match="different golden queries"):
        compare_runs(old, new)


def test_runner_rejects_bad_inputs():
    query = EvalQuery(query_id="q", query="question", relevant_chunk_ids={"A"})
    retriever = StaticRetriever({"question": ["A"]})
    with pytest.raises(ValueError, match="at least one query"):
        evaluate_retriever(retriever, [], dataset="d", configuration_name="c")
    with pytest.raises(ValueError, match="unique positive"):
        evaluate_retriever(retriever, [query], dataset="d", configuration_name="c", k_values=(0,))
    with pytest.raises(ValueError, match="query IDs must be unique"):
        evaluate_retriever(retriever, [query, query], dataset="d", configuration_name="c")
