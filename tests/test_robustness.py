"""Synthetic, offline checks for paired quality and latency evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from groundtruth_rag.evaluation.runner import evaluate_retriever
from groundtruth_rag.robustness import (
    RobustnessConfig, analyze, render_report, run_latency_trials, write_report,
)
from groundtruth_rag.retrieval.static import StaticRetriever
from groundtruth_rag.schemas import EvalQuery


def runs(old, new, *, relevant=None):
    labels = relevant or {"q1": {"a"}, "q2": {"b"}, "q3": {"c"}, "q4": {"d"}}
    queries = [EvalQuery(query_id=q, query=q, relevant_chunk_ids=ids) for q, ids in labels.items()]
    meta = {"retriever": {"type": "hybrid"}, "corpus_fingerprint": "corpus",
            "dataset_sha256": "dataset"}
    def make(rankings, name):
        return evaluate_retriever(StaticRetriever(rankings), queries, dataset="fixed",
                                  configuration_name=name, configuration=meta)
    return make(old, "baseline"), make(new, "candidate")


def config(metrics=("recall@3", "mrr", "ndcg@3", "hit_rate@3"), iterations=1000, seed=2026):
    return RobustnessConfig.model_validate({"quality": {"metrics": list(metrics),
                                            "bootstrap_iterations": iterations,
                                            "random_seed": seed}})


def test_unchanged_and_no_source_text(tmp_path, monkeypatch):
    ranking = {q: ["a", "b", "c", "d"] for q in ("q1", "q2", "q3", "q4")}
    base, cand = runs(ranking, ranking)
    report = analyze(base, cand, config())
    assert all(row["assessment"] == "UNCHANGED" for row in report["metrics"].values())
    assert all(row["ci_lower"] == row["ci_upper"] == 0 for row in report["metrics"].values())
    monkeypatch.chdir(tmp_path)
    write_report(report, Path("reports/robustness/demo"))
    payload = (tmp_path / "reports/robustness/demo/robustness.json").read_text()
    assert json.loads(payload)["schema_version"] == "1.0"
    assert "per_query_results" not in payload and "source_text" not in payload and '"hits"' not in payload
    assert "# Retrieval Robustness Analysis" in render_report(report)


@pytest.mark.parametrize(("old", "new", "expected"), [
    (["x", "y", "z", "a", "b", "c", "d"], ["a", "b", "c", "d"], "STABLE_IMPROVEMENT"),
    (["a", "b", "c", "d"], ["x", "y", "z", "a", "b", "c", "d"], "STABLE_REGRESSION"),
])
def test_consistent_effect(old, new, expected):
    labels = {q: {"a"} for q in ("q1", "q2", "q3", "q4")}
    base, cand = runs({q: old for q in ("q1", "q2", "q3", "q4")},
                      {q: new for q in ("q1", "q2", "q3", "q4")}, relevant=labels)
    report = analyze(base, cand, config())
    assert report["metrics"]["recall@3"]["assessment"] == expected
    assert report["metrics"]["mrr"]["assessment"] == expected
    assert report["metrics"]["ndcg@3"]["assessment"] == expected
    assert report["metrics"]["hit_rate@3"]["assessment"] == expected


def test_mixed_effect_and_seed_reproducibility():
    labels = {q: {"a"} for q in ("q1", "q2", "q3", "q4")}
    base, cand = runs({"q1": ["x", "y", "a"], "q2": ["x", "y", "a"],
                       "q3": ["a", "x", "y"], "q4": ["a", "x", "y"]},
                      {"q1": ["a", "x", "y"], "q2": ["a", "x", "y"],
                       "q3": ["x", "y", "a"], "q4": ["x", "y", "a"]}, relevant=labels)
    first = analyze(base, cand, config())
    assert first == analyze(base, cand, config())
    assert first["metrics"]["mrr"]["assessment"] == "INCONCLUSIVE"
    assert analyze(base, cand, config(seed=99))["random_seed"] == 99


@pytest.mark.parametrize("quality", [{"bootstrap_iterations": 0}, {"confidence_level": 1},
                                      {"confidence_level": 0}, {"metrics": ["made_up"]}])
def test_bad_config(quality):
    with pytest.raises(ValidationError):
        RobustnessConfig.model_validate({"quality": quality})


def test_compatibility_checks():
    rankings = {q: ["a", "b", "c", "d"] for q in ("q1", "q2", "q3", "q4")}
    base, cand = runs(rankings, rankings)
    for mutation in (
        lambda r: setattr(r.per_query_results[0], "query_id", "other"),
        lambda r: r.configuration.update(corpus_fingerprint="other"),
        lambda r: setattr(r.per_query_results[0], "relevant_chunk_ids", ["other"]),
        lambda r: r.configuration["retriever"].update(type="bm25"),
    ):
        changed = cand.model_copy(deep=True)
        mutation(changed)
        with pytest.raises(ValueError):
            analyze(base, changed, config())


def test_multiple_relevant_chunks_and_metric_definitions():
    labels = {"q1": {"a", "b"}, "q2": {"a", "b"}}
    base, cand = runs({"q1": ["x", "a", "y", "b"], "q2": ["x", "a", "y", "b"]},
                      {"q1": ["a", "b", "x"], "q2": ["a", "b", "x"]}, relevant=labels)
    result = analyze(base, cand, config())
    assert result["metrics"]["recall@3"]["baseline"] == 0.5
    assert result["metrics"]["recall@3"]["candidate"] == 1.0
    assert result["metrics"]["mrr"]["candidate"] == 1.0


def test_latency_warmup_and_aggregation(monkeypatch, tmp_path):
    from groundtruth_rag import robustness
    context = {"chunks": [object()], "queries": [object()], "dataset_path": Path("dataset"),
               "benchmark": SimpleNamespace(k_values=[1, 3, 5, 10]),
               "fingerprint": "corpus", "dataset_sha256": "dataset"}
    resolved = {"candidate": {"config": object(), "retriever_type": "hybrid",
                              "config_sha256": "hash"}}
    monkeypatch.setattr(robustness, "validate_experiment", lambda *_: (
        SimpleNamespace(experiment_id="experiment"), resolved, context))
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(len(calls))
        value = float(len(calls) - 1)
        return SimpleNamespace(latency=SimpleNamespace(model_dump=lambda: {
            "mean_ms": value, "p50_ms": value, "p95_ms": value}))
    class Fake:
        def index(self, chunks):
            assert chunks == context["chunks"]
    report = run_latency_trials(tmp_path, Path("spec"), "candidate", config(),
                                builder=lambda _: Fake(), evaluator=evaluate)
    assert len(calls) == 6
    assert [row["p95_ms"] for row in report["trials"]] == [1, 2, 3, 4, 5]
    assert report["summary"]["mean_trial_mean_ms"] == 3
    assert report["summary"]["median_trial_p95_ms"] == 3
    assert report["summary"]["min_p95_ms"] == 1
    assert report["summary"]["max_p95_ms"] == 5
    assert report["warmup_runs"] == 1
    assert report["measured_runs"] == 5
    assert report["environment"]["os"]


def test_zero_latency_and_nonfinite_rejection(monkeypatch, tmp_path):
    from groundtruth_rag import robustness
    context = {"chunks": [], "queries": [object()], "dataset_path": Path("dataset"),
               "benchmark": SimpleNamespace(k_values=[1]), "fingerprint": "corpus",
               "dataset_sha256": "dataset"}
    resolved = {"candidate": {"config": object(), "retriever_type": "hybrid",
                              "config_sha256": "hash"}}
    monkeypatch.setattr(robustness, "validate_experiment", lambda *_: (
        SimpleNamespace(experiment_id="experiment"), resolved, context))
    class Fake:
        def index(self, chunks):
            pass
    def evaluator(value):
        return lambda *args, **kwargs: SimpleNamespace(latency=SimpleNamespace(
            model_dump=lambda: {"mean_ms": value, "p50_ms": value, "p95_ms": value}))
    zero = run_latency_trials(tmp_path, Path("spec"), "candidate", config(),
                              builder=lambda _: Fake(), evaluator=evaluator(0))
    assert zero["summary"]["median_trial_p95_ms"] == 0
    with pytest.raises(ValueError, match="invalid measured latency"):
        run_latency_trials(tmp_path, Path("spec"), "candidate", config(),
                           builder=lambda _: Fake(), evaluator=evaluator(float("nan")))


def test_nonfinite_quality_rejected():
    ranking = {q: ["a", "b", "c", "d"] for q in ("q1", "q2", "q3", "q4")}
    base, cand = runs(ranking, ranking)
    cand.per_query_results[0].metrics["reciprocal_rank"] = float("nan")
    with pytest.raises(ValueError):
        analyze(base, cand, config())
