"""Offline experiment validation, policy, and report checks."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from groundtruth_rag.cli import app
from groundtruth_rag.evaluation.runner import evaluate_retriever
from groundtruth_rag.experiments import (
    load_experiment_report, pareto_status, promotion_status,
    run_experiment, validate_experiment,
)
from groundtruth_rag.ingestion.corpus import write_corpus
from groundtruth_rag.labeling.manifest import CorpusManifest, file_sha256
from groundtruth_rag.reporting.query_regression import create_query_baseline
from groundtruth_rag.reporting.regression import create_baseline
from groundtruth_rag.retrieval.dense import corpus_fingerprint
from groundtruth_rag.retrieval.static import StaticRetriever
from groundtruth_rag.schemas import DocumentChunk, EvalQuery


def _write(root: Path, name: str, value) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value) if name.endswith(".yaml") else json.dumps(value),
                    encoding="utf-8")
    return path


@pytest.fixture
def package(tmp_path: Path):
    root = tmp_path
    chunks = [DocumentChunk(chunk_id="a", document_id="doc", source="doc.pdf", page=1,
                            text="relevant passage"),
              DocumentChunk(chunk_id="b", document_id="doc", source="doc.pdf", page=1,
                            text="other passage")]
    corpus = root / "datasets/osha/corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    write_corpus(corpus, chunks)
    fingerprint = corpus_fingerprint(chunks)
    source_hash = "1" * 64
    manifest = CorpusManifest(corpus_name="synthetic", source_document="osha3120.pdf",
                              source_relative_path="../raw/doc.pdf", source_sha256=source_hash,
                              ingested_at="2026-01-01T00:00:00Z", chunk_size=150,
                              chunk_overlap=30, document_count=1, page_count=1, chunk_count=2,
                              corpus_fingerprint=fingerprint, ingestion_version="word-v1",
                              corpus_schema_version="1.0")
    _write(root, "datasets/osha/corpus_manifest.json", manifest.model_dump(mode="json"))
    dataset = root / "benchmarks/osha/eval.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(json.dumps({"query_id": "q001", "query": "Synthetic question?",
                                   "relevant_chunk_ids": ["a"], "reference_answer": "a",
                                   "tags": ["synthetic"]}) + "\n", encoding="utf-8")
    dataset_hash = file_sha256(dataset)
    _write(root, "benchmarks/osha/review_manifest.json",
           {"schema_version": "1.0", "total_questions": 1, "verified": 1,
            "skipped": 0, "skipped_ids": [], "corpus_fingerprint": fingerprint,
            "dataset_sha256": dataset_hash})
    configs = {
        "dense": {"name": "dense", "retriever": {"type": "dense", "model": "fixed"}},
        "bm25": {"name": "bm25", "retriever": {"type": "bm25"}},
        "hybrid": {"name": "hybrid", "retriever": {"type": "hybrid", "model": "fixed",
                                                "dense_candidates": 20, "bm25_candidates": 20}},
        "hybrid_rerank": {"name": "hybrid_rerank", "retriever": {
            "type": "hybrid_rerank", "model": "fixed", "reranker_model": "fixed-reranker",
            "candidate_pool": 20}},
    }
    for name, config in configs.items():
        _write(root, f"configs/{name}.yaml", config)
    query = EvalQuery(query_id="q001", query="Synthetic question?", relevant_chunk_ids={"a"})
    runs = {}
    for name, config in configs.items():
        run = evaluate_retriever(
            StaticRetriever({query.query: ["a", "b"]}), [query], dataset=str(dataset),
            configuration_name=config["name"], configuration={**config,
                "corpus_fingerprint": fingerprint, "dataset_sha256": dataset_hash},
        )
        runs[name] = run
        _write(root, f"reports/original/{name}.json", run.model_dump(mode="json"))
    summary = {"schema_version": "1.0", "corpus_fingerprint": fingerprint,
               "dataset": str(dataset), "dataset_sha256": dataset_hash,
               "query_count": 1, "k_values": [1, 3, 5, 10],
               "runs": {name: {"run_id": run.run_id, "metrics": run.aggregate_metrics,
                               "latency": run.latency.model_dump()}
                        for name, run in runs.items()}}
    summary_path = _write(root, "reports/original/summary.json", summary)
    baseline_path = root / "baselines/osha/v1.json"
    baseline = create_baseline(summary_path, baseline_path, "synthetic-v1")
    create_query_baseline(summary_path, baseline, root / "baselines/osha/v1.query.json")
    _write(root, "configs/regression/osha.yaml", {
        "retrievers": ["hybrid", "hybrid_rerank"],
        "quality": {"recall@3": {"warn_drop": 0.1, "fail_drop": 0.2}},
        "query_regression": {"enabled": False, "critical_k": [3]},
    })
    _write(root, "benchmarks/osha/benchmark.yaml", {
        "schema_version": "1.0", "benchmark_id": "osha", "title": "synthetic",
        "source": {"document": "osha3120.pdf",
                   "url": "https://www.osha.gov/sites/default/files/publications/osha3120.pdf",
                   "sha256": source_hash, "document_count": 1, "page_count": 1},
        "corpus": {"chunk_size_words": 150, "chunk_overlap_words": 30,
                   "chunk_count": 2, "fingerprint": fingerprint},
        "golden": {"dataset": "benchmarks/osha/eval.jsonl",
                   "review_manifest": "benchmarks/osha/review_manifest.json",
                   "total_questions": 1, "verified_queries": 1, "skipped_ids": []},
        "k_values": [1, 3, 5, 10],
        "retriever_configs": {name: f"configs/{name}.yaml" for name in configs},
        "baseline": "baselines/osha/v1.json",
    })
    experiment = {
        "schema_version": "1.0", "experiment_id": "synthetic-v1",
        "benchmark": {"id": "osha", "spec": "benchmarks/osha/benchmark.yaml",
                      "corpus": "datasets/osha/corpus.jsonl",
                      "manifest": "datasets/osha/corpus_manifest.json",
                      "dataset": "benchmarks/osha/eval.jsonl",
                      "baseline": "baselines/osha/v1.json",
                      "query_baseline": "baselines/osha/v1.query.json",
                      "regression_policy": "configs/regression/osha.yaml"},
        "pareto": {"quality_metrics": ["recall@3", "mrr"], "include_latency": False},
        "candidates": [
            {"id": "hybrid-depth-10", "base_config": "configs/hybrid.yaml",
             "overrides": {"dense_candidates": 10, "bm25_candidates": 10}},
            {"id": "hybrid-depth-30", "base_config": "configs/hybrid.yaml",
             "overrides": {"dense_candidates": 30, "bm25_candidates": 30}},
        ],
    }
    spec = _write(root, "experiments/osha/retrieval-v1.yaml", experiment)
    return root, spec, experiment


def test_valid_spec_and_dry_run_no_models(package, monkeypatch):
    root, path, _ = package
    monkeypatch.setattr("groundtruth_rag.experiments.build_retriever",
                        lambda config: pytest.fail("dry-run loaded a model"))
    result = run_experiment(root, path, root / "reports/experiments/test", dry_run=True)
    assert result["dry_run"] and len(result["candidates"]) == 2
    assert result["candidates"][0]["resolved_config"]["retriever"]["dense_candidates"] == 10
    assert not (root / "reports/experiments/test").exists()


def test_fixed_corpus_and_dataset_compatibility(package):
    root, path, _ = package
    corpus = root / "datasets/osha/corpus.jsonl"
    original = corpus.read_bytes()
    corpus.write_bytes(original.replace(b"relevant passage", b"different passage"))
    with pytest.raises(ValueError, match="fixed corpus"):
        validate_experiment(root, path)
    corpus.write_bytes(original)
    dataset = root / "benchmarks/osha/eval.jsonl"
    dataset.write_text(dataset.read_text().replace("Synthetic question?", "Different question?"),
                       encoding="utf-8")
    with pytest.raises(ValueError, match="golden dataset"):
        validate_experiment(root, path)


@pytest.mark.parametrize("change,match", [
    (lambda x: x.update(schema_version="2.0"), "schema_version"),
    (lambda x: x["candidates"].append(x["candidates"][0]), "duplicate candidate ID"),
    (lambda x: x["candidates"][0]["overrides"].update(unknown=1), "unknown override"),
    (lambda x: x["candidates"][0]["overrides"].update(dense_candidates="ten"), "valid integer"),
    (lambda x: x["candidates"][0]["overrides"].update(dense_candidates=0), "positive"),
    (lambda x: x["candidates"][0].update(base_config="configs/missing.yaml"), "not a canonical"),
    (lambda x: x["candidates"][1]["overrides"].update(dense_candidates=10,
                                                           bm25_candidates=10), "duplicate resolved"),
])
def test_invalid_spec_and_overrides(package, change, match):
    root, path, experiment = package
    change(experiment)
    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    with pytest.raises((ValueError, ValidationError), match=match):
        validate_experiment(root, path)


def test_candidate_execution_provenance_and_inspection(package):
    root, path, _ = package
    output = root / "reports/experiments/test"
    before = (root / "configs/hybrid.yaml").read_bytes()
    result = run_experiment(
        root, path, output, candidate_id="hybrid-depth-10",
        builder=lambda config: StaticRetriever({"Synthetic question?": ["a", "b"]}),
    )
    row = result["candidates"][0]
    assert row["candidate_id"] == "hybrid-depth-10"
    assert row["regression_status"] == "PASS"
    assert row["promotion_status"] == "REVIEW"
    assert row["pareto"] == "NOT_EVALUATED"
    assert row["metrics"]["recall@3"] == row["baseline_metrics"]["recall@3"]
    assert row["config_sha256"]
    run = json.loads((output / "candidates/hybrid-depth-10/hybrid.json").read_text())
    assert run["configuration"]["experiment_id"] == "synthetic-v1"
    assert run["configuration"]["benchmark_spec_version"] == "1.0"
    assert run["configuration"]["config_sha256"] == row["config_sha256"]
    assert run["configuration"]["base_config_sha256"] == row["base_config_sha256"]
    assert "source passage" not in (output / "experiment.json").read_text()
    assert (root / "configs/hybrid.yaml").read_bytes() == before
    assert load_experiment_report(output / "experiment.json")["schema_version"] == "1.0"
    assert "Comparison" in (output / "EXPERIMENT.md").read_text()


def test_candidate_failure_rejected(package):
    root, path, _ = package
    report = run_experiment(
        root, path, root / "reports/experiments/failed", candidate_id="hybrid-depth-10",
        builder=lambda config: StaticRetriever({"Synthetic question?": ["b"]}),
    )
    assert report["candidates"][0]["regression_status"] == "FAIL"
    assert report["candidates"][0]["promotion_status"] == "REJECT"


def test_full_set_promotion_and_inspect(package):
    root, path, _ = package
    output = root / "reports/experiments/full"
    before_baseline = (root / "baselines/osha/v1.json").read_bytes()
    before_query = (root / "baselines/osha/v1.query.json").read_bytes()
    report = run_experiment(
        root, path, output,
        builder=lambda config: StaticRetriever({"Synthetic question?": ["a", "b"]}),
    )
    assert report["complete_candidate_set"]
    assert all(row["pareto"] == "PARETO" for row in report["candidates"])
    assert all(row["promotion_status"] == "PROMOTE_CANDIDATE" for row in report["candidates"])
    result = CliRunner().invoke(app, ["experiment-inspect", str(output / "experiment.json"),
                                      "--candidate", "hybrid-depth-10"])
    assert result.exit_code == 0 and "PROMOTE_CANDIDATE" in result.output
    missing = CliRunner().invoke(app, ["experiment-inspect", str(output / "experiment.json"),
                                       "--candidate", "missing"])
    assert missing.exit_code == 2
    assert (root / "baselines/osha/v1.json").read_bytes() == before_baseline
    assert (root / "baselines/osha/v1.query.json").read_bytes() == before_query


def test_critical_query_review_and_enabled_gate_reject(package):
    root, path, _ = package
    policy_path = root / "configs/regression/osha.yaml"
    policy = yaml.safe_load(policy_path.read_text())
    policy["quality"]["recall@3"] = {"warn_drop": 2, "fail_drop": 3}
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    output = root / "reports/experiments/critical"
    report = run_experiment(
        root, path, output,
        builder=lambda config: StaticRetriever({"Synthetic question?": ["b"]}),
    )
    assert all(row["regression_status"] == "PASS" for row in report["candidates"])
    assert all(row["new_critical_failures"] == 1 for row in report["candidates"])
    assert all(row["promotion_status"] == "REVIEW" for row in report["candidates"])
    policy["query_regression"] = {"enabled": True, "critical_k": [3],
                                  "fail_on_new_failure_at_k": True}
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    report = run_experiment(
        root, path, root / "reports/experiments/critical-gated",
        builder=lambda config: StaticRetriever({"Synthetic question?": ["b"]}),
    )
    assert all(row["query_gate_status"] == "FAIL" for row in report["candidates"])
    assert all(row["promotion_status"] == "REJECT" for row in report["candidates"])


def test_pareto_and_promotion_policy():
    rows = [
        {"candidate_id": "a", "metrics": {"recall@3": 0.8, "mrr": 0.8},
         "latency": {"p95_ms": 20}},
        {"candidate_id": "b", "metrics": {"recall@3": 0.7, "mrr": 0.7},
         "latency": {"p95_ms": 10}},
    ]
    assert pareto_status(rows, ["recall@3", "mrr"], include_latency=False) == {
        "a": "PARETO", "b": "DOMINATED"}
    assert pareto_status(rows, ["recall@3", "mrr"], include_latency=True) == {
        "a": "PARETO", "b": "PARETO"}
    rows[1]["latency"]["p95_ms"] = 30
    assert pareto_status(rows, ["recall@3", "mrr"], include_latency=True)["b"] == "DOMINATED"
    row = {"regression_status": "PASS", "query_gate_status": None,
           "new_critical_failures": 0, "pareto": "PARETO",
           "baseline_latency_p95_ms": 10, "latency": {"p95_ms": 11}}
    assert promotion_status(row, review_on_new_critical=True,
                            latency_review_increase_pct=None) == "PROMOTE_CANDIDATE"
    for key, value in (("regression_status", "WARN"), ("new_critical_failures", 1),
                       ("pareto", "DOMINATED"), ("query_gate_status", "WARN")):
        altered = {**row, key: value}
        assert promotion_status(altered, review_on_new_critical=True,
                                latency_review_increase_pct=None) == "REVIEW"
    assert promotion_status({**row, "regression_status": "FAIL"},
                            review_on_new_critical=True,
                            latency_review_increase_pct=None) == "REJECT"
    assert promotion_status(row, review_on_new_critical=True,
                            latency_review_increase_pct=10) == "REVIEW"
