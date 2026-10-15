import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from groundtruth_rag.cli import app
from groundtruth_rag.reporting.regression import (
    Aggregate, Baseline, BenchmarkSummary, RegressionPolicy, compare,
    create_baseline, load_baseline, load_summary, render_markdown,
    write_report,
)


def aggregate(recall: float = 0.8, mrr: float = 0.7, p95: float = 100) -> Aggregate:
    return Aggregate(run_id="run-1", metrics={"recall@3": recall, "mrr": mrr, "ndcg@3": 0.75},
                     latency={"mean_ms": 50, "p50_ms": 60, "p95_ms": p95})


def samples() -> tuple[Baseline, BenchmarkSummary]:
    run = aggregate()
    baseline = Baseline(name="sample-v1", domain="sample", created_at="2026-10-15T00:00:00Z",
                        corpus_fingerprint="corpus-1", dataset_sha256="dataset-1",
                        dataset_name="eval.jsonl", query_count=29, k_values=[1, 3, 5, 10],
                        runs={"dense": run, "bm25": run.model_copy(deep=True)})
    candidate = BenchmarkSummary(schema_version="1.0", corpus_fingerprint="corpus-1",
                                 dataset="eval.jsonl", dataset_sha256="dataset-1", query_count=29,
                                 k_values=[1, 3, 5, 10],
                                 runs={"dense": run.model_copy(deep=True),
                                       "bm25": run.model_copy(deep=True)})
    return baseline, candidate


def policy(*, latency: bool = True, override: dict | None = None) -> RegressionPolicy:
    return RegressionPolicy.model_validate({
        "retrievers": ["dense", "bm25"],
        "quality": {"recall@3": {"warn_drop": 0.02, "fail_drop": 0.03}},
        "latency": {"p95_ms": {"enabled": latency, "warn_increase_pct": 25,
                               "fail_increase_pct": 50}},
        "per_retriever": override or {},
    })


@pytest.mark.parametrize("candidate_recall,expected", [
    (0.8, "PASS"), (0.79, "PASS"), (0.78, "WARN"), (0.77, "FAIL"),
    (0.81, "PASS"),
])
def test_quality_thresholds(candidate_recall: float, expected: str):
    baseline, candidate = samples()
    candidate.runs["dense"].metrics["recall@3"] = candidate_recall
    report = compare(baseline, candidate, policy(latency=False))
    assert report["overall_status"] == expected
    row = next(row for row in report["details"] if row["retriever"] == "dense")
    assert row["status"] == expected
    assert row["improved"] is (candidate_recall > 0.8)
    assert row["direction"] == "higher"


@pytest.mark.parametrize("candidate_p95,expected", [
    (100, "PASS"), (124, "PASS"), (125, "WARN"), (150, "FAIL"), (90, "PASS"),
])
def test_latency_thresholds(candidate_p95: float, expected: str):
    baseline, candidate = samples()
    candidate.runs["dense"].latency["p95_ms"] = candidate_p95
    report = compare(baseline, candidate, policy())
    assert report["overall_status"] == expected
    row = next(row for row in report["details"] if row["retriever"] == "dense"
               and row["metric"] == "p95_ms")
    assert row["status"] == expected
    assert row["improved"] is (candidate_p95 < 100)
    assert row["direction"] == "lower"


def test_overall_fail_warn_and_per_retriever_override():
    baseline, candidate = samples()
    candidate.runs["dense"].metrics["recall@3"] = 0.78
    candidate.runs["bm25"].metrics["recall@3"] = 0.76
    report = compare(baseline, candidate, policy(latency=False))
    assert report["overall_status"] == "FAIL"
    assert report["retriever_status"] == {"dense": "WARN", "bm25": "FAIL"}
    overrides = {"bm25": {"quality": {"recall@3": {"warn_drop": 0.05, "fail_drop": 0.06}}}}
    report = compare(baseline, candidate, policy(latency=False, override=overrides))
    assert report["overall_status"] == "WARN"
    assert report["retriever_status"] == {"dense": "WARN", "bm25": "PASS"}


def test_disabled_metric_and_zero_baseline_latency():
    baseline, candidate = samples()
    candidate.runs["dense"].latency["p95_ms"] = 1000
    report = compare(baseline, candidate, policy(latency=False))
    assert report["overall_status"] == "PASS"
    assert all(row["metric"] != "p95_ms" for row in report["details"])
    baseline.runs["dense"].latency["p95_ms"] = 0
    with pytest.raises(ValueError, match="zero baseline"):
        compare(baseline, candidate, policy())
    candidate.runs["dense"].latency["p95_ms"] = 0
    assert compare(baseline, candidate, policy())["overall_status"] == "PASS"


@pytest.mark.parametrize("change,message", [
    (lambda b, c: setattr(c, "corpus_fingerprint", "other"), "corpus fingerprint"),
    (lambda b, c: setattr(c, "dataset_sha256", "other"), "dataset hash"),
    (lambda b, c: setattr(c, "query_count", 28), "query count"),
    (lambda b, c: c.k_values.remove(3), "candidate lacks K"),
    (lambda b, c: c.runs.pop("dense"), "missing retriever"),
    (lambda b, c: c.runs["dense"].metrics.pop("recall@3"), "missing metric"),
])
def test_incompatible_benchmarks(change, message: str):
    baseline, candidate = samples()
    change(baseline, candidate)
    with pytest.raises(ValueError, match=message):
        compare(baseline, candidate, policy(latency=False))


def test_invalid_numeric_and_policy_are_rejected():
    with pytest.raises(ValidationError):
        aggregate(recall=float("nan"))
    with pytest.raises(ValidationError):
        aggregate(p95=float("inf"))
    with pytest.raises(ValidationError, match="failure threshold"):
        RegressionPolicy.model_validate({"retrievers": ["dense"],
                                         "quality": {"mrr": {"warn_drop": 0.03,
                                                             "fail_drop": 0.02}}})
    with pytest.raises(ValidationError, match="unique"):
        RegressionPolicy.model_validate({"retrievers": ["dense", "dense"]})
    with pytest.raises(ValidationError, match="enabled metric"):
        RegressionPolicy.model_validate({"retrievers": ["dense"]})


def test_duplicate_keys_and_nonfinite_summary_are_rejected(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"runs":{},"runs":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_summary(summary)
    _, candidate = samples()
    candidate.runs["dense"].metrics["recall@3"] = float("nan")
    summary.write_text(candidate.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_summary(summary)


@pytest.mark.parametrize("metric", ["mean_ms", "p50_ms", "p95_ms"])
def test_all_latency_metrics_are_lower_is_better(metric: str):
    baseline, candidate = samples()
    old = baseline.runs["dense"].latency[metric]
    candidate.runs["dense"].latency[metric] = old * 1.3
    rules = RegressionPolicy.model_validate({
        "retrievers": ["dense"],
        "latency": {metric: {"warn_increase_pct": 25, "fail_increase_pct": 50}},
    })
    report = compare(baseline, candidate, rules)
    assert report["overall_status"] == "WARN"
    assert report["details"][0]["relative_delta_percent"] == pytest.approx(30)


def test_baseline_creation_preserves_precision_and_never_overwrites(tmp_path: Path):
    _, candidate = samples()
    candidate.runs["dense"].metrics["recall@3"] = 0.12345678901234566
    summary = tmp_path / "summary.json"
    summary.write_text(candidate.model_dump_json(), encoding="utf-8")
    output = tmp_path / "v1.json"
    created = create_baseline(summary, output, "sample-v1", domain="sample")
    assert created.query_count == 29
    assert load_baseline(output).runs["dense"].metrics["recall@3"] == 0.12345678901234566
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        create_baseline(summary, output, "sample-v1", domain="sample")
    assert output.read_bytes() == before
    create_baseline(summary, output, "sample-v2", domain="sample", force=True)
    assert load_baseline(output).name == "sample-v2"


def test_json_and_markdown_report_contract(tmp_path: Path):
    baseline, candidate = samples()
    candidate.runs["dense"].metrics["recall@3"] = 0.78
    candidate.runs["bm25"].metrics["recall@3"] = 0.81
    report = compare(baseline, candidate, policy(latency=False))
    report["candidate_summary"] = "candidate.json"
    assert report["schema_version"] == "1.0"
    json_path, markdown_path = write_report(report, tmp_path / "regression")
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["overall_status"] == "WARN"
    assert {"schema_version", "details", "compatibility_checks", "retriever_status"} <= set(persisted)
    markdown = markdown_path.read_text(encoding="utf-8")
    for heading in ("Provenance", "Overall Status", "Retriever Summary", "Metric Details",
                    "Regressions", "Improvements", "Compatibility Checks"):
        assert f"## {heading}" in markdown
    assert "dense recall@3" in markdown
    assert "bm25 recall@3" in markdown
    assert "100 (not gated)" in markdown
    assert render_markdown(report) == markdown


def test_cli_exit_codes_and_fail_on_warn(tmp_path: Path):
    _, candidate = samples()
    summary = tmp_path / "summary.json"
    summary.write_text(candidate.model_dump_json(), encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("retrievers: [dense, bm25]\nquality:\n  recall@3:\n    warn_drop: 0.02\n    fail_drop: 0.03\n", encoding="utf-8")
    runner = CliRunner()
    created = runner.invoke(app, ["baseline-create", "--summary", str(summary),
                                  "--output", str(baseline_path), "--name", "sample-v1"])
    assert created.exit_code == 0, created.output
    assert runner.invoke(app, ["baseline-create", "--summary", str(summary),
                               "--output", str(baseline_path), "--name", "sample-v1"]).exit_code != 0

    def run() -> object:
        return runner.invoke(app, ["regress", "--baseline", str(baseline_path),
                                   "--candidate", str(summary), "--policy", str(policy_path),
                                   "--output", str(tmp_path / "report")])

    assert run().exit_code == 0
    candidate.runs["dense"].metrics["recall@3"] = 0.78
    summary.write_text(candidate.model_dump_json(), encoding="utf-8")
    assert "Overall: WARN" in run().output
    warned = runner.invoke(app, ["regress", "--baseline", str(baseline_path),
                                 "--candidate", str(summary), "--policy", str(policy_path),
                                 "--output", str(tmp_path / "report"), "--fail-on-warn"])
    assert warned.exit_code == 1
    candidate.runs["dense"].metrics["recall@3"] = 0.76
    summary.write_text(candidate.model_dump_json(), encoding="utf-8")
    assert run().exit_code == 1
