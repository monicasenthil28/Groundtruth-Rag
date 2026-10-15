"""Offline query-rank diagnostics using synthetic fixed rankings."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from groundtruth_rag.cli import app
from groundtruth_rag.evaluation.runner import evaluate_retriever
from groundtruth_rag.reporting.query_regression import (
    QueryBaseline, compare_queries, create_query_baseline, load_query_baseline,
)
from groundtruth_rag.reporting.regression import (
    create_baseline, load_policy, load_summary, render_markdown,
)
from groundtruth_rag.retrieval.static import StaticRetriever
from groundtruth_rag.schemas import EvalQuery


KS = [1, 3, 5, 10]
RELEVANT = ["relevant-a", "relevant-b"]
OTHER = [f"other-{number}" for number in range(10)]


def _write_run(directory: Path, ranking: list[str], *, query_id: str = "q001",
               relevant: list[str] | None = None, corpus: str = "corpus",
               dataset_hash: str = "golden", name: str = "dense") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    query = EvalQuery(query_id=query_id, query="Synthetic question?",
                      relevant_chunk_ids=set(relevant or RELEVANT))
    run = evaluate_retriever(
        StaticRetriever({query.query: ranking}), [query], dataset="eval.jsonl",
        configuration_name="dense_fixture", configuration={
            "retriever": {"type": name}, "corpus_fingerprint": corpus,
            "dataset_sha256": dataset_hash,
        }, k_values=KS,
    )
    (directory / f"{name}.json").write_text(run.model_dump_json(), encoding="utf-8")
    summary = {"schema_version": "1.0", "corpus_fingerprint": corpus,
               "dataset": "eval.jsonl", "dataset_sha256": dataset_hash,
               "query_count": 1, "k_values": KS,
               "runs": {name: {"run_id": run.run_id, "metrics": run.aggregate_metrics,
                               "latency": run.latency.model_dump()}}}
    path = directory / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


@pytest.fixture
def pair(tmp_path: Path):
    before = _write_run(tmp_path / "before", [*OTHER[:2], RELEVANT[0], *OTHER[2:9]])
    after = _write_run(tmp_path / "after", [*OTHER[:2], RELEVANT[0], *OTHER[2:9]])
    aggregate = create_baseline(before, tmp_path / "aggregate.json", "synthetic-v1")
    query = create_query_baseline(before, aggregate, tmp_path / "query.json")
    return tmp_path, aggregate, query, after


def _analyze(pair, *, critical=None, warn=3):
    _, aggregate, query, after = pair
    return compare_queries(query, aggregate, after, load_summary(after), ["dense"],
                           critical_k=critical or [3], warn_rank_drop=warn)["retrievers"]["dense"]["queries"][0]


def _replace(pair, ranking, **kwargs):
    root, _, _, _ = pair
    path = _write_run(root / "after", ranking, **kwargs)
    return pair[:3] + (path,)


def test_identical_rankings_and_text_free_baseline(pair):
    root, _, query, _ = pair
    row = _analyze(pair)
    assert row["flags"] == ["UNCHANGED"]
    assert row["baseline_first_relevant_rank"] == row["candidate_first_relevant_rank"] == 3
    content = (root / "query.json").read_text()
    assert not any(key in content for key in ("Synthetic question", "source", "hits", "reference_answer"))
    assert load_query_baseline(root / "query.json") == query
    assert query.schema_version == "1.0"


@pytest.mark.parametrize("rank,expected", [(1, "RANK_IMPROVED"), (2, "RANK_IMPROVED"),
                                            (4, "RANK_REGRESSED"), (6, "RANK_REGRESSED")])
def test_rank_movement(pair, rank, expected):
    ranking = OTHER.copy()
    ranking.insert(rank - 1, RELEVANT[0])
    row = _analyze(_replace(pair, ranking))
    assert expected in row["flags"]
    assert row["rank_delta"] == rank - 3


@pytest.mark.parametrize("k,old,new", [(1, 1, 2), (3, 3, 4), (5, 5, 6), (10, 10, None)])
def test_new_failure_at_each_k(tmp_path, k, old, new):
    initial = OTHER.copy()
    initial.insert(old - 1, RELEVANT[0])
    before = _write_run(tmp_path / "before", initial)
    candidate = OTHER.copy()
    if new is not None:
        candidate.insert(new - 1, RELEVANT[0])
    after = _write_run(tmp_path / "after", candidate)
    aggregate = create_baseline(before, tmp_path / "aggregate.json", "synthetic-v1")
    query = create_query_baseline(before, aggregate, tmp_path / "query.json")
    row = compare_queries(query, aggregate, after, load_summary(after), ["dense"],
                          critical_k=[k], warn_rank_drop=None)["retrievers"]["dense"]["queries"][0]
    assert f"NEW_FAILURE_AT_{k}" in row["flags"]
    assert row["severity"] == "FAIL"


@pytest.mark.parametrize("k,old,new", [(1, 2, 1), (3, 4, 3), (5, 6, 5), (10, None, 10)])
def test_recovery_at_each_k(tmp_path, k, old, new):
    initial = OTHER.copy()
    if old is not None:
        initial.insert(old - 1, RELEVANT[0])
    before = _write_run(tmp_path / "before", initial)
    candidate = OTHER.copy()
    candidate.insert(new - 1, RELEVANT[0])
    after = _write_run(tmp_path / "after", candidate)
    aggregate = create_baseline(before, tmp_path / "aggregate.json", "synthetic-v1")
    query = create_query_baseline(before, aggregate, tmp_path / "query.json")
    row = compare_queries(query, aggregate, after, load_summary(after), ["dense"],
                          critical_k=[k], warn_rank_drop=None)["retrievers"]["dense"]["queries"][0]
    assert f"RECOVERED_AT_{k}" in row["flags"]
    assert row["severity"] == "INFO"


def test_multiple_relevant_chunks_recall_and_persistent_failure(pair):
    row = _analyze(_replace(pair, [RELEVANT[0], RELEVANT[1], *OTHER[:8]]))
    assert row["candidate_first_relevant_rank"] == 1
    assert row["candidate_recall_at_k"]["3"] == 1.0
    assert "RECALL_IMPROVED" in row["flags"]
    missing = _analyze(_replace(pair, OTHER))
    assert missing["candidate_first_relevant_rank"] is None
    assert "STILL_FAILED_AT_K" in missing["flags"]
    assert "RECALL_REGRESSED" in missing["flags"]


def test_incompatible_provenance_query_set_and_labels(pair):
    root, aggregate, query, after = pair
    with pytest.raises(ValueError, match="query baseline provenance"):
        compare_queries(query.model_copy(update={"corpus_fingerprint": "wrong"}), aggregate,
                        after, load_summary(after), ["dense"], critical_k=[3], warn_rank_drop=3)
    with pytest.raises(ValueError, match="query baseline provenance"):
        compare_queries(query.model_copy(update={"dataset_sha256": "wrong"}), aggregate,
                        after, load_summary(after), ["dense"], critical_k=[3], warn_rank_drop=3)
    changed_id = _replace(pair, OTHER, query_id="q999")
    with pytest.raises(ValueError, match="INCOMPATIBLE QUERY SET"):
        _analyze(changed_id)
    changed_labels = _replace(pair, OTHER, relevant=["different"])
    with pytest.raises(ValueError, match="INCOMPATIBLE GOLDEN LABELS"):
        _analyze(changed_labels)
    assert root.exists()


def test_duplicate_query_ids_rejected(pair):
    _, aggregate, query, after = pair
    query.retrievers["dense"].queries.append(query.retrievers["dense"].queries[0])
    with pytest.raises(ValueError, match="duplicate query IDs"):
        QueryBaseline.model_validate(query.model_dump())
    assert aggregate and after


def test_cli_disabled_gate_and_markdown(pair):
    root, _, _, after = _replace(pair, OTHER)
    policy = root / "policy.yaml"
    policy.write_text("schema_version: '1.0'\nretrievers: [dense]\nquality:\n"
                      "  recall@3: {warn_drop: 2, fail_drop: 3}\n"
                      "query_regression: {enabled: false, critical_k: [3]}\n", encoding="utf-8")
    args = ["regress", "--baseline", str(root / "aggregate.json"),
            "--query-baseline", str(root / "query.json"), "--candidate", str(after),
            "--policy", str(policy), "--output", str(root / "report")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    report = json.loads((root / "report/regression.json").read_text())
    assert report["overall_status"] == "PASS"
    assert report["query_analysis"]["retrievers"]["dense"]["new_failures"]["3"] == ["q001"]
    markdown = render_markdown(report)
    assert all(heading in markdown for heading in ("Query-Level Changes", "New Failures",
                                                   "Recoveries", "Rank Regressions",
                                                   "Rank Improvements", "Persistent Failures"))
    policy.write_text(policy.read_text().replace("enabled: false", "enabled: true")
                      .replace("critical_k: [3]", "critical_k: [3], fail_on_new_failure_at_k: true"))
    gated = CliRunner().invoke(app, args)
    assert gated.exit_code == 1
    report = json.loads((root / "report/regression.json").read_text())
    assert report["query_gate_status"] == report["overall_status"] == "FAIL"
    assert load_policy(policy).query_regression.enabled


def test_committed_osha_query_baseline_is_text_free():
    root = Path(__file__).resolve().parents[1]
    baseline = load_query_baseline(root / "baselines/osha/v1.query.json")
    assert baseline.query_count == 29
    assert set(baseline.retrievers) == {"dense", "bm25", "hybrid", "hybrid_rerank"}
    assert baseline.corpus_fingerprint == (
        "55bebfded1835db967ef9fed5688ff0d8ad919a42c308cbc555972419df3531e"
    )
    for retriever in baseline.retrievers.values():
        ids = {row.query_id for row in retriever.queries}
        assert len(ids) == 29 and "q013" not in ids and "q019" in ids
    assert "query\"" not in (root / "baselines/osha/v1.query.json").read_text()


def test_query_gate_new_failure_tolerance(pair):
    root, _, _, after = _replace(pair, OTHER)
    policy = root / "policy.yaml"
    policy.write_text("schema_version: '1.0'\nretrievers: [dense]\nquality:\n"
                      "  recall@3: {warn_drop: 2, fail_drop: 3}\n"
                      "query_regression: {enabled: true, critical_k: [3], "
                      "max_new_failures_warn: 1, max_new_failures_fail: 2}\n", encoding="utf-8")
    args = ["regress", "--baseline", str(root / "aggregate.json"),
            "--query-baseline", str(root / "query.json"), "--candidate", str(after),
            "--policy", str(policy), "--output", str(root / "report")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    report = json.loads((root / "report/regression.json").read_text())
    assert report["query_gate_status"] == "PASS"
