"""Offline state and provenance checks for controlled promotion."""

import json
from types import SimpleNamespace

import pytest
import yaml

from groundtruth_rag import promotion
from groundtruth_rag.evaluation.runner import evaluate_retriever
from groundtruth_rag.experiments import _digest
from groundtruth_rag.reporting.benchmark import summarize_runs
from groundtruth_rag.reporting.query_regression import create_query_baseline
from groundtruth_rag.reporting.regression import create_baseline
from groundtruth_rag.retrieval.factory import RunConfig
from groundtruth_rag.retrieval.static import StaticRetriever
from groundtruth_rag.schemas import EvalQuery


def fixture(tmp_path, monkeypatch):
    root = tmp_path
    def write(relative, content):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
    config = RunConfig.model_validate({"name": "rerank-pool-30", "retriever": {
        "type": "hybrid_rerank", "candidate_pool": 30}})
    digest = _digest(config.model_dump())
    query = EvalQuery(query_id="q019", query="question", relevant_chunk_ids={"a"})
    meta = {**config.model_dump(), "corpus_fingerprint": "fingerprint",
            "dataset_sha256": "dataset", "config_sha256": digest,
            "experiment_id": "experiment", "candidate_id": "rerank-pool-30"}
    run = evaluate_retriever(StaticRetriever({"q019": ["a"]}), [query], dataset="dataset",
                             configuration_name=config.name, configuration=meta)
    candidate_dir = "reports/experiments/demo/candidates/rerank-pool-30"
    summary = {"schema_version": "1.0", "corpus_fingerprint": "fingerprint",
               "dataset": "dataset", "dataset_sha256": "dataset", "query_count": 1,
               "k_values": run.k_values, "runs": {"hybrid_rerank": {
                   "run_id": run.run_id, "metrics": run.aggregate_metrics,
                   "latency": run.latency.model_dump()}}}
    write(f"{candidate_dir}/summary.json", json.dumps(summary))
    write(f"{candidate_dir}/hybrid_rerank.json", run.model_dump_json())
    write(f"{candidate_dir}/regression.json", json.dumps({
        "overall_status": "PASS", "query_gate_status": "PASS",
        "corpus_fingerprint": "fingerprint", "dataset_sha256": "dataset", "query_count": 1}))
    write("reports/experiments/demo/experiment.json", json.dumps({
        "schema_version": "1.0", "complete_candidate_set": True,
        "experiment_id": "experiment", "benchmark_id": "osha",
        "corpus_fingerprint": "fingerprint", "dataset_sha256": "dataset",
        "candidates": [{"candidate_id": "rerank-pool-30", "config_sha256": digest,
                        "retriever_type": "hybrid_rerank", "regression_status": "PASS",
                        "promotion_status": "PROMOTE_CANDIDATE", "run_id": run.run_id}]}))
    write("reports/robustness/demo/robustness.json", json.dumps({
        "schema_version": "1.0", "candidate": "rerank-pool-30",
        "candidate_run_id": run.run_id, "baseline_run_id": "baseline-id",
        "retriever_type": "hybrid_rerank", "corpus_fingerprint": "fingerprint",
        "dataset_sha256": "dataset", "query_count": 1,
        "metrics": {"recall@3": {"assessment": "UNCHANGED"},
                    "mrr": {"assessment": "INCONCLUSIVE"},
                    "ndcg@3": {"assessment": "UNCHANGED"}},
        "latency": {"summary": {"median_trial_p95_ms": 1365.47},
                    "measured_runs": 5, "warmup_runs": 1}}))
    write("baselines/osha/v1.json", "{}")
    write("baselines/osha/v1.query.json", "{}")
    write("configs/hybrid_rerank.yaml", "name: v1\n")
    for name in ("dense", "bm25", "hybrid"):
        write(f"configs/{name}.yaml", yaml.safe_dump({"name": name, "retriever": {"type": name}}))
    write("configs/versions/osha/registry.yaml", yaml.safe_dump({
        "schema_version": "1.0", "benchmark_id": "osha", "active": "v1",
        "versions": {"v1": {"config": "configs/hybrid_rerank.yaml",
                            "baseline": "baselines/osha/v1.json",
                            "query_baseline": "baselines/osha/v1.query.json",
                            "status": "active"}}}))
    write("experiments/osha/spec.yaml", "schema_version: '1.0'\n")
    manifest = promotion.PromotionManifest.model_validate({
        "schema_version": "1.0", "promotion_id": "osha-v2", "benchmark_id": "osha",
        "version": "v2", "candidate": {"experiment_id": "experiment",
            "candidate_id": "rerank-pool-30", "experiment_spec": "experiments/osha/spec.yaml",
            "experiment_report": "reports/experiments/demo/experiment.json"},
        "source": {"base_config": "configs/hybrid_rerank.yaml",
                   "resolved_config_sha256": digest},
        "evidence": {"regression_report": f"{candidate_dir}/regression.json",
                     "robustness_report": "reports/robustness/demo/robustness.json",
                     "regression": "PASS", "experiment_decision": "PROMOTE_CANDIDATE",
                     "robustness": {"recall@3": "UNCHANGED", "mrr": "INCONCLUSIVE",
                                    "ndcg@3": "UNCHANGED"}},
        "human_approval": {"approved": False},
        "target": {"config_path": "configs/versions/osha/hybrid_rerank_v2.yaml",
                   "aggregate_baseline": "baselines/osha/v2.json",
                   "query_baseline": "baselines/osha/v2.query.json",
                   "registry": "configs/versions/osha/registry.yaml",
                   "benchmark_output": "reports/benchmarks/osha/v2",
                   "promotion_output": "reports/promotions/osha/v2"}})
    manifest_path = root / "promotions/osha/v2.yaml"
    promotion.save_manifest(manifest_path, manifest)
    benchmark = SimpleNamespace(corpus=SimpleNamespace(chunk_size_words=150,
        chunk_overlap_words=30), golden=SimpleNamespace(skipped_ids=["q013"]),
        retriever_configs={name: f"configs/{name}.yaml" for name in ("dense", "bm25", "hybrid")}
        | {"hybrid_rerank": "configs/hybrid_rerank.yaml"},
        schema_version="1.0", k_values=[1, 3, 5, 10])
    experiment = SimpleNamespace(experiment_id="experiment", benchmark=SimpleNamespace(
        id="osha", baseline="baselines/osha/v1.json",
        query_baseline="baselines/osha/v1.query.json"))
    context = {"benchmark": benchmark, "queries": [query], "fingerprint": "fingerprint",
               "dataset_sha256": "dataset", "baseline": SimpleNamespace(runs={
                   "hybrid_rerank": SimpleNamespace(run_id="baseline-id")})}
    resolved = {"rerank-pool-30": {"config": config, "config_sha256": digest,
                "base_config": "configs/hybrid_rerank.yaml", "retriever_type": "hybrid_rerank"}}
    monkeypatch.setattr(promotion, "validate_experiment", lambda *_: (experiment, resolved, context))
    return root, manifest_path, run, context


def write_four_run_benchmark(root, manifest, rerank_run):
    query = EvalQuery(query_id="q019", query="question", relevant_chunk_ids={"a"})
    runs = {"hybrid_rerank": rerank_run}
    for name in ("dense", "bm25", "hybrid"):
        config = {"name": name, "retriever": {"type": name},
                  "corpus_fingerprint": "fingerprint", "dataset_sha256": "dataset"}
        runs[name] = evaluate_retriever(StaticRetriever({"q019": ["a"]}), [query],
                                        dataset="dataset", configuration_name=name,
                                        configuration=config)
    summary, _, _ = summarize_runs(runs)
    summary["dataset_sha256"] = "dataset"
    output = root / manifest.target.benchmark_output
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name, run in runs.items():
        (output / f"{name}.json").write_text(run.model_dump_json(), encoding="utf-8")
    return {name: run.run_id for name, run in runs.items()}


def test_check_only_marks_ready(tmp_path, monkeypatch):
    root, path, _, _ = fixture(tmp_path, monkeypatch)
    checked = promotion.check(root, path)
    assert checked.state == "READY_FOR_HUMAN_APPROVAL"
    assert not checked.human_approval.approved
    assert promotion.config_status(root, "osha")["active"] == "v1"
    assert not (root / checked.target.config_path).exists()
    report = (root / checked.target.promotion_output / "PROMOTION.md").read_text()
    assert "1365.47" in report and "source text" not in report
    promotion.write_promotion_report(root, checked, {"status": "PASS", "forensics_summary": {"x": 1}})
    completed = (root / checked.target.promotion_output / "PROMOTION.md").read_text()
    assert "completed on four v2 stage runs" in completed


@pytest.mark.parametrize(("artifact", "change", "message"), [
    ("manifest", lambda m: setattr(m.candidate, "candidate_id", "missing"), "candidate absent"),
    ("manifest", lambda m: setattr(m.source, "resolved_config_sha256", "0" * 64), "config hash"),
    ("report", lambda r: r["candidates"][0].update(regression_status="FAIL"), "not eligible"),
    ("report", lambda r: r["candidates"][0].update(promotion_status="REJECT"), "not eligible"),
    ("report", lambda r: r.update(corpus_fingerprint="wrong"), "not eligible"),
    ("report", lambda r: r.update(dataset_sha256="wrong"), "not eligible"),
])
def test_precheck_blocks_bad_evidence(tmp_path, monkeypatch, artifact, change, message):
    root, path, _, _ = fixture(tmp_path, monkeypatch)
    if artifact == "manifest":
        manifest = promotion.load_manifest(path)
        change(manifest)
        promotion.save_manifest(path, manifest)
    else:
        report_path = root / "reports/experiments/demo/experiment.json"
        report = json.loads(report_path.read_text())
        change(report)
        report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        promotion.check(root, path)
    assert promotion.load_manifest(path).state == "DRAFT"


def test_approval_staging_and_no_activation(tmp_path, monkeypatch):
    root, path, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="approval"):
        promotion.apply(root, path)
    promotion.check(root, path)
    approved = promotion.approve(root, path, "Reviewer")
    assert approved.human_approval.approved_by == "Reviewer"
    assert approved.human_approval.approved_at is not None
    staged = promotion.apply(root, path)
    assert staged.state == "STAGED"
    parsed = promotion.load_config(root / staged.target.config_path)
    assert _digest(parsed.model_dump()) == staged.source.resolved_config_sha256
    assert parsed.retriever.candidate_pool == 30
    registry = promotion.load_registry(root / staged.target.registry)
    assert registry.active == "v1" and registry.versions["v2"].status == "pending_validation"
    assert not (root / staged.target.aggregate_baseline).exists()
    with pytest.raises(ValueError, match="baselines are required"):
        promotion.activate(root, path)


def test_failed_benchmark_stays_v1(tmp_path, monkeypatch):
    root, path, _, _ = fixture(tmp_path, monkeypatch)
    promotion.check(root, path)
    promotion.approve(root, path)
    promotion.apply(root, path)
    monkeypatch.setattr(promotion, "benchmark_status", lambda *_: {"ready": True, "errors": []})
    def broken(_):
        raise RuntimeError("model unavailable")
    with pytest.raises(RuntimeError, match="model unavailable"):
        promotion.benchmark_version(root, path, builder=broken)
    assert promotion.load_manifest(path).state == "FAILED_VALIDATION"
    assert promotion.load_registry(root / "configs/versions/osha/registry.yaml").active == "v1"
    assert not (root / "baselines/osha/v2.json").exists()
    with pytest.raises(ValueError):
        promotion.activate(root, path)


def test_activation_and_rollback_preserve_v2(tmp_path, monkeypatch):
    root, path, run, _ = fixture(tmp_path, monkeypatch)
    promotion.check(root, path)
    promotion.approve(root, path)
    promotion.apply(root, path)
    manifest = promotion.load_manifest(path)
    run_ids = write_four_run_benchmark(root, manifest, run)
    manifest.state = "BASELINED"
    manifest.validation_run_ids = run_ids
    promotion.save_manifest(path, manifest)
    output = root / manifest.target.benchmark_output
    aggregate = create_baseline(output / "summary.json", root / manifest.target.aggregate_baseline,
                                "osha-v2", domain="osha")
    create_query_baseline(output / "summary.json", aggregate, root / manifest.target.query_baseline)
    registry = promotion.load_registry(root / manifest.target.registry)
    registry.versions["v2"].baseline = manifest.target.aggregate_baseline
    registry.versions["v2"].query_baseline = manifest.target.query_baseline
    registry.versions["v2"].status = "baselined"
    promotion.save_registry(root / manifest.target.registry, registry)
    report = {"status": "PASS", "run_ids": manifest.validation_run_ids,
              "corpus_fingerprint": "fingerprint", "dataset_sha256": "dataset"}
    target = root / manifest.target.promotion_output / "validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report))
    active = promotion.activate(root, path)
    assert active.state == "ACTIVE"
    assert promotion.config_status(root, "osha")["active"] == "v2"
    rolled = promotion.rollback(root, path, "v1")
    assert rolled.state == "ROLLED_BACK"
    assert promotion.config_status(root, "osha")["active"] == "v1"
    assert (root / manifest.target.aggregate_baseline).exists()
    assert (root / manifest.target.query_baseline).exists()


def test_invalid_transition_and_missing_baselines(tmp_path, monkeypatch):
    root, path, _, _ = fixture(tmp_path, monkeypatch)
    manifest = promotion.load_manifest(path)
    with pytest.raises(ValueError, match="invalid promotion transition"):
        promotion.transition(manifest, "ACTIVE")
    promotion.check(root, path)
    promotion.approve(root, path)
    promotion.apply(root, path)
    manifest = promotion.load_manifest(path)
    manifest.state = "BASELINED"
    manifest.validation_run_ids = {"hybrid_rerank": "run-id"}
    promotion.save_manifest(path, manifest)
    registry = promotion.load_registry(root / manifest.target.registry)
    registry.versions["v2"].status = "baselined"
    promotion.save_registry(root / manifest.target.registry, registry)
    report = {"status": "PASS", "run_ids": manifest.validation_run_ids,
              "corpus_fingerprint": "fingerprint", "dataset_sha256": "dataset"}
    target = root / manifest.target.promotion_output / "validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report))
    with pytest.raises(OSError):
        promotion.activate(root, path)
    assert promotion.load_registry(root / manifest.target.registry).active == "v1"


@pytest.mark.parametrize(("field", "value", "message"), [
    ("chunk_size_words", 151, "fixed chunking"),
    ("chunk_overlap_words", 31, "fixed chunking"),
    ("skipped_ids", [], "fixed chunking"),
])
def test_fixed_benchmark_guards(tmp_path, monkeypatch, field, value, message):
    root, path, _, context = fixture(tmp_path, monkeypatch)
    target = context["benchmark"].golden if field == "skipped_ids" else context["benchmark"].corpus
    setattr(target, field, value)
    with pytest.raises(ValueError, match=message):
        promotion.check(root, path)


def test_q019_must_remain_present(tmp_path, monkeypatch):
    root, path, _, context = fixture(tmp_path, monkeypatch)
    context["queries"] = [EvalQuery(query_id="q001", query="question", relevant_chunk_ids={"a"})]
    with pytest.raises(ValueError, match="reviewed query set"):
        promotion.check(root, path)


def test_missing_query_baseline_blocks_activation(tmp_path, monkeypatch):
    root, path, run, _ = fixture(tmp_path, monkeypatch)
    promotion.check(root, path)
    promotion.approve(root, path)
    promotion.apply(root, path)
    manifest = promotion.load_manifest(path)
    manifest.state = "BASELINED"
    manifest.validation_run_ids = {"hybrid_rerank": run.run_id}
    promotion.save_manifest(path, manifest)
    registry = promotion.load_registry(root / manifest.target.registry)
    registry.versions["v2"].status = "baselined"
    promotion.save_registry(root / manifest.target.registry, registry)
    with pytest.raises(OSError):
        promotion.activate(root, path)
    assert promotion.load_registry(root / manifest.target.registry).active == "v1"


def test_baseline_pair_generated_from_four_runs(tmp_path, monkeypatch):
    root, path, run, _ = fixture(tmp_path, monkeypatch)
    promotion.check(root, path)
    promotion.approve(root, path)
    promotion.apply(root, path)
    manifest = promotion.load_manifest(path)
    run_ids = write_four_run_benchmark(root, manifest, run)
    manifest.state = "VALIDATED"
    manifest.validation_run_ids = run_ids
    promotion.save_manifest(path, manifest)
    report_path = root / manifest.target.promotion_output / "validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"status": "PASS", "run_ids": run_ids}), encoding="utf-8")
    baselined = promotion.baseline_version(root, path)
    assert baselined.state == "BASELINED"
    assert (root / baselined.target.aggregate_baseline).is_file()
    assert (root / baselined.target.query_baseline).is_file()
    assert promotion.load_registry(root / baselined.target.registry).active == "v1"
    assert (root / "baselines/osha/v1.json").read_text() == "{}"
    with pytest.raises(ValueError, match="validation must PASS"):
        promotion.baseline_version(root, path)
