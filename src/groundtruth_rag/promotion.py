"""Explicit, versioned retrieval promotion without implicit activation."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundtruth_rag import __version__
from groundtruth_rag.benchmark_setup import benchmark_status
from groundtruth_rag.evaluation.runner import evaluate_retriever
from groundtruth_rag.experiments import _digest, load_experiment_report, validate_experiment
from groundtruth_rag.labeling.manifest import file_sha256
from groundtruth_rag.labeling.models import atomic_write
from groundtruth_rag.reporting.benchmark import summarize_runs
from groundtruth_rag.reporting.forensics import analyze_forensics, write_forensics
from groundtruth_rag.reporting.query_regression import (
    _load_run, compare_queries, create_query_baseline, load_query_baseline,
)
from groundtruth_rag.reporting.regression import (
    apply_query_gate, compare, create_baseline, load_baseline, load_policy, load_summary,
)
from groundtruth_rag.retrieval.factory import build_retriever, load_config
from groundtruth_rag.robustness import analyze as analyze_robustness
from groundtruth_rag.robustness import load_config as load_robustness_config
from groundtruth_rag.robustness import load_run as load_robustness_run

State = Literal["DRAFT", "READY_FOR_HUMAN_APPROVAL", "APPROVED", "STAGED",
                "VALIDATION_RUNNING", "VALIDATED", "BASELINED", "ACTIVE",
                "ROLLED_BACK", "FAILED_VALIDATION"]
TRANSITIONS = {
    "DRAFT": {"READY_FOR_HUMAN_APPROVAL"},
    "READY_FOR_HUMAN_APPROVAL": {"APPROVED"},
    "APPROVED": {"STAGED"},
    "STAGED": {"VALIDATION_RUNNING"},
    "VALIDATION_RUNNING": {"VALIDATED", "FAILED_VALIDATION"},
    "VALIDATED": {"BASELINED", "FAILED_VALIDATION"},
    "BASELINED": {"ACTIVE", "FAILED_VALIDATION"},
    "ACTIVE": {"ROLLED_BACK"},
    "ROLLED_BACK": {"ACTIVE"},
    "FAILED_VALIDATION": {"VALIDATION_RUNNING"},
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateRef(StrictModel):
    experiment_id: str
    candidate_id: str
    experiment_spec: str
    experiment_report: str


class SourceRef(StrictModel):
    base_config: str
    resolved_config_sha256: str = Field(min_length=64, max_length=64)


class EvidenceRef(StrictModel):
    regression_report: str
    robustness_report: str
    regression: Literal["PASS"]
    experiment_decision: Literal["PROMOTE_CANDIDATE"]
    robustness: dict[str, str]


class Approval(StrictModel):
    required: Literal[True] = True
    approved: bool = False
    approved_at: datetime | None = None
    approved_by: str | None = None

    @model_validator(mode="after")
    def valid_approval(self) -> "Approval":
        if self.approved != (self.approved_at is not None):
            raise ValueError("approval timestamp and flag must agree")
        return self


class Target(StrictModel):
    config_path: str
    aggregate_baseline: str
    query_baseline: str
    registry: str
    benchmark_output: str
    promotion_output: str


class PromotionManifest(StrictModel):
    schema_version: Literal["1.0"]
    promotion_id: str
    benchmark_id: str
    version: str
    state: State = "DRAFT"
    candidate: CandidateRef
    source: SourceRef
    evidence: EvidenceRef
    human_approval: Approval
    target: Target
    validation_run_ids: dict[str, str] | None = None


class RegistryEntry(StrictModel):
    config: str
    baseline: str | None = None
    query_baseline: str | None = None
    status: str | None = None
    promotion_id: str | None = None
    resolved_config_sha256: str | None = None


class Registry(StrictModel):
    schema_version: Literal["1.0"]
    benchmark_id: str
    active: str
    versions: dict[str, RegistryEntry]

    @model_validator(mode="after")
    def active_exists(self) -> "Registry":
        if self.active not in self.versions:
            raise ValueError("active registry version does not exist")
        return self


def path_in(root: Path, relative: str, prefix: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to((root / prefix).resolve()):
        raise ValueError(f"path must be under {prefix}: {relative}")
    return path


def load_manifest(path: Path) -> PromotionManifest:
    return PromotionManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def save_manifest(path: Path, manifest: PromotionManifest) -> None:
    atomic_write(path, yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False))


def load_registry(path: Path) -> Registry:
    return Registry.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def save_registry(path: Path, registry: Registry) -> None:
    atomic_write(path, yaml.safe_dump(registry.model_dump(mode="json"), sort_keys=False))


def transition(manifest: PromotionManifest, state: State) -> PromotionManifest:
    if state not in TRANSITIONS[manifest.state]:
        raise ValueError(f"invalid promotion transition: {manifest.state} -> {state}")
    return manifest.model_copy(update={"state": state})


def _json(path: Path) -> dict:
    from groundtruth_rag.reporting.regression import _read_json
    return _read_json(path)


def precheck(root: Path, manifest: PromotionManifest) -> dict:
    ref, target = manifest.candidate, manifest.target
    if manifest.state not in {"DRAFT", "READY_FOR_HUMAN_APPROVAL"} and not manifest.human_approval.approved:
        raise ValueError("approved state lacks human approval")
    specification = path_in(root, ref.experiment_spec, "experiments")
    experiment, resolved, context = validate_experiment(root, specification)
    if experiment.experiment_id != ref.experiment_id or experiment.benchmark.id != manifest.benchmark_id:
        raise ValueError("experiment or benchmark identity differs")
    item = resolved.get(ref.candidate_id)
    if item is None:
        raise ValueError(f"candidate absent from experiment: {ref.candidate_id}")
    if (item["config_sha256"] != manifest.source.resolved_config_sha256
            or item["base_config"] != manifest.source.base_config):
        raise ValueError("resolved candidate config hash or base config differs")
    if item["retriever_type"] != "hybrid_rerank":
        raise ValueError("promotion requires the same Hybrid + CrossEncoder retriever family")
    benchmark = context["benchmark"]
    if (benchmark.corpus.chunk_size_words != 150 or benchmark.corpus.chunk_overlap_words != 30
            or "q013" not in benchmark.golden.skipped_ids
            or "q013" in {q.query_id for q in context["queries"]}
            or "q019" not in {q.query_id for q in context["queries"]}):
        raise ValueError("fixed chunking or reviewed query set differs")
    if not path_in(root, experiment.benchmark.baseline, "baselines").is_file():
        raise ValueError("v1 aggregate baseline missing")
    if not path_in(root, experiment.benchmark.query_baseline, "baselines").is_file():
        raise ValueError("v1 query baseline missing")
    report = load_experiment_report(path_in(root, ref.experiment_report, "reports/experiments"))
    row = next((r for r in report["candidates"] if r.get("candidate_id") == ref.candidate_id), None)
    if row is None:
        raise ValueError("candidate absent from experiment report")
    if (not report.get("complete_candidate_set") or report.get("experiment_id") != ref.experiment_id
            or report.get("benchmark_id") != manifest.benchmark_id
            or report.get("corpus_fingerprint") != context["fingerprint"]
            or report.get("dataset_sha256") != context["dataset_sha256"]
            or row.get("config_sha256") != item["config_sha256"]
            or row.get("retriever_type") != item["retriever_type"]
            or row.get("regression_status") != "PASS"
            or row.get("promotion_status") != "PROMOTE_CANDIDATE"):
        raise ValueError("experiment candidate is not eligible for review")
    candidate_dir = path_in(root, ref.experiment_report, "reports/experiments").parent / "candidates" / ref.candidate_id
    regression_path = path_in(root, manifest.evidence.regression_report, "reports/experiments")
    if regression_path != candidate_dir / "regression.json":
        raise ValueError("regression artifact path differs from candidate")
    regression = _json(regression_path)
    if (regression.get("overall_status") != "PASS" or regression.get("query_gate_status") == "FAIL"
            or regression.get("corpus_fingerprint") != context["fingerprint"]
            or regression.get("dataset_sha256") != context["dataset_sha256"]
            or regression.get("query_count") != len(context["queries"])):
        raise ValueError("candidate regression failed or provenance differs")
    run = _load_run(candidate_dir / "summary.json", load_summary(candidate_dir / "summary.json"),
                    item["retriever_type"])
    if (run.run_id != row.get("run_id") or run.configuration.get("config_sha256") != item["config_sha256"]
            or run.configuration.get("experiment_id") != ref.experiment_id
            or run.configuration.get("candidate_id") != ref.candidate_id
            or run.configuration.get("dataset_sha256") != context["dataset_sha256"]):
        raise ValueError("candidate run disagrees with resolved experiment")
    robust = _json(path_in(root, manifest.evidence.robustness_report, "reports/robustness"))
    baseline_run = context["baseline"].runs[item["retriever_type"]]
    if (robust.get("schema_version") != "1.0" or robust.get("candidate") != ref.candidate_id
            or robust.get("candidate_run_id") != run.run_id
            or robust.get("baseline_run_id") != baseline_run.run_id
            or robust.get("retriever_type") != item["retriever_type"]
            or robust.get("corpus_fingerprint") != context["fingerprint"]
            or robust.get("dataset_sha256") != context["dataset_sha256"]
            or robust.get("query_count") != len(context["queries"])):
        raise ValueError("robustness artifact provenance differs")
    for metric, expected in manifest.evidence.robustness.items():
        if robust.get("metrics", {}).get(metric, {}).get("assessment") != expected:
            raise ValueError(f"robustness assessment differs: {metric}")
    registry = load_registry(path_in(root, target.registry, "configs/versions"))
    if (registry.benchmark_id != manifest.benchmark_id or "v1" not in registry.versions
            or registry.versions["v1"].baseline != experiment.benchmark.baseline
            or registry.versions["v1"].query_baseline != experiment.benchmark.query_baseline
            or registry.versions["v1"].config != benchmark.retriever_configs["hybrid_rerank"]):
        raise ValueError("v1 registry differs from canonical benchmark")
    if registry.active not in {"v1", manifest.version}:
        raise ValueError("unexpected active version")
    if registry.active == manifest.version and manifest.state != "ACTIVE":
        raise ValueError("registry activated without matching manifest state")
    path_in(root, target.config_path, "configs/versions")
    path_in(root, target.aggregate_baseline, "baselines")
    path_in(root, target.query_baseline, "baselines")
    path_in(root, target.benchmark_output, "reports/benchmarks")
    path_in(root, target.promotion_output, "reports/promotions")
    if (manifest.version == "v1" or target.aggregate_baseline == experiment.benchmark.baseline
            or target.query_baseline == experiment.benchmark.query_baseline):
        raise ValueError("promotion target would overwrite v1")
    if manifest.state in {"STAGED", "VALIDATION_RUNNING", "VALIDATED", "BASELINED", "ACTIVE", "ROLLED_BACK"}:
        staged = load_config(path_in(root, target.config_path, "configs/versions"))
        if _digest(staged.model_dump()) != item["config_sha256"]:
            raise ValueError("staged config differs from resolved candidate")
    return {"experiment": experiment, "item": item, "context": context,
            "experiment_row": row, "regression": regression, "robustness": robust,
            "registry": registry}


def check(root: Path, manifest_path: Path) -> PromotionManifest:
    manifest = load_manifest(manifest_path)
    precheck(root, manifest)
    if manifest.state == "DRAFT":
        manifest = transition(manifest, "READY_FOR_HUMAN_APPROVAL")
        save_manifest(manifest_path, manifest)
    write_promotion_report(root, manifest)
    return manifest


def approve(root: Path, manifest_path: Path, reviewer: str | None = None) -> PromotionManifest:
    manifest = check(root, manifest_path)
    if manifest.state != "READY_FOR_HUMAN_APPROVAL":
        raise ValueError("promotion is not awaiting human approval")
    if reviewer is not None and not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    manifest = transition(manifest, "APPROVED")
    manifest.human_approval = Approval(approved=True, approved_at=datetime.now(timezone.utc),
                                       approved_by=reviewer)
    save_manifest(manifest_path, manifest)
    write_promotion_report(root, manifest)
    return manifest


def apply(root: Path, manifest_path: Path) -> PromotionManifest:
    manifest = load_manifest(manifest_path)
    evidence = precheck(root, manifest)
    if manifest.state != "APPROVED" or not manifest.human_approval.approved:
        raise ValueError("explicit human approval is required before staging")
    target = path_in(root, manifest.target.config_path, "configs/versions")
    if target.exists():
        raise ValueError("versioned config already exists; refusing replacement")
    registry = evidence["registry"]
    if registry.active != "v1" or manifest.version in registry.versions:
        raise ValueError("registry is not in initial v1 state")
    resolved = evidence["item"]["config"]
    atomic_write(target, yaml.safe_dump(resolved.model_dump(), sort_keys=False))
    registry.versions[manifest.version] = RegistryEntry(
        config=manifest.target.config_path, status="pending_validation",
        promotion_id=manifest.promotion_id,
        resolved_config_sha256=manifest.source.resolved_config_sha256)
    save_registry(path_in(root, manifest.target.registry, "configs/versions"), registry)
    manifest = transition(manifest, "STAGED")
    save_manifest(manifest_path, manifest)
    write_promotion_report(root, manifest)
    return manifest


def write_promotion_report(root: Path, manifest: PromotionManifest, validation: dict | None = None) -> None:
    output = path_in(root, manifest.target.promotion_output, "reports/promotions")
    output.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "1.0", "promotion_id": manifest.promotion_id,
               "candidate": manifest.candidate.candidate_id, "state": manifest.state,
               "approval": manifest.human_approval.model_dump(mode="json"),
               "regression": manifest.evidence.regression,
               "experiment_decision": manifest.evidence.experiment_decision,
               "robustness": manifest.evidence.robustness,
               "active_version": load_registry(path_in(root, manifest.target.registry, "configs/versions")).active,
               "validation": validation,
               "aggregate_baseline": manifest.target.aggregate_baseline if manifest.state in {"BASELINED", "ACTIVE", "ROLLED_BACK"} else None,
               "query_baseline": manifest.target.query_baseline if manifest.state in {"BASELINED", "ACTIVE", "ROLLED_BACK"} else None}
    atomic_write(output / "promotion.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# Retrieval Promotion", "", "## Candidate Identity", "",
             f"- Promotion: {manifest.promotion_id}",
             f"- Candidate: {manifest.candidate.candidate_id}",
             f"- Experiment: {manifest.candidate.experiment_id}",
             f"- State: {manifest.state}", "", "## Evidence Summary", "",
             f"- Regression: {manifest.evidence.regression}",
             f"- Experiment decision: {manifest.evidence.experiment_decision}",
             "- Query diagnostics: included in the saved regression report."]
    lines.extend(f"- Robustness {metric}: {assessment}"
                 for metric, assessment in manifest.evidence.robustness.items())
    if validation and validation.get("forensics_summary"):
        lines.append("- Forensics: completed on four v2 stage runs; see forensics.json.")
    else:
        lines.append("- Forensics: unavailable for the single Phase 10 candidate run; four v2 stage runs required.")
    robust_path = path_in(root, manifest.evidence.robustness_report, "reports/robustness")
    robust = _json(robust_path)
    latency = robust.get("latency")
    latency_note = (f"Median trial p95 {latency['summary']['median_trial_p95_ms']:.2f} ms "
                    f"({latency['measured_runs']} measured trials, {latency['warmup_runs']} excluded warmup)."
                    if latency else "Repeated latency trials were not attached to this evidence.")
    lines.extend(["", "## Latency Caveat", "",
                  latency_note + " Environment-sensitive; not proof of lower latency or production readiness.",
                  "", "## Human Approval", "",
                  f"Approved: {'YES' if manifest.human_approval.approved else 'NO'}",
                  "", "## Validation Benchmark", "",
                  f"Status: {validation['status'] if validation else 'NOT_RUN'}",
                  "", "## Baseline v2", "",
                  ("Created from validated v2 runs; v1 remains intact." if manifest.state in
                   {"BASELINED", "ACTIVE", "ROLLED_BACK"} else
                   "Not created; v1 remains intact."),
                  "", "## Activation And Rollback", "",
                  f"Active version: {payload['active_version']}",
                  "Rollback: groundtruth promotion-rollback osha --to v1"])
    atomic_write(output / "PROMOTION.md", "\n".join(lines) + "\n")


def benchmark_version(root: Path, manifest_path: Path, *, builder=None, evaluator=None) -> PromotionManifest:
    manifest = load_manifest(manifest_path)
    evidence = precheck(root, manifest)
    if manifest.state not in {"STAGED", "FAILED_VALIDATION"} or not manifest.human_approval.approved:
        raise ValueError("approved staged configuration required for validation benchmark")
    registry = evidence["registry"]
    if registry.active != "v1" or registry.versions[manifest.version].status != "pending_validation":
        raise ValueError("v1 must remain active during validation")
    status = benchmark_status(root, evidence["context"]["benchmark"])
    if not status["ready"]:
        raise ValueError("fixed benchmark preparation failed: " + "; ".join(status["errors"]))
    output = path_in(root, manifest.target.benchmark_output, "reports/benchmarks")
    if output.exists() and any(output.iterdir()):
        raise ValueError("validation benchmark output exists; refusing overwrite")
    manifest = transition(manifest, "VALIDATION_RUNNING")
    save_manifest(manifest_path, manifest)
    context = evidence["context"]
    configs = context["benchmark"].retriever_configs
    build, evaluate = builder or build_retriever, evaluator or evaluate_retriever
    runs = {}
    try:
        for name in ("dense", "bm25", "hybrid", "hybrid_rerank"):
            config_path = (path_in(root, manifest.target.config_path, "configs/versions")
                           if name == "hybrid_rerank" else path_in(root, configs[name], "configs"))
            config = load_config(config_path)
            if config.retriever.type != name:
                raise ValueError(f"{name}: versioned config retriever type differs")
            retriever = build(config)
            retriever.index(context["chunks"])
            run = evaluate(retriever, context["queries"], dataset=str(context["dataset_path"]),
                           configuration_name=config.name,
                           configuration={**config.model_dump(),
                                          "corpus_fingerprint": context["fingerprint"],
                                          "dataset_sha256": context["dataset_sha256"],
                                          "config_sha256": file_sha256(config_path),
                                          "source_sha256": context["manifest"].source_sha256,
                                          "benchmark_id": manifest.benchmark_id,
                                          "benchmark_spec_version": context["benchmark"].schema_version,
                                          "promotion_id": manifest.promotion_id,
                                          "candidate_id": manifest.candidate.candidate_id,
                                          "resolved_config_sha256": manifest.source.resolved_config_sha256,
                                          "package_version": __version__,
                                          "python_version": sys.version.split()[0]},
                           k_values=context["benchmark"].k_values)
            if (run.query_count != len(context["queries"])
                    or {q.query_id for q in run.per_query_results} !=
                    {q.query_id for q in context["queries"]}):
                raise ValueError("validation run query set differs")
            runs[name] = run
        summary, markdown, analysis = summarize_runs(runs)
        summary["dataset_sha256"] = context["dataset_sha256"]
        output.mkdir(parents=True, exist_ok=False)
        for name, run in runs.items():
            atomic_write(output / f"{name}.json", run.model_dump_json(indent=2) + "\n")
        atomic_write(output / "summary.json", json.dumps(summary, indent=2) + "\n")
        atomic_write(output / "SUMMARY.md", "# Candidate Validation Benchmark (v2)\n\n" + markdown)
        atomic_write(output / "error_analysis.json", json.dumps(analysis, indent=2) + "\n")
    except Exception:
        failed = transition(manifest, "FAILED_VALIDATION")
        save_manifest(manifest_path, failed)
        write_promotion_report(root, failed, {"status": "FAILED_VALIDATION"})
        raise
    write_promotion_report(root, manifest, {"status": "VALIDATION_RUNNING"})
    return manifest


def validate_version(root: Path, manifest_path: Path) -> dict:
    manifest = load_manifest(manifest_path)
    if manifest.state != "VALIDATION_RUNNING":
        raise ValueError("validation benchmark has not completed")
    evidence = precheck(root, manifest)
    context = evidence["context"]
    output = path_in(root, manifest.target.benchmark_output, "reports/benchmarks")
    try:
        summary = load_summary(output / "summary.json")
        if (summary.corpus_fingerprint != context["fingerprint"]
                or summary.dataset_sha256 != context["dataset_sha256"]
                or summary.query_count != len(context["queries"])
                or set(summary.runs) != {"dense", "bm25", "hybrid", "hybrid_rerank"}):
            raise ValueError("v2 benchmark identity or retriever set differs")
        runs = {name: _load_run(output / "summary.json", summary, name) for name in summary.runs}
        for name, run in runs.items():
            if (run.configuration.get("promotion_id") != manifest.promotion_id
                    or run.configuration.get("resolved_config_sha256") != manifest.source.resolved_config_sha256
                    or "q013" in {q.query_id for q in run.per_query_results}
                    or "q019" not in {q.query_id for q in run.per_query_results}):
                raise ValueError(f"{name}: v2 run provenance or query set differs")
        policy = load_policy(path_in(root, evidence["experiment"].benchmark.regression_policy, "configs"))
        regression = compare(context["baseline"], summary, policy)
        diagnostics = compare_queries(context["query_baseline"], context["baseline"],
                                      output / "summary.json", summary, list(summary.runs),
                                      critical_k=policy.query_regression.critical_k,
                                      warn_rank_drop=policy.query_regression.warn_rank_drop)
        regression["query_analysis"] = diagnostics
        apply_query_gate(regression, diagnostics, policy.query_regression)
        forensics = analyze_forensics(runs["dense"], runs["bm25"], runs["hybrid"],
                                      runs["hybrid_rerank"])
        robust = analyze_robustness(
            load_robustness_run(root / "reports/benchmarks" / manifest.benchmark_id / "hybrid_rerank.json"),
            runs["hybrid_rerank"],
            load_robustness_config(root / "configs/robustness" / f"{manifest.benchmark_id}.yaml"))
        status = regression["overall_status"]
        if status != "PASS":
            raise ValueError(f"v2 regression status is {status}; validation requires PASS")
        result = {"schema_version": "1.0", "status": "PASS", "promotion_id": manifest.promotion_id,
                  "corpus_fingerprint": context["fingerprint"],
                  "dataset_sha256": context["dataset_sha256"],
                  "query_count": summary.query_count,
                  "run_ids": {name: run.run_id for name, run in runs.items()},
                  "regression_status": status,
                  "query_gate_status": regression.get("query_gate_status", "DISABLED"),
                  "forensics_summary": forensics.summary,
                  "robustness": {name: row["assessment"] for name, row in robust["metrics"].items()}}
        validation_dir = path_in(root, manifest.target.promotion_output, "reports/promotions")
        atomic_write(validation_dir / "validation.json", json.dumps(result, indent=2) + "\n")
        atomic_write(validation_dir / "regression.json", json.dumps(regression, indent=2) + "\n")
        atomic_write(validation_dir / "robustness.json", json.dumps(robust, indent=2) + "\n")
        write_forensics(forensics, validation_dir)
        manifest = transition(manifest, "VALIDATED")
        manifest.validation_run_ids = result["run_ids"]
        save_manifest(manifest_path, manifest)
        write_promotion_report(root, manifest, result)
        return result
    except Exception:
        failed = transition(manifest, "FAILED_VALIDATION")
        save_manifest(manifest_path, failed)
        write_promotion_report(root, failed, {"status": "FAILED_VALIDATION"})
        raise


def baseline_version(root: Path, manifest_path: Path) -> PromotionManifest:
    manifest = load_manifest(manifest_path)
    if manifest.state != "VALIDATED" or not manifest.validation_run_ids:
        raise ValueError("v2 validation must PASS before baselining")
    evidence = precheck(root, manifest)
    output = path_in(root, manifest.target.benchmark_output, "reports/benchmarks")
    result = _json(path_in(root, manifest.target.promotion_output, "reports/promotions") / "validation.json")
    summary = load_summary(output / "summary.json")
    if set(summary.runs) != {"dense", "bm25", "hybrid", "hybrid_rerank"}:
        raise ValueError("v2 baseline requires all four retriever runs")
    if (result.get("status") != "PASS" or result.get("run_ids") != manifest.validation_run_ids
            or {name: run.run_id for name, run in summary.runs.items()} != manifest.validation_run_ids
            or summary.corpus_fingerprint != evidence["context"]["fingerprint"]
            or summary.dataset_sha256 != evidence["context"]["dataset_sha256"]):
        raise ValueError("v2 validation evidence differs from completed benchmark")
    for name in summary.runs:
        _load_run(output / "summary.json", summary, name)
    aggregate_path = path_in(root, manifest.target.aggregate_baseline, "baselines")
    query_path = path_in(root, manifest.target.query_baseline, "baselines")
    if aggregate_path.exists() or query_path.exists():
        raise ValueError("v2 baseline exists; refusing replacement")
    aggregate = create_baseline(output / "summary.json", aggregate_path,
                                f"{manifest.benchmark_id}-{manifest.version}",
                                domain=manifest.benchmark_id)
    create_query_baseline(output / "summary.json", aggregate, query_path)
    registry = evidence["registry"]
    entry = registry.versions[manifest.version]
    entry.baseline = manifest.target.aggregate_baseline
    entry.query_baseline = manifest.target.query_baseline
    entry.status = "baselined"
    save_registry(path_in(root, manifest.target.registry, "configs/versions"), registry)
    manifest = transition(manifest, "BASELINED")
    save_manifest(manifest_path, manifest)
    write_promotion_report(root, manifest, result)
    return manifest


def activate(root: Path, manifest_path: Path) -> PromotionManifest:
    manifest = load_manifest(manifest_path)
    if manifest.state not in {"BASELINED", "ROLLED_BACK"} or not manifest.human_approval.approved:
        raise ValueError("approval, validation, and v2 baselines are required before activation")
    evidence = precheck(root, manifest)
    registry = evidence["registry"]
    entry = registry.versions.get(manifest.version)
    if entry is None or entry.status not in {"baselined", "rolled_back"} or registry.active != "v1":
        raise ValueError("registry is not ready for activation")
    result = _json(path_in(root, manifest.target.promotion_output, "reports/promotions") / "validation.json")
    summary_path = path_in(root, manifest.target.benchmark_output, "reports/benchmarks") / "summary.json"
    summary = load_summary(summary_path)
    if set(summary.runs) != {"dense", "bm25", "hybrid", "hybrid_rerank"}:
        raise ValueError("activation requires complete four-retriever validation")
    if (result.get("status") != "PASS" or result.get("run_ids") != manifest.validation_run_ids
            or result.get("corpus_fingerprint") != evidence["context"]["fingerprint"]
            or result.get("dataset_sha256") != evidence["context"]["dataset_sha256"]
            or summary.corpus_fingerprint != evidence["context"]["fingerprint"]
            or summary.dataset_sha256 != evidence["context"]["dataset_sha256"]
            or {name: row.run_id for name, row in summary.runs.items()} != manifest.validation_run_ids):
        raise ValueError("v2 validation evidence is missing or incompatible")
    for name in summary.runs:
        _load_run(summary_path, summary, name)
    aggregate = load_baseline(path_in(root, manifest.target.aggregate_baseline, "baselines"))
    query = load_query_baseline(path_in(root, manifest.target.query_baseline, "baselines"))
    if (aggregate.corpus_fingerprint != evidence["context"]["fingerprint"]
            or aggregate.dataset_sha256 != evidence["context"]["dataset_sha256"]
            or aggregate.query_count != len(evidence["context"]["queries"])
            or {name: row.run_id for name, row in aggregate.runs.items()} != manifest.validation_run_ids
            or any(aggregate.runs[name].metrics != summary.runs[name].metrics
                   for name in summary.runs)
            or query.corpus_fingerprint != aggregate.corpus_fingerprint
            or query.dataset_sha256 != aggregate.dataset_sha256
            or query.query_count != aggregate.query_count
            or {name: row.run_id for name, row in query.retrievers.items()} != manifest.validation_run_ids):
        raise ValueError("v2 baseline pair does not match validated run IDs")
    registry.active = manifest.version
    entry.status = "active"
    save_registry(path_in(root, manifest.target.registry, "configs/versions"), registry)
    manifest = transition(manifest, "ACTIVE")
    save_manifest(manifest_path, manifest)
    write_promotion_report(root, manifest, result)
    return manifest


def rollback(root: Path, manifest_path: Path, to_version: str) -> PromotionManifest:
    manifest = load_manifest(manifest_path)
    registry_path = path_in(root, manifest.target.registry, "configs/versions")
    registry = load_registry(registry_path)
    if (manifest.state != "ACTIVE" or registry.active != manifest.version
            or to_version != "v1" or to_version not in registry.versions):
        raise ValueError("rollback requires active v2 and an existing v1 target")
    old = registry.versions[to_version]
    if not all((root / relative).is_file() for relative in (old.config, old.baseline, old.query_baseline)):
        raise ValueError("rollback target is incomplete")
    registry.active = to_version
    registry.versions[manifest.version].status = "rolled_back"
    save_registry(registry_path, registry)
    manifest = transition(manifest, "ROLLED_BACK")
    save_manifest(manifest_path, manifest)
    write_promotion_report(root, manifest)
    return manifest


def config_status(root: Path, benchmark_id: str) -> dict:
    registry = load_registry(root / "configs/versions" / benchmark_id / "registry.yaml")
    if registry.benchmark_id != benchmark_id:
        raise ValueError("registry benchmark identity differs")
    entry = registry.versions[registry.active]
    if not entry.baseline or not entry.query_baseline:
        raise ValueError("active version is missing baselines")
    for relative in (entry.config, entry.baseline, entry.query_baseline):
        if not (root / relative).is_file():
            raise ValueError(f"active artifact missing: {relative}")
    return {"benchmark": benchmark_id, "active": registry.active, "config": entry.config,
            "baseline": entry.baseline, "query_baseline": entry.query_baseline,
            "previous": "v1" if registry.active != "v1" else None,
            "rollback_available": registry.active != "v1" and "v1" in registry.versions}
