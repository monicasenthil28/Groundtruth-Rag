"""Paired query bootstrap and separately measured retrieval latency."""

import json
import math
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, median

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from groundtruth_rag.evaluation.metrics import query_metrics
from groundtruth_rag.evaluation.runner import BenchmarkRun, evaluate_retriever
from groundtruth_rag.experiments import validate_experiment
from groundtruth_rag.retrieval.factory import build_retriever

SCHEMA_VERSION = "1.0"
SUPPORTED = {f"{name}@{k}" for name in ("recall", "hit_rate", "ndcg", "precision")
             for k in (1, 3, 5, 10)} | {"mrr"}


class QualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    bootstrap_iterations: StrictInt = Field(default=10000, gt=0)
    random_seed: StrictInt = 2026
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    metrics: list[str] = Field(default_factory=lambda: ["recall@3", "mrr", "ndcg@3", "hit_rate@3"])

    @model_validator(mode="after")
    def check_metrics(self) -> "QualityConfig":
        if not self.metrics or len(set(self.metrics)) != len(self.metrics) or set(self.metrics) - SUPPORTED:
            raise ValueError("metrics must be unique supported retrieval metrics")
        return self


class LatencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    warmup_runs: StrictInt = Field(default=1, ge=0)
    measured_runs: StrictInt = Field(default=5, gt=0)


class RobustnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    quality: QualityConfig = Field(default_factory=QualityConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)

    @model_validator(mode="after")
    def check_version(self) -> "RobustnessConfig":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported robustness config schema")
        return self


def load_config(path: Path) -> RobustnessConfig:
    return RobustnessConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_run(path: Path) -> BenchmarkRun:
    from groundtruth_rag.reporting.regression import _read_json
    return BenchmarkRun.model_validate(_read_json(path))


def _checked_rows(run: BenchmarkRun) -> dict:
    rows = {row.query_id: row for row in run.per_query_results}
    if len(rows) != run.query_count:
        raise ValueError("duplicate query IDs")
    for row in rows.values():
        ids = [hit.chunk_id for hit in row.hits]
        expected = query_metrics(row.relevant_chunk_ids, ids, run.k_values)
        if set(expected) != set(row.metrics) or any(
            not math.isfinite(row.metrics[key]) or not math.isclose(row.metrics[key], value, abs_tol=1e-10)
            for key, value in expected.items()
        ):
            raise ValueError(f"stored per-query metrics disagree with hits: {row.query_id}")
    for key, value in run.aggregate_metrics.items():
        metric = "reciprocal_rank" if key == "mrr" else key
        if metric not in next(iter(rows.values())).metrics or not math.isfinite(value):
            raise ValueError(f"invalid aggregate metric: {key}")
        if not math.isclose(value, mean(row.metrics[metric] for row in rows.values()), abs_tol=1e-10):
            raise ValueError(f"stored aggregate disagrees with per-query results: {key}")
    return rows


def analyze(baseline: BenchmarkRun, candidate: BenchmarkRun, config: RobustnessConfig) -> dict:
    if not config.quality.enabled:
        raise ValueError("quality analysis is disabled")
    base_type = baseline.configuration.get("retriever", {}).get("type")
    candidate_type = candidate.configuration.get("retriever", {}).get("type")
    if not base_type or base_type != candidate_type:
        raise ValueError("baseline retriever family differs from candidate")
    if baseline.configuration.get("corpus_fingerprint") != candidate.configuration.get("corpus_fingerprint"):
        raise ValueError("corpus fingerprint differs")
    base_hash = baseline.configuration.get("dataset_sha256")
    if not base_hash or base_hash != candidate.configuration.get("dataset_sha256"):
        raise ValueError("dataset hash differs")
    if baseline.k_values != candidate.k_values:
        raise ValueError("K values differ")
    base_rows, candidate_rows = _checked_rows(baseline), _checked_rows(candidate)
    if base_rows.keys() != candidate_rows.keys():
        raise ValueError("query IDs differ")
    ids = sorted(base_rows)
    for query_id in ids:
        left, right = base_rows[query_id], candidate_rows[query_id]
        if set(left.relevant_chunk_ids) != set(right.relevant_chunk_ids) or left.query != right.query:
            raise ValueError(f"golden judgments or query differ: {query_id}")
    selected = config.quality.metrics
    if any(metric != "mrr" and int(metric.split("@")[1]) not in baseline.k_values for metric in selected):
        raise ValueError("selected metric K is absent from run")
    rng = np.random.default_rng(config.quality.random_seed)
    samples = rng.integers(0, len(ids), size=(config.quality.bootstrap_iterations, len(ids)))
    alpha = (1 - config.quality.confidence_level) / 2
    metrics = {}
    for name in selected:
        key = "reciprocal_rank" if name == "mrr" else name
        base = np.array([base_rows[q].metrics[key] for q in ids], dtype=float)
        cand = np.array([candidate_rows[q].metrics[key] for q in ids], dtype=float)
        differences = cand - base
        draws = differences[samples].mean(axis=1)
        observed = float(differences.mean())
        lower, upper = (float(v) for v in np.quantile(draws, [alpha, 1 - alpha]))
        assessment = ("UNCHANGED" if np.array_equal(base, cand) else
                      "STABLE_IMPROVEMENT" if observed > 0 and lower > 0 else
                      "STABLE_REGRESSION" if observed < 0 and upper < 0 else "INCONCLUSIVE")
        metrics[name] = {
            "baseline": float(base.mean()), "candidate": float(cand.mean()), "delta": observed,
            "bootstrap_mean_delta": float(draws.mean()),
            "bootstrap_median_delta": float(np.median(draws)),
            "ci_lower": lower, "ci_upper": upper,
            "bootstrap_fraction_positive": float(np.mean(draws > 0)),
            "bootstrap_fraction_negative": float(np.mean(draws < 0)),
            "assessment": assessment,
        }
    return {"schema_version": SCHEMA_VERSION, "candidate": candidate.configuration_name,
            "baseline_retriever": baseline.configuration_name, "retriever_type": base_type,
            "baseline_run_id": baseline.run_id, "candidate_run_id": candidate.run_id,
            "corpus_fingerprint": baseline.configuration["corpus_fingerprint"],
            "dataset_sha256": base_hash, "query_count": len(ids),
            "bootstrap_iterations": config.quality.bootstrap_iterations,
            "random_seed": config.quality.random_seed,
            "confidence_level": config.quality.confidence_level,
            "metrics": metrics,
            "warning": f"Confidence intervals reflect resampling of this {len(ids)}-query benchmark and should not be interpreted as universal retrieval performance."}


def render_report(report: dict) -> str:
    level = report["confidence_level"] * 100
    lines = ["# Retrieval Robustness Analysis", "", "## Identity", "",
             f"- Candidate: {report['candidate']}",
             f"- Baseline retriever: {report['baseline_retriever']}",
             f"- Corpus fingerprint: {report['corpus_fingerprint']}",
             f"- Dataset SHA-256: {report['dataset_sha256']}",
             f"- Queries: {report['query_count']}",
             f"- Bootstrap iterations: {report['bootstrap_iterations']}",
             f"- Confidence level: {level:g}%", f"- Seed: {report['random_seed']}",
             "", "## Quality Robustness", "",
             f"| Metric | Baseline | Candidate | Delta | {level:g}% CI | Assessment |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for name, row in report["metrics"].items():
        lines.append(f"| {name} | {row['baseline']:.6f} | {row['candidate']:.6f} | "
                     f"{row['delta']:+.6f} | [{row['ci_lower']:+.6f}, {row['ci_upper']:+.6f}] | "
                     f"{row['assessment']} |")
    lines.extend(["", "## Query Stability", "",
                  "Fractions below are bootstrap sample proportions, not probabilities of correctness.", ""])
    for name, row in report["metrics"].items():
        lines.append(f"- {name}: positive {row['bootstrap_fraction_positive']:.3f}; "
                     f"negative {row['bootstrap_fraction_negative']:.3f}")
    lines.extend(["", "## Latency Stability", ""])
    latency = report.get("latency")
    if latency:
        lines.extend(["| Trial | Mean ms | p50 ms | p95 ms |",
                      "| ---: | ---: | ---: | ---: |"])
        for i, trial in enumerate(latency["trials"], 1):
            lines.append(f"| {i} | {trial['mean_ms']:.2f} | {trial['p50_ms']:.2f} | {trial['p95_ms']:.2f} |")
        lines.append(f"\nMedian trial p95: {latency['summary']['median_trial_p95_ms']:.2f} ms; "
                     f"range: {latency['summary']['min_p95_ms']:.2f}-{latency['summary']['max_p95_ms']:.2f} ms.")
    else:
        lines.append("Not measured; quality bootstrap does not estimate latency stability.")
    lines.extend(["", "## Interpretation", "", report["warning"],
                  "These descriptive intervals describe benchmark-sample stability, not population-level proof or formal universal statistical significance.",
                  "Regression policy (PASS/WARN/FAIL), experiment decision (PROMOTE_CANDIDATE/REVIEW/REJECT), and robustness assessments remain independent.",
                  "Latency trials, when present, describe this execution environment only."])
    return "\n".join(lines) + "\n"


def write_report(report: dict, output: Path) -> None:
    if not output.resolve().is_relative_to(Path("reports/robustness").resolve()):
        raise ValueError("robustness output must be under reports/robustness/")
    output.mkdir(parents=True, exist_ok=True)
    (output / "robustness.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "ROBUSTNESS.md").write_text(render_report(report), encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_metadata() -> dict:
    data = {"os": platform.system(), "python": sys.version.split()[0],
            "cpu_identifier": platform.processor() or None,
            "torch": _package_version("torch"),
            "sentence_transformers": _package_version("sentence-transformers"),
            "transformers": _package_version("transformers"),
            "numpy": _package_version("numpy")}
    try:
        import torch
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["gpu_identifier"] = torch.cuda.get_device_name(0) if data["cuda_available"] else None
    except (ImportError, RuntimeError):
        data["cuda_available"] = False
        data["gpu_identifier"] = None
    return data


def run_latency_trials(root: Path, specification: Path, candidate_id: str,
                       config: RobustnessConfig, *, builder=None, evaluator=None) -> dict:
    if not config.latency.enabled:
        raise ValueError("latency trials are disabled")
    experiment, resolved, context = validate_experiment(root, specification)
    if candidate_id not in resolved:
        raise ValueError(f"unknown candidate: {candidate_id}")
    item = resolved[candidate_id]
    retriever = (builder or build_retriever)(item["config"])
    retriever.index(context["chunks"])
    evaluate = evaluator or evaluate_retriever
    def trial():
        return evaluate(retriever, context["queries"], dataset=str(context["dataset_path"]),
                        configuration_name=candidate_id, k_values=context["benchmark"].k_values)
    for _ in range(config.latency.warmup_runs):
        trial()
    measured = [trial().latency.model_dump() for _ in range(config.latency.measured_runs)]
    for row in measured:
        if any(not math.isfinite(value) or value < 0 for value in row.values()):
            raise ValueError("invalid measured latency")
    p95 = [row["p95_ms"] for row in measured]
    return {"schema_version": SCHEMA_VERSION, "experiment_id": experiment.experiment_id,
            "candidate": candidate_id, "retriever_type": item["retriever_type"],
            "config_sha256": item["config_sha256"],
            "corpus_fingerprint": context["fingerprint"],
            "dataset_sha256": context["dataset_sha256"],
            "query_count": len(context["queries"]),
            "warmup_runs": config.latency.warmup_runs,
            "measured_runs": config.latency.measured_runs,
            "environment": environment_metadata(), "trials": measured,
            "summary": {"mean_trial_mean_ms": mean(row["mean_ms"] for row in measured),
                        "median_trial_p50_ms": median(row["p50_ms"] for row in measured),
                        "median_trial_p95_ms": median(p95),
                        "min_p95_ms": min(p95), "max_p95_ms": max(p95)}}
