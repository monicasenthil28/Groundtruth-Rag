"""Validated, advisory retrieval experiments over a fixed benchmark."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from groundtruth_rag import __version__
from groundtruth_rag.benchmark_setup import ReviewManifest, load_spec
from groundtruth_rag.datasets import load_dataset
from groundtruth_rag.evaluation.runner import evaluate_retriever
from groundtruth_rag.ingestion.corpus import load_corpus
from groundtruth_rag.labeling.manifest import CorpusManifest, file_sha256
from groundtruth_rag.labeling.models import atomic_write
from groundtruth_rag.reporting.query_regression import compare_queries, load_query_baseline
from groundtruth_rag.reporting.regression import (
    apply_query_gate, compare, load_baseline, load_policy, load_summary, write_report,
)
from groundtruth_rag.retrieval.dense import corpus_fingerprint
from groundtruth_rag.retrieval.factory import RunConfig, build_retriever, load_config

OVERRIDES = {
    "hybrid": {"rrf_k", "dense_candidates", "bm25_candidates"},
    "hybrid_rerank": {"rrf_k", "dense_candidates", "bm25_candidates", "candidate_pool"},
}
VERSIONS = ("sentence-transformers", "transformers", "torch", "numpy", "rank-bm25", "pypdf")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkRef(StrictModel):
    id: str = Field(min_length=1)
    spec: str
    corpus: str
    manifest: str
    dataset: str
    baseline: str
    query_baseline: str
    regression_policy: str


class CandidateSpec(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    base_config: str
    overrides: dict[str, StrictInt] = Field(min_length=1)


class ParetoSpec(StrictModel):
    quality_metrics: list[str] = Field(min_length=1)
    include_latency: bool = False
    latency_metric: Literal["mean_ms", "p50_ms", "p95_ms"] = "p95_ms"

    @model_validator(mode="after")
    def unique_metrics(self) -> "ParetoSpec":
        if len(set(self.quality_metrics)) != len(self.quality_metrics):
            raise ValueError("duplicate Pareto quality metrics")
        return self


class PromotionSpec(StrictModel):
    review_on_new_critical: bool = True
    latency_review_increase_pct: float | None = Field(default=None, ge=0)


class ExperimentSpec(StrictModel):
    schema_version: Literal["1.0"]
    experiment_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    benchmark: BenchmarkRef
    pareto: ParetoSpec
    promotion: PromotionSpec = Field(default_factory=PromotionSpec)
    candidates: list[CandidateSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "ExperimentSpec":
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate candidate ID")
        return self


def load_experiment(path: Path) -> ExperimentSpec:
    return ExperimentSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes experiment root: {relative}")
    return path


def _digest(value: dict) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _versions() -> dict[str, str | None]:
    result = {}
    for package in VERSIONS:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def resolve_candidates(root: Path, experiment: ExperimentSpec, benchmark_spec) -> dict[str, dict]:
    resolved = {}
    seen = set()
    canonical = {_within(root, path): name for name, path in benchmark_spec.retriever_configs.items()}
    for candidate in experiment.candidates:
        path = _within(root, candidate.base_config)
        if path not in canonical:
            raise ValueError(f"{candidate.id}: base config is not a canonical benchmark config")
        config = load_config(path)
        kind = config.retriever.type
        if canonical[path] != kind or kind not in OVERRIDES:
            raise ValueError(f"{candidate.id}: unsupported retriever type")
        if set(candidate.overrides) - OVERRIDES[kind]:
            raise ValueError(f"{candidate.id}: unknown override field")
        settings = config.retriever.model_dump()
        settings.update(candidate.overrides)
        if any(value <= 0 for value in candidate.overrides.values()):
            raise ValueError(f"{candidate.id}: override values must be positive")
        effective = RunConfig.model_validate({"name": candidate.id, "retriever": settings})
        identity = _digest(effective.retriever.model_dump())
        if identity in seen:
            raise ValueError("duplicate resolved configuration")
        seen.add(identity)
        resolved[candidate.id] = {
            "id": candidate.id, "base_config": candidate.base_config,
            "retriever_type": kind, "config": effective,
            "base_config_sha256": file_sha256(path),
            "config_sha256": _digest(effective.model_dump()),
        }
    return resolved


def validate_experiment(root: Path, spec_path: Path) -> tuple[ExperimentSpec, dict[str, dict], dict]:
    experiment = load_experiment(spec_path)
    ref = experiment.benchmark
    benchmark = load_spec(_within(root, ref.spec))
    if benchmark.benchmark_id != ref.id:
        raise ValueError("benchmark ID differs from experiment specification")
    if _within(root, ref.dataset) != _within(root, benchmark.golden.dataset):
        raise ValueError("experiment dataset differs from benchmark specification")
    if _within(root, ref.baseline) != _within(root, benchmark.baseline):
        raise ValueError("experiment baseline differs from benchmark specification")
    corpus_path = _within(root, ref.corpus)
    chunks = load_corpus(corpus_path)
    manifest = CorpusManifest.model_validate_json(
        _within(root, ref.manifest).read_text(encoding="utf-8")
    )
    fingerprint = corpus_fingerprint(chunks)
    if (fingerprint != benchmark.corpus.fingerprint or manifest.corpus_fingerprint != fingerprint
            or len(chunks) != benchmark.corpus.chunk_count
            or manifest.chunk_size != benchmark.corpus.chunk_size_words
            or manifest.chunk_overlap != benchmark.corpus.chunk_overlap_words
            or manifest.page_count != benchmark.source.page_count
            or manifest.document_count != benchmark.source.document_count
            or manifest.chunk_count != benchmark.corpus.chunk_count
            or manifest.source_document != benchmark.source.document
            or manifest.source_sha256 != benchmark.source.sha256):
        raise ValueError("fixed corpus is incompatible with benchmark specification")
    dataset_path = _within(root, ref.dataset)
    dataset_hash = file_sha256(dataset_path)
    review = ReviewManifest.model_validate_json(
        _within(root, benchmark.golden.review_manifest).read_text(encoding="utf-8")
    )
    queries = load_dataset(dataset_path)
    known = {chunk.chunk_id for chunk in chunks}
    if (dataset_hash != review.dataset_sha256
            or len(queries) != benchmark.golden.verified_queries
            or any(query.query_id in benchmark.golden.skipped_ids for query in queries)
            or any(query.relevant_chunk_ids - known for query in queries)):
        raise ValueError("golden dataset is incompatible with fixed corpus")
    baseline = load_baseline(_within(root, ref.baseline))
    query_baseline = load_query_baseline(_within(root, ref.query_baseline))
    policy = load_policy(_within(root, ref.regression_policy))
    if 3 not in benchmark.k_values:
        raise ValueError("experiment comparison report requires K=3")
    if (baseline.corpus_fingerprint != fingerprint or baseline.dataset_sha256 != dataset_hash
            or baseline.query_count != len(queries) or baseline.k_values != benchmark.k_values
            or baseline.dataset_name != dataset_path.name
            or query_baseline.corpus_fingerprint != fingerprint
            or query_baseline.dataset_sha256 != dataset_hash
            or query_baseline.query_count != len(queries)
            or query_baseline.k_values != benchmark.k_values
            or query_baseline.benchmark_name != baseline.name):
        raise ValueError("baseline or query baseline is incompatible with benchmark")
    labels = {query.query_id: query.relevant_chunk_ids for query in queries}
    for name, ranking in query_baseline.retrievers.items():
        if name not in baseline.runs or ranking.run_id != baseline.runs[name].run_id:
            raise ValueError("query baseline run ID differs from aggregate baseline")
        if {row.query_id: set(row.relevant_chunk_ids) for row in ranking.queries} != labels:
            raise ValueError("query baseline labels differ from golden dataset")
    resolved = resolve_candidates(root, experiment, benchmark)
    if any(item["retriever_type"] not in policy.retrievers for item in resolved.values()):
        raise ValueError("regression policy does not include a candidate retriever")
    for metric in experiment.pareto.quality_metrics:
        if any(metric not in baseline.runs[item["retriever_type"]].metrics
               for item in resolved.values()):
            raise ValueError(f"Pareto metric unavailable in baseline: {metric}")
    context = {"benchmark": benchmark, "chunks": chunks, "queries": queries,
               "manifest": manifest, "baseline": baseline,
               "query_baseline": query_baseline, "policy": policy,
               "fingerprint": fingerprint, "dataset_sha256": dataset_hash,
               "dataset_path": dataset_path}
    return experiment, resolved, context


def pareto_status(rows: list[dict], metrics: list[str], *, include_latency: bool,
                  latency_metric: str = "p95_ms") -> dict[str, str]:
    result = {}
    for candidate in rows:
        dominated = False
        for other in rows:
            if other["candidate_id"] == candidate["candidate_id"]:
                continue
            quality = all(other["metrics"][metric] >= candidate["metrics"][metric]
                          for metric in metrics)
            cost = not include_latency or other["latency"][latency_metric] <= candidate["latency"][latency_metric]
            strict = (any(other["metrics"][metric] > candidate["metrics"][metric]
                          for metric in metrics) or
                      (include_latency and other["latency"][latency_metric] < candidate["latency"][latency_metric]))
            if quality and cost and strict:
                dominated = True
                break
        result[candidate["candidate_id"]] = "DOMINATED" if dominated else "PARETO"
    return result


def promotion_status(row: dict, *, review_on_new_critical: bool,
                     latency_review_increase_pct: float | None) -> str:
    if row["regression_status"] == "FAIL" or row.get("query_gate_status") == "FAIL":
        return "REJECT"
    if row["regression_status"] == "WARN" or row.get("query_gate_status") == "WARN":
        return "REVIEW"
    if review_on_new_critical and row["new_critical_failures"]:
        return "REVIEW"
    if row["pareto"] == "DOMINATED":
        return "REVIEW"
    if latency_review_increase_pct is not None and row["baseline_latency_p95_ms"] > 0:
        increase = (row["latency"]["p95_ms"] / row["baseline_latency_p95_ms"] - 1) * 100
        if increase >= latency_review_increase_pct:
            return "REVIEW"
    return "PROMOTE_CANDIDATE"


def _summary(run, kind: str, fingerprint: str, dataset_path: Path, dataset_hash: str) -> dict:
    return {"schema_version": "1.0", "corpus_fingerprint": fingerprint,
            "dataset": str(dataset_path), "dataset_sha256": dataset_hash,
            "query_count": run.query_count, "k_values": run.k_values,
            "runs": {kind: {"run_id": run.run_id, "metrics": run.aggregate_metrics,
                            "latency": run.latency.model_dump()}}}


def render_experiment(report: dict) -> str:
    lines = [f"# Retrieval Experiment: {report['experiment_id']}", "",
             f"- Benchmark: {report['benchmark_id']}",
             f"- Corpus fingerprint: {report['corpus_fingerprint']}",
             f"- Dataset SHA-256: {report['dataset_sha256']}",
             "- Promotion is advisory; canonical configurations are unchanged.",
             "- Latency is environment-sensitive.", "", "## Comparison", "",
             "| Candidate | Retriever | Recall@3 | MRR | nDCG@3 | Hit@3 | p95 ms | Recall@3 delta | MRR delta | nDCG@3 delta | Regression | Pareto | Decision |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"]
    for row in report["candidates"]:
        for label, values, latency, status, pareto, decision in (
            ("v1 baseline", row["baseline_metrics"], row["baseline_latency_p95_ms"], "BASELINE", "-", "-"),
            (row["candidate_id"], row["metrics"], row["latency"]["p95_ms"],
             row["regression_status"], row["pareto"], row["promotion_status"]),
        ):
            delta = {key: values[key] - row["baseline_metrics"][key]
                     for key in ("recall@3", "mrr", "ndcg@3")}
            lines.append(f"| {label} | {row['retriever_type']} | {values['recall@3']:.4f} | "
                         f"{values['mrr']:.4f} | {values['ndcg@3']:.4f} | "
                         f"{values['hit_rate@3']:.4f} | {latency:.2f} | "
                         f"{delta['recall@3']:+.4f} | {delta['mrr']:+.4f} | "
                         f"{delta['ndcg@3']:+.4f} | {status} | {pareto} | {decision} |")
    lines.extend(["", "## Diagnostics", ""])
    for row in report["candidates"]:
        lines.append(f"- {row['candidate_id']}: {row['new_critical_failures']} new critical query misses; "
                     f"forensics {row['forensics_status']}")
    return "\n".join(lines) + "\n"


def run_experiment(root: Path, spec_path: Path, output: Path, *, candidate_id: str | None = None,
                   dry_run: bool = False, builder=None) -> dict:
    experiment, resolved, context = validate_experiment(root, spec_path)
    if candidate_id is not None and candidate_id not in resolved:
        raise ValueError(f"unknown candidate: {candidate_id}")
    selected = {key: value for key, value in resolved.items()
                if candidate_id is None or key == candidate_id}
    if dry_run:
        return {"experiment_id": experiment.experiment_id, "dry_run": True,
                "corpus_fingerprint": context["fingerprint"],
                "dataset_sha256": context["dataset_sha256"],
                "candidates": [{"candidate_id": key, "base_config": item["base_config"],
                                "config_sha256": item["config_sha256"],
                                "resolved_config": item["config"].model_dump()}
                               for key, item in selected.items()]}
    if not output.resolve().is_relative_to((root / "reports").resolve()):
        raise ValueError("experiment output must be under reports/")
    build = builder or build_retriever
    rows = []
    for key, item in selected.items():
        kind = item["retriever_type"]
        directory = output / "candidates" / key
        config = item["config"]
        retriever = build(config)
        retriever.index(context["chunks"])
        metadata = {**config.model_dump(), "experiment_id": experiment.experiment_id,
                    "candidate_id": key, "base_config": item["base_config"],
                    "base_config_sha256": item["base_config_sha256"],
                    "config_sha256": item["config_sha256"],
                    "corpus_fingerprint": context["fingerprint"],
                    "dataset_sha256": context["dataset_sha256"],
                    "source_sha256": context["manifest"].source_sha256,
                    "benchmark_id": experiment.benchmark.id,
                    "benchmark_spec_version": context["benchmark"].schema_version,
                    "package_version": __version__, "python_version": sys.version.split()[0],
                    "dependency_versions": _versions()}
        run = evaluate_retriever(
            retriever, context["queries"], dataset=str(context["dataset_path"]),
            configuration_name=config.name, configuration=metadata,
            k_values=context["benchmark"].k_values,
        )
        atomic_write(directory / f"{kind}.json", run.model_dump_json(indent=2) + "\n")
        atomic_write(directory / "summary.json", json.dumps(
            _summary(run, kind, context["fingerprint"], context["dataset_path"],
                     context["dataset_sha256"]), indent=2) + "\n")
        summary = load_summary(directory / "summary.json")
        policy = context["policy"].model_copy(update={"retrievers": [kind]})
        regression = compare(context["baseline"], summary, policy)
        analysis = compare_queries(
            context["query_baseline"], context["baseline"], directory / "summary.json",
            summary, [kind], critical_k=policy.query_regression.critical_k,
            warn_rank_drop=policy.query_regression.warn_rank_drop,
        )
        regression["query_analysis"] = analysis
        apply_query_gate(regression, analysis, policy.query_regression)
        regression["candidate_summary"] = str(directory / "summary.json")
        write_report(regression, directory)
        new_critical = sum(len(analysis["retrievers"][kind]["new_failures"][str(k)])
                           for k in policy.query_regression.critical_k)
        baseline_run = context["baseline"].runs[kind]
        rows.append({"candidate_id": key, "retriever_type": kind,
                     "base_config": item["base_config"],
                     "base_config_sha256": item["base_config_sha256"],
                     "config_sha256": item["config_sha256"],
                     "resolved_config": config.model_dump(),
                     "run_id": run.run_id, "timestamp": run.timestamp.isoformat(),
                     "metrics": run.aggregate_metrics, "latency": run.latency.model_dump(),
                     "baseline_metrics": baseline_run.metrics,
                     "baseline_latency_p95_ms": baseline_run.latency["p95_ms"],
                     "regression_status": regression["overall_status"],
                     "query_gate_status": regression.get("query_gate_status"),
                     "new_critical_failures": new_critical,
                     "forensics_status": "NOT_AVAILABLE: four compatible stage runs required"})
    complete = candidate_id is None
    frontier = pareto_status(rows, experiment.pareto.quality_metrics,
                             include_latency=experiment.pareto.include_latency,
                             latency_metric=experiment.pareto.latency_metric) if complete else {}
    for row in rows:
        row["pareto"] = frontier.get(row["candidate_id"], "NOT_EVALUATED")
        decision = promotion_status(
            row, review_on_new_critical=experiment.promotion.review_on_new_critical,
            latency_review_increase_pct=experiment.promotion.latency_review_increase_pct,
        )
        row["promotion_status"] = "REVIEW" if not complete and decision == "PROMOTE_CANDIDATE" else decision
    result = {"schema_version": "1.0", "experiment_id": experiment.experiment_id,
              "benchmark_id": experiment.benchmark.id,
              "benchmark_spec_version": context["benchmark"].schema_version,
              "timestamp": datetime.now(timezone.utc).isoformat(),
              "corpus_fingerprint": context["fingerprint"],
              "dataset_sha256": context["dataset_sha256"],
              "pareto_policy": experiment.pareto.model_dump(),
              "complete_candidate_set": complete, "candidates": rows}
    atomic_write(output / "experiment.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    atomic_write(output / "EXPERIMENT.md", render_experiment(result))
    return result


def load_experiment_report(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0" or not isinstance(data.get("candidates"), list):
        raise ValueError("unsupported experiment report schema")
    return data
