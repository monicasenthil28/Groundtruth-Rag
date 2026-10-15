"""Versioned retrieval baselines and policy-based aggregate regression checks."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from groundtruth_rag.evaluation.runner import BenchmarkRun
from groundtruth_rag.labeling.models import atomic_write

QUALITY_PATTERN = re.compile(r"^(recall|precision|hit_rate|ndcg)@([1-9][0-9]*)$")
LATENCY_NAMES = {"mean_ms", "p50_ms", "p95_ms"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromotionProvenance(StrictModel):
    promotion_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    resolved_config_sha256: str = Field(min_length=64, max_length=64)


class Aggregate(StrictModel):
    run_id: str = Field(min_length=1)
    metrics: dict[str, FiniteFloat]
    latency: dict[str, FiniteFloat]
    configuration_name: str | None = None
    package_version: str | None = None
    python_version: str | None = None

    @model_validator(mode="after")
    def valid_values(self) -> "Aggregate":
        if set(self.latency) != LATENCY_NAMES:
            raise ValueError("latency must contain mean_ms, p50_ms, and p95_ms")
        if any(value < 0 for value in self.latency.values()):
            raise ValueError("latency values must be nonnegative")
        for name, value in self.metrics.items():
            if name != "mrr" and not QUALITY_PATTERN.fullmatch(name):
                raise ValueError(f"unsupported aggregate metric: {name}")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        return self


class BenchmarkSummary(StrictModel):
    schema_version: Literal["1.0"]
    corpus_fingerprint: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    dataset_sha256: str | None = None
    query_count: int = Field(gt=0)
    k_values: list[int]
    runs: dict[str, Aggregate]

    @model_validator(mode="after")
    def valid_summary(self) -> "BenchmarkSummary":
        if not self.runs or not self.k_values or any(k <= 0 for k in self.k_values):
            raise ValueError("summary needs retrievers and positive K values")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError("duplicate K values")
        return self


class Baseline(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    created_at: datetime
    source_summary_schema_version: Literal["1.0"] = "1.0"
    corpus_fingerprint: str = Field(min_length=1)
    dataset_sha256: str | None = None
    dataset_name: str = Field(min_length=1)
    query_count: int = Field(gt=0)
    skipped_count: int | None = Field(default=None, ge=0)
    k_values: list[int]
    runs: dict[str, Aggregate]
    promotion_provenance: PromotionProvenance | None = None

    @model_validator(mode="after")
    def valid_baseline(self) -> "Baseline":
        if not self.runs or not self.k_values or any(k <= 0 for k in self.k_values):
            raise ValueError("baseline needs retrievers and positive K values")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError("duplicate K values")
        return self


class MetricRule(StrictModel):
    enabled: bool = True
    warn_drop: FiniteFloat | None = None
    fail_drop: FiniteFloat | None = None
    warn_increase_pct: FiniteFloat | None = None
    fail_increase_pct: FiniteFloat | None = None


class RetrieverRules(StrictModel):
    quality: dict[str, MetricRule] = Field(default_factory=dict)
    latency: dict[str, MetricRule] = Field(default_factory=dict)


class QueryRegressionPolicy(StrictModel):
    enabled: bool = False
    critical_k: list[int] = Field(default_factory=list)
    fail_on_new_failure_at_k: bool = False
    warn_rank_drop: int | None = Field(default=None, ge=1)
    max_new_failures_warn: int | None = Field(default=None, ge=0)
    max_new_failures_fail: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_thresholds(self) -> "QueryRegressionPolicy":
        if len(set(self.critical_k)) != len(self.critical_k) or any(k <= 0 for k in self.critical_k):
            raise ValueError("critical K values must be unique and positive")
        if (self.max_new_failures_warn is not None and self.max_new_failures_fail is not None
                and self.max_new_failures_fail < self.max_new_failures_warn):
            raise ValueError("failure tolerance must be at least warning tolerance")
        return self


class RegressionPolicy(RetrieverRules):
    schema_version: Literal["1.0"] = "1.0"
    retrievers: list[str]
    per_retriever: dict[str, RetrieverRules] = Field(default_factory=dict)
    query_regression: QueryRegressionPolicy = Field(default_factory=QueryRegressionPolicy)

    @model_validator(mode="after")
    def valid_policy(self) -> "RegressionPolicy":
        if not self.retrievers or len(set(self.retrievers)) != len(self.retrievers):
            raise ValueError("retrievers must be nonempty and unique")
        if set(self.per_retriever) - set(self.retrievers):
            raise ValueError("override names must be configured retrievers")
        for rules in [self, *self.per_retriever.values()]:
            for name, rule in rules.quality.items():
                if name != "mrr" and not QUALITY_PATTERN.fullmatch(name):
                    raise ValueError(f"unsupported quality metric: {name}")
                _check_rule(rule, "quality", name)
            for name, rule in rules.latency.items():
                if name not in LATENCY_NAMES:
                    raise ValueError(f"unsupported latency metric: {name}")
                _check_rule(rule, "latency", name)
        if any(not _effective_rules(self, name) for name in self.retrievers):
            raise ValueError("every retriever needs at least one enabled metric")
        return self


def _check_rule(rule: MetricRule, kind: str, name: str) -> None:
    warn = rule.warn_drop if kind == "quality" else rule.warn_increase_pct
    fail = rule.fail_drop if kind == "quality" else rule.fail_increase_pct
    other = (rule.warn_increase_pct, rule.fail_increase_pct) if kind == "quality" else (
        rule.warn_drop, rule.fail_drop,
    )
    if any(value is not None for value in other):
        raise ValueError(f"{name}: wrong threshold direction")
    if rule.enabled and (warn is None or fail is None):
        raise ValueError(f"{name}: enabled rule needs warning and failure thresholds")
    if any(value is not None and value < 0 for value in (warn, fail)):
        raise ValueError(f"{name}: thresholds must be nonnegative")
    if warn is not None and fail is not None and fail < warn:
        raise ValueError(f"{name}: failure threshold must be at least warning threshold")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_summary(path: Path) -> BenchmarkSummary:
    data = _read_json(path)
    summary = BenchmarkSummary.model_validate(data)
    hashes: set[str] = set()
    for name, aggregate in summary.runs.items():
        sibling = path.parent / f"{name}.json"
        if not sibling.exists():
            continue
        run = BenchmarkRun.model_validate(_read_json(sibling))
        if (run.run_id != aggregate.run_id or run.dataset != summary.dataset
                or run.query_count != summary.query_count or run.k_values != summary.k_values
                or run.configuration.get("corpus_fingerprint") != summary.corpus_fingerprint
                or run.aggregate_metrics != aggregate.metrics
                or run.latency.model_dump() != aggregate.latency):
            raise ValueError(f"{name}: run JSON contradicts summary")
        digest = run.configuration.get("dataset_sha256")
        if digest:
            hashes.add(digest)
        aggregate.configuration_name = run.configuration_name
        aggregate.package_version = run.configuration.get("package_version")
        aggregate.python_version = run.configuration.get("python_version")
    if len(hashes) > 1 or (hashes and summary.dataset_sha256 not in (None, *hashes)):
        raise ValueError("dataset hashes disagree across benchmark runs")
    if hashes:
        summary.dataset_sha256 = next(iter(hashes))
    return summary


def create_baseline(summary_path: Path, output: Path, name: str, *, domain: str | None = None,
                    force: bool = False,
                    promotion_provenance: PromotionProvenance | None = None) -> Baseline:
    summary = load_summary(summary_path)
    baseline = Baseline(
        name=name, domain=domain or summary_path.parent.name,
        created_at=datetime.now(timezone.utc), corpus_fingerprint=summary.corpus_fingerprint,
        dataset_sha256=summary.dataset_sha256, dataset_name=Path(summary.dataset).name,
        query_count=summary.query_count, k_values=summary.k_values, runs=summary.runs,
        promotion_provenance=promotion_provenance,
    )
    content = baseline.model_dump_json(indent=2) + "\n"
    if force:
        atomic_write(output, content)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    return baseline


def load_baseline(path: Path) -> Baseline:
    return Baseline.model_validate(_read_json(path))


def load_policy(path: Path) -> RegressionPolicy:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RegressionPolicy.model_validate(value)


def _effective_rules(policy: RegressionPolicy, retriever: str) -> dict[str, tuple[str, MetricRule]]:
    rules = {name: ("higher", rule) for name, rule in policy.quality.items()}
    rules.update({name: ("lower", rule) for name, rule in policy.latency.items()})
    override = policy.per_retriever.get(retriever)
    if override:
        rules.update({name: ("higher", rule) for name, rule in override.quality.items()})
        rules.update({name: ("lower", rule) for name, rule in override.latency.items()})
    return {name: value for name, value in rules.items() if value[1].enabled}


def compare(baseline: Baseline, candidate: BenchmarkSummary,
            policy: RegressionPolicy) -> dict:
    checks: list[str] = []
    if baseline.source_summary_schema_version != candidate.schema_version:
        raise ValueError("INCOMPATIBLE BENCHMARKS: summary schema version differs")
    if baseline.corpus_fingerprint != candidate.corpus_fingerprint:
        raise ValueError("INCOMPATIBLE BENCHMARKS: corpus fingerprint differs")
    checks.append("corpus fingerprint matches")
    if baseline.dataset_sha256 or candidate.dataset_sha256:
        if not baseline.dataset_sha256 or baseline.dataset_sha256 != candidate.dataset_sha256:
            raise ValueError("INCOMPATIBLE BENCHMARKS: golden dataset hash differs or is missing")
        checks.append("golden dataset hash matches")
    elif baseline.dataset_name != Path(candidate.dataset).name:
        raise ValueError("INCOMPATIBLE BENCHMARKS: dataset identity differs")
    else:
        checks.append("dataset name matches; hash unavailable")
    if baseline.query_count != candidate.query_count:
        raise ValueError("INCOMPATIBLE BENCHMARKS: query count differs")
    checks.append("query count matches")
    details: list[dict] = []
    for retriever in policy.retrievers:
        if retriever not in baseline.runs or retriever not in candidate.runs:
            raise ValueError(f"INCOMPATIBLE BENCHMARKS: missing retriever {retriever}")
        old_run, new_run = baseline.runs[retriever], candidate.runs[retriever]
        for name, (direction, rule) in _effective_rules(policy, retriever).items():
            if direction == "higher":
                match = QUALITY_PATTERN.fullmatch(name)
                if match and int(match.group(2)) not in baseline.k_values:
                    raise ValueError(f"INCOMPATIBLE BENCHMARKS: baseline lacks K for {name}")
                if match and int(match.group(2)) not in candidate.k_values:
                    raise ValueError(f"INCOMPATIBLE BENCHMARKS: candidate lacks K for {name}")
                if name not in old_run.metrics or name not in new_run.metrics:
                    raise ValueError(f"INCOMPATIBLE BENCHMARKS: missing metric {retriever}.{name}")
                old, new = old_run.metrics[name], new_run.metrics[name]
                regression = max(0.0, old - new)
                warn, fail = rule.warn_drop, rule.fail_drop
                tolerance = {"warn_drop": warn, "fail_drop": fail}
            else:
                if name not in old_run.latency or name not in new_run.latency:
                    raise ValueError(f"INCOMPATIBLE BENCHMARKS: missing metric {retriever}.{name}")
                old, new = old_run.latency[name], new_run.latency[name]
                if old == 0 and new > 0:
                    raise ValueError(f"INCOMPATIBLE BENCHMARKS: cannot compare {retriever}.{name} from zero baseline")
                regression = max(0.0, (new - old) / old * 100) if old else 0.0
                warn, fail = rule.warn_increase_pct, rule.fail_increase_pct
                tolerance = {"warn_increase_pct": warn, "fail_increase_pct": fail}
            delta = new - old
            status = "FAIL" if regression >= fail and regression > 0 else (
                "WARN" if regression >= warn and regression > 0 else "PASS"
            )
            details.append({
                "retriever": retriever, "metric": name, "direction": direction,
                "baseline_value": old, "candidate_value": new, "absolute_delta": delta,
                "relative_delta_percent": delta / abs(old) * 100 if old else None,
                "threshold": tolerance, "status": status, "improved": delta > 0 if direction == "higher" else delta < 0,
            })
    checks.extend(["configured retrievers and metrics present", "required K values present", "numeric values finite"])
    overall = "FAIL" if any(row["status"] == "FAIL" for row in details) else (
        "WARN" if any(row["status"] == "WARN" for row in details) else "PASS"
    )
    return {
        "schema_version": "1.0", "baseline_name": baseline.name, "domain": baseline.domain,
        "candidate_summary": None, "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_fingerprint": baseline.corpus_fingerprint,
        "dataset_sha256": baseline.dataset_sha256,
        "dataset_name": baseline.dataset_name,
        "query_count": candidate.query_count, "k_values": candidate.k_values,
        "compatibility_checks": checks, "overall_status": overall, "details": details,
        "retriever_values": {
            name: {
                "baseline": {**baseline.runs[name].metrics, **baseline.runs[name].latency},
                "candidate": {**candidate.runs[name].metrics, **candidate.runs[name].latency},
            } for name in policy.retrievers
        },
        "retriever_status": {name: ("FAIL" if any(row["retriever"] == name and row["status"] == "FAIL" for row in details)
                                    else "WARN" if any(row["retriever"] == name and row["status"] == "WARN" for row in details)
                                    else "PASS") for name in policy.retrievers},
    }


def apply_query_gate(report: dict, analysis: dict, rules: QueryRegressionPolicy) -> None:
    """Optionally combine query diagnostics with the aggregate gate."""
    if not rules.enabled:
        return
    new_count = sum(len(data["new_failures"][str(k)])
                    for data in analysis["retrievers"].values() for k in rules.critical_k)
    severity = "PASS"
    if ((rules.fail_on_new_failure_at_k and new_count > 0)
            or (rules.max_new_failures_fail is not None
                and new_count > rules.max_new_failures_fail)):
        severity = "FAIL"
    elif ((rules.max_new_failures_warn is not None
           and new_count > rules.max_new_failures_warn)
          or any(row["severity"] == "WARN" for data in analysis["retrievers"].values()
                 for row in data["queries"])):
        severity = "WARN"
    report["query_gate_status"] = severity
    if severity == "FAIL" or (severity == "WARN" and report["overall_status"] == "PASS"):
        report["overall_status"] = severity


def render_markdown(report: dict) -> str:
    rows = report["details"]
    by_key = {(row["retriever"], row["metric"]): row for row in rows}

    def value(retriever: str, metric: str, side: str) -> str:
        row = by_key.get((retriever, metric))
        if row:
            return format(row[f"{side}_value"], ".6g")
        measured = report["retriever_values"][retriever][side].get(metric)
        return f"{measured:.6g} (not gated)" if measured is not None else "n/a"

    lines = ["# Retrieval Regression Report", "", "## Provenance", "",
             f"- Baseline: {report['baseline_name']}",
             f"- Candidate: {report['candidate_summary']}",
             f"- Corpus fingerprint: {report['corpus_fingerprint'][:12]}",
             f"- Dataset SHA-256: {report['dataset_sha256'] or 'unavailable'}",
             f"- Query count: {report['query_count']}", f"- Timestamp: {report['timestamp']}",
             "", "## Overall Status", "", report["overall_status"], "", "## Retriever Summary", "",
             "| Retriever | Status | Recall@3 baseline | Recall@3 candidate | MRR baseline | MRR candidate | nDCG@3 baseline | nDCG@3 candidate | p95 baseline | p95 candidate |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for retriever, status in report["retriever_status"].items():
        cells = [retriever, status]
        for metric in ("recall@3", "mrr", "ndcg@3", "p95_ms"):
            cells.extend([value(retriever, metric, "baseline"), value(retriever, metric, "candidate")])
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Metric Details", ""])
    for retriever in report["retriever_status"]:
        lines.extend([f"### {retriever}", "", "| Metric | Baseline | Candidate | Delta | Tolerance | Result |",
                      "| --- | ---: | ---: | ---: | --- | --- |"])
        for row in rows:
            if row["retriever"] != retriever:
                continue
            tolerance = ", ".join(f"{key}={number:g}" for key, number in row["threshold"].items())
            lines.append(f"| {row['metric']} | {row['baseline_value']:.6g} | {row['candidate_value']:.6g} | "
                         f"{row['absolute_delta']:+.6g} | {tolerance} | {row['status']} |")
        lines.append("")
    for heading, selected in (("Regressions", [row for row in rows if row["status"] != "PASS"]),
                              ("Improvements", [row for row in rows if row["improved"]])):
        lines.extend([f"## {heading}", ""])
        lines.extend(f"- {row['retriever']} {row['metric']}: {row['absolute_delta']:+.6g} ({row['status']})"
                     for row in selected)
        if not selected:
            lines.append("None.")
        lines.append("")
    analysis = report.get("query_analysis")
    if analysis and analysis["enabled"]:
        lines.extend(["## Query-Level Changes", ""])
        lines.append(f"Query gate: {report.get('query_gate_status', 'disabled')}")
        lines.append("")
        for name, data in analysis["retrievers"].items():
            lines.extend([f"### {name}", "",
                          f"- New failures: {sum(map(len, data['new_failures'].values()))} across K values",
                          f"- Recoveries: {sum(map(len, data['recoveries'].values()))} across K values",
                          f"- Rank improved: {len(data['rank_improvements'])}",
                          f"- Rank regressed: {len(data['rank_regressions'])}", ""])
        for heading, prefix in (("New Failures", "NEW_FAILURE_AT_"),
                                ("Recoveries", "RECOVERED_AT_"),
                                ("Rank Regressions", "RANK_REGRESSED"),
                                ("Rank Improvements", "RANK_IMPROVED"),
                                ("Persistent Failures", "STILL_FAILED_AT_K")):
            lines.extend([f"### {heading}", "",
                          "| Retriever | Query | K | Baseline Rank | Candidate Rank |",
                          "| --- | --- | ---: | ---: | ---: |"])
            for name, data in analysis["retrievers"].items():
                for row in data["queries"]:
                    for flag in row["flags"]:
                        if flag.startswith(prefix):
                            k = flag[len(prefix):] if prefix.endswith("_") else "-"
                            old = row["baseline_first_relevant_rank"]
                            new = row["candidate_first_relevant_rank"]
                            lines.append(f"| {name} | {row['query_id']} | {k} | "
                                         f"{old if old is not None else '-'} | "
                                         f"{new if new is not None else '-'} |")
            lines.append("")
    lines.extend(["## Compatibility Checks", ""])
    lines.extend(f"- {check}" for check in report["compatibility_checks"])
    return "\n".join(lines) + "\n"


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    json_path, markdown_path = output_dir / "regression.json", output_dir / "REGRESSION.md"
    atomic_write(json_path, json.dumps(report, indent=2, allow_nan=False) + "\n")
    atomic_write(markdown_path, render_markdown(report))
    return json_path, markdown_path
