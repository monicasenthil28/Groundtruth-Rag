"""Command line entry point."""

import json
import sys
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from groundtruth_rag import __version__
from groundtruth_rag.benchmark_setup import (
    SourceHashMismatchError, SourceUnavailableError, benchmark_status as inspect_benchmark,
    load_spec, setup_benchmark, setup_with_release_fallback, validate_golden,
    verify_benchmark_package,
)
from groundtruth_rag.datasets import DatasetValidationError, load_dataset, tag_counts
from groundtruth_rag.evaluation.runner import BenchmarkRun, evaluate_retriever
from groundtruth_rag.ingestion.corpus import load_corpus
from groundtruth_rag.ingestion.pipeline import ingest as ingest_documents
from groundtruth_rag.labeling.audit import (
    MIN_OSHA_BENCHMARK_QUERIES, audit_dataset as run_audit,
)
from groundtruth_rag.labeling.interactive import run_label_loop
from groundtruth_rag.labeling.manifest import CorpusManifest, file_sha256, prepare_osha as prepare_osha_corpus
from groundtruth_rag.labeling.models import atomic_write, load_questions
from groundtruth_rag.labeling.review import export_review as write_review, label_status as get_label_status
from groundtruth_rag.labeling.session import LabelSession, load_session
from groundtruth_rag.reporting.benchmark import summarize_runs
from groundtruth_rag.reporting.compare import compare_runs
from groundtruth_rag.reporting.regression import (
    apply_query_gate, compare as compare_regression, create_baseline, load_baseline, load_policy,
    load_summary, write_report,
)
from groundtruth_rag.reporting.query_regression import (
    compare_queries, create_query_baseline, load_query_baseline,
)
from groundtruth_rag.reporting.forensics import (
    analyze_forensics, load_forensics, load_run as load_forensics_run, write_forensics,
)
from groundtruth_rag.experiments import load_experiment_report, run_experiment
from groundtruth_rag.robustness import (
    analyze as analyze_robustness, load_config as load_robustness_config,
    load_run as load_robustness_run, run_latency_trials, write_report as write_robustness,
)
from groundtruth_rag.promotion import (
    activate as activate_promotion, apply as apply_promotion, approve as approve_promotion,
    baseline_version, benchmark_version, check as check_promotion,
    config_status as get_config_status, load_manifest as load_promotion_manifest,
    precheck as promotion_precheck, rollback as rollback_promotion,
    validate_version,
)
from groundtruth_rag.retrieval.bm25 import BM25Retriever
from groundtruth_rag.retrieval.dense import DenseRetriever
from groundtruth_rag.retrieval.factory import build_retriever, load_config
from groundtruth_rag.retrieval.dense import corpus_fingerprint
from groundtruth_rag.retrieval.static import StaticRetriever

app = typer.Typer(help="RAG retrieval evaluation and regression platform.")


@app.callback()
def main() -> None:
    """Groundtruth-Rag command line interface."""


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command("validate-dataset")
def validate_dataset(path: Path) -> None:
    """Validate a golden query JSONL file."""
    try:
        queries = load_dataset(path)
    except DatasetValidationError as exc:
        typer.echo(f"Invalid dataset: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Dataset: {path}")
    typer.echo(f"Queries: {len(queries)}")
    typer.echo(f"Tags: {json.dumps(tag_counts(queries), sort_keys=True)}")
    typer.echo("Validation: passed")


@app.command()
def ingest(
    input: Path = typer.Option(..., "--input", help="Document file or directory."),
    output: Path = typer.Option(..., "--output", help="Corpus JSONL path."),
    chunk_size: int = typer.Option(150, "--chunk-size"),
    chunk_overlap: int = typer.Option(30, "--chunk-overlap"),
) -> None:
    """Create a versioned chunk corpus from PDF, text, or Markdown."""
    try:
        summary = ingest_documents(input, output, chunk_size, chunk_overlap)
    except (OSError, ValueError) as exc:
        typer.echo(f"Ingestion failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    for ignored in summary.ignored:
        typer.echo(f"Skipped unsupported file: {ignored}", err=True)
    typer.echo(f"Documents: {summary.documents}; pages: {summary.pages}; chunks: {summary.chunks}")
    typer.echo(f"Chunk size: {chunk_size}; overlap: {chunk_overlap}")
    typer.echo(f"Corpus: {output}")


@app.command()
def evaluate(
    corpus: Path = typer.Option(..., "--corpus"),
    dataset: Path = typer.Option(..., "--dataset"),
    config: Path = typer.Option(..., "--config"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Evaluate a configured retriever against a fixed-chunk golden dataset."""
    try:
        if output.resolve() in {corpus.resolve(), dataset.resolve(), config.resolve()}:
            raise ValueError("output must not overwrite corpus, dataset, or config")
        chunks = load_corpus(corpus)
        queries = load_dataset(dataset)
        known_ids = {chunk.chunk_id for chunk in chunks}
        for query in queries:
            unknown = query.relevant_chunk_ids - known_ids
            if unknown:
                raise ValueError(f"{query.query_id}: relevance IDs absent from corpus: {', '.join(sorted(unknown))}")
        run_config = load_config(config)
        retriever = build_retriever(run_config)
        retriever.index(chunks)
        run = evaluate_retriever(
            retriever, queries, dataset=str(dataset.resolve()),
            configuration_name=run_config.name,
            configuration={**run_config.model_dump(), "corpus_fingerprint": corpus_fingerprint(chunks)},
        )
        output.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, DatasetValidationError, ValidationError) as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Run: {output}; queries: {run.query_count}")
    typer.echo("Metrics: " + ", ".join(f"{name}={value:.4f}" for name, value in run.aggregate_metrics.items()))
    typer.echo(f"Latency ms: mean={run.latency.mean_ms:.2f}, p50={run.latency.p50_ms:.2f}, p95={run.latency.p95_ms:.2f}")


@app.command("prepare-osha")
def prepare_osha(
    source: Path = typer.Option(Path("datasets/raw/osha3120.pdf"), "--source"),
    corpus: Path = typer.Option(Path("datasets/osha/corpus.jsonl"), "--corpus"),
    manifest: Path = typer.Option(Path("datasets/osha/corpus_manifest.json"), "--manifest"),
) -> None:
    """Ingest the supplied OSHA PDF once with fixed 150/30-word chunking."""
    try:
        result = prepare_osha_corpus(source, corpus, manifest)
    except (OSError, ValueError) as exc:
        typer.echo(f"OSHA corpus preparation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Corpus: {corpus}; manifest: {manifest}")
    typer.echo(f"Documents: {result.document_count}; pages: {result.page_count}; chunks: {result.chunk_count}")


@app.command()
def label(
    corpus: Path = typer.Option(..., "--corpus"),
    questions: Path = typer.Option(..., "--questions"),
    output: Path = typer.Option(..., "--output"),
    candidate_depth: int = typer.Option(10, "--candidate-depth", min=1),
    reconcile_output: bool = typer.Option(False, "--reconcile-output"),
) -> None:
    """Review and explicitly verify chunk relevance labels in the terminal."""
    try:
        if output.resolve() in {corpus.resolve(), questions.resolve()}:
            raise ValueError("output must not overwrite corpus or draft questions")
        chunks = load_corpus(corpus)
        drafts = load_questions(questions)
        fingerprint = corpus_fingerprint(chunks)
        manifest_path = corpus.parent / "corpus_manifest.json"
        if manifest_path.exists():
            manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            if manifest.corpus_fingerprint != fingerprint:
                raise ValueError("corpus differs from manifest")
        elif corpus.parent.name.lower() == "osha":
            raise ValueError(f"canonical OSHA manifest missing: {manifest_path}")
        if reconcile_output and not typer.confirm("Replace golden output from saved progress?", default=False):
            raise ValueError("output reconciliation was not confirmed")
        state, progress = load_session(questions, output, drafts, fingerprint,
                                       reconcile_output=reconcile_output)
        session = LabelSession(state, progress, output, {chunk.chunk_id for chunk in chunks})
        dense = DenseRetriever()
        bm25 = BM25Retriever()
        dense.index(chunks)
        bm25.index(chunks)
        run_label_loop(session, chunks, dense, bm25, candidate_depth)
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        typer.echo(f"Labeling failed: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("label-status")
def label_status(
    questions: Path = typer.Option(..., "--questions"),
    dataset: Path = typer.Option(..., "--dataset"),
) -> None:
    """Show human review progress."""
    try:
        status = get_label_status(questions, dataset)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Status failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Total: {status['total']}; verified: {status['verified']}; "
               f"unlabeled: {status['unlabeled']}; in progress: {status['in_progress']}; "
               f"skipped: {status['skipped']}; reviewed: {status['reviewed']}/{status['total']}; "
               f"benchmark queries: {status['benchmark_queries']}; "
               f"reviewed completion: {status['reviewed_completion_pct']}%; "
               f"verified answerable: {status['completion_pct']}%")


@app.command("audit-dataset")
def audit_dataset(
    corpus: Path = typer.Option(..., "--corpus"),
    questions: Path = typer.Option(..., "--questions"),
    dataset: Path = typer.Option(..., "--dataset"),
    manifest: Path | None = typer.Option(None, "--manifest"),
) -> None:
    """Audit human labels, provenance, and fixed-corpus compatibility."""
    try:
        report = run_audit(corpus, questions, dataset, manifest or corpus.parent / "corpus_manifest.json")
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Audit failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Total draft questions: {report.question_count}; verified: {report.verified_count}; "
               f"skipped: {report.skipped_count}; unlabeled: {report.unlabeled_count}; "
               f"in progress: {report.in_progress_count}; "
               f"answerable benchmark queries: {report.benchmark_query_count}; "
               f"minimum required: {report.minimum_required}; "
               f"reference answers: {report.reference_answer_count}")
    typer.echo(f"Skipped IDs: {', '.join(report.skipped_ids) or 'none'}")
    typer.echo(f"Tags: {json.dumps(report.tag_counts, sort_keys=True)}")
    typer.echo(f"Source/pages: {json.dumps(report.source_pages, sort_keys=True)}")
    for warning in report.warnings:
        typer.echo(f"Warning: {warning}")
    for error in report.errors:
        typer.echo(f"Error: {error}", err=True)
    typer.echo(f"Audit: {'passed' if report.passed else 'failed'}")
    if not report.passed:
        raise typer.Exit(1)


@app.command("export-review")
def export_review(
    corpus: Path = typer.Option(..., "--corpus"),
    dataset: Path = typer.Option(..., "--dataset"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Export verified labels and their source passages to Markdown."""
    try:
        if output.resolve() in {corpus.resolve(), dataset.resolve()}:
            raise ValueError("output must not overwrite corpus or dataset")
        count = write_review(corpus, dataset, output)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Review export failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Review: {output}; questions: {count}")


@app.command("benchmark-osha")
def benchmark_osha(
    corpus: Path = typer.Option(Path("datasets/osha/corpus.jsonl"), "--corpus"),
    questions: Path | None = typer.Option(None, "--questions"),
    dataset: Path = typer.Option(Path("benchmarks/osha/eval.jsonl"), "--dataset"),
    manifest: Path = typer.Option(Path("datasets/osha/corpus_manifest.json"), "--manifest"),
    output_dir: Path = typer.Option(Path("reports/benchmarks/osha"), "--output-dir"),
) -> None:
    """Run four retrievers only after the human golden dataset passes audit."""
    try:
        root = Path(__file__).resolve().parents[2]
        spec = load_spec(root / "benchmarks/osha/benchmark.yaml")
        if questions is not None:
            report = run_audit(corpus, questions, dataset, manifest)
            if not report.passed:
                raise ValueError("audit failed: " + "; ".join(report.errors))
            verified_count = report.verified_count
        else:
            status = inspect_benchmark(root, spec)
            if not status["ready"]:
                raise ValueError("benchmark preparation failed: " + "; ".join(status["errors"]))
            verified_count = validate_golden(root, spec, corpus)
        if verified_count < MIN_OSHA_BENCHMARK_QUERIES:
            raise ValueError("first benchmark requires at least 25 human-verified answerable questions")
        chunks = load_corpus(corpus)
        queries = load_dataset(dataset)
        provenance = CorpusManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
        if (provenance.corpus_name != "OSHA 3120 Lockout/Tagout"
                or not provenance.source_document.lower().endswith(".pdf")
                or (provenance.chunk_size, provenance.chunk_overlap) != (150, 30)):
            raise ValueError("benchmark requires the canonical OSHA PDF manifest and 150/30 chunking")
        fingerprint = corpus_fingerprint(chunks)
        runs: dict[str, BenchmarkRun] = {}
        for name in ("dense", "bm25", "hybrid", "hybrid_rerank"):
            config_path = root / spec.retriever_configs[name]
            run_config = load_config(config_path)
            retriever = build_retriever(run_config)
            retriever.index(chunks)
            runs[name] = evaluate_retriever(
                retriever, queries, dataset=str(dataset.resolve()),
                configuration_name=run_config.name,
                configuration={**run_config.model_dump(), "corpus_fingerprint": fingerprint,
                               "dataset_sha256": file_sha256(dataset),
                               "config_sha256": file_sha256(config_path),
                               "source_sha256": provenance.source_sha256,
                               "benchmark_id": spec.benchmark_id,
                               "benchmark_spec_version": spec.schema_version,
                               "package_version": __version__, "python_version": sys.version.split()[0]},
            )
        summary, markdown, analysis = summarize_runs(runs)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, run in runs.items():
            atomic_write(output_dir / f"{name}.json", run.model_dump_json(indent=2) + "\n")
        atomic_write(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
        atomic_write(output_dir / "SUMMARY.md", markdown)
        atomic_write(output_dir / "error_analysis.json", json.dumps(analysis, indent=2) + "\n")
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        typer.echo(f"Benchmark blocked or failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Benchmark results: {output_dir}; verified questions: {verified_count}")


@app.command("benchmark-setup")
def benchmark_setup_command(
    benchmark_id: str,
    source: Path | None = typer.Option(None, "--source"),
    release_fallback: bool = typer.Option(False, "--release-fallback"),
) -> None:
    """Acquire the pinned public source and rebuild the canonical corpus."""
    if benchmark_id != "osha":
        raise typer.BadParameter("only osha is supported")
    root = Path(__file__).resolve().parents[2]
    try:
        if source is not None and release_fallback:
            raise ValueError("--source and --release-fallback cannot be combined")
        spec = load_spec(root / "benchmarks/osha/benchmark.yaml")
        if release_fallback:
            built, source_status = setup_with_release_fallback(root, spec)
        else:
            source_exists = (root / "datasets/raw/osha3120.pdf").exists()
            built = setup_benchmark(root, spec, source)
            if source is not None:
                source_status = "VERIFIED_SOURCE_OK"
            elif source_exists:
                source_status = "VERIFIED_LOCAL_SOURCE_OK"
            else:
                source_status = "OFFICIAL_SOURCE_OK"
    except SourceHashMismatchError as exc:
        typer.echo(f"SOURCE_HASH_MISMATCH: {exc}", err=True)
        raise typer.Exit(1) from exc
    except SourceUnavailableError as exc:
        typer.echo(f"SOURCE_UNAVAILABLE: {exc}", err=True)
        raise typer.Exit(1) from exc
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        typer.echo(f"BENCHMARK REPRODUCTION FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(source_status)
    typer.echo("OSHA benchmark ready; corpus " + ("rebuilt" if built else "reused"))


@app.command("benchmark-status")
def benchmark_status_command(benchmark_id: str) -> None:
    """Show source, corpus, golden dataset, and baseline readiness."""
    if benchmark_id != "osha":
        raise typer.BadParameter("only osha is supported")
    root = Path(__file__).resolve().parents[2]
    try:
        spec = load_spec(root / "benchmarks/osha/benchmark.yaml")
        status = inspect_benchmark(root, spec)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Benchmark status failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo("OSHA Benchmark")
    typer.echo(f"Source: {status['source']} (SHA-256: {spec.source.sha256})")
    typer.echo(f"Corpus: {status['corpus']} (documents: {spec.source.document_count}; "
               f"pages: {spec.source.page_count}; chunks: {spec.corpus.chunk_count})")
    typer.echo(f"Golden dataset: {status['golden']} (reviewed: {spec.golden.total_questions}; "
               f"verified: {spec.golden.verified_queries}; skipped: {len(spec.golden.skipped_ids)}; "
               f"skipped IDs: {', '.join(spec.golden.skipped_ids)})")
    typer.echo(f"Baseline: {status['baseline']} ({spec.baseline})")
    for error in status["errors"]:
        typer.echo(f"Mismatch: {error}", err=True)
    typer.echo(f"Ready to benchmark: {'YES' if status['ready'] else 'NO'}")


@app.command("benchmark-verify")
def benchmark_verify_command(benchmark_id: str) -> None:
    """Verify versioned benchmark inputs without a source PDF or corpus."""
    if benchmark_id != "osha":
        raise typer.BadParameter("only osha is supported")
    root = Path(__file__).resolve().parents[2]
    try:
        spec = load_spec(root / "benchmarks/osha/benchmark.yaml")
        count = verify_benchmark_package(root, spec)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Benchmark package integrity: FAIL: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Benchmark package integrity: PASS; benchmark: {spec.benchmark_id}; queries: {count}")


@app.command("baseline-create")
def baseline_create(
    summary: Path = typer.Option(..., "--summary"),
    output: Path = typer.Option(..., "--output"),
    name: str = typer.Option(..., "--name"),
    domain: str | None = typer.Option(None, "--domain"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Capture validated aggregate results without rerunning retrieval."""
    try:
        if output.resolve() == summary.resolve():
            raise ValueError("baseline output must differ from benchmark summary")
        baseline = create_baseline(summary, output, name, domain=domain, force=force)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Baseline creation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Baseline: {output}; queries: {baseline.query_count}")


@app.command()
def regress(
    baseline: Path = typer.Option(..., "--baseline"),
    candidate: Path = typer.Option(..., "--candidate"),
    policy: Path = typer.Option(..., "--policy"),
    output: Path = typer.Option(..., "--output"),
    fail_on_warn: bool = typer.Option(False, "--fail-on-warn"),
    query_baseline: Path | None = typer.Option(None, "--query-baseline"),
) -> None:
    """Compare a candidate benchmark with a versioned baseline."""
    try:
        base = load_baseline(baseline)
        current = load_summary(candidate)
        rules = load_policy(policy)
        report = compare_regression(base, current, rules)
        if query_baseline is not None:
            analysis = compare_queries(
                load_query_baseline(query_baseline), base, candidate, current,
                rules.retrievers, critical_k=rules.query_regression.critical_k,
                warn_rank_drop=rules.query_regression.warn_rank_drop,
            )
            report["query_analysis"] = analysis
            apply_query_gate(report, analysis, rules.query_regression)
        elif rules.query_regression.enabled:
            raise ValueError("query regression policy requires --query-baseline")
        report["candidate_summary"] = str(candidate)
        _, markdown_path = write_report(report, output)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Regression comparison failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"{base.domain} Retrieval Regression; baseline: {base.name}; "
               f"candidate queries: {current.query_count}")
    for name, status in report["retriever_status"].items():
        typer.echo(f"{name}: {status}")
    typer.echo(f"Overall: {report['overall_status']}; report: {markdown_path}")
    if report["overall_status"] == "FAIL" or (fail_on_warn and report["overall_status"] == "WARN"):
        raise typer.Exit(1)


@app.command("query-baseline-create")
def query_baseline_create(
    summary: Path = typer.Option(..., "--summary"),
    baseline: Path = typer.Option(..., "--baseline"),
    output: Path = typer.Option(..., "--output"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Extract text-free query ranks from existing benchmark runs."""
    try:
        if output.resolve() in {summary.resolve(), baseline.resolve()}:
            raise ValueError("query baseline output must differ from inputs")
        result = create_query_baseline(summary, load_baseline(baseline), output, force=force)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Query baseline creation failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Query baseline: {output}; queries: {result.query_count}")


@app.command("analyze-failures")
def analyze_failures(
    dense: Path = typer.Option(..., "--dense"),
    bm25: Path = typer.Option(..., "--bm25"),
    hybrid: Path = typer.Option(..., "--hybrid"),
    rerank: Path = typer.Option(..., "--rerank"),
    output: Path = typer.Option(..., "--output"),
    target_k: int = typer.Option(3, "--target-k"),
) -> None:
    """Trace relevant chunks through saved retrieval stage rankings."""
    try:
        report = analyze_forensics(
            load_forensics_run(dense), load_forensics_run(bm25),
            load_forensics_run(hybrid), load_forensics_run(rerank), target_k=target_k,
        )
        _, markdown = write_forensics(report, output)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Failure analysis failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Forensics: {markdown}; queries: {report.query_count}; target K: {target_k}")


@app.command("failure-inspect")
def failure_inspect(path: Path, query: str = typer.Option(..., "--query")) -> None:
    """Inspect one query in a saved text-free forensic report."""
    try:
        report = load_forensics(path)
        record = next((item for item in report.queries if item.query_id == query), None)
        if record is None:
            raise ValueError(f"query not found: {query}")
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Failure inspection failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Query: {record.query_id}\nRelevant chunks: {record.relevant_chunk_count}"
               f"\nTarget K: {report.target_k}")
    for stage in ("dense", "bm25", "hybrid", "rerank"):
        rank = record.first_relevant_rank[stage]
        typer.echo(f"{stage} first relevant: {rank if rank is not None else 'not established within saved depth'}")
    typer.echo(f"Flags: {', '.join(record.flags) or 'none'}")


@app.command("experiment-run")
def experiment_run(
    specification: Path,
    output: Path = typer.Option(..., "--output"),
    candidate: str | None = typer.Option(None, "--candidate"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Validate or run fixed-benchmark retrieval candidates."""
    try:
        root = Path(__file__).resolve().parents[2]
        result = run_experiment(root, specification, output, candidate_id=candidate,
                                dry_run=dry_run)
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        typer.echo(f"Experiment failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    if dry_run:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"Experiment report: {output / 'EXPERIMENT.md'}")
        for row in result["candidates"]:
            typer.echo(f"{row['candidate_id']}: {row['regression_status']}; "
                       f"{row['pareto']}; {row['promotion_status']}")


@app.command("experiment-inspect")
def experiment_inspect(path: Path, candidate: str = typer.Option(..., "--candidate")) -> None:
    """Inspect one advisory candidate decision without rerunning retrieval."""
    try:
        report = load_experiment_report(path)
        row = next((item for item in report["candidates"]
                    if item["candidate_id"] == candidate), None)
        if row is None:
            raise ValueError(f"candidate not found: {candidate}")
    except (OSError, ValueError) as exc:
        typer.echo(f"Experiment inspection failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Candidate: {candidate}\nStatus: {row['promotion_status']}\n"
               f"Regression: {row['regression_status']}\nPareto: {row['pareto']}")
    for metric in ("recall@3", "mrr", "ndcg@3"):
        old, new = row["baseline_metrics"][metric], row["metrics"][metric]
        typer.echo(f"{metric}: baseline {old:.4f}; candidate {new:.4f}; delta {new - old:+.4f}")
    typer.echo(f"New critical query failures: {row['new_critical_failures']}")
    typer.echo(f"Forensics: {row['forensics_status']}")


@app.command("robustness-analyze")
def robustness_analyze(
    baseline_run: Path = typer.Option(..., "--baseline-run"),
    candidate_run: Path = typer.Option(..., "--candidate-run"),
    config: Path = typer.Option(..., "--config"),
    output: Path = typer.Option(..., "--output"),
    latency: Path | None = typer.Option(None, "--latency"),
) -> None:
    """Analyze saved paired runs; retrieval is never executed."""
    try:
        report = analyze_robustness(load_robustness_run(baseline_run),
                                    load_robustness_run(candidate_run),
                                    load_robustness_config(config))
        if latency is not None:
            measurement = json.loads(latency.read_text(encoding="utf-8"))
            if (measurement.get("schema_version") != report["schema_version"]
                    or measurement.get("candidate") != report["candidate"]
                    or measurement.get("corpus_fingerprint") != report["corpus_fingerprint"]
                    or measurement.get("dataset_sha256") != report["dataset_sha256"]):
                raise ValueError("latency trial identity differs from quality report")
            report["latency"] = measurement
        write_robustness(report, output)
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        typer.echo(f"Robustness analysis failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Robustness report: {output / 'ROBUSTNESS.md'}")


@app.command("latency-trials")
def latency_trials(
    specification: Path = typer.Option(..., "--experiment"),
    candidate: str = typer.Option(..., "--candidate"),
    config: Path = typer.Option(..., "--config"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Measure repeated full-query trials using a validated experiment candidate."""
    try:
        root = Path(__file__).resolve().parents[2]
        if not output.resolve().is_relative_to((root / "reports/robustness").resolve()):
            raise ValueError("latency output must be under reports/robustness/")
        result = run_latency_trials(root, specification, candidate, load_robustness_config(config))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        typer.echo(f"Latency trials failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Latency trials: {output}")


@app.command("promotion-check")
def promotion_check(manifest: Path) -> None:
    """Verify saved evidence and mark a draft ready for human review."""
    try:
        state = check_promotion(Path(__file__).resolve().parents[2], manifest)
    except (OSError, ValueError, RuntimeError, ValidationError, KeyError) as exc:
        typer.echo(f"Promotion blocked: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Promotion state: {state.state}")


@app.command("promotion-approve")
def promotion_approve(manifest: Path, reviewer: str | None = typer.Option(None, "--reviewer")) -> None:
    """Record an explicit, interactive human approval; never stages or activates."""
    try:
        root = Path(__file__).resolve().parents[2]
        current = load_promotion_manifest(manifest)
        evidence = promotion_precheck(root, current)
        if current.state != "READY_FOR_HUMAN_APPROVAL":
            raise ValueError("promotion is not ready for human approval")
        typer.echo(f"Candidate: {current.candidate.candidate_id}")
        typer.echo(f"Regression: {evidence['regression']['overall_status']}")
        for metric, assessment in current.evidence.robustness.items():
            typer.echo(f"Robustness {metric}: {assessment}")
        latency = evidence["robustness"].get("latency")
        if latency:
            typer.echo(f"Latency: environment-sensitive; median p95 "
                       f"{latency['summary']['median_trial_p95_ms']:.2f} ms")
        if not typer.confirm("Proceed with promotion approval?", default=False):
            raise ValueError("human approval declined")
        state = approve_promotion(root, manifest, reviewer)
    except (OSError, ValueError, RuntimeError, ValidationError, KeyError) as exc:
        typer.echo(f"Promotion approval blocked: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Promotion state: {state.state}")


@app.command("promotion-apply")
def promotion_apply(manifest: Path) -> None:
    """Stage the exact approved candidate config without activating it."""
    try:
        state = apply_promotion(Path(__file__).resolve().parents[2], manifest)
    except (OSError, ValueError, RuntimeError, ValidationError, KeyError) as exc:
        typer.echo(f"Promotion staging blocked: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"PROMOTION_STAGED: {state.target.config_path}; active version remains v1")


@app.command("benchmark-version")
def benchmark_version_command(benchmark: str, version: str) -> None:
    """Run an isolated candidate validation benchmark; requires prior approval."""
    root = Path(__file__).resolve().parents[2]
    manifest = root / "promotions" / benchmark / f"{version}.yaml"
    try:
        state = benchmark_version(root, manifest)
    except (OSError, ValueError, RuntimeError, ValidationError, KeyError) as exc:
        typer.echo(f"Version benchmark failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Candidate validation benchmark: {state.target.benchmark_output}")


@app.command("promotion-validate")
def promotion_validate(manifest: Path) -> None:
    """Compare completed v2 runs with v1 and record validation evidence."""
    try:
        result = validate_version(Path(__file__).resolve().parents[2], manifest)
    except (OSError, ValueError, RuntimeError, ValidationError, KeyError) as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Validation: {result['status']}")


@app.command("promotion-baseline")
def promotion_baseline(manifest: Path) -> None:
    """Create v2 baseline pair only from validated v2 run files."""
    try:
        state = baseline_version(Path(__file__).resolve().parents[2], manifest)
    except (OSError, ValueError, RuntimeError, ValidationError, KeyError) as exc:
        typer.echo(f"Baselining blocked: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Promotion state: {state.state}; active version remains v1")


@app.command("promotion-activate")
def promotion_activate(manifest: Path) -> None:
    """Explicitly activate a validated, baselined version."""
    try:
        root = Path(__file__).resolve().parents[2]
        state = load_promotion_manifest(manifest)
        if state.state not in {"BASELINED", "ROLLED_BACK"}:
            raise ValueError("v2 must be validated and baselined first")
        evidence = promotion_precheck(root, state)
        typer.echo("QUALITY: " + ", ".join(
            f"{metric} {assessment}" for metric, assessment in state.evidence.robustness.items()))
        latency = evidence["robustness"].get("latency")
        typer.echo("LATENCY: " + (f"median p95 {latency['summary']['median_trial_p95_ms']:.2f} ms "
                                  "in one environment; not a hard gate."
                                  if latency else "repeated trials unavailable; review required."))
        if not typer.confirm("Activate this retrieval version?", default=False):
            raise ValueError("activation declined")
        state = activate_promotion(root, manifest)
    except (OSError, ValueError, RuntimeError, ValidationError, KeyError) as exc:
        typer.echo(f"Activation blocked: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Active version: {state.version}")


@app.command("promotion-rollback")
def promotion_rollback(benchmark: str, to: str = typer.Option(..., "--to")) -> None:
    """Switch the active registry back to an existing version without deleting artifacts."""
    try:
        root = Path(__file__).resolve().parents[2]
        current = get_config_status(root, benchmark)
        state = rollback_promotion(root, root / "promotions" / benchmark /
                                   f"{current['active']}.yaml", to)
    except (OSError, ValueError, RuntimeError, ValidationError, KeyError) as exc:
        typer.echo(f"Rollback blocked: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Active version: {to}; preserved {state.version} artifacts")


@app.command("config-status")
def config_status_command(benchmark: str) -> None:
    """Resolve the active retrieval config and baselines without loading models."""
    try:
        result = get_config_status(Path(__file__).resolve().parents[2], benchmark)
    except (OSError, ValueError, RuntimeError, ValidationError, KeyError) as exc:
        typer.echo(f"Config status failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Benchmark: {result['benchmark']}\nActive retrieval version: {result['active']}\n"
               f"Config: {result['config']}\nBaseline: {result['baseline']}\n"
               f"Query baseline: {result['query_baseline']}\n"
               f"Previous: {result['previous'] or 'none'}\n"
               f"Rollback available: {'YES' if result['rollback_available'] else 'NO'}")


@app.command("evaluate-demo")
def evaluate_demo(output: Path) -> None:
    """Evaluate fixed synthetic rankings; this is not a benchmark."""
    demo = Path(__file__).resolve().parents[2] / "datasets" / "demo"
    try:
        queries = load_dataset(demo / "queries.jsonl")
        rankings = json.loads((demo / "rankings.json").read_text(encoding="utf-8"))
        run = evaluate_retriever(
            StaticRetriever(rankings), queries, dataset="demo/queries.jsonl",
            configuration_name="static-demo", configuration={"source": "fixed synthetic rankings"},
        )
        output.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Demo run: {output}")
    typer.echo(f"Queries: {run.query_count}")


@app.command()
def compare(run_a: Path, run_b: Path) -> None:
    """Compare aggregate metrics from two compatible benchmark run files."""
    try:
        baseline = BenchmarkRun.model_validate_json(run_a.read_text(encoding="utf-8"))
        candidate = BenchmarkRun.model_validate_json(run_b.read_text(encoding="utf-8"))
        rows = compare_runs(baseline, candidate)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Cannot compare runs: {exc}", err=True)
        raise typer.Exit(1) from exc
    table = Table(title=f"{baseline.configuration_name} vs {candidate.configuration_name}")
    for column in ("Metric", "Baseline", "Candidate", "Absolute Delta", "Relative Delta %", "Prefer"):
        table.add_column(column)
    for row in rows:
        relative = f"{row.relative_delta_pct:+.2f}%" if row.relative_delta_pct is not None else "N/A (zero baseline)"
        table.add_row(row.name, f"{row.baseline:.4f}", f"{row.candidate:.4f}",
                      f"{row.absolute_delta:+.4f}", relative, row.direction)
    Console(width=120).print(table)


if __name__ == "__main__":
    app()
