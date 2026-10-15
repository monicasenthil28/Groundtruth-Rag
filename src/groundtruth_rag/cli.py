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
    compare as compare_regression, create_baseline, load_baseline, load_policy,
    load_summary, write_report,
)
from groundtruth_rag.reporting.query_regression import (
    compare_queries, create_query_baseline, load_query_baseline,
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
            query_rules = rules.query_regression
            if query_rules.enabled:
                new_count = sum(len(data["new_failures"][str(k)])
                                for data in analysis["retrievers"].values()
                                for k in query_rules.critical_k)
                severity = "PASS"
                if ((query_rules.fail_on_new_failure_at_k and new_count > 0)
                        or (query_rules.max_new_failures_fail is not None
                            and new_count > query_rules.max_new_failures_fail)):
                    severity = "FAIL"
                elif ((query_rules.max_new_failures_warn is not None
                       and new_count > query_rules.max_new_failures_warn)
                      or any(row["severity"] == "WARN" for data in analysis["retrievers"].values()
                             for row in data["queries"])):
                    severity = "WARN"
                report["query_gate_status"] = severity
                if severity == "FAIL" or (severity == "WARN" and report["overall_status"] == "PASS"):
                    report["overall_status"] = severity
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
