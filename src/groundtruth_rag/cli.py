"""Command line entry point."""

import json
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from groundtruth_rag import __version__
from groundtruth_rag.datasets import DatasetValidationError, load_dataset, tag_counts
from groundtruth_rag.evaluation.runner import BenchmarkRun, evaluate_retriever
from groundtruth_rag.ingestion.corpus import load_corpus
from groundtruth_rag.ingestion.pipeline import ingest as ingest_documents
from groundtruth_rag.reporting.compare import compare_runs
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
