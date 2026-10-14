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
from groundtruth_rag.reporting.compare import compare_runs
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
