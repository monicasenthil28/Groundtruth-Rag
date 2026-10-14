"""Command line entry point."""

import typer

from groundtruth_rag import __version__

app = typer.Typer(help="RAG retrieval evaluation and regression platform.")


@app.callback()
def main() -> None:
    """Groundtruth-Rag command line interface."""


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
