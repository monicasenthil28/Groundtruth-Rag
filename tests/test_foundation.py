import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from groundtruth_rag.cli import app
from groundtruth_rag.retrieval.base import Retriever
from groundtruth_rag.schemas import DocumentChunk, EvalQuery, RetrievalHit, RetrievalRunResult


def test_schema_validation() -> None:
    chunk = DocumentChunk(chunk_id="c1", document_id="d1", source="fixture", text="Example")
    query = EvalQuery(query_id="q1", query="What?", relevant_chunk_ids={chunk.chunk_id})
    assert query.relevant_chunk_ids == {"c1"}
    with pytest.raises(ValidationError):
        DocumentChunk(chunk_id="", document_id="d1", source="fixture", text="Example")
    with pytest.raises(ValidationError):
        RetrievalRunResult(query_id="q1", hits=[RetrievalHit(chunk_id="c1", score=1, rank=2, retriever="test")], latency_ms=1)


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_retriever_requires_both_methods() -> None:
    with pytest.raises(TypeError):
        Retriever()

    class IncompleteRetriever(Retriever):
        def index(self, chunks):
            pass

    with pytest.raises(TypeError):
        IncompleteRetriever()
