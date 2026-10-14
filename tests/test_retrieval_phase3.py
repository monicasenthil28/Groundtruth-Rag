import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from groundtruth_rag.cli import app
from groundtruth_rag.ingestion.chunking import chunk_pages
from groundtruth_rag.ingestion.corpus import load_corpus, write_corpus
from groundtruth_rag.ingestion.loaders import LoadedPage, discover_documents, load_document
from groundtruth_rag.ingestion.pipeline import ingest
from groundtruth_rag.retrieval.bm25 import BM25Retriever, tokenize
from groundtruth_rag.retrieval.dense import DenseRetriever
from groundtruth_rag.retrieval.factory import RunConfig, build_retriever, load_config
from groundtruth_rag.retrieval.hybrid import HybridRetriever, rrf_fuse
from groundtruth_rag.retrieval.rerank import RerankedHybridRetriever
from groundtruth_rag.retrieval.static import StaticRetriever
from groundtruth_rag.schemas import DocumentChunk, RetrievalHit


def chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(chunk_id=chunk_id, document_id="d", source="synthetic", text=text)


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class FakeReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def predict(self, pairs):
        return np.asarray([self.scores[text] for _, text in pairs])


def test_chunk_ids_overlap_and_provenance():
    page = LoadedPage("doc", "source.pdf", 7, "one two three four five six")
    first = chunk_pages([page], 4, 2)
    assert [item.text for item in first] == ["one two three four", "three four five six"]
    assert [item.chunk_id for item in first] == [item.chunk_id for item in chunk_pages([page], 4, 2)]
    assert first[1].metadata["word_start"] == 2
    assert first[1].metadata["word_end"] == 6
    assert all(item.page == 7 and item.source == "source.pdf" for item in first)
    changed = chunk_pages([LoadedPage("doc", "source.pdf", 7, "one two three changed five six")], 4, 2)
    assert changed[0].chunk_id != first[0].chunk_id
    assert chunk_pages([LoadedPage("doc", "source.pdf", 8, page.text)], 4, 2)[0].chunk_id != first[0].chunk_id
    for size, overlap in ((0, 0), (4, -1), (4, 4), (4, 5)):
        with pytest.raises(ValueError):
            chunk_pages([page], size, overlap)


def test_document_loading_and_corpus_jsonl(tmp_path: Path, monkeypatch):
    (tmp_path / "b.md").write_text("Markdown text", encoding="utf-8")
    (tmp_path / "a.txt").write_text("Text file", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"unused")
    files, ignored = discover_documents(tmp_path)
    assert [source for _, source in files] == ["a.txt", "b.md"]
    assert [item.name for item in ignored] == ["ignored.bin"]
    assert load_document(files[0][0], files[0][1])[0].page is None

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Reader:
        is_encrypted = False
        pages = [Page("Page one"), Page("Page two")]

    monkeypatch.setattr("groundtruth_rag.ingestion.loaders.PdfReader", lambda path: Reader())
    pdf_pages = load_document(tmp_path / "manual.pdf", "manual.pdf")
    assert [(page.page, page.source, page.text) for page in pdf_pages] == [
        (1, "manual.pdf", "Page one"), (2, "manual.pdf", "Page two")]
    corpus = tmp_path / "corpus.jsonl"
    chunks = chunk_pages(pdf_pages, 2, 0)
    write_corpus(corpus, chunks)
    assert load_corpus(corpus) == chunks
    assert json.loads(corpus.read_text(encoding="utf-8").splitlines()[0])["schema_version"] == "1.0"
    corpus.write_text('{"schema_version":"2.0"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_corpus(corpus)
    with pytest.raises(ValueError, match="overwrite an input"):
        ingest(tmp_path / "a.txt", tmp_path / "a.txt", 3, 0)


def test_dense_ranking_tie_break_and_cache(tmp_path: Path):
    chunks = [chunk("B", "beta"), chunk("A", "alpha"), chunk("C", "gamma")]
    vectors = {"beta": [1, 0], "alpha": [1, 0], "gamma": [0, 1], "q": [3, 0]}
    encoder = FakeEncoder(vectors)
    dense = DenseRetriever("fake-v1", 2, tmp_path, encoder)
    dense.index(chunks)
    assert [hit.chunk_id for hit in dense.retrieve("q", 2)] == ["A", "B"]
    assert len(dense.retrieve("q", 10)) == 3
    cached_encoder = FakeEncoder(vectors)
    cached = DenseRetriever("fake-v1", 2, tmp_path, cached_encoder)
    cached.index(chunks)
    assert cached_encoder.calls == []
    assert [hit.chunk_id for hit in cached.retrieve("q", 1)] == ["A"]
    changed = DenseRetriever("fake-v1", 2, tmp_path, FakeEncoder({**vectors, "changed": [0, 1]}))
    changed.index([chunk("B", "changed"), *chunks[1:]])
    assert len(list(tmp_path.glob("*.npz"))) == 2
    DenseRetriever("fake-v2", 2, tmp_path, FakeEncoder(vectors)).index(chunks)
    assert len(list(tmp_path.glob("*.npz"))) == 3
    with pytest.raises(ValueError, match="empty corpus"):
        dense.index([])
    with pytest.raises(ValueError, match="indexed"):
        dense.retrieve("q", 1)


def test_bm25_keyword_retrieval():
    chunks = [chunk("A", "red apple fruit"), chunk("B", "blue bicycle wheels"),
              chunk("C", "green forest trees"), chunk("D", "yellow river water")]
    bm25 = BM25Retriever()
    bm25.index(chunks)
    hits = bm25.retrieve("BICYCLE", 2)
    assert hits[0].chunk_id == "B" and hits[0].retriever == "bm25"
    assert [hit.rank for hit in hits] == [1, 2]
    assert tokenize("Blue-bike, BLUE") == ["blue", "bike", "blue"]
    with pytest.raises(ValueError, match="no word tokens"):
        bm25.index([chunk("X", "!!!")])


def test_rrf_exact_scores_and_order():
    def hits(ids, label):
        return [RetrievalHit(chunk_id=item, score=1, rank=rank, retriever=label)
                for rank, item in enumerate(ids, 1)]

    fused = rrf_fuse(hits("ABC", "dense"), hits("BCA", "bm25"), 60)
    assert [hit.chunk_id for hit in fused] == ["B", "A", "C"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].metadata["dense_rank"] == 2
    assert fused[0].metadata["bm25_rank"] == 1


def test_hybrid_and_reranker_ordering():
    chunks = [chunk("A", "alpha"), chunk("B", "beta")]
    hybrid = HybridRetriever(StaticRetriever({"q": ["A", "B"]}),
                             StaticRetriever({"q": ["B", "A"]}), 60, 2, 2)
    reranked = RerankedHybridRetriever(hybrid, 2, reranker=FakeReranker({"alpha": .2, "beta": .9}))
    reranked.index(chunks)
    results = reranked.retrieve("q", 2)
    assert [hit.chunk_id for hit in results] == ["B", "A"]
    assert results[0].score == .9
    assert results[0].metadata["crossencoder_score"] == .9
    assert "rrf_score" in results[0].metadata


def test_configs_and_factory(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    for name in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        config = load_config(root / "configs" / f"{name}.yaml")
        retriever = build_retriever(config, encoder=FakeEncoder({}),
                                    reranker=FakeReranker({}), cache_dir=tmp_path)
        assert retriever is not None
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "bad", "retriever": {"type": "unknown"}})
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "bad", "retriever": {"type": "hybrid", "dense_candidates": 0}})
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "bad", "retriever": {"type": "hybrid_rerank", "candidate_pool": 0}})


def test_cli_ingest_and_bm25_evaluate(tmp_path: Path):
    source = tmp_path / "notes.txt"
    source.write_text("red apple fruit. blue bicycle wheels. green tree forest.", encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    runner = CliRunner()
    ingested = runner.invoke(app, ["ingest", "--input", str(source), "--output", str(corpus),
                                    "--chunk-size", "3", "--chunk-overlap", "0"])
    assert ingested.exit_code == 0, ingested.output
    chunks = load_corpus(corpus)
    assert len(chunks) == 3
    dataset = tmp_path / "queries.jsonl"
    dataset.write_text(json.dumps({"query_id": "q", "query": "bicycle",
                                   "relevant_chunk_ids": [chunks[1].chunk_id]}) + "\n", encoding="utf-8")
    output = tmp_path / "run.json"
    config = Path(__file__).resolve().parents[1] / "configs" / "bm25.yaml"
    evaluated = runner.invoke(app, ["evaluate", "--corpus", str(corpus), "--dataset", str(dataset),
                                    "--config", str(config), "--output", str(output)])
    assert evaluated.exit_code == 0, evaluated.output
    assert "Metrics:" in evaluated.output
    assert json.loads(output.read_text(encoding="utf-8"))["query_count"] == 1
    dataset.write_text(json.dumps({"query_id": "q", "query": "bicycle",
                                   "relevant_chunk_ids": ["missing"]}) + "\n", encoding="utf-8")
    bad = runner.invoke(app, ["evaluate", "--corpus", str(corpus), "--dataset", str(dataset),
                             "--config", str(config), "--output", str(output)])
    assert bad.exit_code != 0
    assert "absent from corpus" in bad.output
    protected = runner.invoke(app, ["evaluate", "--corpus", str(corpus), "--dataset", str(dataset),
                                    "--config", str(config), "--output", str(dataset)])
    assert protected.exit_code != 0
    assert "must not overwrite" in protected.output
