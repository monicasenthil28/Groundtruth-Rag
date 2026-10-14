import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner
from pydantic import ValidationError

from groundtruth_rag.cli import app
from groundtruth_rag.datasets import load_dataset
from groundtruth_rag.evaluation.runner import evaluate_retriever
from groundtruth_rag.ingestion.corpus import load_corpus, write_corpus
from groundtruth_rag.labeling.audit import audit_dataset
from groundtruth_rag.labeling.candidates import discover_candidates, search_corpus
from groundtruth_rag.labeling.manifest import CorpusManifest, file_sha256, prepare_osha
from groundtruth_rag.labeling.models import LabelState, load_questions, progress_path
from groundtruth_rag.labeling.review import export_review, label_status
from groundtruth_rag.labeling.session import LabelSession, load_session
from groundtruth_rag.reporting.benchmark import summarize_runs
from groundtruth_rag.retrieval.dense import corpus_fingerprint
from groundtruth_rag.retrieval.static import StaticRetriever
from groundtruth_rag.schemas import DocumentChunk, EvalQuery


def fixture_files(tmp_path: Path):
    chunks = [DocumentChunk(chunk_id="a", document_id="d", source="public.txt", page=1,
                            text="red apple"),
              DocumentChunk(chunk_id="b", document_id="d", source="public.txt", page=2,
                            text="blue bicycle")]
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(corpus, chunks)
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"query_id": "q1", "query": "What is red?", "tags": ["definition"],
                    "status": "unlabeled"}) + "\n", encoding="utf-8")
    dataset = tmp_path / "eval.jsonl"
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(CorpusManifest(
        corpus_name="synthetic", source_document="public.txt", source_relative_path="public.txt",
        source_sha256="0" * 64,
        ingested_at=datetime.now(timezone.utc), chunk_size=2, chunk_overlap=0,
        document_count=1, page_count=2, chunk_count=2,
        corpus_fingerprint=corpus_fingerprint(chunks), ingestion_version="word-v1",
        corpus_schema_version="1.0",
    ).model_dump_json(), encoding="utf-8")
    return chunks, corpus, questions, dataset, manifest


def test_draft_parse_and_no_relevance_ids(tmp_path: Path):
    _, _, questions, _, _ = fixture_files(tmp_path)
    assert load_questions(questions)[0].status == "unlabeled"
    questions.write_text('{"query_id":"q1","query":"x","relevant_chunk_ids":["a"]}\n',
                         encoding="utf-8")
    with pytest.raises(ValueError, match="extra_forbidden"):
        load_questions(questions)
    record = {"query_id": "q1", "query": "x", "status": "unlabeled"}
    with pytest.raises(ValidationError, match="duplicate query IDs"):
        LabelState.model_validate({"corpus_fingerprint": "fingerprint", "questions": [record, record]})


def test_confirmation_resume_multi_select_and_append(tmp_path: Path):
    chunks, _, questions, dataset, _ = fixture_files(tmp_path)
    drafts = load_questions(questions)
    state, progress = load_session(questions, dataset, drafts, corpus_fingerprint(chunks))
    session = LabelSession(state, progress, dataset, {"a", "b"})
    with pytest.raises(ValueError, match="absent from corpus"):
        session.select("q1", ["missing"])
    session.select("q1", ["a", "b"])
    assert progress.exists() and not dataset.exists()
    with pytest.raises(ValueError, match="confirmation"):
        session.verify("q1", confirmed=False)
    session.edit("q1", answer="Both passages need review", notes="Checked source")
    session.verify("q1", confirmed=True)
    assert load_dataset(dataset)[0].relevant_chunk_ids == {"a", "b"}
    with pytest.raises(ValueError, match="edit confirmation"):
        session.select("q1", ["a"])
    resumed, _ = load_session(questions, dataset, drafts, corpus_fingerprint(chunks))
    assert resumed.questions[0].status == "verified"
    dataset.write_text("{broken json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="use --reconcile-output"):
        load_session(questions, dataset, drafts, corpus_fingerprint(chunks))
    load_session(questions, dataset, drafts, corpus_fingerprint(chunks), reconcile_output=True)
    assert load_dataset(dataset)[0].relevant_chunk_ids == {"a", "b"}
    questions.write_text(questions.read_text(encoding="utf-8") + json.dumps({
        "query_id": "q2", "query": "What is blue?", "tags": [], "status": "unlabeled"}) + "\n",
        encoding="utf-8")
    expanded, _ = load_session(questions, dataset, load_questions(questions), corpus_fingerprint(chunks))
    assert [item.query_id for item in expanded.questions] == ["q1", "q2"]
    assert label_status(questions, dataset)["verified"] == 1
    session.select("q1", ["a"], allow_verified=True)
    assert not dataset.exists()
    session.verify("q1", confirmed=True)
    assert load_dataset(dataset)[0].relevant_chunk_ids == {"a"}


def test_candidate_union_and_corpus_search(tmp_path: Path):
    chunks, _, _, _, _ = fixture_files(tmp_path)
    dense = StaticRetriever({"query": ["a", "b"]})
    bm25 = StaticRetriever({"query": ["b", "a"]})
    found = discover_candidates("query", chunks, dense, bm25, 2)
    assert {item.chunk.chunk_id for item in found} == {"a", "b"}
    assert len(found[0].matches) == 3
    assert [item.chunk.chunk_id for item in search_corpus(chunks, "BICYCLE")] == ["b"]


def test_audit_and_review_export(tmp_path: Path):
    chunks, corpus, questions, dataset, manifest = fixture_files(tmp_path)
    state, progress = load_session(questions, dataset, load_questions(questions), corpus_fingerprint(chunks))
    session = LabelSession(state, progress, dataset, {"a", "b"})
    session.select("q1", ["a"])
    session.verify("q1", confirmed=True)
    report = audit_dataset(corpus, questions, dataset, manifest)
    assert report.passed
    assert report.reference_answer_count == 0
    assert report.source_pages == {"public.txt": [1]}
    review = tmp_path / "review.md"
    assert export_review(corpus, dataset, review) == 1
    assert "red apple" in review.read_text(encoding="utf-8")
    bad_manifest = CorpusManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
    bad_manifest.corpus_fingerprint = "different"
    manifest.write_text(bad_manifest.model_dump_json(), encoding="utf-8")
    assert not audit_dataset(corpus, questions, dataset, manifest).passed
    manifest.write_text(CorpusManifest.model_validate_json(manifest.read_text(encoding="utf-8")).model_copy(
        update={"corpus_fingerprint": corpus_fingerprint(chunks)}).model_dump_json(), encoding="utf-8")
    record = json.loads(dataset.read_text(encoding="utf-8"))
    record["reference_answer"] = "unconfirmed edit"
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert any("not confirmed" in error for error in audit_dataset(corpus, questions, dataset, manifest).errors)
    with pytest.raises(ValueError, match="different corpus fingerprint"):
        load_session(questions, dataset, load_questions(questions), "different")


def test_cli_label_and_status_with_stub_retriever(tmp_path: Path, monkeypatch):
    chunks, corpus, questions, dataset, _ = fixture_files(tmp_path)
    monkeypatch.setattr("groundtruth_rag.cli.DenseRetriever",
                        lambda: StaticRetriever({"What is red?": ["a", "b"]}))
    runner = CliRunner()
    result = runner.invoke(app, ["label", "--corpus", str(corpus), "--questions", str(questions),
                                 "--output", str(dataset), "--candidate-depth", "2"],
                           input="select 1\nconfirm\ny\n")
    assert result.exit_code == 0, result.output
    assert load_dataset(dataset)[0].relevant_chunk_ids == {"a"}
    status = runner.invoke(app, ["label-status", "--questions", str(questions),
                                 "--dataset", str(dataset)])
    assert "verified: 1" in status.output
    assert progress_path(questions).exists()


def test_benchmark_summary_uses_run_values():
    query = EvalQuery(query_id="q1", query="question", relevant_chunk_ids={"a"})
    rankings = {"dense": ["b", "a"], "bm25": ["a", "b"],
                "hybrid": ["a", "b"], "hybrid_rerank": ["a", "b"]}
    runs = {name: evaluate_retriever(StaticRetriever({"question": ids}), [query],
                                     dataset="synthetic", configuration_name=name,
                                     configuration={"corpus_fingerprint": "same"})
            for name, ids in rankings.items()}
    summary, markdown, analysis = summarize_runs(runs)
    assert summary["runs"]["dense"]["metrics"]["recall@1"] == 0
    assert summary["runs"]["bm25"]["metrics"]["recall@1"] == 1
    assert "| dense |" in markdown
    assert analysis["reranking_moves_first_relevant_up"] == []
    runs["bm25"].configuration["corpus_fingerprint"] = "other"
    with pytest.raises(ValueError, match="incompatible corpus"):
        summarize_runs(runs)


def test_status_skip_and_missing_osha_source(tmp_path: Path):
    chunks, corpus, questions, dataset, manifest = fixture_files(tmp_path)
    state, progress = load_session(questions, dataset, load_questions(questions), corpus_fingerprint(chunks))
    session = LabelSession(state, progress, dataset, {"a", "b"})
    session.skip("q1")
    status = label_status(questions, dataset)
    assert status["skipped"] == 1 and status["verified"] == 0
    with pytest.raises(ValueError, match="cannot read dataset"):
        audit_dataset(corpus, questions, dataset, manifest)
    result = CliRunner().invoke(app, ["prepare-osha", "--source", str(tmp_path / "missing.pdf")])
    assert result.exit_code != 0
    assert "OSHA PDF missing" in result.output
    blocked = CliRunner().invoke(app, ["benchmark-osha", "--corpus", str(corpus),
                                       "--questions", str(questions), "--dataset", str(dataset),
                                       "--manifest", str(manifest)])
    assert blocked.exit_code != 0
    assert "Benchmark blocked" in blocked.output


def test_prepare_osha_manifest_from_supplied_pdf(tmp_path: Path, monkeypatch):
    source = tmp_path / "raw" / "osha3120.pdf"
    source.parent.mkdir()
    source.write_bytes(b"synthetic PDF fixture, not an OSHA document")
    corpus = tmp_path / "osha" / "corpus.jsonl"
    manifest_path = tmp_path / "osha" / "corpus_manifest.json"

    class Page:
        def extract_text(self):
            return " ".join(f"word{number}" for number in range(160))

    class Reader:
        is_encrypted = False
        pages = [Page()]

    monkeypatch.setattr("groundtruth_rag.ingestion.loaders.PdfReader", lambda path: Reader())
    manifest = prepare_osha(source, corpus, manifest_path)
    assert (manifest.chunk_size, manifest.chunk_overlap) == (150, 30)
    assert manifest.source_sha256 == file_sha256(source)
    assert manifest.chunk_count == 2 and manifest.page_count == 1
    assert manifest.corpus_fingerprint == corpus_fingerprint(load_corpus(corpus))
    with pytest.raises(ValueError, match="refusing to regenerate"):
        prepare_osha(source, corpus, manifest_path)
