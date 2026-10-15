"""Offline checks for the versioned OSHA benchmark package."""

import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from groundtruth_rag.benchmark_setup import (
    BenchmarkSpec, SourceHashMismatchError, SourceUnavailableError, benchmark_paths,
    benchmark_status, load_spec, setup_benchmark, setup_with_release_fallback,
    validate_golden,
)
from groundtruth_rag.cli import app
from groundtruth_rag.datasets import DatasetValidationError, load_dataset
from groundtruth_rag.ingestion.corpus import load_corpus
from groundtruth_rag.ingestion.pipeline import ingest
from groundtruth_rag.labeling.manifest import file_sha256
from groundtruth_rag.retrieval.dense import corpus_fingerprint


@pytest.fixture
def package(tmp_path: Path, monkeypatch):
    source = tmp_path / "input" / "osha3120.pdf"
    source.parent.mkdir()
    source.write_bytes(b"fixed synthetic PDF bytes")

    class Page:
        def extract_text(self):
            return " ".join(f"word{number}" for number in range(160))

    class Reader:
        is_encrypted = False
        pages = [Page()]

    monkeypatch.setattr("groundtruth_rag.ingestion.loaders.PdfReader", lambda path: Reader())
    scratch = tmp_path / "scratch" / "corpus.jsonl"
    scratch.parent.mkdir()
    ingest(source, scratch, 150, 30)
    chunks = load_corpus(scratch)
    root = tmp_path / "clone"
    golden = root / "benchmarks/osha"
    golden.mkdir(parents=True)
    records = [
        {"query_id": f"q{number:03d}", "query": f"Question {number}?",
         "relevant_chunk_ids": [chunks[0].chunk_id],
         "reference_answer": "Grounded answer", "tags": ["policy"]}
        for number in range(1, 31) if number != 13
    ]
    dataset = golden / "eval.jsonl"
    dataset.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    digest = file_sha256(dataset)
    review = {"schema_version": "1.0", "total_questions": 30, "verified": 29,
              "skipped": 1, "skipped_ids": ["q013"],
              "corpus_fingerprint": corpus_fingerprint(chunks), "dataset_sha256": digest}
    (golden / "review_manifest.json").write_text(json.dumps(review), encoding="utf-8")
    baseline_path = root / "baselines/osha/v1.json"
    baseline_path.parent.mkdir(parents=True)
    baseline = {"schema_version": "1.0", "name": "test", "domain": "osha",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_summary_schema_version": "1.0",
                "corpus_fingerprint": corpus_fingerprint(chunks),
                "dataset_sha256": digest, "dataset_name": "eval.jsonl", "query_count": 29,
                "k_values": [1, 3, 5, 10],
                "runs": {name: {"run_id": f"test-{name}", "metrics": {"mrr": 1.0},
                               "latency": {"mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 1.0}}
                         for name in ("dense", "bm25", "hybrid", "hybrid_rerank")}}
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    data = {"schema_version": "1.0", "benchmark_id": "osha", "title": "OSHA 3120 Lockout/Tagout",
            "source": {"document": "osha3120.pdf",
                       "url": "https://www.osha.gov/sites/default/files/publications/osha3120.pdf",
                       "sha256": file_sha256(source), "document_count": 1, "page_count": 1},
            "corpus": {"chunk_size_words": 150, "chunk_overlap_words": 30,
                       "chunk_count": 2, "fingerprint": corpus_fingerprint(chunks)},
            "golden": {"dataset": "benchmarks/osha/eval.jsonl",
                       "review_manifest": "benchmarks/osha/review_manifest.json",
                       "total_questions": 30, "verified_queries": 29, "skipped_ids": ["q013"]},
            "k_values": [1, 3, 5, 10],
            "retriever_configs": {name: f"configs/{name}.yaml"
                                  for name in ("dense", "bm25", "hybrid", "hybrid_rerank")},
            "baseline": "baselines/osha/v1.json"}
    spec_path = golden / "benchmark.yaml"
    spec_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    configs = root / "configs"
    configs.mkdir()
    for name in data["retriever_configs"]:
        (configs / f"{name}.yaml").write_text(yaml.safe_dump({"name": name,
                                                                "retriever": {"type": name}}),
                                                  encoding="utf-8")
    return root, source, data, spec_path


def test_spec_parsing_and_schema(package):
    _, _, data, path = package
    assert load_spec(path).corpus.chunk_size_words == 150
    data["schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        BenchmarkSpec.model_validate(data)


def test_spec_rejects_wrong_source_and_overlap(package):
    _, _, data, _ = package
    data["source"]["url"] = "https://example.invalid/osha3120.pdf"
    with pytest.raises(ValidationError, match="official publication"):
        BenchmarkSpec.model_validate(data)
    data["source"]["url"] = "https://www.osha.gov/sites/default/files/publications/osha3120.pdf"
    data["corpus"]["chunk_overlap_words"] = 150
    with pytest.raises(ValidationError, match="overlap"):
        BenchmarkSpec.model_validate(data)


def test_source_copy_hash_and_reuse(package):
    root, source, _, path = package
    spec = load_spec(path)
    assert setup_benchmark(root, spec, source)
    canonical, corpus, manifest = benchmark_paths(root)
    assert file_sha256(canonical) == spec.source.sha256
    assert corpus_fingerprint(load_corpus(corpus)) == spec.corpus.fingerprint
    assert manifest.exists()
    before = corpus.read_bytes()
    assert not setup_benchmark(root, spec, source)
    assert corpus.read_bytes() == before
    assert benchmark_status(root, spec)["ready"]


def test_mock_download(package):
    root, source, _, path = package
    spec = load_spec(path)
    assert setup_benchmark(root, spec, download=lambda url: io.BytesIO(source.read_bytes()))
    assert benchmark_status(root, spec)["ready"]


def test_official_source_succeeds_without_release(package):
    root, source, _, path = package
    spec = load_spec(path)

    def no_release(directory):
        pytest.fail("release fallback must not run after official success")

    built, outcome = setup_with_release_fallback(
        root, spec, official_download=lambda url: io.BytesIO(source.read_bytes()),
        release_download=no_release,
    )
    assert built and outcome == "OFFICIAL_SOURCE_OK"
    assert benchmark_status(root, spec)["ready"]


def test_existing_verified_source_does_not_claim_official_download(package):
    root, source, _, path = package
    spec = load_spec(path)
    setup_benchmark(root, spec, source)

    def no_download(url):
        pytest.fail("existing verified source must be reused")

    built, outcome = setup_with_release_fallback(root, spec, official_download=no_download)
    assert not built and outcome == "VERIFIED_LOCAL_SOURCE_OK"


def test_official_unavailable_uses_verified_release(package):
    root, source, _, path = package
    spec = load_spec(path)

    def unavailable(url):
        raise OSError("HTTP Error 403: Forbidden")

    def release(directory):
        (directory / "osha3120.pdf").write_bytes(source.read_bytes())

    built, outcome = setup_with_release_fallback(
        root, spec, official_download=unavailable, release_download=release,
    )
    assert built and outcome == "RELEASE_FALLBACK_OK"
    assert file_sha256(benchmark_paths(root)[0]) == spec.source.sha256
    assert benchmark_status(root, spec)["ready"]


def test_both_transports_unavailable_stops_before_corpus(package):
    root, _, _, path = package
    spec = load_spec(path)

    def unavailable(url):
        raise OSError("HTTP Error 403: Forbidden")

    def missing_release(directory):
        raise OSError("release asset unavailable")

    with pytest.raises(SourceUnavailableError, match="Release asset unavailable"):
        setup_with_release_fallback(root, spec, official_download=unavailable,
                                    release_download=missing_release)
    source, corpus, manifest = benchmark_paths(root)
    assert not source.exists() and not corpus.exists() and not manifest.exists()
    assert not benchmark_status(root, spec)["ready"]


def test_fallback_wrong_sha_stops_before_corpus(package):
    root, _, _, path = package
    spec = load_spec(path)

    def unavailable(url):
        raise OSError("HTTP Error 403: Forbidden")

    def bad_release(directory):
        (directory / "osha3120.pdf").write_bytes(b"wrong PDF")

    with pytest.raises(SourceHashMismatchError, match="source SHA-256 mismatch"):
        setup_with_release_fallback(root, spec, official_download=unavailable,
                                    release_download=bad_release)
    assert not benchmark_paths(root)[1].exists()


def test_official_wrong_sha_does_not_try_release(package):
    root, _, _, path = package
    spec = load_spec(path)

    def no_release(directory):
        pytest.fail("hash mismatch must not trigger release fallback")

    with pytest.raises(SourceHashMismatchError, match="source SHA-256 mismatch"):
        setup_with_release_fallback(root, spec,
                                    official_download=lambda url: io.BytesIO(b"wrong PDF"),
                                    release_download=no_release)
    assert not benchmark_paths(root)[1].exists()


def test_default_release_download_uses_pinned_asset(monkeypatch, tmp_path: Path):
    from groundtruth_rag.benchmark_setup import _download_release_asset

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr("groundtruth_rag.benchmark_setup.subprocess.run", fake_run)
    _download_release_asset(tmp_path)
    command, options = calls[0]
    assert command[:3] == ["gh", "release", "download"]
    assert "benchmark-source-v1" in command
    assert "monicasenthil28/Groundtruth-Rag" in command
    assert "osha3120.pdf" in command
    assert options["check"] is True


def test_cli_source_failure_prevents_benchmark_setup(package, monkeypatch):
    from groundtruth_rag.cli import app

    def unavailable(root, spec):
        raise SourceUnavailableError("official and release unavailable")

    monkeypatch.setattr("groundtruth_rag.cli.setup_with_release_fallback", unavailable)
    result = CliRunner().invoke(app, ["benchmark-setup", "osha", "--release-fallback"])
    assert result.exit_code == 1
    assert "SOURCE_UNAVAILABLE" in result.output
    assert not benchmark_paths(package[0])[1].exists()


def test_wrong_supplied_source_and_existing_source(package):
    root, source, _, path = package
    spec = load_spec(path)
    bad = source.parent / "bad.pdf"
    bad.write_bytes(b"wrong PDF")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        setup_benchmark(root, spec, bad)
    assert not benchmark_paths(root)[0].exists()
    canonical = benchmark_paths(root)[0]
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"wrong PDF")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        setup_benchmark(root, spec)


@pytest.mark.parametrize("field,word", [
    ("page_count", "pages mismatch"),
    ("chunk_count", "chunks mismatch"),
    ("fingerprint", "corpus fingerprint mismatch"),
])
def test_rebuild_mismatch_stops_before_corpus_write(package, field, word):
    root, source, data, _ = package
    parent = "source" if field == "page_count" else "corpus"
    data[parent][field] = (data[parent][field] + 1 if field != "fingerprint" else "0" * 64)
    spec = BenchmarkSpec.model_validate(data)
    with pytest.raises(ValueError, match=word):
        setup_benchmark(root, spec, source)
    assert not benchmark_paths(root)[1].exists()


def test_golden_queries_and_skipped_id(package):
    root, source, _, path = package
    spec = load_spec(path)
    setup_benchmark(root, spec, source)
    labels = load_dataset(root / spec.golden.dataset)
    assert len(labels) == validate_golden(root, spec, benchmark_paths(root)[1]) == 29
    assert "q013" not in {label.query_id for label in labels}


def test_missing_chunk_rejected(package):
    root, source, _, path = package
    spec = load_spec(path)
    setup_benchmark(root, spec, source)
    dataset = root / spec.golden.dataset
    lines = dataset.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["relevant_chunk_ids"] = ["missing"]
    lines[0] = json.dumps(row)
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = root / spec.golden.review_manifest
    data = json.loads(review.read_text(encoding="utf-8"))
    data["dataset_sha256"] = file_sha256(dataset)
    review.write_text(json.dumps(data), encoding="utf-8")
    baseline = root / spec.baseline
    baseline_data = json.loads(baseline.read_text(encoding="utf-8"))
    baseline_data["dataset_sha256"] = data["dataset_sha256"]
    baseline.write_text(json.dumps(baseline_data), encoding="utf-8")
    with pytest.raises(ValueError, match="missing relevant chunks"):
        validate_golden(root, spec, benchmark_paths(root)[1])


def test_duplicate_query_rejected(package):
    root, _, _, path = package
    spec = load_spec(path)
    dataset = root / spec.golden.dataset
    lines = dataset.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[0]
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="duplicate query_id"):
        load_dataset(dataset)


def test_duplicate_relevance_id_rejected(package):
    root, _, _, path = package
    spec = load_spec(path)
    dataset = root / spec.golden.dataset
    lines = dataset.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["relevant_chunk_ids"] *= 2
    lines[0] = json.dumps(row)
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="duplicate relevant chunk IDs"):
        load_dataset(dataset)


def test_skipped_query_rejected_even_with_matching_review_hash(package):
    root, source, _, path = package
    spec = load_spec(path)
    setup_benchmark(root, spec, source)
    dataset = root / spec.golden.dataset
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    rows[12]["query_id"] = "q013"
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    review = root / spec.golden.review_manifest
    data = json.loads(review.read_text(encoding="utf-8"))
    data["dataset_sha256"] = file_sha256(dataset)
    review.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="skipped query"):
        validate_golden(root, spec, benchmark_paths(root)[1])


def test_review_hash_mismatch_rejected(package):
    root, source, _, path = package
    spec = load_spec(path)
    setup_benchmark(root, spec, source)
    review = root / spec.golden.review_manifest
    data = json.loads(review.read_text(encoding="utf-8"))
    data["dataset_sha256"] = "0" * 64
    review.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="golden dataset hash mismatch"):
        validate_golden(root, spec, benchmark_paths(root)[1])


def test_baseline_mismatch_rejected(package):
    root, source, _, path = package
    spec = load_spec(path)
    setup_benchmark(root, spec, source)
    baseline = root / spec.baseline
    data = json.loads(baseline.read_text(encoding="utf-8"))
    data["query_count"] = 28
    baseline.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline incompatible"):
        validate_golden(root, spec, benchmark_paths(root)[1])


def test_status_missing_source_and_incompatible_corpus(package):
    root, source, _, path = package
    spec = load_spec(path)
    missing = benchmark_status(root, spec)
    assert missing["source"] == "missing" and not missing["ready"]
    setup_benchmark(root, spec, source)
    corpus = benchmark_paths(root)[1]
    corpus.write_text(corpus.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    bad = benchmark_status(root, spec)
    assert bad["corpus"] == "FAIL" and not bad["ready"]


def test_status_reports_wrong_manifest_page_count(package):
    root, source, _, path = package
    spec = load_spec(path)
    setup_benchmark(root, spec, source)
    manifest = benchmark_paths(root)[2]
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["page_count"] = 2
    manifest.write_text(json.dumps(data), encoding="utf-8")
    status = benchmark_status(root, spec)
    assert status["corpus"] == "FAIL"
    assert any("manifest pages mismatch" in error for error in status["errors"])


def test_cli_status_ready_and_missing(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("groundtruth_rag.cli.inspect_benchmark", lambda root, spec: {
        "source": "PASS", "corpus": "PASS", "golden": "PASS", "baseline": "PASS",
        "ready": True, "errors": []})
    ready = runner.invoke(app, ["benchmark-status", "osha"])
    assert ready.exit_code == 0 and "Ready to benchmark: YES" in ready.output
    monkeypatch.setattr("groundtruth_rag.cli.inspect_benchmark", lambda root, spec: {
        "source": "missing", "corpus": "missing", "golden": "unavailable",
        "baseline": "unavailable", "ready": False, "errors": []})
    missing = runner.invoke(app, ["benchmark-status", "osha"])
    assert missing.exit_code == 0 and "Ready to benchmark: NO" in missing.output


def test_cli_setup_reuses_existing(monkeypatch):
    monkeypatch.setattr("groundtruth_rag.cli.setup_benchmark", lambda root, spec, source: False)
    result = CliRunner().invoke(app, ["benchmark-setup", "osha"])
    assert result.exit_code == 0 and "corpus reused" in result.output


def test_real_committed_dataset_matches_baseline():
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / "benchmarks/osha/benchmark.yaml")
    labels = load_dataset(root / spec.golden.dataset)
    assert len(labels) == 29
    assert "q013" not in {label.query_id for label in labels}
    assert file_sha256(root / spec.golden.dataset) == (
        "78dd4d33573577c9ce5b285699afea6c91ad6bfcce2499f8a629960cb058f85e"
    )


def test_local_pdf_clean_room(tmp_path: Path):
    original = Path(__file__).resolve().parents[1]
    source = original / "datasets/raw/osha3120.pdf"
    if not source.exists():
        pytest.skip("local OSHA PDF is not available")
    spec = load_spec(original / "benchmarks/osha/benchmark.yaml")
    for relative in (spec.golden.dataset, spec.golden.review_manifest, spec.baseline,
                     *spec.retriever_configs.values()):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, target)
    assert setup_benchmark(tmp_path, spec, source)
    assert benchmark_status(tmp_path, spec)["ready"]
    assert len(load_corpus(benchmark_paths(tmp_path)[1])) == 91
