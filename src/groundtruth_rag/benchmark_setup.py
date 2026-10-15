"""Reproduce a fixed public-source benchmark without versioning source text."""

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundtruth_rag.datasets import load_dataset
from groundtruth_rag.ingestion.corpus import CORPUS_SCHEMA_VERSION, load_corpus
from groundtruth_rag.ingestion.pipeline import ingest
from groundtruth_rag.labeling.manifest import INGESTION_VERSION, CorpusManifest, file_sha256
from groundtruth_rag.labeling.models import atomic_write
from groundtruth_rag.reporting.regression import load_baseline
from groundtruth_rag.retrieval.factory import load_config
from groundtruth_rag.retrieval.dense import corpus_fingerprint


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceSpec(StrictModel):
    document: str
    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(gt=0)
    page_count: int = Field(gt=0)

    @model_validator(mode="after")
    def official_source(self) -> "SourceSpec":
        if self.document != "osha3120.pdf" or self.url != (
            "https://www.osha.gov/sites/default/files/publications/osha3120.pdf"
        ):
            raise ValueError("OSHA source must be the configured official publication")
        return self


class CorpusSpec(StrictModel):
    chunk_size_words: int = Field(gt=0)
    chunk_overlap_words: int = Field(ge=0)
    chunk_count: int = Field(gt=0)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_overlap(self) -> "CorpusSpec":
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class GoldenSpec(StrictModel):
    dataset: str
    review_manifest: str
    total_questions: int = Field(gt=0)
    verified_queries: int = Field(gt=0)
    skipped_ids: list[str]


class BenchmarkSpec(StrictModel):
    schema_version: Literal["1.0"]
    benchmark_id: Literal["osha"]
    title: str
    source: SourceSpec
    corpus: CorpusSpec
    golden: GoldenSpec
    k_values: list[int]
    retriever_configs: dict[str, str]
    baseline: str

    @model_validator(mode="after")
    def valid_scope(self) -> "BenchmarkSpec":
        if self.golden.total_questions != self.golden.verified_queries + len(self.golden.skipped_ids):
            raise ValueError("question counts and skipped IDs disagree")
        if len(set(self.golden.skipped_ids)) != len(self.golden.skipped_ids):
            raise ValueError("duplicate skipped IDs")
        if not self.k_values or any(k <= 0 for k in self.k_values) or len(set(self.k_values)) != len(self.k_values):
            raise ValueError("K values must be unique and positive")
        if set(self.retriever_configs) != {"dense", "bm25", "hybrid", "hybrid_rerank"}:
            raise ValueError("four canonical retriever configurations are required")
        return self


class ReviewManifest(StrictModel):
    schema_version: Literal["1.0"]
    total_questions: int
    verified: int
    skipped: int
    skipped_ids: list[str]
    corpus_fingerprint: str
    dataset_sha256: str


def load_spec(path: Path) -> BenchmarkSpec:
    return BenchmarkSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def benchmark_paths(root: Path) -> tuple[Path, Path, Path]:
    return (root / "datasets/raw/osha3120.pdf", root / "datasets/osha/corpus.jsonl",
            root / "datasets/osha/corpus_manifest.json")


def _check_source(source: Path, spec: BenchmarkSpec) -> None:
    if not source.is_file():
        raise ValueError(f"source missing: {source}")
    actual = file_sha256(source)
    if actual != spec.source.sha256:
        raise ValueError(f"source SHA-256 mismatch: expected {spec.source.sha256}, got {actual}")


def _check_corpus(corpus: Path, manifest_path: Path, spec: BenchmarkSpec) -> CorpusManifest:
    if not corpus.is_file() or not manifest_path.is_file():
        raise ValueError("corpus or manifest missing")
    chunks = load_corpus(corpus)
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    document_count = len({chunk.source for chunk in chunks})
    fingerprint = corpus_fingerprint(chunks)
    for label, actual, expected in (
        ("documents", document_count, spec.source.document_count),
        ("chunks", len(chunks), spec.corpus.chunk_count),
        ("corpus fingerprint", fingerprint, spec.corpus.fingerprint),
        ("manifest fingerprint", manifest.corpus_fingerprint, spec.corpus.fingerprint),
        ("manifest chunks", manifest.chunk_count, spec.corpus.chunk_count),
        ("manifest pages", manifest.page_count, spec.source.page_count),
        ("manifest documents", manifest.document_count, spec.source.document_count),
        ("manifest source hash", manifest.source_sha256, spec.source.sha256),
        ("chunk size", manifest.chunk_size, spec.corpus.chunk_size_words),
        ("chunk overlap", manifest.chunk_overlap, spec.corpus.chunk_overlap_words),
    ):
        if actual != expected:
            raise ValueError(f"{label} mismatch: expected {expected}, got {actual}")
    if manifest.source_document != spec.source.document:
        raise ValueError("manifest source document mismatch")
    return manifest


def verify_benchmark_package(root: Path, spec: BenchmarkSpec) -> int:
    """Check committed benchmark inputs without source, corpus, network, or models."""
    dataset_path = root / spec.golden.dataset
    review_path = root / spec.golden.review_manifest
    labels = load_dataset(dataset_path)
    review = ReviewManifest.model_validate_json(review_path.read_text(encoding="utf-8"))
    for label, actual, expected in (
        ("golden query count", len(labels), spec.golden.verified_queries),
        ("reviewed questions", review.total_questions, spec.golden.total_questions),
        ("verified questions", review.verified, spec.golden.verified_queries),
        ("skipped questions", review.skipped, len(spec.golden.skipped_ids)),
        ("skipped IDs", review.skipped_ids, spec.golden.skipped_ids),
        ("review corpus fingerprint", review.corpus_fingerprint, spec.corpus.fingerprint),
        ("golden dataset hash", file_sha256(dataset_path), review.dataset_sha256),
    ):
        if actual != expected:
            raise ValueError(f"{label} mismatch: expected {expected}, got {actual}")
    for label in labels:
        if label.query_id in spec.golden.skipped_ids:
            raise ValueError(f"skipped query in golden dataset: {label.query_id}")
        if not label.reference_answer or not label.reference_answer.strip() or not label.tags:
            raise ValueError(f"{label.query_id}: reference answer and tags required")
    baseline = load_baseline(root / spec.baseline)
    if (baseline.query_count != len(labels) or baseline.corpus_fingerprint != spec.corpus.fingerprint
            or baseline.dataset_sha256 != review.dataset_sha256 or baseline.k_values != spec.k_values
            or baseline.dataset_name != dataset_path.name or set(baseline.runs) != set(spec.retriever_configs)):
        raise ValueError("baseline incompatible with benchmark specification or golden dataset")
    for name, relative_path in spec.retriever_configs.items():
        config = load_config(root / relative_path)
        if config.retriever.type != name:
            raise ValueError(f"{name}: retriever config type mismatch")
    return len(labels)


def validate_golden(root: Path, spec: BenchmarkSpec, corpus: Path) -> int:
    count = verify_benchmark_package(root, spec)
    chunk_ids = {chunk.chunk_id for chunk in load_corpus(corpus)}
    for label in load_dataset(root / spec.golden.dataset):
        unknown = label.relevant_chunk_ids - chunk_ids
        if unknown:
            raise ValueError(f"{label.query_id}: missing relevant chunks: {', '.join(sorted(unknown))}")
    return count


def setup_benchmark(root: Path, spec: BenchmarkSpec, supplied_source: Path | None = None,
                    download=None) -> bool:
    source, corpus, manifest = benchmark_paths(root)
    if supplied_source is not None:
        _check_source(supplied_source, spec)
    if source.exists():
        _check_source(source, spec)
    else:
        source.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=source.parent) as temp_dir:
            temporary = Path(temp_dir) / spec.source.document
            if supplied_source is not None:
                shutil.copyfile(supplied_source, temporary)
            else:
                opener = download or urlopen
                with opener(spec.source.url) as response, temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
            _check_source(temporary, spec)
            temporary.replace(source)
    if corpus.exists() or manifest.exists():
        _check_corpus(corpus, manifest, spec)
        validate_golden(root, spec, corpus)
        return False
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=corpus.parent) as temp_dir:
        temp_corpus = Path(temp_dir) / "corpus.jsonl"
        summary = ingest(source, temp_corpus, spec.corpus.chunk_size_words,
                         spec.corpus.chunk_overlap_words)
        chunks = load_corpus(temp_corpus)
        fingerprint = corpus_fingerprint(chunks)
        for label, actual, expected in (
            ("documents", summary.documents, spec.source.document_count),
            ("pages", summary.pages, spec.source.page_count),
            ("chunks", summary.chunks, spec.corpus.chunk_count),
            ("corpus fingerprint", fingerprint, spec.corpus.fingerprint),
        ):
            if actual != expected:
                raise ValueError(f"{label} mismatch: expected {expected}, got {actual}")
        validate_golden(root, spec, temp_corpus)
        provenance = CorpusManifest(
            corpus_name=spec.title, source_document=spec.source.document,
            source_relative_path=os.path.relpath(source.resolve(), corpus.parent.resolve()),
            source_sha256=spec.source.sha256, ingested_at=datetime.now(timezone.utc),
            chunk_size=spec.corpus.chunk_size_words, chunk_overlap=spec.corpus.chunk_overlap_words,
            document_count=summary.documents, page_count=summary.pages, chunk_count=summary.chunks,
            corpus_fingerprint=fingerprint, ingestion_version=INGESTION_VERSION,
            corpus_schema_version=CORPUS_SCHEMA_VERSION,
        )
        temp_corpus.replace(corpus)
        atomic_write(manifest, provenance.model_dump_json(indent=2) + "\n")
    return True


def benchmark_status(root: Path, spec: BenchmarkSpec) -> dict:
    source, corpus, manifest = benchmark_paths(root)
    result = {"source": "missing", "corpus": "missing", "golden": "unavailable",
              "baseline": "unavailable", "ready": False, "errors": []}
    if source.exists():
        try:
            _check_source(source, spec)
            result["source"] = "PASS"
        except (OSError, ValueError) as exc:
            result["source"] = "FAIL"
            result["errors"].append(str(exc))
    if corpus.exists() or manifest.exists():
        try:
            _check_corpus(corpus, manifest, spec)
            result["corpus"] = "PASS"
        except (OSError, ValueError) as exc:
            result["corpus"] = "FAIL"
            result["errors"].append(str(exc))
    if result["corpus"] == "PASS":
        try:
            result["query_count"] = validate_golden(root, spec, corpus)
            result["golden"] = "PASS"
            result["baseline"] = "PASS"
        except (OSError, ValueError) as exc:
            result["golden"] = "FAIL"
            result["errors"].append(str(exc))
    result["ready"] = all(result[key] == "PASS" for key in ("source", "corpus", "golden", "baseline"))
    return result
