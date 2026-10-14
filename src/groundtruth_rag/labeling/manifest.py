"""Canonical source and corpus provenance for fixed-chunk labeling."""

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from groundtruth_rag.ingestion.corpus import CORPUS_SCHEMA_VERSION, load_corpus
from groundtruth_rag.ingestion.pipeline import ingest
from groundtruth_rag.labeling.models import atomic_write
from groundtruth_rag.retrieval.dense import corpus_fingerprint

INGESTION_VERSION = "word-v1"


class CorpusManifest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    corpus_name: str
    source_document: str
    source_relative_path: str
    source_sha256: str
    ingested_at: datetime
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    document_count: int = Field(gt=0)
    page_count: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    corpus_fingerprint: str
    ingestion_version: str
    corpus_schema_version: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_osha(source: Path, corpus_path: Path, manifest_path: Path) -> CorpusManifest:
    if not source.is_file():
        raise ValueError(f"OSHA PDF missing: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError("OSHA source must be the supplied PDF")
    if corpus_path.exists() or manifest_path.exists():
        raise ValueError("canonical corpus or manifest already exists; refusing to regenerate")
    if source.resolve() in {corpus_path.resolve(), manifest_path.resolve()}:
        raise ValueError("output must not overwrite the source PDF")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(source)
    summary = ingest(source, corpus_path, 150, 30)
    chunks = load_corpus(corpus_path)
    manifest = CorpusManifest(
        corpus_name="OSHA 3120 Lockout/Tagout", source_document=source.name,
        source_relative_path=os.path.relpath(source.resolve(), corpus_path.parent.resolve()),
        source_sha256=source_hash, ingested_at=datetime.now(timezone.utc),
        chunk_size=150, chunk_overlap=30, document_count=summary.documents,
        page_count=summary.pages, chunk_count=summary.chunks,
        corpus_fingerprint=corpus_fingerprint(chunks),
        ingestion_version=INGESTION_VERSION, corpus_schema_version=CORPUS_SCHEMA_VERSION,
    )
    atomic_write(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    return manifest
