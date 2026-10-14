"""Versioned, model-independent corpus JSONL."""

import json
from pathlib import Path

from pydantic import ValidationError

from groundtruth_rag.schemas import DocumentChunk

CORPUS_SCHEMA_VERSION = "1.0"
_CHUNK_FIELDS = set(DocumentChunk.model_fields)


def write_corpus(path: Path, chunks: list[DocumentChunk]) -> None:
    if not chunks:
        raise ValueError("cannot write an empty corpus")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("corpus contains duplicate chunk IDs")
    lines = [json.dumps({"schema_version": CORPUS_SCHEMA_VERSION, **chunk.model_dump()},
                        ensure_ascii=False, sort_keys=True) for chunk in chunks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_corpus(path: Path) -> list[DocumentChunk]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{path}: cannot read corpus: {exc}") from exc
    if not lines:
        raise ValueError(f"{path}: corpus is empty")
    chunks: list[DocumentChunk] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, 1):
        location = f"{path}:{number}"
        if not line.strip():
            raise ValueError(f"{location}: blank JSONL line")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{location}: expected a JSON object")
        if record.pop("schema_version", None) != CORPUS_SCHEMA_VERSION:
            raise ValueError(f"{location}: unsupported or missing corpus schema_version")
        if set(record) != _CHUNK_FIELDS:
            raise ValueError(f"{location}: chunk fields do not match corpus schema")
        try:
            chunk = DocumentChunk.model_validate(record, strict=True)
        except ValidationError as exc:
            raise ValueError(f"{location}: {exc.errors()[0]['msg']}") from exc
        if chunk.chunk_id in seen:
            raise ValueError(f"{location}: duplicate chunk_id {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        chunks.append(chunk)
    return chunks
