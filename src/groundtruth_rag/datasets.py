"""Strict JSONL golden dataset loading and validation."""

import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from groundtruth_rag.schemas import EvalQuery


class DatasetValidationError(ValueError):
    pass


def load_dataset(path: Path) -> list[EvalQuery]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetValidationError(f"{path}: cannot read dataset: {exc.strerror}") from exc
    if not lines:
        raise DatasetValidationError(f"{path}: dataset is empty")

    queries: list[EvalQuery] = []
    seen: set[str] = set()
    allowed = {"query_id", "query", "relevant_chunk_ids", "reference_answer", "tags"}
    for line_number, line in enumerate(lines, 1):
        location = f"{path}:{line_number}"
        if not line.strip():
            raise DatasetValidationError(f"{location}: blank JSONL line")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"{location}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise DatasetValidationError(f"{location}: expected a JSON object")
        unknown = set(record) - allowed
        if unknown:
            raise DatasetValidationError(f"{location}: unknown fields: {', '.join(sorted(unknown))}")
        query_id = record.get("query_id")
        location = f"{location} (query_id={query_id!r})"
        for field in ("query_id", "query"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise DatasetValidationError(f"{location}: {field} must be a non-empty string")
        ids = record.get("relevant_chunk_ids")
        if not isinstance(ids, list) or not ids:
            raise DatasetValidationError(f"{location}: relevant_chunk_ids must be a non-empty list")
        if any(not isinstance(item, str) or not item.strip() for item in ids):
            raise DatasetValidationError(f"{location}: relevant_chunk_ids must contain non-empty strings")
        if len(ids) != len(set(ids)):
            raise DatasetValidationError(f"{location}: duplicate relevant chunk IDs")
        tags = record.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise DatasetValidationError(f"{location}: tags must be a list of non-empty strings")
        if len(tags) != len(set(tags)):
            raise DatasetValidationError(f"{location}: duplicate tags")
        answer = record.get("reference_answer")
        if answer is not None and not isinstance(answer, str):
            raise DatasetValidationError(f"{location}: reference_answer must be a string or null")
        if query_id in seen:
            raise DatasetValidationError(f"{location}: duplicate query_id")
        try:
            query = EvalQuery.model_validate(record)
        except ValidationError as exc:
            raise DatasetValidationError(f"{location}: {exc.errors()[0]['msg']}") from exc
        seen.add(query_id)
        queries.append(query)
    return queries


def tag_counts(queries: list[EvalQuery]) -> dict[str, int]:
    return dict(sorted(Counter(tag for query in queries for tag in query.tags).items()))
