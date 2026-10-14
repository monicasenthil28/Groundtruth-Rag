"""Draft questions and persisted human review state."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundtruth_rag.schemas import EvalQuery

Status = Literal["unlabeled", "in_progress", "skipped", "verified"]
STATE_VERSION = "1.0"


class LabelingQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    notes: str | None = None
    status: Status = "unlabeled"

    @model_validator(mode="after")
    def validate_content(self) -> "LabelingQuestion":
        if not self.query_id.strip() or not self.query.strip():
            raise ValueError("query_id and query must not be blank")
        if any(not tag.strip() for tag in self.tags) or len(self.tags) != len(set(self.tags)):
            raise ValueError("tags must be unique non-empty strings")
        return self


class LabelRecord(LabelingQuestion):
    selected_chunk_ids: list[str] = Field(default_factory=list)
    verified_at: datetime | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> "LabelRecord":
        if len(self.selected_chunk_ids) != len(set(self.selected_chunk_ids)):
            raise ValueError("selected chunk IDs must be unique")
        if self.status == "verified" and not self.selected_chunk_ids:
            raise ValueError("verified questions require relevant chunks")
        if self.status == "verified" and self.verified_at is None:
            raise ValueError("verified questions require confirmation time")
        return self

    def to_eval_query(self) -> EvalQuery:
        if self.status != "verified":
            raise ValueError("only verified labels belong in the golden dataset")
        return EvalQuery(query_id=self.query_id, query=self.query,
                         relevant_chunk_ids=set(self.selected_chunk_ids),
                         reference_answer=self.reference_answer, tags=set(self.tags))


class LabelState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = STATE_VERSION
    corpus_fingerprint: str
    questions: list[LabelRecord]

    @model_validator(mode="after")
    def validate_ids(self) -> "LabelState":
        ids = [item.query_id for item in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("label progress contains duplicate query IDs")
        return self


def load_questions(path: Path) -> list[LabelingQuestion]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read draft questions {path}: {exc}") from exc
    if not lines:
        raise ValueError(f"draft questions are empty: {path}")
    questions: list[LabelingQuestion] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank line")
        try:
            question = LabelingQuestion.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if question.query_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate query_id {question.query_id}")
        if question.status != "unlabeled":
            raise ValueError(f"{path}:{line_number}: draft question must start unlabeled")
        seen.add(question.query_id)
        questions.append(question)
    return questions


def progress_path(questions_path: Path) -> Path:
    return questions_path.with_name(f"{questions_path.stem}.progress.json")


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def expected_dataset(state: LabelState) -> list[EvalQuery]:
    return [record.to_eval_query() for record in state.questions if record.status == "verified"]


def save_state(state: LabelState, progress: Path, dataset: Path) -> None:
    atomic_write(progress, state.model_dump_json(indent=2) + "\n")
    verified = expected_dataset(state)
    if verified:
        lines = [json.dumps({"query_id": item.query_id, "query": item.query,
                             "relevant_chunk_ids": sorted(item.relevant_chunk_ids),
                             "reference_answer": item.reference_answer, "tags": sorted(item.tags)},
                            ensure_ascii=False) for item in verified]
        atomic_write(dataset, "\n".join(lines) + "\n")
    elif dataset.exists():
        dataset.unlink()


def new_state(questions: list[LabelingQuestion], fingerprint: str) -> LabelState:
    return LabelState(corpus_fingerprint=fingerprint,
                      questions=[LabelRecord.model_validate(question.model_dump()) for question in questions])
