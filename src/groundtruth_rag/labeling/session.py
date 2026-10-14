"""Resumable state transitions with explicit human confirmation."""

from pathlib import Path
from datetime import datetime, timezone

from groundtruth_rag.datasets import DatasetValidationError, load_dataset
from groundtruth_rag.labeling.models import (
    LabelRecord, LabelState, LabelingQuestion, expected_dataset, new_state,
    progress_path, save_state,
)


def load_session(questions_path: Path, dataset_path: Path,
                 questions: list[LabelingQuestion], fingerprint: str,
                 *, reconcile_output: bool = False) -> tuple[LabelState, Path]:
    progress = progress_path(questions_path)
    if progress.exists():
        state = LabelState.model_validate_json(progress.read_text(encoding="utf-8"))
        if state.corpus_fingerprint != fingerprint:
            raise ValueError("label progress belongs to a different corpus fingerprint")
        saved = [(item.query_id, item.query) for item in state.questions]
        current = [(item.query_id, item.query) for item in questions]
        if current[:len(saved)] != saved:
            raise ValueError("draft questions changed since label progress was saved")
        if len(current) > len(saved):
            state.questions.extend(LabelRecord.model_validate(item.model_dump())
                                   for item in questions[len(saved):])
        expected = expected_dataset(state)
        if dataset_path.exists():
            invalid_output = False
            try:
                actual = load_dataset(dataset_path)
            except DatasetValidationError:
                if not reconcile_output:
                    raise ValueError("golden dataset is invalid; use --reconcile-output after review") from None
                invalid_output = True
                actual = []
            if invalid_output or {item.query_id: item for item in actual} != {
                item.query_id: item for item in expected
            }:
                if not reconcile_output:
                    raise ValueError("golden dataset differs from progress; use --reconcile-output after review")
                save_state(state, progress, dataset_path)
        elif expected:
            save_state(state, progress, dataset_path)
        return state, progress
    if dataset_path.exists():
        raise ValueError("golden dataset exists without progress; refusing to overwrite it")
    return new_state(questions, fingerprint), progress


class LabelSession:
    def __init__(self, state: LabelState, progress: Path, dataset: Path,
                 valid_chunk_ids: set[str]) -> None:
        self.state = state
        self.progress = progress
        self.dataset = dataset
        self.valid_chunk_ids = valid_chunk_ids

    def _record(self, query_id: str) -> LabelRecord:
        return next(item for item in self.state.questions if item.query_id == query_id)

    def _editable(self, record: LabelRecord, allow_verified: bool) -> None:
        if record.status == "verified" and not allow_verified:
            raise ValueError("verified label requires explicit edit confirmation")

    def select(self, query_id: str, chunk_ids: list[str], *, allow_verified: bool = False) -> None:
        record = self._record(query_id)
        self._editable(record, allow_verified)
        if not chunk_ids or len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("select one or more unique chunk IDs")
        unknown = set(chunk_ids) - self.valid_chunk_ids
        if unknown:
            raise ValueError(f"chunk IDs absent from corpus: {', '.join(sorted(unknown))}")
        record.selected_chunk_ids = chunk_ids
        record.status = "in_progress"
        record.verified_at = None
        self.persist()

    def edit(self, query_id: str, *, answer: str | None = None, notes: str | None = None,
             tags: list[str] | None = None, allow_verified: bool = False) -> None:
        record = self._record(query_id)
        self._editable(record, allow_verified)
        if tags is not None:
            if len(tags) != len(set(tags)) or any(not tag.strip() for tag in tags):
                raise ValueError("tags must be unique non-empty strings")
            record.tags = tags
        if answer is not None:
            record.reference_answer = answer.strip() or None
        if notes is not None:
            record.notes = notes.strip() or None
        record.status = "in_progress"
        record.verified_at = None
        self.persist()

    def skip(self, query_id: str, *, allow_verified: bool = False) -> None:
        record = self._record(query_id)
        self._editable(record, allow_verified)
        record.status = "skipped"
        record.selected_chunk_ids = []
        record.verified_at = None
        self.persist()

    def verify(self, query_id: str, *, confirmed: bool) -> None:
        if not confirmed:
            raise ValueError("human confirmation is required")
        record = self._record(query_id)
        if not record.selected_chunk_ids:
            raise ValueError("select at least one relevant chunk before verification")
        if set(record.selected_chunk_ids) - self.valid_chunk_ids:
            raise ValueError("selected chunks are absent from corpus")
        record.status = "verified"
        record.verified_at = datetime.now(timezone.utc)
        self.persist()

    def persist(self) -> None:
        save_state(self.state, self.progress, self.dataset)
