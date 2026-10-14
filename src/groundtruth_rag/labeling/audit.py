"""Audit a human-labeled dataset against its fixed corpus and progress state."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from groundtruth_rag.datasets import load_dataset
from groundtruth_rag.ingestion.corpus import load_corpus
from groundtruth_rag.labeling.manifest import CorpusManifest, file_sha256
from groundtruth_rag.labeling.models import LabelState, load_questions, progress_path
from groundtruth_rag.retrieval.dense import corpus_fingerprint


@dataclass
class AuditReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    question_count: int = 0
    verified_count: int = 0
    reference_answer_count: int = 0
    tag_counts: dict[str, int] = field(default_factory=dict)
    source_pages: dict[str, list[int | None]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors


def audit_dataset(corpus_path: Path, questions_path: Path, dataset_path: Path,
                  manifest_path: Path) -> AuditReport:
    chunks = load_corpus(corpus_path)
    questions = load_questions(questions_path)
    labels = load_dataset(dataset_path)
    report = AuditReport(question_count=len(questions), verified_count=len(labels))
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    fingerprint = corpus_fingerprint(chunks)
    if not manifest_path.exists():
        report.errors.append(f"corpus manifest missing: {manifest_path}")
    else:
        manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.corpus_fingerprint != fingerprint or manifest.chunk_count != len(chunks):
            report.errors.append("corpus fingerprint or chunk count differs from manifest")
        if manifest.corpus_name.startswith("OSHA") and (manifest.chunk_size, manifest.chunk_overlap) != (150, 30):
            report.errors.append("OSHA corpus must use 150-word chunks with 30-word overlap")
        if manifest.corpus_name.startswith("OSHA"):
            source = (corpus_path.parent / manifest.source_relative_path).resolve()
            if not source.is_file():
                report.errors.append(f"raw OSHA PDF missing: {source}")
            elif file_sha256(source) != manifest.source_sha256:
                report.errors.append("raw OSHA PDF hash differs from manifest")

    progress = progress_path(questions_path)
    state = None
    if not progress.exists():
        report.errors.append(f"human verification progress missing: {progress}")
    else:
        state = LabelState.model_validate_json(progress.read_text(encoding="utf-8"))
        if state.corpus_fingerprint != fingerprint:
            report.errors.append("label progress belongs to a different corpus fingerprint")

    draft_by_id = {item.query_id: item for item in questions}
    labels_by_id = {item.query_id: item for item in labels}
    if set(labels_by_id) != set(draft_by_id):
        report.errors.append("not every draft question has a verified golden label")
    progress_by_id = {item.query_id: item for item in state.questions} if state else {}
    if state and set(progress_by_id) != set(draft_by_id):
        report.errors.append("label progress does not match draft questions")

    pages: dict[str, set[int | None]] = {}
    tags: Counter[str] = Counter()
    for label in labels:
        draft = draft_by_id.get(label.query_id)
        if draft and label.query != draft.query:
            report.errors.append(f"{label.query_id}: query text differs from draft")
        record = progress_by_id.get(label.query_id)
        if record is None or record.status != "verified" or record.to_eval_query() != label:
            report.errors.append(f"{label.query_id}: not confirmed in human label progress")
        unknown = label.relevant_chunk_ids - set(by_id)
        if unknown:
            report.errors.append(f"{label.query_id}: orphaned chunk IDs: {', '.join(sorted(unknown))}")
        if len(label.relevant_chunk_ids) > 5:
            report.warnings.append(f"{label.query_id}: {len(label.relevant_chunk_ids)} relevant chunks; review breadth")
        for chunk_id in label.relevant_chunk_ids - unknown:
            chunk = by_id[chunk_id]
            pages.setdefault(chunk.source, set()).add(chunk.page)
        tags.update(label.tags)
        report.reference_answer_count += bool(label.reference_answer)
    report.tag_counts = dict(sorted(tags.items()))
    report.source_pages = {source: sorted(values, key=lambda page: (-1 if page is None else page))
                           for source, values in sorted(pages.items())}
    return report
