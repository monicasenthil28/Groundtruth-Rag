"""Progress summaries and a readable passage review export."""

from collections import Counter
from pathlib import Path

from groundtruth_rag.datasets import load_dataset
from groundtruth_rag.ingestion.corpus import load_corpus
from groundtruth_rag.labeling.models import LabelState, atomic_write, load_questions, progress_path


def label_status(questions_path: Path, dataset_path: Path) -> dict[str, int | float]:
    questions = load_questions(questions_path)
    progress = progress_path(questions_path)
    if progress.exists():
        state = LabelState.model_validate_json(progress.read_text(encoding="utf-8"))
        saved = {item.query_id: item for item in state.questions}
        if not set(saved).issubset({item.query_id for item in questions}):
            raise ValueError("progress contains questions absent from draft")
        counts = Counter(saved[item.query_id].status if item.query_id in saved else "unlabeled"
                         for item in questions)
    else:
        if dataset_path.exists():
            raise ValueError("golden dataset exists without human verification progress")
        counts = Counter({"unlabeled": len(questions)})
    verified = counts["verified"]
    if dataset_path.exists():
        labels = load_dataset(dataset_path)
        if len(labels) != verified:
            raise ValueError("golden dataset count differs from verified progress")
    elif verified:
        raise ValueError("verified progress exists but golden dataset is missing")
    total = len(questions)
    return {"total": total, "verified": verified, "unlabeled": counts["unlabeled"],
            "in_progress": counts["in_progress"], "skipped": counts["skipped"],
            "completion_pct": round(verified / total * 100, 1)}


def export_review(corpus_path: Path, dataset_path: Path, output_path: Path) -> int:
    chunks = {chunk.chunk_id: chunk for chunk in load_corpus(corpus_path)}
    labels = load_dataset(dataset_path)
    lines = ["# Golden Dataset Review", "", "Human review copy. Verify each cited passage against the source document.", ""]
    for label in labels:
        lines.extend([f"## {label.query_id}: {label.query}", "",
                      f"Reference answer: {label.reference_answer or '(none)'}", ""])
        for chunk_id in sorted(label.relevant_chunk_ids):
            if chunk_id not in chunks:
                raise ValueError(f"{label.query_id}: orphaned chunk ID {chunk_id}")
            chunk = chunks[chunk_id]
            lines.extend([f"### {chunk.source}, page {chunk.page or 'n/a'}", "",
                          f"Chunk ID: `{chunk_id}`", "", chunk.text, ""])
    atomic_write(output_path, "\n".join(lines) + "\n")
    return len(labels)
