"""Local execution smoke for all real retrievers; not a benchmark."""

from pathlib import Path

from groundtruth_rag.retrieval.factory import build_retriever, load_config
from groundtruth_rag.schemas import DocumentChunk


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    corpus = root / "datasets" / "demo" / "chunks.jsonl"
    chunks = [DocumentChunk.model_validate_json(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    query = "Which object has wheels?"
    for name in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        retriever = build_retriever(load_config(root / "configs" / f"{name}.yaml"))
        retriever.index(chunks)
        hits = retriever.retrieve(query, 2)
        print(f"{name}: {[hit.chunk_id for hit in hits]}")


if __name__ == "__main__":
    main()
