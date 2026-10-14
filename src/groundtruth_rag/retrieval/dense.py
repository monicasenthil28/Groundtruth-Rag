"""In-memory normalized dot-product retrieval with an on-disk embedding cache."""

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from zipfile import BadZipFile
from pathlib import Path
from typing import Any

import numpy as np

from groundtruth_rag.retrieval.base import Retriever
from groundtruth_rag.retrieval.common import ranked_indices, validate_chunks, validate_k
from groundtruth_rag.schemas import DocumentChunk, RetrievalHit

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def corpus_fingerprint(chunks: list[DocumentChunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(json.dumps([chunk.chunk_id, chunk.text], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class DenseRetriever(Retriever):
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 32,
                 cache_dir: Path | None = None, encoder: Any = None) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model
        self.batch_size = batch_size
        self.cache_dir = cache_dir or Path(os.environ.get("GROUNDTRUTH_CACHE_DIR") or ".cache/groundtruth-rag")
        self._encoder = encoder
        self._chunks: list[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None

    def _model(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Dense retrieval requires: pip install -e '.[models]'") from exc
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    @staticmethod
    def _normalized(values: Any, expected_rows: int) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1 and expected_rows == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
            raise ValueError("encoder returned an unexpected embedding shape")
        if not np.isfinite(matrix).all():
            raise ValueError("encoder returned non-finite embeddings")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("encoder returned a zero embedding")
        return matrix / norms

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        self._embeddings = None
        self._chunks = validate_chunks(chunks)
        encoder = self._model()
        cache_key = hashlib.sha256(json.dumps({
            "model": self.model_name, "corpus": corpus_fingerprint(self._chunks),
            "normalization": "l2-float32-v1",
        }, sort_keys=True).encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.npz"
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    matrix = self._normalized(cached["embeddings"], len(self._chunks))
                self._embeddings = matrix
                return
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                pass
        values = encoder.encode(
            [chunk.text for chunk in self._chunks], batch_size=self.batch_size,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )
        matrix = self._normalized(values, len(self._chunks))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, embeddings=matrix)
            temporary.replace(cache_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self._embeddings = matrix

    def retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        validate_k(k)
        if self._embeddings is None:
            raise ValueError("dense retriever must be indexed before retrieval")
        query_vector = self._normalized(self._model().encode(
            [query], batch_size=1, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ), 1)[0]
        if query_vector.shape[0] != self._embeddings.shape[1]:
            raise ValueError("query embedding dimension differs from corpus")
        scores = self._embeddings @ query_vector
        indices = ranked_indices(scores, self._chunks, k)
        return [RetrievalHit(chunk_id=self._chunks[i].chunk_id, score=float(scores[i]),
                             rank=rank, retriever="dense") for rank, i in enumerate(indices, 1)]
