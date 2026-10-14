"""Validated YAML configuration and retriever construction."""

from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from groundtruth_rag.retrieval.base import Retriever
from groundtruth_rag.retrieval.bm25 import BM25Retriever
from groundtruth_rag.retrieval.dense import DEFAULT_MODEL, DenseRetriever
from groundtruth_rag.retrieval.hybrid import HybridRetriever
from groundtruth_rag.retrieval.rerank import DEFAULT_RERANKER_MODEL, RerankedHybridRetriever


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DenseSettings(StrictConfig):
    type: Literal["dense"]
    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    batch_size: int = Field(default=32, gt=0)


class BM25Settings(StrictConfig):
    type: Literal["bm25"]


class HybridSettings(StrictConfig):
    type: Literal["hybrid"]
    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    batch_size: int = Field(default=32, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    dense_candidates: int = Field(default=20, gt=0)
    bm25_candidates: int = Field(default=20, gt=0)


class RerankSettings(HybridSettings):
    type: Literal["hybrid_rerank"] = "hybrid_rerank"
    candidate_pool: int = Field(default=20, gt=0)
    reranker_model: str = Field(default=DEFAULT_RERANKER_MODEL, min_length=1)


RetrieverSettings = Annotated[
    Union[DenseSettings, BM25Settings, HybridSettings, RerankSettings],
    Field(discriminator="type"),
]


class RunConfig(StrictConfig):
    name: str = Field(min_length=1)
    retriever: RetrieverSettings


def load_config(path: Path) -> RunConfig:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    return RunConfig.model_validate(content)


def build_retriever(config: RunConfig, *, encoder: Any = None, reranker: Any = None,
                    cache_dir: Path | None = None) -> Retriever:
    settings = config.retriever
    if isinstance(settings, BM25Settings):
        return BM25Retriever()
    dense = DenseRetriever(settings.model, settings.batch_size, cache_dir, encoder)
    if isinstance(settings, DenseSettings):
        return dense
    hybrid = HybridRetriever(dense, BM25Retriever(), settings.rrf_k,
                             settings.dense_candidates, settings.bm25_candidates)
    if isinstance(settings, RerankSettings):
        return RerankedHybridRetriever(hybrid, settings.candidate_pool,
                                       settings.reranker_model, reranker)
    return hybrid
