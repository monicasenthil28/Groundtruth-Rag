"""Pure retrieval metrics over ordered chunk IDs."""

import math
from collections.abc import Collection, Sequence


def _validate(relevant: Collection[str], retrieved: Sequence[str], k: int | None = None) -> None:
    if not relevant:
        raise ValueError("relevant chunk IDs must not be empty")
    if len(relevant) != len(set(relevant)):
        raise ValueError("relevant chunk IDs must be unique")
    if len(retrieved) != len(set(retrieved)):
        raise ValueError("retrieved chunk IDs must be unique")
    if k is not None and k <= 0:
        raise ValueError("k must be positive")


def recall_at_k(relevant: Collection[str], retrieved: Sequence[str], k: int) -> float:
    _validate(relevant, retrieved, k)
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def precision_at_k(relevant: Collection[str], retrieved: Sequence[str], k: int) -> float:
    _validate(relevant, retrieved, k)
    considered = retrieved[:k]
    return len(set(considered) & set(relevant)) / len(considered) if considered else 0.0


def hit_rate_at_k(relevant: Collection[str], retrieved: Sequence[str], k: int) -> float:
    _validate(relevant, retrieved, k)
    return float(bool(set(retrieved[:k]) & set(relevant)))


def reciprocal_rank(relevant: Collection[str], retrieved: Sequence[str]) -> float:
    _validate(relevant, retrieved)
    return next((1.0 / rank for rank, item in enumerate(retrieved, 1) if item in relevant), 0.0)


def mean_reciprocal_rank(scores: Sequence[float]) -> float:
    if not scores:
        raise ValueError("reciprocal rank scores must not be empty")
    if any(not math.isfinite(score) or score < 0 or score > 1 for score in scores):
        raise ValueError("reciprocal rank scores must be finite values from 0 to 1")
    return sum(scores) / len(scores)


def ndcg_at_k(relevant: Collection[str], retrieved: Sequence[str], k: int) -> float:
    _validate(relevant, retrieved, k)
    relevant_set = set(relevant)
    dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(retrieved[:k], 1) if item in relevant_set)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant_set)) + 1))
    return dcg / idcg


def query_metrics(relevant: Collection[str], retrieved: Sequence[str], k_values: Sequence[int]) -> dict[str, float]:
    values: dict[str, float] = {}
    for k in k_values:
        values[f"recall@{k}"] = recall_at_k(relevant, retrieved, k)
        values[f"precision@{k}"] = precision_at_k(relevant, retrieved, k)
        values[f"hit_rate@{k}"] = hit_rate_at_k(relevant, retrieved, k)
        values[f"ndcg@{k}"] = ndcg_at_k(relevant, retrieved, k)
    values["reciprocal_rank"] = reciprocal_rank(relevant, retrieved)
    return values
