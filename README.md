# Groundtruth-Rag

## RAG Retrieval Evaluation & Regression Platform

Groundtruth-Rag is a domain-agnostic foundation for comparing retrieval architectures against a human-verified golden dataset. It is intended to make changes to chunking, embeddings, retrieval, and reranking measurable instead of relying on subjective chatbot testing.

The planned comparison is **Dense** vs **Dense + BM25** vs **Dense + BM25 + CrossEncoder**. Those real retrievers are not implemented yet.

## Current Scope

- Typed contracts for chunks, evaluation queries, hits, and retrieval runs
- Abstract retriever and separate retrieval/generation metric interfaces
- Deterministic Recall@K, Precision@K, Hit Rate@K, reciprocal rank, MRR, and binary nDCG@K
- Retrieval evaluation runner with per-query results, aggregate metrics, and mean/p50/p95 retrieval latency
- JSONL dataset validation, versioned JSON run results, and metric-by-metric comparison
- A fixed-ranking, synthetic demo for offline evaluation; it is not semantic search or a benchmark
- Package and test foundation for Python 3.11+

See [architecture](docs/ARCHITECTURE.md) for the planned evaluation flow.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall src
groundtruth --help
groundtruth version
groundtruth validate-dataset datasets/demo/queries.jsonl
mkdir -p reports/tmp
groundtruth evaluate-demo reports/tmp/baseline.json
groundtruth evaluate-demo reports/tmp/candidate.json
groundtruth compare reports/tmp/baseline.json reports/tmp/candidate.json
```

These two demo runs contain identical fixed rankings; their latency will vary. Temporary results in `reports/tmp/` are ignored by Git.

## Golden Dataset

One UTF-8 JSON object per line. Required fields are `query_id` (unique non-empty string), `query` (non-empty string), and `relevant_chunk_ids` (non-empty array of unique chunk ID strings). Optional fields are `reference_answer` (string or null) and `tags` (array of unique strings). Blank lines and duplicate IDs are rejected.

```json
{"query_id":"q001","query":"Example question","relevant_chunk_ids":["doc1:p3:c2"],"tags":["definition"]}
```

Retrieval evaluation uses human relevance judgments and ordered chunk IDs. Generation evaluation remains a separate, unimplemented interface; no answer-quality claims are produced. `datasets/demo/` contains four synthetic chunks, three queries, and predetermined rankings for tests.

The optional `benchmark` dependency group records libraries planned for future retrieval, ingestion, and reporting work. It is not needed to exercise the current contracts.

## Benchmark Results

Benchmark pending — results will be generated from the real evaluation dataset.
