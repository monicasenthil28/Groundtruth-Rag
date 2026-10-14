# Groundtruth-Rag

## RAG Retrieval Evaluation & Regression Platform

Groundtruth-Rag is a domain-agnostic foundation for comparing retrieval architectures against a human-verified golden dataset. It is intended to make changes to chunking, embeddings, retrieval, and reranking measurable instead of relying on subjective chatbot testing.

The supported comparison is **Dense** vs **Dense + BM25** vs **Dense + BM25 + CrossEncoder**. BM25 is also available as a standalone configuration. No real benchmark has been run yet.

## Current Scope

- Typed contracts for chunks, evaluation queries, hits, and retrieval runs
- Abstract retriever and separate retrieval/generation metric interfaces
- Deterministic Recall@K, Precision@K, Hit Rate@K, reciprocal rank, MRR, and binary nDCG@K
- Retrieval evaluation runner with per-query results, aggregate metrics, and mean/p50/p95 retrieval latency
- JSONL dataset validation, versioned JSON run results, and metric-by-metric comparison
- A fixed-ranking, synthetic demo for offline evaluation; it is not semantic search or a benchmark
- PDF, text, and Markdown ingestion with deterministic word chunking and versioned corpus JSONL
- Dense, BM25, Hybrid RRF, and Hybrid + CrossEncoder retrievers
- Package and test foundation for Python 3.11+

See [architecture](docs/ARCHITECTURE.md) for the planned evaluation flow.

## Development

```bash
python -m pip install -e ".[dev,models]"
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

## Ingest And Evaluate

```bash
groundtruth ingest --input path/to/documents --output datasets/corpus.jsonl --chunk-size 150 --chunk-overlap 30
groundtruth evaluate --corpus datasets/corpus.jsonl --dataset datasets/eval.jsonl --config configs/dense.yaml --output reports/dense.json
groundtruth evaluate --corpus datasets/corpus.jsonl --dataset datasets/eval.jsonl --config configs/hybrid_rerank.yaml --output reports/hybrid_rerank.json
groundtruth compare reports/dense.json reports/hybrid_rerank.json
```

`ingest` accepts one `.pdf`, `.txt`, or `.md` file, or a directory of those files. PDF page numbers are preserved. Chunk size and overlap are word counts. Each corpus JSONL line contains `schema_version: "1.0"` and a `DocumentChunk`. Create output parent directories before running commands.

## Retrieval Configurations

`configs/dense.yaml` uses normalized MiniLM embeddings and cosine-equivalent dot product. `configs/bm25.yaml` uses lowercase regex word tokens. `configs/hybrid.yaml` combines Dense and BM25 rankings with Reciprocal Rank Fusion (RRF). `configs/hybrid_rerank.yaml` applies a CrossEncoder to a bounded hybrid candidate pool. The model package is installed with the `models` extra; normal tests use fakes and do not download models.

Raw BM25 and cosine scores have different scales, so averaging them would make the blend arbitrary. RRF combines rank positions instead. See [retrieval design](docs/RETRIEVAL.md) for cache, candidate depth, and latency details.

## Golden Dataset

One UTF-8 JSON object per line. Required fields are `query_id` (unique non-empty string), `query` (non-empty string), and `relevant_chunk_ids` (non-empty array of unique chunk ID strings). Optional fields are `reference_answer` (string or null) and `tags` (array of unique strings). Blank lines and duplicate IDs are rejected.

```json
{"query_id":"q001","query":"Example question","relevant_chunk_ids":["doc1:p3:c2"],"tags":["definition"]}
```

Retrieval evaluation uses human relevance judgments and ordered chunk IDs. Generation evaluation remains a separate, unimplemented interface; no answer-quality claims are produced. `datasets/demo/` contains four synthetic chunks, three queries, and predetermined rankings for tests.

**Keep chunking fixed across compared runs.** Changing chunk size or overlap changes chunk boundaries and IDs, invalidating current chunk-level relevance labels. Relabel or add source-span judgments before comparing different chunking strategies.

## Creating a Golden Dataset

1. Place the exact source PDF at `datasets/raw/osha3120.pdf`. Raw files are ignored by Git. Run `groundtruth prepare-osha` once to produce the local `datasets/osha/corpus.jsonl` with fixed 150-word chunks and 30-word overlap, plus `datasets/osha/corpus_manifest.json` with source and corpus hashes. The command refuses to replace an existing canonical corpus.
2. Review or append answerable, unlabeled questions in `datasets/osha/questions_draft.jsonl`. Do not put `relevant_chunk_ids` in this file. The ten initial questions are prompts for human review, not labels.
3. Run `groundtruth label --corpus datasets/osha/corpus.jsonl --questions datasets/osha/questions_draft.jsonl --output datasets/osha/eval.jsonl`. Candidate discovery unions Dense, BM25, and Hybrid results. Use `more`, `search <phrase>`, or `inspect <chunk ID>` to check source passages. Select one or several IDs and explicitly confirm each answer. Progress is saved beside the draft file and resumes after interruption.
4. Run `groundtruth label-status --questions datasets/osha/questions_draft.jsonl --dataset datasets/osha/eval.jsonl`, then `groundtruth audit-dataset --corpus datasets/osha/corpus.jsonl --questions datasets/osha/questions_draft.jsonl --dataset datasets/osha/eval.jsonl`. Optional review copy: `groundtruth export-review --corpus datasets/osha/corpus.jsonl --dataset datasets/osha/eval.jsonl --output reports/tmp/osha-review.md`.
5. After at least 25 answerable questions are explicitly verified and audit passes, run `groundtruth benchmark-osha`. It evaluates Dense, BM25, Hybrid, and Hybrid + CrossEncoder on the same corpus and labels. Run JSON, summary, and ranking-only error analysis go to `reports/benchmarks/osha/`.

The OSHA corpus and in-progress/final local labels are ignored to keep source text and unfinished work out of tooling commits. A later, separate data commit may include human-verified labels after review. See [golden dataset guidance](docs/GOLDEN_DATASET.md).

## OSHA Benchmark

OSHA benchmark pending human verification. The exact source PDF is not present in this checkout, and no OSHA corpus or final labels have been generated.

## Benchmark Results

Benchmark pending — results will be generated from the real evaluation dataset.
