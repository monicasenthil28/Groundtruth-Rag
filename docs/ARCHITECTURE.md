# Architecture

RAG teams frequently change chunking, embeddings, retrieval methods, and rerankers without knowing whether retrieval quality improved. Groundtruth-Rag will run repeatable evaluations against a human-verified golden dataset. The framework is domain agnostic; an OSHA Lockout/Tagout corpus is planned as the first real benchmark, not embedded in the core.

```text
Document Corpus
      |
      v
Chunking
      |
      v
Human Golden Labels
      |
      v
Retriever Configuration
      |
      v
Evaluation Runner
      |
      v
Retrieval Metrics
      |
      v
Optional Generation Evaluation
      |
      v
Benchmark Comparison
      |
      v
Regression Gate
```

## Responsibilities

- **Ingestion (implemented):** PDF, UTF-8 text, and Markdown loaders preserve document source and PDF page numbers. Deterministic word chunking produces stable IDs and position metadata; versioned corpus JSONL stores the resulting chunks.
- **Retrieval (implemented):** Dense, BM25, Hybrid RRF, and Hybrid + CrossEncoder implement `Retriever.index(chunks)` and `retrieve(query, k)`. Dense uses in-memory normalized vectors and an ignored local embedding cache. All pipelines return the same `RetrievalHit` contract; the evaluation runner has no retriever-specific path.
- **Human labeling (tooling implemented):** Separate draft questions carry no relevance IDs. A resumable terminal review shows Dense, BM25, and Hybrid candidates plus corpus search. Explicit human confirmation writes strict golden `EvalQuery` records. Audit checks progress, provenance, coverage, and fixed-corpus compatibility. OSHA source PDF and human labels are still pending.
- **Evaluation:** The runner matches `EvalQuery.relevant_chunk_ids` to ranked chunk IDs. Recall@K, Precision@K, Hit Rate@K, reciprocal rank, MRR, and binary nDCG@K are implemented as pure functions. Default K values are 1, 3, 5, and 10. Invalid K, empty relevance, and duplicate retrieved IDs are rejected. Precision divides by the number of returned hits considered. The `RetrievalMetric` abstraction remains available for future metrics.
- **Generation evaluation:** Optional answer-level assessment, separate from retrieval metrics. Planned metrics are faithfulness and answer relevance; the current `GenerationMetric` is an interface only.
- **Reporting:** The runner emits versioned JSON with per-query hits, metrics, latency, and run configuration. Comparison accepts compatible dataset identifiers and K values, reporting each metric and mean/p50/p95 latency separately. No composite score or regression gate exists yet.
- **Regression testing:** Compare a candidate run to an established baseline and apply explicit thresholds. No gate is active until real benchmark results exist.

The CLI provides `--help`, `version`, `ingest`, `prepare-osha`, `label`, `label-status`, `audit-dataset`, `export-review`, `benchmark-osha`, `validate-dataset`, `evaluate`, `evaluate-demo`, and `compare`. `benchmark-osha` refuses to run until audit passes and at least 25 questions are human verified. `evaluate-demo` uses fixed synthetic rankings only. Generation evaluation and regression gates remain future work.

Golden queries are stored as JSONL, one `EvalQuery` per line. Raw records are checked before Pydantic converts relevance IDs to a set, so duplicate IDs cannot disappear during parsing. The demo corpus and rankings exist solely to exercise the engine offline. `schema_version` in run JSON makes future format migration explicit.

Chunk-level relevance IDs are tied to one chunking configuration. Keep chunk size and overlap fixed in comparisons until source-span judgments or relabeling are available. See [retrieval design](RETRIEVAL.md).
