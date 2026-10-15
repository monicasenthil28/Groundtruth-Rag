# Architecture

RAG teams frequently change chunking, embeddings, retrieval methods, and rerankers without knowing whether retrieval quality improved. Groundtruth-Rag runs repeatable evaluations against a human-verified golden dataset. The framework is domain agnostic; the OSHA Lockout/Tagout corpus is the first local benchmark, not embedded in the core.

```text
Public Source
      |
      v
SHA-256 Verification
      |
      v
Deterministic Corpus Build
      |
      v
Committed Golden Labels
      |
      v
Benchmark Runner
      |
      v
Baseline / Regression Policy
      |
      v
PASS / WARN / FAIL
```

## Responsibilities

- **Ingestion (implemented):** PDF, UTF-8 text, and Markdown loaders preserve document source and PDF page numbers. Deterministic word chunking produces stable IDs and position metadata; versioned corpus JSONL stores the resulting chunks.
- **Retrieval (implemented):** Dense, BM25, Hybrid RRF, and Hybrid + CrossEncoder implement `Retriever.index(chunks)` and `retrieve(query, k)`. Dense uses in-memory normalized vectors and an ignored local embedding cache. All pipelines return the same `RetrievalHit` contract; the evaluation runner has no retriever-specific path.
- **Human labeling (tooling implemented):** Separate draft questions carry no relevance IDs. A resumable terminal review shows Dense, BM25, and Hybrid candidates plus corpus search. Explicit human confirmation writes strict golden `EvalQuery` records. Audit checks progress, provenance, coverage, and fixed-corpus compatibility. Local OSHA labels have 29 verified queries and one reviewed skip.
- **Evaluation:** The runner matches `EvalQuery.relevant_chunk_ids` to ranked chunk IDs. Recall@K, Precision@K, Hit Rate@K, reciprocal rank, MRR, and binary nDCG@K are implemented as pure functions. Default K values are 1, 3, 5, and 10. Invalid K, empty relevance, and duplicate retrieved IDs are rejected. Precision divides by the number of returned hits considered. The `RetrievalMetric` abstraction remains available for future metrics.
- **Generation evaluation:** Optional answer-level assessment, separate from retrieval metrics. Planned metrics are faithfulness and answer relevance; the current `GenerationMetric` is an interface only.
- **Reporting:** The runner emits versioned JSON with per-query hits, metrics, latency, and run configuration. The aggregate baseline contains no source text or relevance labels. Regression reports contain metric deltas, compatibility checks, and PASS/WARN/FAIL decisions.
- **Regression testing:** Compare a candidate summary to a versioned baseline after corpus, dataset, query-count, retriever, metric, and K compatibility checks. Quality uses absolute-drop thresholds; latency uses independently configurable relative-increase thresholds. No composite score is used.

The CLI provides `--help`, `version`, `ingest`, `prepare-osha`, `label`, `label-status`, `audit-dataset`, `export-review`, `benchmark-setup`, `benchmark-status`, `benchmark-osha`, `baseline-create`, `regress`, `validate-dataset`, `evaluate`, `evaluate-demo`, and `compare`. `benchmark-osha` refuses to run until the pinned source, corpus, committed golden dataset, and baseline are compatible. The older `--questions` path still audits local review progress when explicitly supplied. `evaluate-demo` uses fixed synthetic rankings only. Generation evaluation remains future work.

Golden queries are stored as JSONL, one `EvalQuery` per line. Raw records are checked before Pydantic converts relevance IDs to a set, so duplicate IDs cannot disappear during parsing. The demo corpus and rankings exist solely to exercise the engine offline. `schema_version` in run JSON makes future format migration explicit.

Chunk-level relevance IDs are tied to one chunking configuration. Keep chunk size and overlap fixed in comparisons until source-span judgments or relabeling are available. See [retrieval design](RETRIEVAL.md).
