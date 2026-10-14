# Architecture

RAG teams frequently change chunking, embeddings, retrieval methods, and rerankers without knowing whether retrieval quality improved. Groundtruth-Rag will run repeatable evaluations against a human-verified golden dataset. The framework is domain agnostic; an OSHA Lockout/Tagout corpus is planned as the first real benchmark, not embedded in the core.

```text
Document Corpus
      |
      v
Chunking
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

- **Ingestion:** Read source documents and produce stable `DocumentChunk` records with provenance. No reader or chunker exists yet.
- **Retrieval:** Implement `Retriever.index(chunks)` and `retrieve(query, k)`. Dense, hybrid, and reranked pipelines will share this contract. Scores may differ by retriever; rank and chunk ID provide the common comparison surface.
- **Evaluation:** The runner matches `EvalQuery.relevant_chunk_ids` to ranked chunk IDs. Recall@K, Precision@K, Hit Rate@K, reciprocal rank, MRR, and binary nDCG@K are implemented as pure functions. Default K values are 1, 3, 5, and 10. Invalid K, empty relevance, and duplicate retrieved IDs are rejected. Precision divides by the number of returned hits considered. The `RetrievalMetric` abstraction remains available for future metrics.
- **Generation evaluation:** Optional answer-level assessment, separate from retrieval metrics. Planned metrics are faithfulness and answer relevance; the current `GenerationMetric` is an interface only.
- **Reporting:** The runner emits versioned JSON with per-query hits, metrics, latency, and run configuration. Comparison accepts compatible dataset identifiers and K values, reporting each metric and mean/p50/p95 latency separately. No composite score or regression gate exists yet.
- **Regression testing:** Compare a candidate run to an established baseline and apply explicit thresholds. No gate is active until real benchmark results exist.

The CLI provides `--help`, `version`, `validate-dataset`, `evaluate-demo`, and `compare`. `evaluate-demo` uses fixed synthetic rankings only. Future real-retriever commands include `ingest`, `evaluate`, and `report`; none are exposed until implemented.

Golden queries are stored as JSONL, one `EvalQuery` per line. Raw records are checked before Pydantic converts relevance IDs to a set, so duplicate IDs cannot disappear during parsing. The demo corpus and rankings exist solely to exercise the engine offline. `schema_version` in run JSON makes future format migration explicit.
